"""Residue templates from the PDB Chemical Component Dictionary (CCD).

This module reads CCD ``.cif`` component files into ``ResidueTemplate``
objects that the ``Protein`` recipe uses to match PDB residues by atom
name and to stamp chemistry (bonds, bond orders, formal charges) onto
mBuild Compounds.

The data model follows the practice of residue-template readers,
inspired by and verified to be compatible with openff-pablo
(https://github.com/openforcefield/openff-pablo), so that a protein
loaded into and modified by mBuild can be exported to software that can
assign parameters. In particular:

- Each atom carries a ``leaving`` flag. Leaving atoms are the atoms that
  are absent from a PDB file when the corresponding inter-residue bond
  (peptide bond, crosslink) exists.
- Protonation variants are generated the way such readers expect:
  removing an acidic proton decrements the formal charge of its heavy
  atom; adding a proton to a basic atom increments it.

The bundled ``.cif`` files in ``mbuild/lib/biomolecules/ccd_cache`` are
unmodified CCD component files (public domain), copied from the set that
openff-pablo vendors.

Attribution: the ``_ACIDIC_PROTONS``, ``_BASIC_ATOMS``, and
``_ATOM_NAME_SYNONYMS`` tables and the protonation-variant, caps, and
histidine-tautomer patch semantics are derived from openff-pablo
(https://github.com/openforcefield/openff-pablo, MIT license,
Copyright (c) 2025 Ashley Mitchell). They are mirrored here, rather than
imported, so that mBuild carries no OpenFF dependency. If a shared,
dependency-free residue-definition package emerges upstream, this
mirror should be replaced by it. ``test_biopolymers.py`` contains a
parity test that runs whenever openff-pablo is importable, to catch
drift between the two. The full openff-pablo license notice is below.
"""

# Portions of this module are derived from openff-pablo
# (https://github.com/openforcefield/openff-pablo), which carries this
# license:
#
# MIT License
#
# Copyright (c) 2025 Ashley Mitchell
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
import itertools
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path

from mbuild.exceptions import MBuildError

logger = logging.getLogger(__name__)

__all__ = ["AtomTemplate", "BondTemplate", "ResidueTemplate", "CCDLibrary"]

CCD_CACHE_DIR = Path(__file__).parent.parent / "lib" / "biomolecules" / "ccd_cache"

#: Directory where downloaded CCD definitions are stored and found
#: again in later sessions.
USER_CCD_CACHE_DIR = Path.home() / ".mbuild" / "ccd_cache"

RCSB_CCD_URL = "https://files.rcsb.org/ligands/download/{}.cif"

#: CCD component types that link into a polymer via the peptide bond.
_PEPTIDE_LINKING_TYPES = {
    "L-PEPTIDE LINKING",
    "PEPTIDE LINKING",
    "D-PEPTIDE LINKING",
}

_CIF_BOND_ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3}

#: Acidic protons per residue, following openff-pablo's ACIDIC_PROTONS.
#: Removing one produces a deprotonated variant (heavy atom charge -1).
_ACIDIC_PROTONS = {
    "ALA": ["HXT", "H2"],
    "ARG": ["HXT", "H2", "HH12", "HH22"],
    "ASN": ["HXT", "H2"],
    "ASP": ["HXT", "H2", "HD2"],
    "CYS": ["HXT", "H2", "HG"],
    "GLN": ["HXT", "H2"],
    "GLU": ["HXT", "H2", "HE2"],
    "GLY": ["HXT", "H2"],
    "HIS": ["HXT", "H2", "HD1", "HE2"],
    "ILE": ["HXT", "H2"],
    "LEU": ["HXT", "H2"],
    "LYS": ["HXT", "H2", "HZ3"],
    "MET": ["HXT", "H2"],
    "PHE": ["HXT", "H2"],
    "PRO": ["HXT"],
    "SER": ["HXT", "H2", "HG"],
    "THR": ["HXT", "H2", "HG1"],
    "TRP": ["HXT", "H2", "HE1"],
    "TYR": ["HXT", "H2", "HH"],
    "VAL": ["HXT", "H2"],
}

#: Basic sites per residue, following openff-pablo's BASIC_ATOMS. Each
#: entry is (heavy atom to protonate, name of the added proton).
_BASIC_ATOMS = {
    "PRO": [("N", "H2")],
    **{
        resname: [("N", "H3")]
        for resname in (
            "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS "
            "MET PHE SER THR TRP TYR VAL".split()
        )
    },
}

#: Extra atom-name synonyms per residue, following openff-pablo's
#: ATOM_NAME_SYNONYMS (e.g. Amber/OpenMM name the N-terminal H "H1").
_ATOM_NAME_SYNONYMS = {
    "NME": {
        "HN2": ["H"],
        "C": ["CH3"],
        "H1": ["1HH3"],
        "H2": ["2HH3"],
        "H3": ["3HH3"],
    },
    "ACE": {"H1": ["1HH3"], "H2": ["2HH3"], "H3": ["3HH3"]},
    "NA": {"NA": ["Na"]},
    "CL": {"CL": ["Cl"]},
    "CYS": {"H": ["H1"], "HB2": ["3HB"]},
    "SER": {"H": ["H1"], "HB2": ["3HB"]},
    **{
        resname: {"H": ["H1"]}
        for resname in (
            "ALA ARG ASN ASP GLN GLU GLY HIS ILE LEU LYS "
            "MET PHE THR TRP TYR VAL".split()
        )
    },
}


#: Matches an atom name that starts with one or more digits, such as
#: ``2HB`` or ``1HH1``.
_DIGIT_FIRST_NAME = re.compile(r"^(\d+)([A-Za-z]\w*)$")


def _wraparound_name(name):
    """Return a digit-first atom name in digit-last form.

    Names such as ``2HB`` and ``1HH1`` are PDB format version 2 atom
    names. Amber-style tools (tleap, and the Amber writers of OpenMM
    and ParmEd) still write them. The wwPDB Format Guide v3.30, section
    "Atom Names", puts the digit last, so version 3 files and the CCD
    write the same atoms as ``HB2`` and ``HH11``. This function rotates
    the leading digits to the end of the name.

    Parameters
    ----------
    name : str
        A PDB atom name with the padding whitespace already removed.

    Returns
    -------
    str or None
        The rotated name, or None when the name does not start with a
        digit.
    """
    match = _DIGIT_FIRST_NAME.match(name)
    if match is None:
        return None
    return match.group(2) + match.group(1)


@dataclass(frozen=True)
class AtomTemplate:
    """One atom of a CCD residue template.

    Attributes
    ----------
    name : str
        Canonical CCD atom name (e.g. "NZ").
    element : str
        Element symbol.
    formal_charge : int
        Integer formal charge in this protonation variant.
    leaving : bool
        True if this atom is absent from a PDB file whenever the
        inter-residue bond associated with it exists.
    synonyms : tuple of str
        Alternative atom names accepted during matching.
    """

    name: str
    element: str
    formal_charge: int = 0
    leaving: bool = False
    synonyms: tuple = ()


@dataclass(frozen=True)
class BondTemplate:
    """One intra-residue bond of a CCD residue template."""

    atom1: str
    atom2: str
    order: int = 1


@dataclass(frozen=True)
class ResidueTemplate:
    """One protonation/termination variant of a CCD residue.

    A residue with a peptide ``linking`` type bonds its ``C`` atom to the
    ``N`` atom of the next residue. The template describes the *unlinked*
    species; the ``prior_fragment`` and ``posterior_fragment`` leaving
    atoms are absent from a PDB file when the corresponding peptide bond
    exists.
    """

    name: str
    description: str
    atoms: tuple
    bonds: tuple
    linking: str = None
    crosslink: tuple = None

    # The lookup maps below are cached: templates are frozen, and the
    # matcher touches every map once per residue x per variant, so
    # rebuilding them there dominated loading time. cached_property
    # writes to the instance __dict__ directly, which a frozen
    # dataclass permits; dataclasses.replace() makes a new instance,
    # so a patched copy never sees a stale cache. Callers must not
    # mutate the returned sets and dicts.
    @functools.cached_property
    def atom_names(self):
        """Return the set of canonical atom names."""
        return {atom.name for atom in self.atoms}

    @functools.cached_property
    def _atom_by_name(self):
        """Return a dict mapping each canonical name to its atom."""
        return {atom.name: atom for atom in self.atoms}

    @functools.cached_property
    def _neighbors(self):
        """Return a dict mapping each atom name to its bonded names."""
        neighbors = {atom.name: set() for atom in self.atoms}
        for bond in self.bonds:
            neighbors.setdefault(bond.atom1, set()).add(bond.atom2)
            neighbors.setdefault(bond.atom2, set()).add(bond.atom1)
        return neighbors

    @functools.cached_property
    def name_to_atom(self):
        """Return a dict mapping canonical names and synonyms to atoms.

        Synonyms that clash with a canonical name of another atom in the
        same template are skipped, so that files using canonical names
        always match unambiguously.
        """
        mapping = {atom.name: atom for atom in self.atoms}
        for atom in self.atoms:
            for synonym in atom.synonyms:
                if synonym not in mapping:
                    mapping[synonym] = atom
        return mapping

    @functools.cached_property
    def _atoms_by_any_name(self):
        """Return a dict mapping every accepted name to its atoms.

        A name maps to more than one atom when it is the canonical name
        of one atom and an alternative name of another. Glycine is the
        common case: the CCD gives ``HA2`` the alternative name ``HA1``
        and ``HA3`` the alternative name ``HA2``.
        """
        mapping = {}
        for atom in self.atoms:
            mapping.setdefault(atom.name, []).append(atom)
        for atom in self.atoms:
            for synonym in atom.synonyms:
                candidates = mapping.setdefault(synonym, [])
                if atom not in candidates:
                    candidates.append(atom)
        return {name: tuple(atoms) for name, atoms in mapping.items()}

    @property
    def formal_charge(self):
        """Return the net formal charge of this variant."""
        return sum(atom.formal_charge for atom in self.atoms)

    def atoms_named(self, name):
        """Return every template atom that a PDB atom name can denote.

        The candidates come in a fixed order: the atom whose canonical
        CCD name is ``name``, then the atoms that carry ``name`` as an
        alternative name in template order, then those two groups again
        for the digit-last rotation of a digit-first name (see
        ``_wraparound_name``). The order is fixed so that a matcher
        that consumes the list gives the same result on every run.

        This method reports more than one atom for a name, and the
        rotated names, which ``name_to_atom`` does not. Use it only to
        resolve a residue whose names the single-atom lookup rejects.

        Parameters
        ----------
        name : str
            A PDB atom name.

        Returns
        -------
        list of AtomTemplate
            The candidate atoms. Empty when this template accepts no
            atom of that name.
        """
        candidates = list(self._atoms_by_any_name.get(name, ()))
        rotated = _wraparound_name(name)
        if rotated is not None:
            candidates.extend(
                atom
                for atom in self._atoms_by_any_name.get(rotated, ())
                if atom not in candidates
            )
        return candidates

    def bonded_names(self, name):
        """Return the canonical names bonded to the named atom.

        Parameters
        ----------
        name : str
            Canonical atom name.

        Returns
        -------
        set of str
            The bonded canonical names; empty for an unknown name.
        """
        return self._neighbors.get(name, set())

    def leaving_fragment_of(self, name):
        """Return the leaving atoms connected to the named atom.

        The fragment is collected by walking bonds outward from the named
        (non-leaving) atom through leaving atoms only. This is the set of
        atoms that must be absent from a PDB file for a bond formed at the
        named atom, and it matches Pablo's leaving-fragment semantics.

        Parameters
        ----------
        name : str
            Canonical name of a non-leaving atom.

        Returns
        -------
        set of str
            The names of the connected leaving atoms.
        """
        atom_by_name = self._atom_by_name
        fragment = set()
        stack = [
            neighbor
            for neighbor in self.bonded_names(name)
            if atom_by_name[neighbor].leaving
        ]
        while stack:
            current = stack.pop()
            if current in fragment:
                continue
            fragment.add(current)
            stack.extend(
                neighbor
                for neighbor in self.bonded_names(current)
                if atom_by_name[neighbor].leaving and neighbor not in fragment
            )
        return fragment

    @functools.cached_property
    def prior_fragment(self):
        """Return leaving atoms absent when bonded to a preceding residue.

        For peptide residues this is the fragment at ``N`` (e.g. {"H2"}).
        """
        if self.linking != "peptide" or "N" not in self.atom_names:
            return set()
        return self.leaving_fragment_of("N")

    @functools.cached_property
    def posterior_fragment(self):
        """Return leaving atoms absent when bonded to a following residue.

        For peptide residues this is the fragment at ``C`` (e.g.
        {"OXT", "HXT"}).
        """
        if self.linking != "peptide" or "C" not in self.atom_names:
            return set()
        return self.leaving_fragment_of("C")

    def deprotonated_at(self, name):
        """Return a copy with proton ``name`` removed.

        The formal charge of the atom bonded to the proton is decremented,
        following Pablo's ``ResidueDefinition.deprotonated_at``.

        Parameters
        ----------
        name : str
            Canonical name of the hydrogen to remove.

        Returns
        -------
        ResidueTemplate
            A new template without the proton.
        """
        atom = self._atom_by_name.get(name)
        if atom is None or atom.element != "H":
            raise MBuildError(
                f"Cannot deprotonate {self.name} at {name}: "
                "no such hydrogen in this template."
            )
        neighbors = self.bonded_names(name)
        if len(neighbors) != 1:
            raise MBuildError(
                f"Cannot deprotonate {self.name} at {name}: "
                f"bonded to {len(neighbors)} atoms."
            )
        heavy = next(iter(neighbors))
        return replace(
            self,
            atoms=tuple(
                replace(a, formal_charge=a.formal_charge - 1) if a.name == heavy else a
                for a in self.atoms
                if a.name != name
            ),
            bonds=tuple(
                bond for bond in self.bonds if name not in (bond.atom1, bond.atom2)
            ),
            description=f"{self.description} -{name}",
        )

    def protonated_at(self, heavy_name, proton_name):
        """Return a copy with a proton added to the named heavy atom.

        The heavy atom's formal charge is incremented and the new proton
        inherits its ``leaving`` flag, following Pablo's
        ``ResidueDefinition.protonated_at``.

        Parameters
        ----------
        heavy_name : str
            Canonical name of the heavy atom to protonate.
        proton_name : str
            Name given to the added proton.

        Returns
        -------
        ResidueTemplate
            A new template with the proton added.
        """
        heavy = self._atom_by_name.get(heavy_name)
        if heavy is None:
            raise MBuildError(
                f"Cannot protonate {self.name} at missing atom {heavy_name}."
            )
        if proton_name in self.atom_names:
            raise MBuildError(
                f"Proton name {proton_name} already exists in {self.name}."
            )
        return replace(
            self,
            atoms=(
                *(
                    replace(a, formal_charge=a.formal_charge + 1)
                    if a.name == heavy_name
                    else a
                    for a in self.atoms
                ),
                AtomTemplate(name=proton_name, element="H", leaving=heavy.leaving),
            ),
            bonds=(*self.bonds, BondTemplate(heavy_name, proton_name)),
            description=f"{self.description} +{proton_name}",
        )


def _tokenize_cif_line(line):
    """Split a CIF data line, honoring double-quoted values."""
    return [
        token[1:-1] if token.startswith('"') and token.endswith('"') else token
        for token in re.findall(r'"[^"]*"|\S+', line)
    ]


def _parse_cif_blocks(text):
    """Parse the key-values and loops of a single-component CCD file.

    Returns (keys, loops) where keys maps "_category.item" to its value
    and loops maps "_category" to a list of row dicts.
    """
    keys = {}
    loops = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line.startswith("_"):
            tokens = _tokenize_cif_line(line)
            if len(tokens) >= 2:
                keys[tokens[0]] = tokens[1]
        elif line == "loop_":
            headers = []
            rows = []
            while index < len(lines):
                line = lines[index].strip()
                if line == "loop_":
                    # Do not consume the terminator: the outer loop
                    # must parse it as the start of the next loop.
                    break
                index += 1
                if line.startswith("_"):
                    headers.append(line.split()[0])
                elif line in ("#", ""):
                    break
                else:
                    tokens = _tokenize_cif_line(line)
                    if len(tokens) == len(headers):
                        rows.append(dict(zip(headers, tokens)))
            if headers:
                category = headers[0].rsplit(".", 1)[0]
                loops[category] = [
                    {header.rsplit(".", 1)[1]: value for header, value in row.items()}
                    for row in rows
                ]
    return keys, loops


def parse_ccd_cif(text):
    """Parse one CCD component ``.cif`` file into a base ResidueTemplate.

    The base template is the unpatched CCD species (e.g. the free amino
    acid with OXT/HXT/H2 present and flagged as leaving atoms).

    Parameters
    ----------
    text : str
        The content of one CCD component ``.cif`` file.

    Returns
    -------
    ResidueTemplate
        The base template, before patches and protonation variants.
    """
    keys, loops = _parse_cif_blocks(text)
    resname = keys.get("_chem_comp.id")
    if resname is None:
        raise MBuildError("CCD cif file has no _chem_comp.id entry.")
    comp_type = keys.get("_chem_comp.type", "NON-POLYMER").upper()
    linking = "peptide" if comp_type in _PEPTIDE_LINKING_TYPES else None

    atoms = []
    for row in loops.get("_chem_comp_atom", []):
        name = row["atom_id"]
        alt = row.get("alt_atom_id", name)
        # CIF marks an unknown value with "?" and an inapplicable value
        # with "."; both mean no formal charge here.
        charge = row.get("charge", "0")
        atoms.append(
            AtomTemplate(
                name=name,
                element=row["type_symbol"].capitalize(),
                formal_charge=0 if charge in ("?", ".") else int(charge),
                leaving=row.get("pdbx_leaving_atom_flag", "N") == "Y",
                synonyms=(alt,) if alt != name else (),
            )
        )
    bonds = tuple(
        BondTemplate(
            atom1=row["atom_id_1"],
            atom2=row["atom_id_2"],
            order=_CIF_BOND_ORDERS.get(row.get("value_order", "SING"), 1),
        )
        for row in loops.get("_chem_comp_bond", [])
    )
    return ResidueTemplate(
        name=resname,
        description=keys.get("_chem_comp.name", resname).strip('"'),
        atoms=tuple(atoms),
        bonds=bonds,
        linking=linking,
    )


def _fix_caps(template):
    """Give ACE/NME peptide linking and mark the reacting H as leaving.

    Mirrors Pablo's ``fix_caps`` patch: the CCD stores caps as
    non-polymers (acetaldehyde, methylamine), but as caps they link via
    the peptide bond and lose one hydrogen.
    """
    if template.name not in ("ACE", "NME"):
        return template
    return replace(
        template,
        linking="peptide",
        atoms=tuple(
            replace(atom, leaving=True) if atom.name == "H" else atom
            for atom in template.atoms
        ),
    )


def _add_disulfide(template):
    """Mark the cysteine thiol for the SG-SG disulfide crosslink.

    Mirrors Pablo's ``add_disulfide_crosslink`` patch: HG becomes a
    leaving atom, and the template records that SG can crosslink to the
    SG of another cysteine. The loader forms the bond only when the PDB
    holds a CONECT record between the two SG atoms and both HG atoms are
    absent.
    """
    if template.name != "CYS":
        return template
    return replace(
        template,
        crosslink=("SG", "SG"),
        atoms=tuple(
            replace(atom, leaving=True) if atom.name == "HG" else atom
            for atom in template.atoms
        ),
    )


def _fix_his_zwitterion(template):
    """Rewrite the zwitterionic HIS tautomer as the neutral form.

    Mirrors Pablo's ``patch_his_sidechain_zwitterion``: removing HD1 and
    HE2 from the CCD histidine cation leaves ND1 +1 / NE2 -1 with the
    double bond on ND1. The neutral form needs the double bond flipped to
    NE2 and both charges zeroed.
    """
    atom_by_name = {atom.name: atom for atom in template.atoms}
    nd1 = atom_by_name.get("ND1")
    ne2 = atom_by_name.get("NE2")
    if not (nd1 and ne2 and nd1.formal_charge == 1 and ne2.formal_charge == -1):
        return template
    new_bonds = []
    for bond in template.bonds:
        pair = {bond.atom1, bond.atom2}
        if pair == {"ND1", "CE1"}:
            new_bonds.append(replace(bond, order=1))
        elif pair == {"NE2", "CE1"}:
            new_bonds.append(replace(bond, order=2))
        else:
            new_bonds.append(bond)
    return replace(
        template,
        atoms=tuple(
            replace(atom, formal_charge=0) if atom.name in ("ND1", "NE2") else atom
            for atom in template.atoms
        ),
        bonds=tuple(new_bonds),
    )


def _add_synonyms(template):
    """Attach the extra atom-name synonyms used by common PDB writers."""
    extra = _ATOM_NAME_SYNONYMS.get(template.name)
    if not extra:
        return template
    return replace(
        template,
        atoms=tuple(
            replace(
                atom,
                synonyms=tuple(
                    dict.fromkeys((*atom.synonyms, *extra.get(atom.name, ())))
                ),
            )
            for atom in template.atoms
        ),
    )


def _protonation_variants(template):
    """Generate all protonation variants of a template, Pablo-style.

    Every combination of removed acidic protons and added basic protons
    is generated. Chemically invalid combinations are dropped: the
    doubly-deprotonated arginine (both HH12 and HH22 removed) and any
    histidine left with a negatively charged ring nitrogen that the
    zwitterion fix cannot rewrite.
    """
    acidic = [
        proton
        for proton in _ACIDIC_PROTONS.get(template.name, [])
        if proton in template.atom_names
    ]
    basic = _BASIC_ATOMS.get(template.name, [])
    variants = []
    for n_removed in range(len(acidic) + 1):
        for removed in itertools.combinations(acidic, n_removed):
            if {"HH12", "HH22"} <= set(removed):
                continue
            variant = template
            for proton in removed:
                variant = variant.deprotonated_at(proton)
            variants.append(variant)
            for heavy_name, proton_name in basic:
                variants.append(variant.protonated_at(heavy_name, proton_name))
    variants = [_fix_his_zwitterion(variant) for variant in variants]
    return [
        variant
        for variant in variants
        if not any(
            atom.formal_charge < 0 and atom.element == "N" for atom in variant.atoms
        )
    ]


class CCDLibrary:
    """A library of CCD residue templates, keyed by residue name.

    Templates load lazily from the bundled ``.cif`` files. Unknown
    residue codes optionally download from RCSB when ``download=True``.

    Parameters
    ----------
    paths : list of pathlib.Path, optional
        Directories searched for ``{RESNAME}.cif`` files, in order. The
        user download cache and the bundled directory are always
        searched last, in that order.
    download : bool, optional, default=False
        Download unknown residue codes from
        ``files.rcsb.org/ligands/download`` into the user cache.
    """

    #: Class-level cache of parsed variant lists, keyed by the resolved
    #: cif path and its modification time. Every Protein() builds a
    #: CCDLibrary, and re-parsing the same files dominated the load
    #: time of every Protein after the first. Templates are frozen
    #: dataclasses, so instances can share them; callers must not
    #: mutate the cached lists.
    _parse_cache = {}

    def __init__(self, paths=None, download=False):
        # The user download cache sits in the search order, so a
        # definition downloaded in one session loads in the next
        # session without download=True.
        self._paths = [Path(p) for p in (paths or [])] + [
            USER_CCD_CACHE_DIR,
            CCD_CACHE_DIR,
        ]
        self._download = download
        self._templates = {}

    def __contains__(self, resname):
        """Return True when the residue name resolves to templates."""
        try:
            self[resname]
        except KeyError:
            return False
        return True

    def __getitem__(self, resname):
        """Return the list of template variants for a residue name."""
        resname = resname.upper()
        if resname not in self._templates:
            self._templates[resname] = self._load(resname)
        return self._templates[resname]

    def _load(self, resname):
        """Read, parse, patch, and expand the templates for one residue.

        The search paths are tried in order; a missing file falls back
        to the RCSB download when ``download=True``. Parsed variant
        lists are cached class-wide, keyed by file path and mtime.
        """
        text = None
        cache_key = None
        for directory in self._paths:
            path = directory / f"{resname}.cif"
            if path.exists():
                resolved = path.resolve()
                cache_key = (str(resolved), resolved.stat().st_mtime_ns)
                cached = CCDLibrary._parse_cache.get(cache_key)
                if cached is not None:
                    return cached
                text = path.read_text()
                break
        if text is None and self._download:
            text = self._download_cif(resname)
        if text is None:
            raise KeyError(
                f"Residue {resname!r} is not in the CCD template library. "
                "Pass download=True to fetch it from RCSB, or add a local "
                "cif file to the library search paths."
            )
        base = _add_synonyms(_add_disulfide(_fix_caps(parse_ccd_cif(text))))
        variants = _protonation_variants(base)
        if cache_key is not None:
            CCDLibrary._parse_cache[cache_key] = variants
        return variants

    def _download_cif(self, resname):
        """Download one CCD ``.cif`` file into the user cache directory."""
        import urllib.request

        cache_dir = USER_CCD_CACHE_DIR
        cached = cache_dir / f"{resname}.cif"
        if cached.exists():
            return cached.read_text()
        url = RCSB_CCD_URL.format(resname)
        logger.info(f"Downloading CCD definition for {resname} from {url}")
        try:
            with urllib.request.urlopen(url) as response:
                text = response.read().decode()
        except Exception as error:
            raise KeyError(
                f"Could not download CCD definition for {resname!r}: {error}"
            ) from error
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(text)
        return text
