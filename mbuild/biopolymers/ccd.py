"""Residue templates from the wwPDB Chemical Component Dictionary (CCD).

A ``ResidueTemplate`` holds the atoms, bonds and formal charges of one
protonation or termination variant of a CCD component. ``CCDLibrary``
reads CCD component ``.cif`` files and returns every variant of a residue,
so that the residues of a PDB file can be matched by atom name.

Each atom carries the CCD leaving-atom flag. Leaving atoms are absent from
a PDB file when the residue bonds to a neighbor (peptide bond or
crosslink). Removing an acidic proton lowers the formal charge of its heavy
atom by one; adding a proton to a basic atom raises it by one.

The bundled files in ``mbuild/lib/biomolecules/ccd_cache`` are unmodified
CCD component files (public domain), except ``I.cif``. The CCD uses the
code ``I`` for a nucleotide, but OpenMM and PACKMOL write ``I`` for an
iodide ion, so the bundled ``I.cif`` describes iodide.

The ``_ACIDIC_PROTONS``, ``_BASIC_ATOMS`` and ``_ATOM_NAME_SYNONYMS``
tables and the cap, disulfide and histidine patches are derived from
openff-pablo (https://github.com/openforcefield/openff-pablo, MIT license).
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
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from mbuild.exceptions import MBuildError
from mbuild.utils.io import import_

logger = logging.getLogger(__name__)

__all__ = ["AtomTemplate", "BondTemplate", "ResidueTemplate", "CCDLibrary"]

CCD_CACHE_DIR = Path(__file__).parent.parent / "lib" / "biomolecules" / "ccd_cache"
USER_CCD_CACHE_DIR = Path.home() / ".mbuild" / "ccd_cache"
RCSB_CCD_URL = "https://files.rcsb.org/ligands/download/{}.cif"

_PEPTIDE_LINKING_TYPES = {
    "L-PEPTIDE LINKING",
    "PEPTIDE LINKING",
    "D-PEPTIDE LINKING",
}

_CIF_BOND_ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3}

# CIF marks an unknown value with "?" and an inapplicable value with ".".
_CIF_NULLS = ("?", ".")

# Protons that may be absent from a residue in a PDB file.
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

# Basic sites as (heavy atom, name of the added proton).
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

# Atom names written by common PDB writers that the CCD does not list.
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

_DIGIT_FIRST_NAME = re.compile(r"^(\d+)([A-Za-z]\w*)$")


@dataclass(frozen=True)
class AtomTemplate:
    """One atom of a residue template.

    Parameters
    ----------
    name : str
        CCD atom name.
    element : str
        Element symbol.
    formal_charge : int, optional, default=0
        Formal charge of the atom in this variant.
    leaving : bool, optional, default=False
        True if the atom is absent when the residue bonds to a neighbor.
    aromatic : bool, optional, default=False
        The CCD aromatic flag of the atom.
    synonyms : tuple of str, optional, default=()
        Other names accepted for the atom.
    """

    name: str
    element: str
    formal_charge: int = 0
    leaving: bool = False
    aromatic: bool = False
    synonyms: tuple = ()


@dataclass(frozen=True)
class BondTemplate:
    """One bond of a residue template, between two atom names."""

    atom1: str
    atom2: str
    order: int = 1


@dataclass(frozen=True)
class ResidueTemplate:
    """One protonation or termination variant of a CCD component.

    Parameters
    ----------
    name : str
        CCD component code, e.g. ``"ALA"``.
    description : str
        Component name, followed by the protons added or removed.
    atoms : tuple of AtomTemplate
    bonds : tuple of BondTemplate
    linking : str, optional, default=None
        ``"peptide"`` for components that link through peptide bonds.
    crosslink : tuple of str, optional, default=None
        Atom names of a crosslink, e.g. ``("SG", "SG")`` for a disulfide.
    """

    name: str
    description: str
    atoms: tuple
    bonds: tuple
    linking: str = None
    crosslink: tuple = None

    # Templates are frozen, so the lookup tables below are computed once.
    # The returned sets and dicts are shared and must not be modified.
    @functools.cached_property
    def atom_names(self):
        """Return the set of CCD atom names."""
        return {atom.name for atom in self.atoms}

    @functools.cached_property
    def name_to_atom(self):
        """Return a dict mapping CCD names and synonyms to atoms.

        A synonym that is the CCD name of another atom is skipped.
        """
        mapping = {atom.name: atom for atom in self.atoms}
        for atom in self.atoms:
            for synonym in atom.synonyms:
                mapping.setdefault(synonym, atom)
        return mapping

    @functools.cached_property
    def _neighbors(self):
        neighbors = {atom.name: set() for atom in self.atoms}
        for bond in self.bonds:
            neighbors.setdefault(bond.atom1, set()).add(bond.atom2)
            neighbors.setdefault(bond.atom2, set()).add(bond.atom1)
        return neighbors

    @functools.cached_property
    def _atoms_by_any_name(self):
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
        """Return every atom that a PDB atom name can refer to.

        Unlike ``name_to_atom``, a name can give more than one atom. For
        example, the CCD names the glycine hydrogens ``HA2``/``HA3`` with
        alternative names ``HA1``/``HA2``. A digit-first PDB v2 name such
        as ``2HB`` also matches its digit-last form ``HB2``.

        Parameters
        ----------
        name : str
            A PDB atom name.

        Returns
        -------
        list of AtomTemplate
            Atoms with that CCD name, then atoms with that synonym, then
            the same for the digit-last form of the name.
        """
        candidates = list(self._atoms_by_any_name.get(name, ()))
        match = _DIGIT_FIRST_NAME.match(name)
        if match is not None:
            rotated = match.group(2) + match.group(1)
            candidates.extend(
                atom
                for atom in self._atoms_by_any_name.get(rotated, ())
                if atom not in candidates
            )
        return candidates

    def bonded_names(self, name):
        """Return the set of atom names bonded to the named atom."""
        return self._neighbors.get(name, set())

    def leaving_fragment_of(self, name):
        """Return the leaving atoms reached from an atom through leaving atoms.

        These are the atoms that are absent from a PDB file when the named
        atom bonds to another residue.

        Parameters
        ----------
        name : str
            CCD name of a non-leaving atom.

        Returns
        -------
        set of str
        """
        fragment = set()
        stack = list(self.bonded_names(name))
        while stack:
            current = stack.pop()
            if current in fragment or not self.name_to_atom[current].leaving:
                continue
            fragment.add(current)
            stack.extend(self.bonded_names(current))
        return fragment

    @property
    def prior_link_atom(self):
        """Return the atom bonded to the preceding residue, or None."""
        if self.linking != "peptide" or "N" not in self.atom_names:
            return None
        return "N"

    @property
    def posterior_link_atom(self):
        """Return the atom bonded to the following residue, or None."""
        if self.linking != "peptide" or "C" not in self.atom_names:
            return None
        return "C"

    @functools.cached_property
    def prior_fragment(self):
        """Return the leaving atoms lost on bonding to a preceding residue."""
        if self.prior_link_atom is None:
            return set()
        return self.leaving_fragment_of(self.prior_link_atom)

    @functools.cached_property
    def posterior_fragment(self):
        """Return the leaving atoms lost on bonding to a following residue."""
        if self.posterior_link_atom is None:
            return set()
        return self.leaving_fragment_of(self.posterior_link_atom)


def _cif_rows(block, category):
    """Return the rows of a CIF category as dicts keyed by item name.

    A category with a single row is written as key-value pairs rather than
    a ``loop_``, and pycifrw then returns strings instead of lists.
    """
    prefix = category + "."
    columns = {}
    for key in block.keys():
        if key.startswith(prefix):
            value = block[key]
            columns[key[len(prefix) :]] = [value] if isinstance(value, str) else value
    return [dict(zip(columns, values)) for values in zip(*columns.values())]


def parse_ccd_cif(filename):
    """Parse a CCD component ``.cif`` file into a ResidueTemplate.

    The template is the CCD species as written, e.g. the free amino acid
    with its leaving atoms ``H2``, ``OXT`` and ``HXT``.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the ``.cif`` file.

    Returns
    -------
    ResidueTemplate
    """
    CifFile = import_("CifFile")
    block = CifFile.ReadCif(str(filename)).first_block()
    resname = block.get("_chem_comp.id")
    if resname is None:
        raise MBuildError(f"{filename} has no _chem_comp.id entry.")
    comp_type = block.get("_chem_comp.type", "NON-POLYMER").upper()
    description = block.get("_chem_comp.name", resname)

    atoms = []
    for row in _cif_rows(block, "_chem_comp_atom"):
        name = row["atom_id"]
        alt = row.get("alt_atom_id", name)
        charge = row.get("charge", "0")
        atoms.append(
            AtomTemplate(
                name=name,
                element=row["type_symbol"].capitalize(),
                formal_charge=0 if charge in _CIF_NULLS else int(charge),
                leaving=row.get("pdbx_leaving_atom_flag") == "Y",
                aromatic=row.get("pdbx_aromatic_flag") == "Y",
                synonyms=() if alt in (name, *_CIF_NULLS) else (alt,),
            )
        )
    bonds = tuple(
        BondTemplate(
            row["atom_id_1"],
            row["atom_id_2"],
            _CIF_BOND_ORDERS.get(row.get("value_order"), 1),
        )
        for row in _cif_rows(block, "_chem_comp_bond")
    )
    return ResidueTemplate(
        name=resname,
        description=" ".join(description.split()),
        atoms=tuple(atoms),
        bonds=bonds,
        linking="peptide" if comp_type in _PEPTIDE_LINKING_TYPES else None,
    )


def _with_leaving(template, name):
    """Return a copy of the template with the named atom marked leaving."""
    return replace(
        template,
        atoms=tuple(
            replace(atom, leaving=True) if atom.name == name else atom
            for atom in template.atoms
        ),
    )


def _patch(template):
    """Apply the cap and disulfide patches and the extra atom synonyms.

    The CCD stores the ACE and NME caps as small molecules; as caps they
    link through a peptide bond and lose ``H``. The ``HG`` of cysteine
    leaves when ``SG`` forms a disulfide.
    """
    if template.name in ("ACE", "NME"):
        template = replace(_with_leaving(template, "H"), linking="peptide")
    elif template.name == "CYS":
        template = replace(_with_leaving(template, "HG"), crosslink=("SG", "SG"))
    extra = _ATOM_NAME_SYNONYMS.get(template.name, {})
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


def _deprotonate(template, name):
    """Remove a proton and lower the charge of its heavy atom by one."""
    (heavy,) = template.bonded_names(name)
    return replace(
        template,
        atoms=tuple(
            replace(atom, formal_charge=atom.formal_charge - 1)
            if atom.name == heavy
            else atom
            for atom in template.atoms
            if atom.name != name
        ),
        bonds=tuple(
            bond for bond in template.bonds if name not in (bond.atom1, bond.atom2)
        ),
        description=f"{template.description} -{name}",
    )


def _protonate(template, heavy, name):
    """Add a proton to a heavy atom and raise its charge by one."""
    leaving = template.name_to_atom[heavy].leaving
    return replace(
        template,
        atoms=(
            *(
                replace(atom, formal_charge=atom.formal_charge + 1)
                if atom.name == heavy
                else atom
                for atom in template.atoms
            ),
            AtomTemplate(name=name, element="H", leaving=leaving),
        ),
        bonds=(*template.bonds, BondTemplate(heavy, name)),
        description=f"{template.description} +{name}",
    )


def _fix_his_zwitterion(template):
    """Rewrite histidine without HE2 as the neutral HD1 tautomer.

    Removing HE2 from the CCD histidine cation leaves ND1 at +1 and NE2 at
    -1. The ring double bond moves from ND1-CE1 to NE2-CE1 and both
    charges become zero.
    """
    charges = {atom.name: atom.formal_charge for atom in template.atoms}
    if charges.get("ND1") != 1 or charges.get("NE2") != -1:
        return template
    orders = {frozenset(("ND1", "CE1")): 1, frozenset(("NE2", "CE1")): 2}
    return replace(
        template,
        atoms=tuple(
            replace(atom, formal_charge=0) if atom.name in ("ND1", "NE2") else atom
            for atom in template.atoms
        ),
        bonds=tuple(
            replace(
                bond, order=orders.get(frozenset((bond.atom1, bond.atom2)), bond.order)
            )
            for bond in template.bonds
        ),
    )


def _protonation_variants(template):
    """Return every protonation variant of a template.

    Every combination of removed acidic protons, each with and without
    the added basic proton, is generated. Arginine without both HH12 and
    HH22 is dropped, as is any variant that leaves a nitrogen negative
    that is not negative in the CCD definition (e.g. a histidine ring).
    """
    acidic = [
        proton
        for proton in _ACIDIC_PROTONS.get(template.name, [])
        if proton in template.atom_names
    ]
    basic = [
        site
        for site in _BASIC_ATOMS.get(template.name, [])
        if site[0] in template.atom_names
    ]
    negative = {
        atom.name
        for atom in template.atoms
        if atom.element == "N" and atom.formal_charge < 0
    }
    variants = []
    for n_removed in range(len(acidic) + 1):
        for removed in itertools.combinations(acidic, n_removed):
            if {"HH12", "HH22"} <= set(removed):
                continue
            variant = template
            for proton in removed:
                variant = _deprotonate(variant, proton)
            variants.append(variant)
            variants.extend(_protonate(variant, *site) for site in basic)
    variants = [_fix_his_zwitterion(variant) for variant in variants]
    return [
        variant
        for variant in variants
        if not any(
            atom.element == "N" and atom.formal_charge < 0 and atom.name not in negative
            for atom in variant.atoms
        )
    ]


class CCDLibrary:
    """A library of CCD residue templates, keyed by residue name.

    ``library[resname]`` returns the list of ``ResidueTemplate`` variants
    of a residue; the first is the CCD species as written. Templates are
    read on first use from ``{RESNAME}.cif`` files in ``paths``, then the
    user download cache (``~/.mbuild/ccd_cache``), then the files bundled
    with mBuild.

    Parameters
    ----------
    paths : list of str or pathlib.Path, optional, default=None
        Directories searched first for ``{RESNAME}.cif`` files.
    download : bool, optional, default=False
        Download residues found in no directory from files.rcsb.org into
        the user download cache.
    """

    # Parsed variants shared by all instances, keyed by file path and
    # modification time.
    _parse_cache = {}

    def __init__(self, paths=None, download=False):
        self._paths = [Path(p) for p in paths or []]
        self._paths += [USER_CCD_CACHE_DIR, CCD_CACHE_DIR]
        self._download = download
        self._templates = {}

    def __contains__(self, resname):
        try:
            self[resname]
        except KeyError:
            return False
        return True

    def __getitem__(self, resname):
        resname = resname.upper()
        if resname not in self._templates:
            self._templates[resname] = self._load(resname)
        return self._templates[resname]

    def _load(self, resname):
        """Find, parse and expand the templates of one residue."""
        for directory in self._paths:
            path = directory / f"{resname}.cif"
            if path.exists():
                break
        else:
            if not self._download:
                raise KeyError(
                    f"Residue {resname!r} is not in the CCD template library. "
                    "Pass download=True to fetch it from RCSB, or add its "
                    ".cif file to one of the library paths."
                )
            path = self._download_cif(resname)
        path = path.resolve()
        key = (str(path), path.stat().st_mtime_ns)
        if key not in CCDLibrary._parse_cache:
            template = _patch(parse_ccd_cif(path))
            CCDLibrary._parse_cache[key] = _protonation_variants(template)
        return CCDLibrary._parse_cache[key]

    @staticmethod
    def _download_cif(resname):
        """Download one CCD ``.cif`` file into the user download cache."""
        url = RCSB_CCD_URL.format(resname)
        logger.info(f"Downloading CCD definition of {resname} from {url}")
        try:
            with urllib.request.urlopen(url) as response:
                text = response.read()
        except Exception as error:
            raise KeyError(
                f"Could not download the CCD definition of {resname!r}: {error}"
            ) from error
        USER_CCD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = USER_CCD_CACHE_DIR / f"{resname}.cif"
        path.write_bytes(text)
        return path
