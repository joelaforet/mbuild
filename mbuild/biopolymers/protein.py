"""Protein recipe: load protonated protein PDB files by template matching.

The loader mirrors the matching model of the OpenFF Pablo PDB reader so
that a protein loaded (and later modified) with mBuild round-trips
through Pablo's ``topology_from_pdb``:

- Residues are matched against CCD templates **by atom name**; chemistry
  (bonds, bond orders, formal charges) comes from the matched template,
  never from the PDB file.
- The absence of whole leaving fragments signals inter-residue bonds:
  a missing ``H2`` means a peptide bond to the preceding residue, a
  missing ``OXT``/``HXT`` means a peptide bond to the following residue,
  and a missing ``HG`` on cysteine means a disulfide, which additionally
  requires a ``CONECT`` record between the two ``SG`` atoms.
- ``TER`` records are hard chain boundaries. Loading is strict: unknown
  residues, unmatched atom names, and unexplained missing atoms raise
  errors that name the residue and suggest a fix.

mBuild's generic PDB path (mdtraj via ``mb.load``) is not reused here
because it hides ``TER`` records and atom serials, both of which this
matching model needs. The input file must be fully protonated (for
example with pdbfixer or reduce) at the desired pH.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from mbuild import clone
from mbuild.biopolymers.ccd import CCDLibrary
from mbuild.box import Box
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError
from mbuild.polymer import Polymer
from mbuild.port import Port

logger = logging.getLogger(__name__)

__all__ = ["Protein", "Chain", "Residue"]


class Chain(Compound):
    """A protein chain. Children are ``Residue`` compounds.

    Parameters
    ----------
    chain_id : str
        The PDB chain identifier (e.g. "A").
    """

    def __init__(self, chain_id=""):
        super().__init__(name=f"Chain_{chain_id or '_'}")
        self.chain_id = chain_id

    def _clone(self, clone_of=None, root_container=None):
        newone = super()._clone(clone_of, root_container)
        newone.chain_id = self.chain_id
        return newone


class Residue(Compound):
    """One residue of a biopolymer. Children are atom particles.

    Attributes
    ----------
    resnum : int
        The PDB residue sequence number.
    icode : str
        The PDB insertion code ("" when absent).
    original_name : str
        The residue name at load time; ``name`` may change when the
        residue is modified.
    hetatm : bool
        True if the residue was read from (or should be written as)
        HETATM records.
    template : mbuild.biopolymers.ccd.ResidueTemplate or None
        The matched template variant, kept for chemistry lookups.
    formal_charge : int
        Net formal charge of the atoms present in this residue, from the
        matched template.
    """

    def __init__(self, resname="RES", resnum=1, icode="", hetatm=False):
        super().__init__(name=resname)
        self.resnum = resnum
        self.icode = icode
        self.original_name = resname
        self.hetatm = hetatm
        self.template = None
        self.formal_charge = 0

    def _clone(self, clone_of=None, root_container=None):
        newone = super()._clone(clone_of, root_container)
        for attribute in (
            "resnum",
            "icode",
            "original_name",
            "hetatm",
            "template",
            "formal_charge",
        ):
            setattr(newone, attribute, getattr(self, attribute))
        return newone


@dataclass
class InterResidueBond:
    """A recorded bond between atoms of two different residues.

    ``leaving1``/``leaving2`` are the atom names that were absent from
    (or removed from) each residue because this bond exists. Together
    with the residue names, linking atom names, and bond order, they are
    exactly the information Pablo's ``with_crosslink`` needs.
    """

    residue1: Residue
    residue2: Residue
    atom1_name: str
    atom2_name: str
    order: int = 1
    leaving1: tuple = ()
    leaving2: tuple = ()
    kind: str = "crosslink"  # "peptide" bonds need no crosslink spec


@dataclass
class _PdbRecord:
    serial: int
    name: str
    alt_loc: str
    resname: str
    chain_id: str
    resnum: int
    icode: str
    pos: np.ndarray
    element: str
    hetatm: bool
    line_no: int


@dataclass
class _PdbResidue:
    resname: str
    chain_id: str
    resnum: int
    icode: str
    records: list = field(default_factory=list)
    ter_after: bool = False

    @property
    def label(self):
        return f"{self.resname} {self.chain_id}:{self.resnum}{self.icode}"


@dataclass
class _Match:
    variant: object
    record_atoms: dict  # id(record) -> AtomTemplate
    missing: set
    expects_prior: bool
    expects_posterior: bool
    expects_crosslink: bool


def _atom_in_residue(residue, atom_name):
    """Return the named particle of a residue, or None.

    Particles are found by scanning names instead of labels, because
    labels can go stale after ``remove()``.
    """
    for particle in residue.particles():
        if particle.name == atom_name:
            return particle
    return None


def _parse_pdb(text):
    """Parse ATOM/HETATM/TER/CONECT/CRYST1 records of the first model."""
    residues = []
    conects = set()
    box = None
    seen_altloc_a = False
    in_extra_model = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        record_type = line[:6]
        if record_type == "ENDMDL":
            in_extra_model = True
        elif record_type.startswith("MODEL") and in_extra_model:
            logger.warning("PDB file has multiple models; only model 1 is read.")
        elif record_type in ("ATOM  ", "HETATM") and not in_extra_model:
            alt_loc = line[16].strip()
            if alt_loc not in ("", "A"):
                raise MBuildError(
                    f"Alternate location {alt_loc!r} on line {line_no} is not "
                    "supported. Keep only one location (altLoc blank or 'A')."
                )
            if alt_loc == "A" and not seen_altloc_a:
                logger.warning("Using alternate location 'A' atoms only.")
                seen_altloc_a = True
            record = _PdbRecord(
                serial=int(line[6:11]),
                name=line[12:16].strip(),
                alt_loc=alt_loc,
                resname=line[17:20].strip(),
                chain_id=line[21].strip(),
                resnum=int(line[22:26]),
                icode=line[26].strip(),
                pos=np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
                / 10.0,
                element=line[76:78].strip(),
                hetatm=record_type == "HETATM",
                line_no=line_no,
            )
            key = (record.resname, record.chain_id, record.resnum, record.icode)
            if not residues or key != (
                residues[-1].resname,
                residues[-1].chain_id,
                residues[-1].resnum,
                residues[-1].icode,
            ):
                residues.append(
                    _PdbResidue(
                        resname=record.resname,
                        chain_id=record.chain_id,
                        resnum=record.resnum,
                        icode=record.icode,
                    )
                )
            residues[-1].records.append(record)
        elif record_type.startswith("TER") and residues:
            residues[-1].ter_after = True
        elif record_type == "CONECT":
            fields = [line[start : start + 5].strip() for start in (6, 11, 16, 21, 26)]
            serials = [int(value) for value in fields if value]
            for partner in serials[1:]:
                conects.add(frozenset((serials[0], partner)))
        elif record_type == "CRYST1":
            lengths = (
                float(line[6:15]) / 10.0,
                float(line[15:24]) / 10.0,
                float(line[24:33]) / 10.0,
            )
            angles = (float(line[33:40]), float(line[40:47]), float(line[47:54]))
            if any(length > 0.2 for length in lengths):
                box = Box(lengths=lengths, angles=angles)
    if not residues:
        raise MBuildError("No ATOM or HETATM records found in the PDB file.")
    return residues, conects, box


def _match_residue(group, variants, prior_possible, posterior_possible):
    """Match one PDB residue against its template variants.

    Returns the valid matches. Raises MBuildError with the per-variant
    rejection reasons when nothing matches.
    """
    matches = []
    reasons = []
    for variant in variants:
        name_to_atom = variant.name_to_atom
        record_atoms = {}
        used = set()
        reason = None
        for record in group.records:
            atom = name_to_atom.get(record.name)
            if atom is None:
                reason = f"atom name {record.name!r} is not in the template"
                break
            if atom.name in used:
                reason = f"two records match template atom {atom.name!r}"
                break
            if record.element and record.element.upper() != atom.element.upper():
                reason = (
                    f"element {record.element!r} of atom {record.name!r} "
                    f"conflicts with template element {atom.element!r}"
                )
                break
            used.add(atom.name)
            record_atoms[id(record)] = atom
        if reason is not None:
            reasons.append(f"{variant.description}: {reason}")
            continue

        missing = variant.atom_names - used
        prior = variant.prior_fragment
        posterior = variant.posterior_fragment
        crosslink_fragment = set()
        if variant.crosslink and variant.crosslink[0] in variant.atom_names:
            crosslink_fragment = variant.leaving_fragment_of(variant.crosslink[0])
        expects_prior = bool(prior) and prior <= missing
        expects_posterior = bool(posterior) and posterior <= missing
        expects_crosslink = bool(crosslink_fragment) and crosslink_fragment <= missing
        explained = (
            (prior if expects_prior else set())
            | (posterior if expects_posterior else set())
            | (crosslink_fragment if expects_crosslink else set())
        )
        if missing != explained:
            reasons.append(
                f"{variant.description}: atoms {sorted(missing - explained)} "
                "are missing but not part of a leaving fragment"
            )
            continue
        if expects_prior and not prior_possible:
            reasons.append(
                f"{variant.description}: expects a bond to a preceding "
                "residue, but none is adjacent"
            )
            continue
        if expects_posterior and not posterior_possible:
            reasons.append(
                f"{variant.description}: expects a bond to a following "
                "residue, but none is adjacent"
            )
            continue
        matches.append(
            _Match(
                variant=variant,
                record_atoms=record_atoms,
                missing=missing,
                expects_prior=expects_prior,
                expects_posterior=expects_posterior,
                expects_crosslink=expects_crosslink,
            )
        )
    if not matches:
        details = "\n  ".join(reasons)
        raise MBuildError(
            f"Could not match residue {group.label} against any "
            f"template variant:\n  {details}\n"
            "Check that the file is fully protonated (e.g. run pdbfixer "
            "or reduce) and uses standard PDB atom names."
        )
    return matches


def _matches_agree(matches, group):
    """Verify that all valid matches assign the same chemistry.

    Matches may differ in absent atoms (e.g. a neutral vs deprotonated
    C-terminal template when OXT itself is absent); they must agree on
    the charges of the atoms present, the bonds among them, and the
    expected links. Returns the first match; raises on disagreement.
    """
    reference = matches[0]

    def fingerprint(match):
        present = {atom.name for atom in match.record_atoms.values()}
        charges = tuple(
            sorted(
                (atom.name, atom.formal_charge) for atom in match.record_atoms.values()
            )
        )
        bonds = tuple(
            sorted(
                (*sorted((bond.atom1, bond.atom2)), bond.order)
                for bond in match.variant.bonds
                if bond.atom1 in present and bond.atom2 in present
            )
        )
        return (
            charges,
            bonds,
            match.expects_prior,
            match.expects_posterior,
            match.expects_crosslink,
        )

    for match in matches[1:]:
        if fingerprint(match) != fingerprint(reference):
            raise MBuildError(
                f"Residue {group.label} matches multiple template variants "
                "that disagree on chemistry. This usually means the "
                "protonation state is incomplete or inconsistent."
            )
    return reference


class Protein(Polymer):
    """A protein loaded from a fully protonated PDB file.

    The hierarchy is ``Protein -> Chain -> Residue -> particles``. Atom
    particles use canonical CCD names; bonds carry the template bond
    orders; each ``Residue`` records its matched template and net formal
    charge.

    Parameters
    ----------
    filename : str, optional
        Path of the PDB file to load. The file must be fully protonated
        at the desired pH (e.g. prepared with pdbfixer or reduce).
    library : CCDLibrary, optional
        The residue template library. A default library over the bundled
        CCD files is created when omitted.
    download : bool, optional, default=False
        Allow the default library to download unknown residue codes from
        RCSB.
    name : str, optional, default="Protein"
    """

    def __init__(self, filename=None, library=None, download=False, name="Protein"):
        super().__init__()
        self.name = name
        self.library = library or CCDLibrary(download=download)
        self.cross_bonds = []
        if filename is not None:
            self._load_pdb(filename)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_pdb(self, filename):
        with open(filename) as handle:
            text = handle.read()
        groups, conects, box = _parse_pdb(text)

        # Peptide adjacency: consecutive peptide-capable residues in the
        # same chain with no TER between them may bond.
        peptide_capable = []
        for group in groups:
            try:
                variants = self.library[group.resname]
            except KeyError as error:
                raise MBuildError(
                    f"Residue {group.label}: {error.args[0]} "
                    "Pass download=True to fetch unknown residues from RCSB."
                ) from error
            peptide_capable.append(variants[0].linking == "peptide")

        def linked(i, j):
            return (
                peptide_capable[i]
                and peptide_capable[j]
                and groups[i].chain_id == groups[j].chain_id
                and not groups[i].ter_after
            )

        matches = []
        for i, group in enumerate(groups):
            prior_possible = i > 0 and linked(i - 1, i)
            posterior_possible = i < len(groups) - 1 and linked(i, i + 1)
            candidates = _match_residue(
                group,
                self.library[group.resname],
                prior_possible,
                posterior_possible,
            )
            matches.append(_matches_agree(candidates, group))

        self._build(groups, matches, conects)
        if box is not None:
            self.box = box

    def _build(self, groups, matches, conects):
        # Build each residue fully while it is detached, and attach whole
        # chains at the end: Compound.add composes the parent's entire
        # bond graph on every attach, so adding particles under an
        # already-attached root is quadratic in protein size.
        chains = {}
        chain_order = []
        serial_to_particle = {}
        serial_info = {}
        residues = []
        for group, match in zip(groups, matches):
            residue = Residue(
                resname=group.resname,
                resnum=group.resnum,
                icode=group.icode,
                hetatm=group.records[0].hetatm,
            )
            residue.template = match.variant
            residue.formal_charge = sum(
                atom.formal_charge for atom in match.record_atoms.values()
            )
            residues.append(residue)

            particles = {}
            for record in group.records:
                atom = match.record_atoms[id(record)]
                particle = Compound(
                    name=atom.name, element=atom.element, pos=record.pos
                )
                particles[atom.name] = particle
                serial_to_particle[record.serial] = particle
                serial_info[record.serial] = (residue, match, atom)
            residue.add([particles[name] for name in particles])
            for bond in match.variant.bonds:
                if bond.atom1 in particles and bond.atom2 in particles:
                    residue.add_bond(
                        (particles[bond.atom1], particles[bond.atom2]),
                        bond_order=float(bond.order),
                    )
            chain_key = group.chain_id
            if chain_key not in chains:
                chains[chain_key] = []
                chain_order.append(chain_key)
            chains[chain_key].append(residue)

        for chain_key in chain_order:
            chain = Chain(chain_id=chain_key)
            chain.add(chains[chain_key])
            self.add(chain)

        self._bond_backbone(groups, matches, residues)
        self._bond_crosslinks(groups, matches, residues, conects, serial_info)
        self._check_conects(conects, serial_to_particle)

    def _bond_backbone(self, groups, matches, residues):
        for i in range(len(groups) - 1):
            here, there = matches[i], matches[i + 1]
            if here.expects_posterior != there.expects_prior:
                raise MBuildError(
                    f"Inconsistent peptide bond between {groups[i].label} and "
                    f"{groups[i + 1].label}: one residue is missing its "
                    "leaving atoms and the other is not."
                )
            if not here.expects_posterior:
                if (
                    groups[i].chain_id == groups[i + 1].chain_id
                    and not groups[i].ter_after
                    and here.variant.linking == "peptide"
                    and there.variant.linking == "peptide"
                ):
                    logger.warning(
                        f"No peptide bond between {groups[i].label} and "
                        f"{groups[i + 1].label} (chain break without TER)."
                    )
                continue
            carbon = _atom_in_residue(residues[i], "C")
            nitrogen = _atom_in_residue(residues[i + 1], "N")
            if carbon is None or nitrogen is None:
                raise MBuildError(
                    f"Cannot form the peptide bond between {groups[i].label} "
                    f"and {groups[i + 1].label}: backbone atom missing."
                )
            self.add_bond((carbon, nitrogen), bond_order=1.0)

    def _bond_crosslinks(self, groups, matches, residues, conects, serial_info):
        expecting = {}
        for group, match, residue in zip(groups, matches, residues):
            if match.expects_crosslink:
                link_name = match.variant.crosslink[0]
                for record in group.records:
                    if match.record_atoms[id(record)].name == link_name:
                        expecting[record.serial] = (group, match, residue, record)
        satisfied = set()
        for serial, (group, match, residue, record) in expecting.items():
            if serial in satisfied:
                continue
            partner_serial = None
            for pair in conects:
                if serial in pair:
                    other = next(iter(pair - {serial}))
                    if other in expecting:
                        partner_serial = other
                        break
            if partner_serial is None:
                raise MBuildError(
                    f"Residue {group.label} is missing its "
                    f"{sorted(match.variant.leaving_fragment_of(match.variant.crosslink[0]))} "
                    "atoms, which signals a crosslink, but no CONECT record "
                    "connects it to a crosslink partner."
                )
            other_group, other_match, other_residue, other_record = expecting[
                partner_serial
            ]
            particle1 = _atom_in_residue(residue, record.name)
            particle2 = _atom_in_residue(other_residue, other_record.name)
            self.add_bond((particle1, particle2), bond_order=1.0)
            self.cross_bonds.append(
                InterResidueBond(
                    residue1=residue,
                    residue2=other_residue,
                    atom1_name=record.name,
                    atom2_name=other_record.name,
                    order=1,
                    leaving1=tuple(
                        sorted(match.variant.leaving_fragment_of(record.name))
                    ),
                    leaving2=tuple(
                        sorted(
                            other_match.variant.leaving_fragment_of(other_record.name)
                        )
                    ),
                )
            )
            satisfied.update((serial, partner_serial))

    def _check_conects(self, conects, serial_to_particle):
        for pair in conects:
            serials = tuple(pair)
            if len(serials) == 1:
                continue  # self-referencing CONECT
            particles = [serial_to_particle.get(serial) for serial in serials]
            if None in particles:
                raise MBuildError(
                    f"CONECT record references unknown atom serial in {serials}."
                )
            if not self.bond_graph.has_edge(*particles):
                raise MBuildError(
                    f"CONECT record between serials {serials} does not "
                    "correspond to any bond the residue templates predict. "
                    "mBuild does not guess chemistry for unknown bonds."
                )

    # ------------------------------------------------------------------
    # Functionalization
    # ------------------------------------------------------------------
    def attach(
        self,
        fragment,
        fragment_atom_name,
        resnum,
        atom_name,
        chain_id=None,
        icode="",
        fragment_resnum=None,
        fragment_resname=None,
        bond_order=1,
        separation=0.15,
    ):
        """Bond a fragment Compound onto a residue of this protein.

        One hydrogen leaves each side: a hydrogen bonded to the named
        protein atom, and a hydrogen bonded to the named fragment atom.
        Ports along the removed-hydrogen vectors align the fragment
        (``force_overlap``), a bond with ``bond_order`` forms between
        the two named atoms, and the new inter-residue bond is recorded
        in ``cross_bonds`` together with the removed (leaving) hydrogen
        names — exactly the information Pablo's ``with_crosslink``
        needs to parameterize the product.

        The fragment is cloned; the original is not changed. Fragment
        residues keep their identity: a fragment whose children are
        ``Residue`` compounds (a single PTM residue, a linear polymer,
        or a branched glycan) is added residue-per-residue metadata
        intact; any other Compound is wrapped into one new ``Residue``.
        To build branched, multiply-linked structures, call ``attach``
        repeatedly — an attached residue is addressable like any other,
        so a later call can target it. Every call records its bond, so
        residues may carry any number of links inside mBuild (note that
        Pablo 0.2.2 supports at most one crosslink per residue
        definition when loading the result).

        Parameters
        ----------
        fragment : mbuild.Compound
            The group to add. Cloned before use.
        fragment_atom_name : str
            Name of the fragment atom that forms the new bond. It must
            have at least one bonded hydrogen.
        resnum : int
            Residue number of the protein attachment site.
        atom_name : str
            Name of the protein atom that forms the new bond. It must
            have at least one bonded hydrogen.
        chain_id : str, optional
            Chain of the attachment site; required when residue numbers
            repeat across chains.
        icode : str, optional
            Insertion code of the attachment site.
        fragment_resnum : int, optional
            Residue number, within the fragment, of the fragment atom;
            required when the fragment atom name repeats across the
            fragment's residues.
        fragment_resname : str, optional
            Residue name given to a fragment that is not made of
            Residue compounds and therefore gets wrapped.
        bond_order : int, optional, default=1
            Order of the new bond.
        separation : float, optional, default=0.15
            Length of the new bond in nanometers.

        Returns
        -------
        InterResidueBond
            The recorded bond, as appended to ``cross_bonds``.
        """
        site_residue = self.get_residue(resnum, chain_id=chain_id, icode=icode)
        site_atom = self.get_atom(resnum, atom_name, chain_id=chain_id, icode=icode)
        site_hydrogen = self._bonded_hydrogen(site_atom, site_residue.name)

        added = clone(fragment)
        if not isinstance(added, Residue):
            fragment_residues = [
                child for child in added.successors() if isinstance(child, Residue)
            ]
            if not fragment_residues:
                added = self._wrap_in_residue(added, fragment_resname)
        frag_residues = (
            [added]
            if isinstance(added, Residue)
            else [c for c in added.successors() if isinstance(c, Residue)]
        )
        for residue in frag_residues:
            self._ensure_unique_atom_names(residue)

        frag_atom, frag_residue = self._find_fragment_atom(
            frag_residues, fragment_atom_name, fragment_resnum
        )
        frag_hydrogen = self._bonded_hydrogen(frag_atom, frag_residue.name)

        site_orientation = site_hydrogen.pos - site_atom.pos
        frag_orientation = frag_hydrogen.pos - frag_atom.pos

        self.remove(site_hydrogen)
        (added if added.parent is None else added.root).remove(frag_hydrogen)

        # Renumber fragment residues into the site's chain.
        chain = next(c for c in site_residue.ancestors() if isinstance(c, Chain))
        next_resnum = max(r.resnum for r in self.residues(chain.chain_id)) + 1
        for offset, residue in enumerate(frag_residues):
            residue.resnum = next_resnum + offset
            residue.hetatm = True
        chain.add(added)

        site_port = Port(
            anchor=site_atom,
            orientation=site_orientation,
            separation=separation / 2,
        )
        site_residue.add(site_port, label="attach_site")
        frag_port = Port(
            anchor=frag_atom,
            orientation=frag_orientation,
            separation=separation / 2,
        )
        added.add(frag_port, label="attach_frag")

        from mbuild.coordinate_transform import force_overlap

        force_overlap(
            move_this=added,
            from_positions=frag_port,
            to_positions=site_port,
            add_bond=True,
            bond_order=float(bond_order),
        )

        record = InterResidueBond(
            residue1=site_residue,
            residue2=frag_residue,
            atom1_name=site_atom.name,
            atom2_name=frag_atom.name,
            order=int(bond_order),
            leaving1=(site_hydrogen.name,),
            leaving2=(frag_hydrogen.name,),
        )
        self.cross_bonds.append(record)
        return record

    @staticmethod
    def _bonded_hydrogen(atom, residue_name):
        """Return one hydrogen bonded to the atom (first by sorted name)."""
        hydrogens = sorted(
            (
                particle
                for particle in atom.direct_bonds()
                if particle.element is not None and particle.element.symbol == "H"
            ),
            key=lambda particle: particle.name,
        )
        if not hydrogens:
            raise MBuildError(
                f"Atom {atom.name} of residue {residue_name} has no bonded "
                "hydrogen to substitute. Pick an atom with a hydrogen."
            )
        return hydrogens[0]

    @staticmethod
    def _wrap_in_residue(compound, fragment_resname):
        """Wrap a plain Compound into a single Residue."""
        resname = (fragment_resname or compound.name or "LIG")[:3].upper()
        if not resname.isalnum():
            resname = "LIG"
        residue = Residue(resname=resname, resnum=1, hetatm=True)
        residue.add(compound)
        logger.info(f"Fragment {compound.name!r} wrapped into residue {resname!r}.")
        return residue

    @staticmethod
    def _ensure_unique_atom_names(residue):
        """Rename particles element+index when names repeat in a residue.

        The PDB export and Pablo's matching need atom names that are
        unique within each residue; fragments from SMILES usually name
        every carbon "C".
        """
        names = [particle.name for particle in residue.particles()]
        if len(set(names)) == len(names):
            return
        counters = {}
        for particle in residue.particles():
            symbol = (
                particle.element.symbol.upper()
                if particle.element is not None
                else particle.name.upper()
            )
            counters[symbol] = counters.get(symbol, 0) + 1
            particle.name = f"{symbol}{counters[symbol]}"
        logger.info(
            f"Renamed atoms of residue {residue.name} to element+index "
            "names so they are unique within the residue."
        )

    @staticmethod
    def _find_fragment_atom(frag_residues, atom_name, fragment_resnum):
        """Locate the named atom among the fragment residues."""
        hits = []
        for residue in frag_residues:
            if fragment_resnum is not None and residue.resnum != fragment_resnum:
                continue
            particle = _atom_in_residue(residue, atom_name)
            if particle is not None:
                hits.append((particle, residue))
        if not hits:
            raise MBuildError(
                f"No fragment atom named {atom_name!r}"
                + (
                    f" in fragment residue {fragment_resnum}"
                    if fragment_resnum is not None
                    else ""
                )
                + f". Fragment atoms are "
                f"{[p.name for r in frag_residues for p in r.particles()]}."
            )
        if len(hits) > 1:
            raise MBuildError(
                f"Fragment atom name {atom_name!r} is ambiguous across "
                "fragment residues; pass fragment_resnum."
            )
        return hits[0]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def chains(self):
        """Return the Chain compounds in file order."""
        return [child for child in self.children if isinstance(child, Chain)]

    def residues(self, chain_id=None):
        """Yield Residue compounds, optionally restricted to one chain.

        The traversal is recursive because attached fragments may sit in
        a grouping compound under a chain.
        """
        for chain in self.chains:
            if chain_id is not None and chain.chain_id != chain_id:
                continue
            stack = list(chain.children)
            while stack:
                child = stack.pop(0)
                if isinstance(child, Residue):
                    yield child
                elif child.children:
                    stack = list(child.children) + stack

    def get_residue(self, resnum, chain_id=None, icode=""):
        """Return the residue with the given number (and chain/icode)."""
        found = [
            residue
            for residue in self.residues(chain_id=chain_id)
            if residue.resnum == resnum and residue.icode == icode
        ]
        if not found:
            raise MBuildError(
                f"No residue with number {resnum}"
                + (f" in chain {chain_id}" if chain_id else "")
            )
        if len(found) > 1:
            raise MBuildError(
                f"Residue number {resnum} is ambiguous across chains "
                f"{[r.parent.chain_id for r in found]}; pass chain_id."
            )
        return found[0]

    def get_atom(self, resnum, atom_name, chain_id=None, icode=""):
        """Return the named atom particle of the given residue."""
        residue = self.get_residue(resnum, chain_id=chain_id, icode=icode)
        particle = _atom_in_residue(residue, atom_name)
        if particle is None:
            raise MBuildError(
                f"Residue {residue.name} {residue.resnum} has no atom "
                f"{atom_name!r}. Its atoms are "
                f"{[p.name for p in residue.particles()]}."
            )
        return particle

    @property
    def net_formal_charge(self):
        """Return the summed formal charge of all residues."""
        return sum(residue.formal_charge for residue in self.residues())
