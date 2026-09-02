"""Protein recipe: load protonated protein PDB files by template matching.

The loader matches residues against chemical templates, in the same
way modern residue-template PDB readers do. A protein that mBuild
loads, and later modifies, therefore round-trips through such tools. A
parity test verifies the matching model against the OpenFF Pablo
reader. The test runs wherever openff-pablo is installed.

- Residues are matched against CCD templates **by atom name**; chemistry
  (bonds, bond orders, formal charges) comes from the matched template,
  never from the PDB file.
- The absence of whole leaving fragments signals inter-residue bonds:
  a missing ``H2`` means a peptide bond to the preceding residue, a
  missing ``OXT``/``HXT`` means a peptide bond to the following residue,
  and a missing ``HG`` on cysteine means a disulfide, which additionally
  requires a ``CONECT`` record between the two ``SG`` atoms.
- A ``TER`` record is a chain boundary when the chain identifier
  changes. It is a boundary as well when the residue numbering does
  not increase across it. It is also a boundary when the two residues
  are too far apart for a peptide bond. Every other ``TER`` is
  advisory: the loader keeps the peptide bond and logs a warning.
  Tools that write a PDB file from a topology put a ``TER`` at the end
  of each topology chain, which is the reason for this rule. Loading
  is strict: unknown residues, unmatched atom names, and unexplained
  missing atoms raise errors that name the residue and suggest a fix.

mBuild's generic PDB path (mdtraj via ``mb.load``) is not reused here
because it hides ``TER`` records and atom serials, both of which this
matching model needs. The input file must be fully protonated (for
example with pdbfixer or reduce) at the desired pH.
"""

import logging
import os
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from mbuild import clone
from mbuild.biopolymers.ccd import _ACIDIC_PROTONS, CCDLibrary
from mbuild.biopolymers.protein_pdb_io import (
    _check_residue_membership,
    _parse_pdb,
    _pdb_name_field,
)
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError
from mbuild.port import Port
from mbuild.utils.io import import_

logger = logging.getLogger(__name__)

__all__ = ["Protein", "Chain", "Residue", "residue_labels", "save", "to_gmso"]

#: File extensions that ``conversion.save`` routes through GMSO.
_GMSO_EXTENSIONS = frozenset((".gro", ".gsd", ".data", ".xyz", ".mcf", ".top"))

#: Longest C to N distance, in nm, that the loader accepts as a peptide
#: bond across an advisory TER record. A peptide C-N bond is about
#: 0.133 nm long. The limit adds a margin of about 50 percent for a
#: strained or low-resolution structure. It still rejects two residues
#: that only share a chain identifier and increasing residue numbers.
_ADVISORY_TER_MAX_C_N = 0.2

#: Extensions whose writers record the unit cell, so a missing box
#: changes the file that ``save`` writes.
_BOXED_EXTENSIONS = frozenset((".gro", ".top"))

#: Extensions whose GMSO writers need force-field parameters. Without
#: them each writer fails inside GMSO, and no message names the cause:
#: the top writer asserts "System not fully typed", the data writer
#: raises an AttributeError that carries a 400-character bond repr, and
#: the mcf writer raises a pydantic ValidationError.
_TYPED_EXTENSIONS = frozenset((".data", ".mcf", ".top"))


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

    Parameters
    ----------
    resname : str, optional, default="RES"
        The residue name.
    resnum : int, optional, default=1
        The PDB residue sequence number.
    icode : str, optional, default=""
        The PDB insertion code.
    hetatm : bool, optional, default=False
        True if the residue was read from (or should be written as)
        HETATM records.

    Attributes
    ----------
    original_name : str
        The residue name at load time; ``name`` may change when the
        residue is modified.
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
        #: Sparse map of atom name -> integer formal charge, filled by
        #: the loader (from the matched template) and fragment loaders.
        self.atom_formal_charges = {}
        #: Map of site label -> atom name for the fragment's covalent
        #: bond sites, set by attachment points in the SMILES (* or
        #: [*:n]) or by particle tags. attach() uses a lone entry when
        #: no fragment atom name is given.
        self.link_atoms = {}

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
        newone.atom_formal_charges = dict(self.atom_formal_charges)
        newone.link_atoms = dict(self.link_atoms)
        return newone


@dataclass
class InterResidueBond:
    """A recorded bond between atoms of two different residues.

    ``leaving1``/``leaving2`` are the atom names that were absent from
    (or removed from) each residue because this bond exists. Together
    with the residue names, linking atom names, and bond order, they
    describe the covalent modification completely; see
    ``Protein.bond_records``.
    """

    residue1: Residue
    residue2: Residue
    atom1_name: str
    atom2_name: str
    order: int = 1
    leaving1: tuple = ()
    leaving2: tuple = ()


@dataclass
class _Match:
    variant: object
    record_atoms: dict  # id(record) -> AtomTemplate
    missing: set
    expects_prior: bool
    expects_posterior: bool
    expects_crosslink: bool


@lru_cache(maxsize=1)
def _rdkit_bond_orders():
    """Return the RDKit bond type of every bond order this recipe uses.

    The table is built on the first call, not at import, because RDKit
    is an optional dependency and the module must import without it.
    ``lru_cache`` then holds the one table, so a caller that reads it
    per bond does not rebuild it. ``Protein.to_rdkit`` reads the table
    forward and
    ``fragments.fragment_from_sdf`` reads it backward, so one table
    keeps the two directions in agreement.

    The table is stricter than the map in ``mbuild.conversion`` by
    intent. That map turns UNSPECIFIED into the order 0.0. This recipe
    needs a real bond order on every bond, so an absent key must fail.

    Returns
    -------
    dict
        Map of bond order (float) -> ``rdkit.Chem.BondType``.
    """
    rdkit = import_("rdkit")  # noqa: F841
    from rdkit import Chem

    return {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
    }


def _chain_of(residue):
    """Return the Chain ancestor of a residue.

    A fragment residue can sit under a wrapper Compound inside its
    Chain, so the direct parent is not always the Chain.
    """
    return next(
        ancestor for ancestor in residue.ancestors() if isinstance(ancestor, Chain)
    )


def _residue_label(residue):
    """Return a short label for a residue, such as ``LYS 5 A``.

    The chain identifier is appended when the chain has one.

    Parameters
    ----------
    residue : Residue
        The residue to label.

    Returns
    -------
    str
        The residue name, its number, and the chain identifier.
    """
    label = f"{residue.name} {residue.resnum}"
    chain_id = _chain_of(residue).chain_id
    return f"{label} {chain_id}" if chain_id else label


def _atom_in_residue(residue, atom_name):
    """Return the named particle of a residue, or None.

    Particles are found by name instead of by label, because labels can
    go stale after ``remove()``.
    """
    return next(residue.particles_by_name(atom_name), None)


def _record_pos(group, atom_name):
    """Return the position of a named PDB record of one residue, or None.

    Parameters
    ----------
    group : _PdbResidue
        The parsed residue whose records are searched.
    atom_name : str
        The record name to find, as the file writes it.

    Returns
    -------
    numpy.ndarray or None
        The position in nm, or None when no record carries the name.
    """
    return next(
        (record.pos for record in group.records if record.name == atom_name), None
    )


def _stamp_template(residue, variant):
    """Write a template variant and its formal charges onto a residue.

    Only the atoms the residue holds contribute to the charges. A
    residue inside a chain is missing the leaving atoms of its peptide
    bonds, and those absent atoms must add no charge.

    Parameters
    ----------
    residue : Residue
        The residue to write. Its particles must already be added.
    variant : mbuild.biopolymers.ccd.ResidueTemplate
        The matched template variant.
    """
    name_to_atom = variant.name_to_atom
    charges = {}
    for particle in residue.particles():
        atom = name_to_atom.get(particle.name)
        if atom is not None and atom.formal_charge:
            charges[particle.name] = atom.formal_charge
    residue.template = variant
    residue.atom_formal_charges = charges
    residue.formal_charge = sum(charges.values())


def _remove_pruning_ports(root, atom, particles):
    """Remove particles bonded to an atom and drop the opened ports.

    ``Compound.remove`` leaves one auto-generated port on the atom per
    severed bond. Those ports are removed here, so the atom keeps only
    the ports the caller made.

    Parameters
    ----------
    root : mbuild.Compound
        The compound that owns the bond graph, for example the Protein
        or a detached fragment.
    atom : mbuild.Compound
        The atom the removed particles are bonded to.
    particles : list of mbuild.Compound
        The particles to remove.
    """
    residue = atom.parent
    old_ports = {p for p in residue.children if isinstance(p, Port)}
    root.remove(particles)
    # Compound.remove removes a Port with three constant-cost
    # operations and then rescans every particle of the root in
    # _prune_ghost_ports. The ports removed here were created by the
    # remove() call above and anchor an atom that is still present, so
    # that rescan finds nothing. Apply the same three operations
    # directly and skip the second full-protein scan.
    for port in [
        p for p in residue.children if isinstance(p, Port) and p not in old_ports
    ]:
        root._remove(port)
        port.parent.children.remove(port)
        root._remove_references(port)


def _ter_break_reason(earlier, later, earlier_variant, later_variant):
    """Report why a TER between two residues ends the chain.

    The wwPDB Format Guide v3.30, section 9 (Coordinate Section, TER),
    states that the TER record ends the chain of ATOM and HETATM
    records that comes before it. A strict reader therefore ends the
    polymer at every TER.

    A preparation tool that writes the file from a topology puts a TER
    at the end of each topology chain. It does not put one at the end
    of each PDB chain. OpenMM's ``PDBFile.writeModel`` prints a TER
    after the last residue of every ``Topology`` chain. With
    ``keepIds=True`` it takes the chain identifier from the chain
    object, so two topology chains can carry one identifier. A cap
    (ACE, NME) or a ligand that the topology holds in its own chain
    then follows a TER inside one PDB chain. Every writer that goes
    through OpenMM inherits this behavior.

    The callers have already checked that the two residues share a
    chain identifier. This function applies the two remaining tests.

    The residue numbering must increase across the TER. The pair
    ``(resnum, icode)`` of the later residue must be greater than the
    pair of the earlier residue. Strict consecutiveness is not
    required. OpenMM writes a numbering gap where a loop is missing,
    and it writes insertion codes, and the loader accepts both shapes
    when no TER is present.

    The candidate peptide bond must also be short enough. The distance
    from the C atom of the earlier residue to the N atom of the later
    residue must stay below ``_ADVISORY_TER_MAX_C_N``. The numbering
    alone cannot tell one polymer from two separate molecules that
    share a chain identifier.

    A TER that passes both tests is advisory. The atom records then
    still describe one polymer. The residue before the TER is missing
    its OXT and HXT atoms, and the residue after it is missing its H2
    atom. Only a peptide bond explains that.

    Parameters
    ----------
    earlier : _PdbResidue
        The residue that carries the TER record.
    later : _PdbResidue
        The residue that follows it in the file.
    earlier_variant : mbuild.biopolymers.ccd.ResidueTemplate
        The base template of the earlier residue. It names the atom
        that carries the posterior peptide bond.
    later_variant : mbuild.biopolymers.ccd.ResidueTemplate
        The base template of the later residue. It names the atom that
        carries the prior peptide bond.

    Returns
    -------
    str or None
        The reason the TER ends the chain, or None when the TER is
        advisory.
    """
    if (later.resnum, later.icode) <= (earlier.resnum, earlier.icode):
        return (
            "the residue numbering does not increase across it "
            f"({earlier.resnum}{earlier.icode} then "
            f"{later.resnum}{later.icode})"
        )
    # The link atom names C and N are the same in PDB format version 2
    # and version 3, so a record can be found by the template name here,
    # before the records are assigned to template atoms.
    carbon = _record_pos(earlier, earlier_variant.posterior_link_atom)
    nitrogen = _record_pos(later, later_variant.prior_link_atom)
    if carbon is None or nitrogen is None:
        return (
            f"{earlier.label} or {later.label} carries no backbone C or "
            "N record, so the peptide bond cannot be measured"
        )
    distance = float(np.linalg.norm(carbon - nitrogen))
    if distance > _ADVISORY_TER_MAX_C_N:
        return (
            f"the C atom of {earlier.label} and the N atom of "
            f"{later.label} are {distance * 10:.2f} A apart, which is "
            f"too far for a peptide bond (limit "
            f"{_ADVISORY_TER_MAX_C_N * 10:.1f} A)"
        )
    return None


def _assign_records(group, variant):
    """Assign the records of one PDB residue to template atoms.

    The first pass gives every record the single atom that
    ``name_to_atom`` reports for its name. That pass resolves files
    written with wwPDB version 3 atom names, and it is the only pass
    such files need.

    Two of its rejection reasons come from the atom names alone: a name
    the template does not carry, and two records that claim the same
    template atom. Both happen on files whose hydrogen names are valid
    but not canonical. Digit-first names such as ``2HB`` are PDB format
    version 2 names, which Amber-style tools still write, and they fail
    the first reason. Glycine written with the version 2 alpha-hydrogen
    names ``HA1``/``HA2`` fails the second reason. The CCD gives the
    version 3 atom ``HA2`` the alternative name ``HA1``, and it gives
    the atom ``HA3`` the alternative name ``HA2``. After either reason, a second
    pass runs ``_assign_records_bipartite`` over the full candidate
    list of every record.

    Parameters
    ----------
    group : _ResidueGroup
        The records of one PDB residue, in file order.
    variant : mbuild.biopolymers.ccd.ResidueTemplate
        The template variant to assign the records to.

    Returns
    -------
    record_atoms : dict
        Maps ``id(record)`` to the assigned AtomTemplate. Complete only
        when ``reason`` is None.
    reason : str or None
        Why the variant does not fit, or None on success.
    fallback : bool
        True when the second pass produced the assignment.
    """
    name_to_atom = variant.name_to_atom
    record_atoms = {}
    used = set()
    reason = None
    names_disagree = False
    for record in group.records:
        atom = name_to_atom.get(record.name)
        if atom is None:
            reason = f"atom name {record.name!r} is not in the template"
            names_disagree = True
            break
        if atom.name in used:
            reason = f"two records match template atom {atom.name!r}"
            names_disagree = True
            break
        if record.element and record.element.upper() != atom.element.upper():
            reason = (
                f"element {record.element!r} of atom {record.name!r} "
                f"conflicts with template element {atom.element!r}"
            )
            break
        used.add(atom.name)
        record_atoms[id(record)] = atom
    if not names_disagree:
        return record_atoms, reason, False
    record_atoms, reason = _assign_records_bipartite(group, variant, reason)
    return record_atoms, reason, reason is None


def _assign_records_bipartite(group, variant, reason):
    """Assign records to template atoms by a bipartite matching.

    Every record gets the candidate atoms of
    ``ResidueTemplate.atoms_named``, kept only where the element of the
    record agrees. A Kuhn augmenting-path search then gives each record
    a distinct template atom. The records are visited in file order and
    the candidates stay in template order, so the same file always
    produces the same assignment.

    The assignment covers every record or none. A partial cover is a
    failure, because chemistry comes from the template and a record
    with no atom has no chemistry.

    Parameters
    ----------
    group : _ResidueGroup
        The records of one PDB residue, in file order.
    variant : mbuild.biopolymers.ccd.ResidueTemplate
        The template variant to assign the records to.
    reason : str
        The rejection reason of the first pass. It is reported again
        when the assignment fails, so that the error message keeps
        naming the record that the user must inspect.

    Returns
    -------
    record_atoms : dict
        Maps ``id(record)`` to the assigned AtomTemplate, or empty on
        failure.
    reason : str or None
        None on success, else the reason passed in.
    """
    candidates = []
    for record in group.records:
        atoms = [
            atom
            for atom in variant.atoms_named(record.name)
            if not record.element or record.element.upper() == atom.element.upper()
        ]
        if not atoms:
            return {}, reason
        candidates.append(atoms)

    holder = {}

    def augment(index, visited):
        """Give record ``index`` an atom, moving earlier records on."""
        for atom in candidates[index]:
            if atom.name in visited:
                continue
            visited.add(atom.name)
            held_by = holder.get(atom.name)
            if held_by is None or augment(held_by, visited):
                holder[atom.name] = index
                return True
        return False

    for index in range(len(candidates)):
        if not augment(index, set()):
            return {}, reason

    atom_of = {index: name for name, index in holder.items()}
    name_to_atom = variant.name_to_atom
    return {
        id(record): name_to_atom[atom_of[index]]
        for index, record in enumerate(group.records)
    }, None


def _match_residue(group, variants, prior_possible, posterior_possible):
    """Match one PDB residue against its template variants.

    Returns the valid matches. Raises MBuildError with the per-variant
    rejection reasons when nothing matches. Logs at info level when the
    second assignment pass rescued the residue, so that the tolerance
    is visible in the log.
    """
    matches = []
    reasons = []
    rescued = []
    for variant in variants:
        record_atoms, reason, fallback = _assign_records(group, variant)
        if reason is not None:
            reasons.append(f"{variant.description}: {reason}")
            continue
        used = {atom.name for atom in record_atoms.values()}

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
        if fallback and not rescued:
            rescued = sorted(
                record.name
                for record in group.records
                if record.name != record_atoms[id(record)].name
            )
    if not matches:
        details = "\n  ".join(reasons)
        # The per-variant reasons name the records that failed, but not
        # the names the template takes. List them once, from the base
        # variant, so the user can compare the file against them.
        accepted = sorted(variants[0].name_to_atom)
        raise MBuildError(
            f"Could not match residue {group.label} against any "
            f"template variant:\n  {details}\n"
            f"The {group.resname} template accepts these {len(accepted)} "
            f"atom names: {', '.join(accepted)}.\n"
            "Check that the file is fully protonated (e.g. run pdbfixer "
            "or reduce) and uses standard PDB atom names."
        )
    if rescued:
        logger.info(
            f"Residue {group.label}: the first-hit atom names did not fit, "
            f"and the second pass read {rescued} as alternative names."
        )
    return matches


def _filter_crosslink_candidates(groups, all_candidates, conects):
    """Reject crosslink candidate matches that CONECT records contradict.

    A bridged cysteine (HG absent) matches two kinds of variants: the
    neutral crosslink variants (``expects_crosslink=True``) and the
    deprotonated thiolate variants (``expects_crosslink=False``). The
    two disagree on the SG formal charge, so ``_matches_agree`` would
    reject every disulfide-containing file. The CONECT records decide
    between them. A candidate whose variant carries a crosslink atom
    must agree with the file. Its crosslink expectation must equal the
    presence of an SS CONECT to a crosslink-capable partner residue. A
    partner is crosslink-capable when any of its own pre-filter
    candidates expects the crosslink. Capability is computed before any
    rejection, so two bridged residues validate each other. Candidates
    whose variant has
    no crosslink atom are kept unchanged. This mirrors openff-pablo's
    ``filter_on_crosslinks`` rule.

    Returns the filtered candidate lists. Raises MBuildError when the
    rule leaves a residue with no candidate.
    """

    def crosslink_serial(group, match):
        """Return the serial of the match's crosslink atom, or None."""
        name = match.variant.crosslink[0]
        for record in group.records:
            atom = match.record_atoms.get(id(record))
            if atom is not None and atom.name == name:
                return record.serial
        return None

    serial_owner = {}
    for index, (group, candidates) in enumerate(zip(groups, all_candidates)):
        for match in candidates:
            if match.expects_crosslink:
                serial = crosslink_serial(group, match)
                if serial is not None:
                    serial_owner[serial] = index

    partners_of = {}
    for pair in conects:
        serials = tuple(pair)
        if len(serials) != 2:
            continue
        partners_of.setdefault(serials[0], set()).add(serials[1])
        partners_of.setdefault(serials[1], set()).add(serials[0])

    filtered = []
    for index, (group, candidates) in enumerate(zip(groups, all_candidates)):
        kept = []
        rejected = []
        for match in candidates:
            if not match.variant.crosslink:
                kept.append(match)
                continue
            serial = crosslink_serial(group, match)
            if serial is None:
                kept.append(match)
                continue
            linked = any(
                serial_owner.get(other, index) != index
                for other in partners_of.get(serial, ())
            )
            if match.expects_crosslink == linked:
                kept.append(match)
            else:
                rejected.append((match, linked))
        if not kept:
            match, linked = rejected[0]
            for candidate in rejected:
                if candidate[1] and not candidate[0].expects_crosslink:
                    match, linked = candidate
                    break
            name = match.variant.crosslink[0]
            leaving = sorted(match.variant.leaving_fragment_of(name))
            if linked:
                raise MBuildError(
                    f"Residue {group.label}: a CONECT record joins its "
                    f"{name} atom to the {name} atom of another residue, "
                    f"which signals a disulfide, but its {leaving} atoms "
                    "are present. Remove the CONECT record, or remove the "
                    f"{leaving} atoms to form the disulfide."
                )
            raise MBuildError(
                f"Residue {group.label} is missing its {leaving} atoms, "
                "which signals a crosslink, but no CONECT record connects "
                f"it to a crosslink partner. Add a CONECT record between "
                f"the two {name} atoms, or restore the {leaving} atoms."
            )
        filtered.append(kept)
    return filtered


def _matches_agree(matches, group):
    """Verify that all valid matches assign the same chemistry.

    Matches may differ in absent atoms. A neutral and a deprotonated
    C-terminal template differ that way when OXT itself is absent. The
    matches must agree on the charges of the atoms present, on the bonds
    among them, and on the expected links. This function returns the
    first match, and it raises on disagreement.

    The reference is the first match, and the match order is the
    variant order of the CCDLibrary, which is the order in which
    ``_protonation_variants`` generates the variants. That order does
    not depend on the file, so the same residue always returns the same
    match. A disagreement that reaches this point raises instead of
    picking a variant. The two variants give the atoms different
    chemistry, and the loader must never guess which one the file means.
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

    reference_fingerprint = fingerprint(reference)
    for match in matches[1:]:
        if fingerprint(match) != reference_fingerprint:
            raise MBuildError(
                f"Residue {group.label} matches multiple template variants "
                "that disagree on chemistry. This usually means the "
                "protonation state is incomplete or inconsistent."
            )
    return reference


class Protein(Compound):
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
        super().__init__(name=name)
        self.library = library or CCDLibrary(download=download)
        self.cross_bonds = []
        #: Map of anchor particle -> tuple of the hydrogen names that
        #: were removed at that particle, by a port that opened there or
        #: by ``deprotonate``. Repeated removals at one atom accumulate
        #: in the entry.
        #: ``record_bond`` reads it for its default leaving-atom lists.
        self._leaving_atoms = {}
        if filename is not None:
            self._load_pdb(filename)

    def _clone(self, clone_of=None, root_container=None):
        newone = super()._clone(clone_of, root_container)
        newone.library = self.library
        # cross_bonds reference live Residue objects; point the records
        # at the cloned residues so the clone stays self-consistent. A
        # record whose residue is no longer part of this Protein is an
        # inconsistent state and must fail here, not far downstream.
        clone_of = clone_of if clone_of is not None else {}

        def mapped(residue):
            if residue in clone_of:
                return clone_of[residue]
            raise MBuildError(
                f"cross_bonds references residue {residue.name} "
                f"{residue.resnum}, which is not part of this Protein. "
                "Remove stale records before cloning."
            )

        newone.cross_bonds = [
            replace(
                bond,
                residue1=mapped(bond.residue1),
                residue2=mapped(bond.residue2),
            )
            for bond in self.cross_bonds
        ]
        # The leaving-atom ledger holds default values for
        # record_bond, not state the protein depends on, so an anchor
        # that left this Protein is dropped instead of raising.
        newone._leaving_atoms = {
            clone_of[anchor]: names
            for anchor, names in self._leaving_atoms.items()
            if anchor in clone_of
        }
        return newone

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_pdb(self, filename):
        """Parse the PDB file, match every residue, and build the hierarchy."""
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

        # Reason text for each TER that ends the chain, keyed by the
        # index of the residue that carries the TER. A residue next to
        # such a TER can fail to match, and the reason explains why.
        ter_notes = {}

        def linked(i, j):
            if not (peptide_capable[i] and peptide_capable[j]):
                return False
            if groups[i].chain_id != groups[j].chain_id:
                return False
            if not groups[i].ter_after:
                return True
            reason = _ter_break_reason(
                groups[i],
                groups[j],
                self.library[groups[i].resname][0],
                self.library[groups[j].resname][0],
            )
            if reason is None:
                return True
            ter_notes[i] = (
                f"A TER record separates {groups[i].label} from "
                f"{groups[j].label}. The loader ends the chain at that "
                f"TER, because {reason}."
            )
            return False

        links = [linked(i, i + 1) for i in range(len(groups) - 1)]

        all_candidates = []
        for i, group in enumerate(groups):
            prior_possible = i > 0 and links[i - 1]
            posterior_possible = i < len(groups) - 1 and links[i]
            try:
                candidates = _match_residue(
                    group,
                    self.library[group.resname],
                    prior_possible,
                    posterior_possible,
                )
            except MBuildError as error:
                # The residue is missing the leaving atoms of a peptide
                # bond that the loader did not make. Name the TER, so
                # the message says what ended the chain.
                note = ter_notes.get(i) or ter_notes.get(i - 1)
                if note is None:
                    raise
                raise MBuildError(f"{error.args[0]}\n{note}") from error
            all_candidates.append(candidates)
        # A bridged cysteine matches both the crosslink variants and the
        # thiolate variants; the CONECT records decide between them
        # before the consensus check.
        all_candidates = _filter_crosslink_candidates(groups, all_candidates, conects)
        matches = [
            _matches_agree(candidates, group)
            for candidates, group in zip(all_candidates, groups)
        ]

        self._build(groups, matches, conects)
        if box is not None:
            self.box = box

    def _build(self, groups, matches, conects):
        """Build chains, residues, particles, and intra-residue bonds."""
        # Build each residue fully while it is detached, and attach whole
        # chains at the end: Compound.add composes the parent's entire
        # bond graph on every attach, so adding particles under an
        # already-attached root is quadratic in protein size.
        chains = {}
        chain_order = []
        serial_to_particle = {}
        residues = []
        for group, match in zip(groups, matches):
            residue = Residue(
                resname=group.resname,
                resnum=group.resnum,
                icode=group.icode,
                hetatm=group.records[0].hetatm,
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
            residue.add([particles[name] for name in particles])
            _stamp_template(residue, match.variant)
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
        self._bond_crosslinks(groups, matches, residues, conects)
        self._check_conects(conects, serial_to_particle)

    def _bond_backbone(self, groups, matches, residues):
        """Form the peptide bonds that the matched variants expect."""
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
            # expects_posterior is only true for a peptide-linking
            # variant, so neither link atom name is None here.
            carbon = _atom_in_residue(residues[i], here.variant.posterior_link_atom)
            nitrogen = _atom_in_residue(residues[i + 1], there.variant.prior_link_atom)
            if carbon is None or nitrogen is None:
                raise MBuildError(
                    f"Cannot form the peptide bond between {groups[i].label} "
                    f"and {groups[i + 1].label}: backbone atom missing."
                )
            self.add_bond((carbon, nitrogen), bond_order=1.0)
            if groups[i].ter_after:
                distance = float(np.linalg.norm(carbon.pos - nitrogen.pos))
                logger.warning(
                    f"A TER record separates {groups[i].label} from "
                    f"{groups[i + 1].label}. The loader kept the peptide "
                    "bond between them, because both residues are "
                    "missing the leaving atoms of that bond and their C "
                    f"and N atoms are {distance * 10:.2f} A apart. Split "
                    "the two residues into two chains in the PDB file if "
                    "they are separate molecules."
                )

    def _bond_crosslinks(self, groups, matches, residues, conects):
        """Form the crosslink bonds and record them in ``cross_bonds``."""
        expecting = {}
        for group, match, residue in zip(groups, matches, residues):
            if match.expects_crosslink:
                link_name = match.variant.crosslink[0]
                for record in group.records:
                    if match.record_atoms[id(record)].name == link_name:
                        expecting[record.serial] = (group, match, residue, record)
        pairs_of = {}
        for pair in conects:
            for serial in pair:
                pairs_of.setdefault(serial, []).append(pair)
        satisfied = set()
        for serial, (group, match, residue, record) in expecting.items():
            if serial in satisfied:
                continue
            partner_serial = None
            for pair in pairs_of.get(serial, ()):
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
        """Verify that every CONECT record maps to a template-predicted bond."""
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
    # Canonical Compound verbs, routed to residue-aware behavior
    # ------------------------------------------------------------------
    def save(self, filename, **kwargs):
        """Save the protein through the module-level ``save``.

        ``conversion.save`` calls the module-level GMSO converter, so a
        method override alone would not reach the GMSO formats. The
        module-level ``save`` holds every routing rule: it sends a
        ``.pdb`` file to ``save_pdb`` and the GMSO formats (.gro, .gsd,
        .data, .xyz, .mcf, .top) through the module-level ``to_gmso``.

        Parameters
        ----------
        filename : str
            Path of the file to write. The extension selects the
            writer.
        **kwargs
            Passed to the selected writer. For ``.pdb`` files only
            ``overwrite`` is accepted.
        """
        return save(self, filename, **kwargs)

    def to_parmed(self, **kwargs):
        """Create a ParmEd structure with residues taken from hierarchy.

        ``conversion.save`` passes ``residues=None`` explicitly. The
        default must therefore fill in whenever the value is None, not
        only when the key is absent. Otherwise every ParmEd-routed
        format (mol2, psf, ...) collapses the protein into one residue.

        Raises MBuildError when a residue name equals an atom name
        present in the protein. The generic converter matches each
        atom's own name against the residue list before it checks the
        atom's ancestors. Such a collision then splits those atoms into
        spurious residues, and nothing says so. A calcium ion residue
        ``CA`` next to alpha-carbon atoms ``CA`` is one example. Rename the
        residue before this export, or write a PDB with ``save_pdb``.

        Parameters
        ----------
        **kwargs
            Passed to ``Compound.to_parmed``. ``residues`` defaults to
            the residue names of this protein.

        Returns
        -------
        parmed.Structure
            The ParmEd structure with per-residue assignments.
        """
        if kwargs.get("residues") is None:
            kwargs["residues"] = sorted({residue.name for residue in self.residues()})
        residue_names = kwargs["residues"]
        if isinstance(residue_names, str):
            residue_names = [residue_names]
        colliding = set(residue_names) & {
            particle.name for particle in self.particles()
        }
        if colliding:
            raise MBuildError(
                f"Residue names {sorted(colliding)} equal atom names in "
                "this protein. The ParmEd converter matches atom names "
                "against the residue list first, so these atoms would "
                "split into spurious residues. Rename the residues "
                "(residue.name = ...) before this export, or write a "
                "PDB with save_pdb()."
            )
        return super().to_parmed(**kwargs)

    def to_gmso(self, **kwargs):
        """Create a GMSO topology that keeps residue identity.

        The work is in the module-level ``to_gmso``, which also serves
        a packed system that holds this protein as a child.

        Parameters
        ----------
        **kwargs
            Passed to the module-level ``to_gmso``.

        Returns
        -------
        gmso.Topology
            The topology with per-site residue names and numbers.
        """
        return to_gmso(self, **kwargs)

    def to_trajectory(self, include_ports=False, chains=None, residues=None, box=None):
        """Create an mdtraj Trajectory that keeps chains and residues.

        The generic converter assigns every atom to one default residue
        unless the caller lists the chain and residue names. This
        override fills both lists from the hierarchy, so each Chain and
        each Residue compound becomes its own mdtraj chain and residue.
        Caller-provided values win.

        Parameters
        ----------
        include_ports : bool, optional, default=False
            Include ghost particles of open ports.
        chains : list of str, optional
            Chain names to map to mdtraj chains. Default: the names of
            this protein's Chain compounds.
        residues : list of str, optional
            Residue names to map to mdtraj residues. Default: the
            residue names of this protein.
        box : mbuild.Box, optional
            The unit cell written to the trajectory. Default: this
            protein's box, or its bounding box with a 0.5 nm pad.

        Returns
        -------
        mdtraj.Trajectory
        """
        if chains is None:
            chains = sorted({chain.name for chain in self.chains})
        if residues is None:
            residues = sorted({residue.name for residue in self.residues()})
        return super().to_trajectory(
            include_ports=include_ports,
            chains=chains,
            residues=residues,
            box=box,
        )

    def to_rdkit(self, embed=False):
        """Create a sanitized RDKit molecule of the (modified) protein.

        Unlike the generic ``Compound.to_rdkit``, this export carries
        the chemistry the recipe knows. It writes the formal charges of
        the matched templates and the fragment records, the bond orders,
        the explicit hydrogens, one conformer, and PDB residue info on
        every atom. The result
        sanitizes, so it is directly usable by RDKit and by tools that
        consume RDKit molecules.

        Parameters
        ----------
        embed : bool, optional, default=False
            Accepted for compatibility with ``Compound.to_rdkit``
            (``Compound.volume`` passes ``embed=True``). The value is
            ignored: the molecule always carries one conformer with the
            protein's real coordinates, and a new embedding would
            replace them with generated ones.

        Returns
        -------
        rdkit.Chem.Mol
            The sanitized molecule.
        """
        if embed:
            logger.debug(
                "Protein.to_rdkit ignores embed=True: the export always "
                "carries the protein's real coordinates."
            )
        rdkit = import_("rdkit")  # noqa: F841
        from rdkit import Chem

        editable = Chem.RWMol()
        particle_index = {}
        particle_residue = self._particle_residues()
        particles = list(self.particles())
        _check_residue_membership(particles, particle_residue)
        for particle in particles:
            chain_id, residue = particle_residue[particle]
            atom = Chem.Atom(particle.element.atomic_number)
            atom.SetFormalCharge(residue.atom_formal_charges.get(particle.name, 0))
            atom.SetNoImplicit(True)
            info = Chem.AtomPDBResidueInfo()
            info.SetName(_pdb_name_field(particle.name))
            info.SetResidueName(residue.name)
            info.SetResidueNumber(residue.resnum)
            info.SetChainId(chain_id or " ")
            info.SetInsertionCode(residue.icode or " ")
            info.SetIsHeteroAtom(residue.hetatm)
            atom.SetPDBResidueInfo(info)
            particle_index[particle] = editable.AddAtom(atom)

        aromatic_pairs = []
        bond_types = _rdkit_bond_orders()
        for particle1, particle2, data in self.bonds(return_bond_order=True):
            if particle1 not in particle_index or particle2 not in particle_index:
                continue
            order = float(data["bond_order"])
            if order <= 0.0:
                raise MBuildError(
                    f"Bond {particle1.name}-{particle2.name} has no bond "
                    "order; cannot export chemistry to RDKit."
                )
            bond_type = bond_types[order]
            editable.AddBond(
                particle_index[particle1], particle_index[particle2], bond_type
            )
            if order == 1.5:
                aromatic_pairs.append((particle1, particle2))
        for particle1, particle2 in aromatic_pairs:
            for particle in (particle1, particle2):
                editable.GetAtomWithIdx(particle_index[particle]).SetIsAromatic(True)
            editable.GetBondBetweenAtoms(
                particle_index[particle1], particle_index[particle2]
            ).SetIsAromatic(True)

        mol = editable.GetMol()
        conformer = Chem.Conformer(len(particles))
        for particle, index in particle_index.items():
            x, y, z = particle.pos * 10.0
            conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
        mol.AddConformer(conformer, assignId=True)
        mol.UpdatePropertyCache(strict=False)
        try:
            Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as error:
            raise MBuildError(
                "RDKit rejected the protein's chemistry during sanitize: "
                f"{error}. If atoms were removed manually, update the "
                "residue's atom_formal_charges accordingly."
            ) from error
        return mol

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def relax_fragments(
        self,
        residues=None,
        n_steps=500,
        tolerance=50.0,
        platform="CPU",
    ):
        """Relax attached fragments while the protein stays fixed.

        Runs a short energy minimization with mBuild's generic
        UFF-style parameters (``OpenMMSimulation`` with
        ``forcefield=None``). The force field does not matter here: the
        goal is only to pull a rigidly placed fragment out of steric
        overlap so a downstream simulation stays stable. Every atom
        outside the given residues gets zero mass, which OpenMM treats
        as immobile, so the protein coordinates do not change.

        Parameters
        ----------
        residues : iterable of Residue, optional
            The residues allowed to move. Default: every HETATM
            residue (i.e. all attached fragments).
        n_steps : int, optional, default=500
            Maximum minimization iterations. It reaches OpenMM as
            ``maxIterations``, where ``0`` means that the minimizer
            runs until it meets ``tolerance``, with no iteration limit.
            The default is finite because ``attach(relax=True)`` calls
            this method: an interactive build must return, and a
            fragment that a rigid placement puts deep inside the
            protein can take a long time to meet the tolerance. Pass
            ``0`` when the converged structure matters more than the
            run time.
        tolerance : float, optional, default=50.0
            Energy tolerance in kJ/mol/nm.
        platform : str, optional, default="CPU"
            OpenMM platform name.
        """
        try:
            from mbuild.simulation import OpenMMSimulation
        except ImportError as error:
            raise MBuildError(
                "relax_fragments() needs mbuild.simulation, which is not "
                f"importable here ({error}). Install the simulation "
                "dependencies (hoomd, openmm; see environment-dev.yml) "
                "to relax fragments."
            ) from error

        targets = (
            list(residues)
            if residues is not None
            else [residue for residue in self.residues() if residue.hetatm]
        )
        if not targets:
            return
        mobile = set()
        for residue in targets:
            mobile.update(residue.particles())
        simulation = OpenMMSimulation(
            self, forcefield=None, kick=False, platform=platform
        )
        for index, particle in enumerate(self.particles()):
            if particle not in mobile:
                simulation.system.setParticleMass(index, 0.0)
        simulation.minimize(n_steps=n_steps, tolerance=tolerance)

    # ------------------------------------------------------------------
    # Functionalization
    # ------------------------------------------------------------------
    def deprotonate(self, resnum, atom_name, chain_id=None, icode=""):
        """Remove the acidic proton of one atom and update its charge.

        The proton comes from the residue's matched template variant:
        the acidic proton that the variant bonds to ``atom_name``. The
        residue is then re-matched to the variant that describes the
        result, so ``template``, ``formal_charge`` and
        ``atom_formal_charges`` all describe the deprotonated residue.

        The call changes nothing and logs a warning when the named atom
        carries no acidic proton, for example because it is already
        deprotonated. A notebook cell that calls this method therefore
        runs a second time without an error.

        The removed proton is written to the leaving-atom ledger, so a
        later ``attach`` at the same atom records every hydrogen that
        left it. A tool that rebuilds the residue from that record
        therefore restores neither the proton nor the charge.

        Parameters
        ----------
        resnum : int
            Residue number of the target residue.
        atom_name : str
            Name of the heavy atom that loses the proton.
        chain_id : str, optional
            Chain of the target residue; required when residue numbers
            repeat across chains.
        icode : str, optional
            Insertion code of the target residue.

        Returns
        -------
        None

        Raises
        ------
        MBuildError
            When the residue or the atom does not exist. The error
            comes from ``get_residue`` and ``get_atom``.

        Notes
        -----
        A protonated amine is not the reactive species in an acylation.
        The neutral amine is the reactive species, and the product is a
        neutral amide. Call this method before ``attach`` so that the
        site starts from the neutral form and the product carries the
        correct charge.

        Examples
        --------
        >>> protein.deprotonate(63, "NZ", chain_id="A")
        >>> protein.attach(fragment, resnum=63, atom_name="NZ", chain_id="A")
        """
        residue = self.get_residue(resnum, chain_id=chain_id, icode=icode)
        atom = self._atom_of(residue, atom_name)
        variant = residue.template
        protons = (
            [
                name
                for name in _ACIDIC_PROTONS.get(residue.name, ())
                if name in variant.bonded_names(atom_name)
                and _atom_in_residue(residue, name) is not None
            ]
            if variant is not None
            else []
        )
        if not protons:
            logger.warning(
                f"Atom {atom_name} of residue {residue.name} {residue.resnum} "
                "carries no acidic proton, so nothing changed. The atom is "
                "already deprotonated, or its protons are not acidic."
            )
            return
        proton_name = protons[0]
        _remove_pruning_ports(self, atom, [_atom_in_residue(residue, proton_name)])
        _stamp_template(residue, variant.deprotonated_at(proton_name))
        self._record_leaving_atoms(atom, [proton_name])
        self._warn_if_variant_is_absent(residue, atom_name, proton_name)
        self._warn_on_split_charge(residue)

    @staticmethod
    def _warn_on_split_charge(residue):
        """Warn when the residue keeps more than one charged atom.

        ``ResidueTemplate.deprotonated_at`` decrements the charge of the
        heavy atom that held the proton. It changes no other atom, so a
        residue whose charge sat on a second atom now holds two charged
        atoms. The residue charge can still be zero. The warning names
        the atoms and their charges and the call proceeds.

        Parameters
        ----------
        residue : Residue
            The residue, with its new template already assigned.
        """
        charges = {
            name: charge
            for name, charge in residue.atom_formal_charges.items()
            if charge
        }
        if len(charges) < 2:
            return
        listing = ", ".join(
            f"{name} {charge:+d}" for name, charge in sorted(charges.items())
        )
        logger.warning(
            f"{_residue_label(residue)} holds {len(charges)} charged atoms "
            f"after this call: {listing}. Load the protein again and "
            "deprotonate another atom if one charged atom is correct for the "
            "chemistry you model."
        )

    def _warn_if_variant_is_absent(self, residue, atom_name, proton_name):
        """Warn when no library variant describes the deprotonated residue.

        ``deprotonate`` builds the new template variant from the old
        one. The library holds fewer variants than that construction can
        produce, so the result can be a residue that no library variant
        describes. The loader matches a file against the library
        variants, so a PDB written from such a residue does not reload.
        The warning names the consequence and the call proceeds.

        Parameters
        ----------
        residue : Residue
            The residue, with its new template already assigned.
        atom_name : str
            Name of the heavy atom that lost the proton.
        proton_name : str
            Name of the removed proton.
        """
        variant = residue.template
        library_variants = self.library[residue.name]
        if any(other.atom_names == variant.atom_names for other in library_variants):
            return
        logger.warning(
            f"{_residue_label(residue)} atom {atom_name} lost {proton_name}. "
            f"The template library holds no {residue.name} variant with the "
            f"atoms of {variant.description}. A PDB written from this protein "
            "does not reload with Protein(). Deprotonate another atom if the "
            "written file must reload."
        )

    def add_port_at(
        self,
        resnum,
        atom_name,
        chain_id=None,
        icode="",
        separation=0.15,
        bond_order=1,
    ):
        """Create and return a real Port at the named atom.

        This is the low-level alternative under ``attach()``:
        the named atom loses ``bond_order`` hydrogens, and a ``Port``
        pointing along the removed hydrogens is added to the residue.
        Use it with ``force_overlap`` for placements ``attach()`` does
        not cover. ``force_overlap`` forms the bond but writes no
        record, so call ``record_bond`` after it:

        >>> port = protein.add_port_at(12, "NZ", chain_id="A")
        >>> force_overlap(fragment, fragment_port, port, add_bond=True)
        >>> protein.record_bond(port.anchor, fragment_atom)

        ``record_bond`` takes the names of the hydrogens removed here
        as the leaving atoms of the record, so ``bond_records``
        describes such a bond the way it describes an ``attach`` bond.

        Parameters
        ----------
        resnum : int
            Residue number of the target residue.
        atom_name : str
            Name of the atom that anchors the port. It must have at
            least ``bond_order`` bonded hydrogens.
        chain_id : str, optional
            Chain of the target residue; required when residue numbers
            repeat across chains.
        icode : str, optional
            Insertion code of the target residue.
        separation : float, optional, default=0.15
            Length of the bond the port will form, in nanometers.
        bond_order : int, optional, default=1
            Order of the bond the port will form; one hydrogen leaves
            per unit.

        Returns
        -------
        mbuild.Port
            The port, anchored at the named atom.
        """
        residue = self.get_residue(resnum, chain_id=chain_id, icode=icode)
        atom = self._atom_of(residue, atom_name)
        hydrogens = self._bonded_hydrogens(atom, residue.name, int(bond_order))
        port = self._port_along_hydrogens(self, atom, hydrogens, separation)
        residue.add(port, label="port[$]")
        return port

    def record_bond(self, atom1, atom2, order=1, leaving1=None, leaving2=None):
        """Record a bond formed outside ``attach`` in ``cross_bonds``.

        ``add_port_at`` with ``force_overlap`` forms a bond but writes
        no record, so ``bond_records`` does not report it and
        ``save_pdb`` cannot describe it to a downstream loader. This
        method adds the record for a bond that already exists.

        The leaving atom names default to the hydrogens that
        ``add_port_at`` removed at each atom. Pass ``leaving1`` or
        ``leaving2`` when the bond replaced other atoms.

        Parameters
        ----------
        atom1, atom2 : mbuild.Compound
            The two bonded atoms. They must be bonded already, and they
            must sit in two different residues of this protein.
        order : int, optional, default=1
            Order of the bond.
        leaving1 : sequence of str, optional
            Names of the atoms removed from the residue of ``atom1``.
        leaving2 : sequence of str, optional
            Names of the atoms removed from the residue of ``atom2``.

        Returns
        -------
        InterResidueBond
            The record, as appended to ``cross_bonds``.

        Raises
        ------
        MBuildError
            When an atom is not in a residue of this protein, when the
            two atoms are not bonded, or when both sit in one residue.
        """
        residues = self._particle_residues()
        for atom in (atom1, atom2):
            if atom not in residues:
                raise MBuildError(
                    f"Atom {atom.name} is not in a residue of this "
                    "Protein, so a bond to it cannot be recorded."
                )
        if atom2 not in atom1.direct_bonds():
            raise MBuildError(
                f"Atoms {atom1.name} and {atom2.name} are not bonded. "
                "Form the bond first (for example with force_overlap), "
                "then record it."
            )
        residue1 = residues[atom1][1]
        residue2 = residues[atom2][1]
        if residue1 is residue2:
            raise MBuildError(
                f"Atoms {atom1.name} and {atom2.name} are both in residue "
                f"{residue1.name} {residue1.resnum}. cross_bonds holds "
                "bonds between two residues."
            )
        record = InterResidueBond(
            residue1=residue1,
            residue2=residue2,
            atom1_name=atom1.name,
            atom2_name=atom2.name,
            order=int(order),
            leaving1=tuple(
                self._leaving_atoms.get(atom1, ()) if leaving1 is None else leaving1
            ),
            leaving2=tuple(
                self._leaving_atoms.get(atom2, ()) if leaving2 is None else leaving2
            ),
        )
        self.cross_bonds.append(record)
        return record

    def attach(
        self,
        fragment,
        fragment_atom_name=None,
        *,
        resnum,
        atom_name,
        chain_id=None,
        icode="",
        fragment_resnum=None,
        fragment_resname=None,
        bond_order=1,
        separation=0.15,
        relax=True,
    ):
        """Bond a fragment Compound onto a residue of this protein.

        ``bond_order`` hydrogens leave each side. They are hydrogens
        bonded to the named protein atom, and hydrogens bonded to the
        named fragment atom. One hydrogen leaves per unit of bond order,
        so a double bond removes two from each atom.

        Ports along the removed-hydrogen vectors align the fragment
        (``force_overlap``). A bond with ``bond_order`` then forms
        between the two named atoms. The new inter-residue bond is
        recorded in ``cross_bonds``, together with the removed (leaving)
        hydrogen names. The record holds everything a downstream tool
        needs to describe the modification (see ``bond_records``).

        The fragment is cloned; the original is not changed. Fragment
        residues keep their identity. A fragment whose children are
        ``Residue`` compounds is added residue-per-residue, with the
        metadata intact: a single PTM residue, a linear polymer, or a
        branched glycan. Any other Compound is wrapped into one new
        ``Residue``. To build branched, multiply-linked structures, call
        ``attach`` repeatedly. An attached residue is addressable like
        any other, so a later call can target it. Every call records its
        bond, so residues may carry any number of links inside mBuild.

        Parameters
        ----------
        fragment : mbuild.Compound
            The group to add. Cloned before use.
        fragment_atom_name : str
            Name of the fragment atom that forms the new bond. It must
            have at least ``bond_order`` bonded hydrogens.
        resnum : int
            Residue number of the protein attachment site.
        atom_name : str
            Name of the protein atom that forms the new bond. It must
            have at least ``bond_order`` bonded hydrogens.
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
        relax : bool, optional, default=True
            When the placed fragment overlaps existing atoms, run a
            short energy minimization that moves only the fragment
            (see ``relax_fragments``).

        Returns
        -------
        InterResidueBond
            The recorded bond, as appended to ``cross_bonds``.
        """
        # fragments.py imports Residue from this module, so a top-level
        # import of fragments here would be circular. Import inside the
        # method instead.
        from mbuild.biopolymers.fragments import _as_residues

        bond_order = int(bond_order)
        site_residue, site_atom, site_hydrogens = self._attachment_site(
            resnum, atom_name, chain_id, icode, bond_order
        )
        self._warn_on_kept_charge(site_residue, resnum, atom_name)
        added, frag_residues = _as_residues(clone(fragment), fragment_resname)
        frag_atom, frag_residue, frag_hydrogens = self._fragment_site(
            frag_residues, fragment_atom_name, fragment_resnum, bond_order
        )

        # One hydrogen leaves per bond order unit on each side (the
        # polymer.add_monomer convention); the port points along the sum
        # of the removed-hydrogen vectors. Both ports are opened while
        # the fragment is still detached, so that each removal runs on
        # its own compound.
        site_port = self._port_along_hydrogens(
            self, site_atom, site_hydrogens, separation
        )
        site_residue.add(site_port, label="attach_site")
        frag_port = self._port_along_hydrogens(
            added, frag_atom, frag_hydrogens, separation
        )
        added.add(frag_port, label="attach_frag")

        self._adopt_fragment(added, frag_residues, site_residue)
        self._align_on_ports(added, frag_port, site_port, bond_order)

        # The record is appended before the relaxation step, so the
        # protein state stays complete and consistent when relaxation
        # fails: the fragment is already bonded at this point.
        #
        # The site side reads the leaving-atom ledger, not the hydrogens
        # this call removed. The ledger also holds a proton that
        # deprotonate() removed at the same atom, so the record names
        # every hydrogen that left the site. A tool that rebuilds the
        # residue from the record then restores no proton and no charge.
        record = InterResidueBond(
            residue1=site_residue,
            residue2=frag_residue,
            atom1_name=site_atom.name,
            atom2_name=frag_atom.name,
            order=bond_order,
            leaving1=self._leaving_atoms[site_atom],
            leaving2=tuple(sorted(h.name for h in frag_hydrogens)),
        )
        self.cross_bonds.append(record)

        self._relax_if_clashing(added, site_atom, frag_atom, frag_residues, relax)
        return record

    @staticmethod
    def _warn_on_kept_charge(residue, resnum, atom_name):
        """Warn when the anchor atom keeps a formal charge across the bond.

        A hydrogen leaves the anchor atom, and the new bond takes its
        place, so the formal charge of the atom does not change. A
        charged anchor therefore gives a charged product. For an
        acylation that product is a protonated amide, which is not a
        real species; the neutral amine is the reactant that gives the
        neutral amide. The warning names ``deprotonate`` as the
        remedy and the call proceeds, because other chemistries do keep
        a charge on the anchor atom.

        The remedy is named for a positive charge only. ``deprotonate``
        removes a proton, which makes a negative anchor more negative.
        For a negative anchor the warning states the charge and names no
        remedy.

        Parameters
        ----------
        residue : Residue
            The residue that holds the anchor atom.
        resnum : int
            Residue number the caller passed to ``attach``.
        atom_name : str
            Name of the anchor atom.
        """
        charge = residue.atom_formal_charges.get(atom_name, 0)
        if not charge:
            return
        state = (
            f"{_residue_label(residue)} atom {atom_name} has formal charge "
            f"{charge:+d} before this bond and {charge:+d} after it."
        )
        if charge < 0:
            logger.warning(state)
            return
        chain_id = _chain_of(residue).chain_id
        call = f'deprotonate({resnum}, "{atom_name}"'
        if chain_id:
            call = f'{call}, chain_id="{chain_id}"'
        logger.warning(
            f"{state} Call {call}) before attach() if a neutral product is "
            "correct for the chemistry you model."
        )

    def _attachment_site(self, resnum, atom_name, chain_id, icode, bond_order):
        """Return the protein-side residue, atom, and leaving hydrogens.

        Parameters
        ----------
        resnum : int
            Residue number of the attachment site.
        atom_name : str
            Name of the protein atom that forms the new bond.
        chain_id : str or None
            Chain of the site; required when residue numbers repeat
            across chains.
        icode : str
            Insertion code of the site.
        bond_order : int
            Order of the new bond. One hydrogen leaves per unit.

        Returns
        -------
        residue : Residue
            The residue that holds the attachment atom.
        atom : mbuild.Compound
            The attachment atom.
        hydrogens : list of mbuild.Compound
            The hydrogens that leave that atom.
        """
        residue = self.get_residue(resnum, chain_id=chain_id, icode=icode)
        atom = self._atom_of(residue, atom_name)
        hydrogens = self._bonded_hydrogens(atom, residue.name, bond_order)
        return residue, atom, hydrogens

    @staticmethod
    def _fragment_site(frag_residues, atom_name, fragment_resnum, bond_order):
        """Return the fragment-side atom, residue, and leaving hydrogens.

        With no atom name, the fragment must carry exactly one labeled
        attachment site in ``Residue.link_atoms``, and that site forms
        the bond.

        Parameters
        ----------
        frag_residues : list of Residue
            The residues of the cloned fragment.
        atom_name : str or None
            Name of the fragment atom that forms the new bond.
        fragment_resnum : int or None
            Residue number, inside the fragment, of that atom.
        bond_order : int
            Order of the new bond. One hydrogen leaves per unit.

        Returns
        -------
        atom : mbuild.Compound
            The fragment attachment atom.
        residue : Residue
            The fragment residue that holds it.
        hydrogens : list of mbuild.Compound
            The hydrogens that leave that atom.
        """
        if atom_name is None:
            linked = [
                (label, residue)
                for residue in frag_residues
                for label in residue.link_atoms
            ]
            if len(linked) != 1:
                raise MBuildError(
                    "attach() bonds one site, but the fragment carries "
                    f"{len(linked)} attachment points. Pass "
                    "fragment_atom_name or mark exactly one site with * "
                    "in the SMILES."
                )
            label, link_residue = linked[0]
            atom_name = link_residue.link_atoms[label]
            fragment_resnum = link_residue.resnum

        atom, residue = Protein._find_fragment_atom(
            frag_residues, atom_name, fragment_resnum
        )
        hydrogens = Protein._bonded_hydrogens(atom, residue.name, bond_order)
        return atom, residue, hydrogens

    def _adopt_fragment(self, added, frag_residues, site_residue):
        """Renumber the fragment residues and add them to the chain.

        The fragment residues continue the residue numbering of the
        site's chain and are marked as HETATM records, so that a
        written PDB separates them from the standard residues.
        """
        chain = _chain_of(site_residue)
        next_resnum = max(r.resnum for r in self.residues(chain.chain_id)) + 1
        for offset, residue in enumerate(frag_residues):
            residue.resnum = next_resnum + offset
            residue.hetatm = True
        chain.add(added)

    @staticmethod
    def _align_on_ports(added, frag_port, site_port, bond_order):
        """Move the fragment onto the site port and bond the anchors.

        ``force_overlap`` superposes the fragment port on the site port
        and adds the bond between the two port anchor atoms.
        """
        from mbuild.coordinate_transform import force_overlap

        force_overlap(
            move_this=added,
            from_positions=frag_port,
            to_positions=site_port,
            add_bond=True,
            bond_order=float(bond_order),
        )

    def _relax_if_clashing(self, added, site_atom, frag_atom, frag_residues, relax):
        """Relax the placed fragment when it overlaps other atoms.

        Port alignment is rigid, so a bulky fragment can land inside the
        protein. The relaxation moves only the fragment residues. It
        needs the simulation dependencies; without them the method warns
        and keeps the rigid placement.

        Parameters
        ----------
        added : mbuild.Compound
            The placed fragment.
        site_atom : mbuild.Compound
            The protein atom of the new bond.
        frag_atom : mbuild.Compound
            The fragment atom of the new bond.
        frag_residues : list of Residue
            The residues that relaxation may move.
        relax : bool
            False leaves the rigid placement in place.
        """
        clashes = self._warn_on_clashes(added, site_atom, frag_atom)
        if not (clashes and relax):
            return
        try:
            import mbuild.simulation  # noqa: F401
        except ImportError as error:
            # The automatic path only warns: the attachment itself
            # is complete, and the user can relax later on a system
            # with the simulation dependencies installed.
            logger.warning(
                "Cannot relax the placed fragment: mbuild.simulation "
                f"is not importable ({error}). Install the simulation "
                "dependencies (hoomd, openmm) or call "
                "relax_fragments() elsewhere. The fragment keeps its "
                "rigid placement."
            )
            return
        logger.info("Relaxing the placed fragment with the protein held fixed.")
        self.relax_fragments(residues=frag_residues)
        if not self._warn_on_clashes(added, site_atom, frag_atom):
            logger.info("Fragment overlaps resolved by relaxation.")

    def _port_along_hydrogens(self, root, atom, hydrogens, separation):
        """Remove the hydrogens and return a Port pointing along them.

        The port points along the sum of the removed-hydrogen vectors,
        or along the first hydrogen when the sum is degenerate.
        ``Compound.remove`` leaves one auto-generated port on the atom
        per severed bond. Those ports are removed here, the same cleanup
        that ``Polymer.add_monomer`` does, so the returned Port is the
        only open port at the atom.

        The removed names are written to the ``_leaving_atoms`` ledger
        under the anchor atom, which is where ``record_bond`` reads its
        default leaving-atom lists. A second port at the same atom, and
        an earlier ``deprotonate`` call, add their removed names to the
        entry, so the entry always names every hydrogen that left that
        atom.
        """
        orientation = sum(h.pos - atom.pos for h in hydrogens)
        if np.linalg.norm(orientation) < 1e-8:
            orientation = hydrogens[0].pos - atom.pos
        _remove_pruning_ports(root, atom, hydrogens)
        self._record_leaving_atoms(atom, [h.name for h in hydrogens])
        return Port(anchor=atom, orientation=orientation, separation=separation / 2)

    def _record_leaving_atoms(self, atom, names):
        """Add removed hydrogen names to the leaving-atom ledger.

        The names are held under the anchor atom, sorted, and merged
        with the names an earlier removal at that atom wrote.

        Parameters
        ----------
        atom : mbuild.Compound
            The atom the hydrogens were bonded to.
        names : list of str
            Names of the removed hydrogens.
        """
        self._leaving_atoms[atom] = tuple(
            sorted(self._leaving_atoms.get(atom, ()) + tuple(names))
        )

    @staticmethod
    def _bonded_hydrogens(atom, residue_name, count):
        """Return ``count`` hydrogens bonded to the atom (sorted by name).

        One hydrogen leaves per unit of bond order. Reactions that
        remove other leaving groups (e.g. condensations) belong in
        future reaction recipes that use ``attach``.
        """
        if not 1 <= count <= 3:
            raise MBuildError(f"bond_order must be 1, 2, or 3; you passed {count}.")
        hydrogens = sorted(
            (
                particle
                for particle in atom.direct_bonds()
                if particle.element is not None and particle.element.symbol == "H"
            ),
            key=lambda particle: particle.name,
        )
        if len(hydrogens) < count:
            raise MBuildError(
                f"Atom {atom.name} of residue {residue_name} has "
                f"{len(hydrogens)} bonded hydrogens, but a bond of order "
                f"{count} must replace {count}. Pick an atom with enough "
                "hydrogens."
            )
        return hydrogens[:count]

    def _warn_on_clashes(self, added, site_atom, frag_atom, cutoff=0.1):
        """Warn when placed fragment atoms overlap the rest of the system.

        Port alignment is rigid; a bulky fragment can land inside the
        protein. The check compares every added atom against every other
        atom, and it leaves out the new bond pair. It warns below
        ``cutoff`` nm, so the user knows to relax the structure before
        simulating.
        """
        from scipy.spatial import cKDTree

        added_particles = list(added.particles())
        added_set = set(added_particles) | {site_atom}
        others = [p for p in self.particles() if p not in added_set]
        if not others or not added_particles:
            return
        tree = cKDTree([p.pos for p in others])
        distances, _ = tree.query(
            [p.pos for p in added_particles if p is not frag_atom]
        )
        n_clashes = int((distances < cutoff).sum())
        if n_clashes:
            logger.warning(
                f"{n_clashes} atoms of the attached fragment sit within "
                f"{cutoff * 10:.1f} A of existing atoms (closest: "
                f"{distances.min() * 10:.2f} A). Relax the structure before "
                "simulating (e.g. relax_fragments(), which holds the "
                "protein fixed)."
            )
        return n_clashes

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

        The traversal is recursive because attached fragments can be
        inside a grouping compound under a chain. The loop below stops
        at each Residue and does not enter it. ``chain.successors()``
        is not used here, because it visits every particle: thousands
        of nodes instead of hundreds of residues. This method backs
        ``get_residue``, ``net_formal_charge``, and the exports, so it
        must stay fast.

        Parameters
        ----------
        chain_id : str, optional
            Yield only the residues of this chain.

        Yields
        ------
        Residue
            The next residue, in hierarchy order.
        """
        for chain in self.chains:
            if chain_id is not None and chain.chain_id != chain_id:
                continue
            stack = deque(chain.children)
            while stack:
                child = stack.popleft()
                if isinstance(child, Residue):
                    yield child
                elif child.children:
                    stack.extendleft(reversed(child.children))

    def get_residue(self, resnum, chain_id=None, icode=""):
        """Return the residue with the given number (and chain/icode).

        Parameters
        ----------
        resnum : int
            The residue number.
        chain_id : str, optional
            The chain to search; required when residue numbers repeat
            across chains.
        icode : str, optional
            The insertion code.

        Returns
        -------
        Residue
            The single residue that matches. Raises MBuildError when
            no residue matches or the match is ambiguous.
        """
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
            # _chain_of, not parent: a wrapped fragment residue's
            # parent is its wrapper Compound, not the Chain.
            raise MBuildError(
                f"Residue number {resnum} is ambiguous across chains "
                f"{[_chain_of(r).chain_id for r in found]}; pass chain_id."
            )
        return found[0]

    def get_atom(self, resnum, atom_name, chain_id=None, icode=""):
        """Return the named atom particle of the given residue.

        Parameters
        ----------
        resnum : int
            The residue number.
        atom_name : str
            The atom name within the residue.
        chain_id : str, optional
            The chain to search; required when residue numbers repeat
            across chains.
        icode : str, optional
            The insertion code.

        Returns
        -------
        mbuild.Compound
            The atom particle. Raises MBuildError when the residue or
            the atom does not exist.
        """
        residue = self.get_residue(resnum, chain_id=chain_id, icode=icode)
        return self._atom_of(residue, atom_name)

    @staticmethod
    def _atom_of(residue, atom_name):
        """Return the named atom of the given residue, or raise."""
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

    def _particle_residues(self):
        """Return a map of particle -> (chain_id, residue).

        The chemistry exports resolve each particle's residue through
        this one map, so the traversal rules (recursive fragment
        residues) stay in ``residues()`` alone.
        ``_residue_of_particles`` calls this method, so
        ``residue_labels`` reads the same walk for a ``Protein``.
        """
        mapping = {}
        for chain in self.chains:
            for residue in self.residues(chain.chain_id):
                for particle in residue.particles():
                    mapping[particle] = (chain.chain_id, residue)
        return mapping

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_pdb(self, filename, overwrite=False):
        """Write a prepared PDB file for downstream residue-template loaders.

        Bonds made through ``add_port_at`` + ``force_overlap`` get
        CONECT records but no ``cross_bonds`` record, unless you call
        ``record_bond`` first. Without that record, downstream loaders
        need residue definitions for such a bond from you.

        mBuild's generic PDB writer (ParmEd via ``save``) cannot express
        residue numbers, chain identifiers, HETATM records, or a
        selective CONECT policy. This recipe therefore has its own
        writer. Its conventions follow the RCSB standard and the
        residue-template readers:

        - ``ATOM`` for residues loaded from ATOM records.
        - ``HETATM`` for attached fragments and heteroatoms.
        - ``TER`` after every chain.
        - ``CRYST1`` when a box is set.
        - ``CONECT`` records **only** for bonds between non-adjacent
          residues: disulfides, attached fragments, and branch links.

        A peptide bond is implied by residue adjacency, so it needs no
        CONECT record. A CONECT that the residue templates cannot
        explain makes a strict loader fail.

        Parameters
        ----------
        filename : str
            Path of the PDB file to write.
        overwrite : bool, optional, default=False
            Overwrite the file if it exists.
        """
        from mbuild.biopolymers.protein_pdb_io import write_pdb

        write_pdb(self, filename, overwrite=overwrite)

    def bond_records(self):
        """Return one plain dict per recorded inter-residue bond.

        Each dict describes a covalent modification completely: which
        residues bond through which atoms, which leaving atoms were
        removed on each side, and the bond order. Downstream tools
        format these records into their own vocabulary (residue
        definitions, crosslink declarations, templates).

        Returns
        -------
        list of dict
            One dict per record, with the keys ``residue_names``,
            ``residue_numbers``, ``atom_names``, ``leaving_atoms``
            (one list per side), and ``bond_order``.
        """
        return [
            {
                "residue_names": (bond.residue1.name, bond.residue2.name),
                "residue_numbers": (bond.residue1.resnum, bond.residue2.resnum),
                "atom_names": (bond.atom1_name, bond.atom2_name),
                "leaving_atoms": (list(bond.leaving1), list(bond.leaving2)),
                "bond_order": bond.order,
            }
            for bond in self.cross_bonds
        ]


def _residue_of_particles(compound):
    """Return a map of particle -> the ``Residue`` that holds it.

    A ``Protein`` answers through its own ``_particle_residues``, which
    walks its chains and its residues. Every other compound is walked
    particle by particle, and each particle takes its nearest
    ``Residue`` ancestor. The second path is needed because
    ``mb.solvate`` and ``mb.fill_box`` return a plain ``Compound`` that
    holds the protein as a child.

    Parameters
    ----------
    compound : mbuild.Compound
        The compound to walk.

    Returns
    -------
    dict
        Map of particle -> ``Residue``. A particle that sits outside a
        ``Residue`` gets no entry.
    """
    resolver = getattr(compound, "_particle_residues", None)
    if resolver is not None:
        return {particle: residue for particle, (_, residue) in resolver().items()}
    mapping = {}
    for particle in compound.particles():
        residue = next(
            (
                ancestor
                for ancestor in particle.ancestors()
                if isinstance(ancestor, Residue)
            ),
            None,
        )
        if residue is not None:
            mapping[particle] = residue
    return mapping


def residue_labels(compound):
    """Return a map of particle -> ``(residue name, residue number)``.

    The number in the label is unique for every ``Residue`` compound
    below ``compound``. GMSO stores a site residue by value. Two
    residues that share a name and a number are therefore one residue to
    GMSO, and every writer that numbers residues from that value merges
    them. The four-chain protein in ``1p3q_noter.pdb`` has 228 residues
    and writes 151 residue numbers to a ``.gro`` file for this reason.

    The first residue with a given ``(name, resnum)`` pair keeps its PDB
    number. Each later residue with the same pair moves into an offset
    block. Its number grows by the span of the residue numbers of
    ``compound``, once for every earlier repeat. The block is as wide as
    the span, so a shifted number cannot equal the number of any other
    residue with the same name. The walk order sets the block index, so
    two calls on the same compound return the same labels.

    The shifted numbers therefore grow with the number of copies. The
    residue field of a ``.gro`` file holds five digits, so a system of
    many copies of one protein can overflow that field.

    Parameters
    ----------
    compound : mbuild.Compound
        The compound to label. A particle that sits outside a
        ``Residue`` gets no entry.

    Returns
    -------
    dict
        Map of particle -> ``(name, number)``.
    """
    particle_residue = _residue_of_particles(compound)
    order = []
    seen = set()
    for residue in particle_residue.values():
        if id(residue) not in seen:
            seen.add(id(residue))
            order.append(residue)
    if not order:
        return {}
    numbers = [residue.resnum for residue in order]
    span = max(numbers) - min(numbers) + 1
    repeats = {}
    labels = {}
    for residue in order:
        key = (residue.name, residue.resnum)
        block = repeats.get(key, 0)
        repeats[key] = block + 1
        labels[id(residue)] = (residue.name, residue.resnum + block * span)
    return {
        particle: labels[id(residue)] for particle, residue in particle_residue.items()
    }


def to_gmso(compound, box=None, **kwargs):
    """Create a GMSO topology whose sites keep residue identity.

    The generic converter numbers residues by counting the occurrences
    of each residue name, so the numbers restart at 0 per name and do
    not match the PDB file. This function rewrites every site's residue
    with the label that ``residue_labels`` gives. The label holds the
    residue name and the PDB number. A residue whose ``(name, number)``
    pair repeats gets a shifted number instead, because GMSO merges
    residues that share a name and a number. Chains stay available
    through each site's molecule/group labels.

    The function takes a compound instead of a ``Protein`` because
    ``mb.solvate`` and ``mb.fill_box`` return a plain ``Compound`` that
    holds the protein as a child. Sites of particles outside a
    ``Residue`` (solvent, ions) keep the residue that the generic
    converter gave them.

    Four kinds of data are not carried over, because GMSO's data model
    has no slot for them: formal charges, bond orders, insertion codes,
    and the HETATM flag. A GMSO site charge is a partial charge, so it
    stays unset for a typing engine to fill.

    Parameters
    ----------
    compound : mbuild.Compound
        The compound to convert. It may be a ``Protein`` or any
        compound that holds ``Residue`` compounds below it.
    box : mbuild.Box, optional
        The unit cell written to the topology.
    **kwargs
        Passed to ``Compound.to_gmso``.

    Returns
    -------
    gmso.Topology
        The topology with per-site residue names and numbers.
    """
    gmso = import_("gmso")  # noqa: F841
    from gmso.abc.abstract_site import Residue as GMSOResidue

    # Compound.to_gmso, not compound.to_gmso: Protein.to_gmso calls
    # this function, so the bound call would recurse.
    topology = Compound.to_gmso(compound, box=box, **kwargs)
    labels = residue_labels(compound)
    particles = list(compound.particles())
    sites = list(topology.sites)
    # The guard compares site count, per-site names, and per-site
    # positions. Names alone cannot detect a reorder of same-name
    # residues; positions can, because every atom sits at its own
    # coordinates.
    aligned = len(sites) == len(particles) and np.allclose(
        topology.positions.to_value("nm"), compound.xyz, atol=1e-6
    )
    if not aligned or any(
        site.name != particle.name for site, particle in zip(sites, particles)
    ):
        raise MBuildError(
            "Site order of the GMSO topology does not match the "
            "compound's particles; cannot restore residue identity."
        )
    for site, particle in zip(sites, particles):
        label = labels.get(particle)
        if label is not None:
            site.residue = GMSOResidue(name=label[0], number=label[1])
    return topology


def save(compound, filename, **kwargs):
    """Save a compound that holds residues, through GMSO where possible.

    ``conversion.save`` calls the module-level GMSO converter, which
    numbers residues per name, so a saved file describes the wrong
    residues. This function routes the GMSO extensions (.gro, .gsd,
    .data, .xyz, .mcf, .top) through the ``to_gmso`` above. A ``.pdb``
    file goes to ``save_pdb`` when the compound provides that method.
    Every other extension goes to ``Compound.save``.

    Use it for a packed system, for example the result of
    ``mb.solvate``. A ``Protein.save`` call reaches this function on
    its own.

    Parameters
    ----------
    compound : mbuild.Compound
        The compound to write.
    filename : str
        Path of the file to write. The extension selects the writer.
    **kwargs
        Passed to the selected writer. Pass ``box`` to set the unit
        cell. A ``.gro`` or ``.top`` write without a box logs a
        warning, because the writer then takes the bounding box of the
        compound, and a packed system carries no box of its own.

    Raises
    ------
    MBuildError
        When a ``.data``, ``.mcf`` or ``.top`` write finds a topology
        that carries no force-field parameters. Those three files hold
        the parameters, so the topology must be typed first.
    """
    extension = os.path.splitext(str(filename))[-1].lower()
    if extension == ".pdb" and hasattr(compound, "save_pdb"):
        # A compound that carries its own PDB writer must use it. The
        # generic ParmEd writer cannot express residue numbers, chain
        # identifiers, HETATM records, or a selective CONECT policy, so
        # it writes a blank chain column and no CONECT records.
        unexpected = set(kwargs) - {"overwrite"}
        if unexpected:
            raise MBuildError(
                "Saving a Protein to .pdb uses save_pdb(), which takes "
                f"only 'overwrite'; the arguments {sorted(unexpected)} "
                "would be ignored."
            )
        return compound.save_pdb(filename, overwrite=kwargs.get("overwrite", False))
    if extension not in _GMSO_EXTENSIONS:
        return Compound.save(compound, filename, **kwargs)
    overwrite = kwargs.pop("overwrite", False)
    if os.path.exists(filename) and not overwrite:
        raise IOError(f"{filename} exists; not overwriting")
    # conversion.save consumes these two and does not hand them to the
    # GMSO writers; drop them the same way.
    kwargs.pop("residues", None)
    kwargs.pop("include_ports", None)
    box = kwargs.pop("box", None)
    if extension in _BOXED_EXTENSIONS and box is None and compound.box is None:
        logger.warning(
            f"No box is set, so the {extension} writer takes the "
            "bounding box of the compound. mb.solvate and mb.fill_box "
            "do not set a box on the packed system. The file then holds "
            "a box that is smaller than the packing box. Pass "
            "box=mb.Box(...) to write the box you packed into."
        )
    topology = to_gmso(compound, box=box)
    # is_typed() reports whether the topology carries any parameters at
    # all, which is the state that mb.solvate and the loader leave. A
    # partly typed topology still fails inside GMSO, and only the
    # writer knows what it needs, so this check does not go further.
    if extension in _TYPED_EXTENSIONS and not topology.is_typed():
        raise MBuildError(
            f"A {extension} file holds force-field parameters. The "
            "topology of this compound carries none, so the writer "
            "would fail inside GMSO. Build the topology with "
            "mbuild.biopolymers.to_gmso(), apply a force field to it, "
            "and save it from GMSO."
        )
    if extension == ".gro":
        # The gro writer reads site.molecule before site.residue, and
        # molecule holds the chain label. Clear it so the writer takes
        # the per-residue name that to_gmso set. Only the gro branch
        # clears it: GMSO's top writer groups sites by molecule, and
        # _get_unique_molecules raises when molecule is None.
        for site in topology.sites:
            site.molecule = None
    topology.save(filename=filename, overwrite=overwrite, **kwargs)
