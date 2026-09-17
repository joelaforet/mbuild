"""Load protonated protein PDB files by template matching.

The loader matches residues against chemical templates, in the same
way modern residue-template PDB readers do. Such tools can therefore
read a protein that mBuild loads and later modifies.

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
matching model needs. The input file must be fully protonated at the
desired pH, for example with PDBFixer or Reduce:
PDBFixer: https://github.com/openmm/pdbfixer
Reduce: https://github.com/rlabduke/reduce
"""

import logging
from collections import deque
from dataclasses import replace
from functools import lru_cache

import numpy as np

from mbuild import clone, force_overlap
from mbuild.biopolymers.ccd import CCDLibrary
from mbuild.biopolymers.fragments import _as_residues, _check_resname
from mbuild.biopolymers.matching import (
    _bridge_scope_conflict,
    _bridge_scope_message,
    _filter_crosslink_candidates,
    _match_residue,
    _matches_agree,
    _ter_break_reason,
)
from mbuild.biopolymers.protein_pdb_io import (
    _check_residue_membership,
    _format_residue_label,
    _parse_pdb,
    _pdb_name_field,
    pdb_text,
    write_pdb,
)
from mbuild.biopolymers.residue import Chain, InterResidueBond, Residue
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError
from mbuild.port import Port
from mbuild.utils.io import import_

logger = logging.getLogger(__name__)


#: Length, in nm, of the bond that a new Port forms. It is a rounded
#: value near the single-bond lengths that ``attach`` and
#: ``add_port_at`` form; mBuild's ``Polymer.from_big_smiles`` uses
#: 0.145 nm for a C-C bond. Each port of a bond takes half of it.
#: This follows the ``Polymer.add_monomer`` convention. Relaxation
#: corrects the length afterwards (see ``relax_fragments``).
_PORT_SEPARATION = 0.15


#: Element symbols, in upper case, of the atoms that join two residues
#: through one covalent bond in a PDB entry. The CYS-CYS disulfide and
#: the SEC-SEC diselenide are the two common bridges.
_BRIDGING_ELEMENTS = frozenset(("S", "SE"))


def _chain_of(residue):
    """Return the Chain ancestor of a residue.

    A fragment residue can sit under a wrapper Compound inside its
    Chain, so the direct parent is not always the Chain.
    """
    return next(
        ancestor for ancestor in residue.ancestors() if isinstance(ancestor, Chain)
    )


def _pdb_label(residue):
    """Return the loader-style label of a residue, such as ``CYS A:22``.

    The loader labels a residue by its PDB fields while it reads the
    file, in ``_PdbResidue.label``. This function writes the same text
    for a built residue. Every error and every warning that the module
    writes after the build names a residue in this one format.

    Parameters
    ----------
    residue : Residue
        The residue to label.

    Returns
    -------
    str
        The residue name, the chain identifier, and the residue number
        with its insertion code.
    """
    chain_id = _chain_of(residue).chain_id
    return _format_residue_label(residue.name, chain_id, residue.resnum, residue.icode)


def _atom_in_residue(residue, atom_name):
    """Return the named particle of a residue, or None.

    Particles are found by name instead of by label, because labels can
    go stale after ``remove()``.
    """
    return next(residue.particles_by_name(atom_name), None)


def _assign_template(residue, variant):
    """Assign a template variant and its formal charges to a residue.

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


def _remove_particles_and_ports(root, atom, particles):
    """Remove particles bonded to an atom and drop the opened ports.

    ``Compound.remove`` leaves one auto-generated port on the atom per
    severed bond. Those ports are removed here, so the atom keeps only
    the ports the caller made. The second ``remove`` call scans the
    whole compound once more for orphaned ports. This cost is accepted
    to stay on the public ``Compound`` API.

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
    root.remove(list(particles))
    new_ports = [
        p for p in residue.children if isinstance(p, Port) and p not in old_ports
    ]
    if new_ports:
        root.remove(new_ports)


@lru_cache(maxsize=1)
def _rdkit_bond_orders():
    """Return the RDKit bond type of every bond order this package uses.

    The table is built on the first call, not at import, because RDKit
    is an optional dependency and the module must import without it.
    ``lru_cache`` then holds the one table, so a caller that reads it
    per bond does not rebuild it. ``Protein.to_rdkit`` is the only
    reader in this package: it maps each mBuild bond order to an RDKit
    bond type. One table keeps that mapping in one place, so a new bond
    order is added once.

    The table is stricter than the map in ``mbuild.conversion`` by
    intent. That map turns UNSPECIFIED into the order 0.0. This
    package needs a real bond order on every bond, so an absent key must fail.

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


class Protein(Compound):
    """A protein loaded from a fully protonated PDB file.

    The hierarchy is ``Protein -> Chain -> Residue -> particles``. Atom
    particles use canonical CCD names; bonds carry the template bond
    orders; each ``Residue`` records its matched template and net formal
    charge.

    A protein holds four charge attributes. Each attribute holds a
    different quantity. ``Compound.charge`` is the partial-charge
    attribute of mBuild's core data model. On a container it is the sum
    over its particles. This module never sets it, so it stays None for
    every particle a protein holds. ``Residue.atom_formal_charges``
    maps an atom name to the integer formal charge that the matched CCD
    template gives that atom. It holds an entry only for atoms whose
    charge is not zero. ``Residue.formal_charge`` is the sum of those
    values.
    ``Protein.net_formal_charge`` is the sum of ``formal_charge`` over
    every residue.

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

    def __init__(
        self,
        filename=None,
        library=None,
        download=False,
        name="Protein",
    ):
        super().__init__(name=name)
        self.library = library or CCDLibrary(download=download)
        #: Inter-residue bonds that residue adjacency does not imply:
        #: the disulfides and other crosslinks found at load time, and
        #: every bond that attach() or record_bond() forms.
        self.cross_bonds = []
        #: Map of anchor particle -> tuple of the hydrogen names that
        #: were removed at that particle to open a port. Repeated ports
        #: at one atom accumulate in the entry.
        #: ``_port_along_hydrogens`` is the only writer, so the entry
        #: names the hydrogens that a bond displaces and no other
        #: removed atom.
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

        crosslink_orders = {}

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
                # The residue bridges to a second residue through a
                # sulfur or selenium atom, which mBuild builds for CYS
                # only. The check sits here and not in _match_residue,
                # because only this scope holds the CONECT records and
                # the other residues, which the message must name.
                message = _bridge_scope_conflict(group, groups, conects, self.library)
                if message is not None:
                    raise MBuildError(message) from error
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
        self._warn_on_renamed_hydrogens(groups, matches)

        self._build(groups, matches, conects, crosslink_orders)
        if box is not None:
            self.box = box

    @staticmethod
    def _warn_on_renamed_hydrogens(groups, matches):
        """Warn once when hydrogens were assigned by geometry and renamed.

        The matcher places a hydrogen whose name no template carries on
        the nearest heavy atom and gives it that atom's CCD hydrogen
        name (see ``matching._hydrogen_candidates_by_geometry``). The
        particles, and every file written from them, then carry the
        CCD names and not the names of the input file. One warning for
        the whole load says so, with an example, so the user is not
        surprised by the renamed atoms and can check the placement.
        """
        renamed = 0
        example = None
        for group, match in zip(groups, matches):
            hit = False
            for record in group.records:
                atom = match.record_atoms[id(record)]
                if record.name != atom.name and atom.element.upper() == "H":
                    hit = True
                    if example is None:
                        example = f"{record.name} of {group.label} is now {atom.name}"
            renamed += hit
        if renamed:
            logger.warning(
                f"{renamed} residues carry hydrogen names that no residue "
                f"template uses (for example, {example}). Each such "
                "hydrogen was assigned to the heavy atom nearest to it "
                "and renamed to the CCD name. Files written from this "
                "structure carry the CCD names."
            )

    def _build(self, groups, matches, conects, crosslink_orders):
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
            _assign_template(residue, match.variant)
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
        self._bond_crosslinks(groups, matches, residues, conects, crosslink_orders)
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

    def _bond_crosslinks(self, groups, matches, residues, conects, crosslink_orders):
        """Form the crosslink bonds and record them in ``cross_bonds``.

        ``crosslink_orders`` holds the bond order of each record of a
        bond-records file, keyed by the two residue addresses of that
        record: chain identifier, residue number, insertion code and
        atom name. A bond that no record names is a disulfide from the
        CCD templates, and it takes the order 1.
        """
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
            order = crosslink_orders.get(
                frozenset(
                    (
                        (group.chain_id, group.resnum, group.icode, record.name),
                        (
                            other_group.chain_id,
                            other_group.resnum,
                            other_group.icode,
                            other_record.name,
                        ),
                    )
                ),
                1,
            )
            self.add_bond((particle1, particle2), bond_order=float(order))
            self.cross_bonds.append(
                InterResidueBond(
                    residue1=residue,
                    residue2=other_residue,
                    atom1_name=record.name,
                    atom2_name=other_record.name,
                    order=order,
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
                residues = [particle.parent for particle in particles]
                symbols = {particle.element.symbol.upper() for particle in particles}
                # A CONECT between two sulfur or selenium atoms asks for
                # a bridge. Only the CYS templates carry the crosslink,
                # so name that limit instead of the generic text.
                if symbols <= _BRIDGING_ELEMENTS and {
                    residue.name for residue in residues
                } != {"CYS"}:
                    raise MBuildError(
                        _bridge_scope_message(
                            _pdb_label(residues[0]),
                            particles[0].name,
                            _pdb_label(residues[1]),
                            particles[1].name,
                            "To load the entry without the bridge, remove "
                            "that CONECT record.",
                        )
                    )
                raise MBuildError(
                    f"CONECT record between serials {serials} does not "
                    "correspond to any bond the residue templates predict. "
                    "mBuild does not guess chemistry for unknown bonds."
                )

    # ------------------------------------------------------------------
    # Canonical Compound verbs, routed to residue-aware behavior
    # ------------------------------------------------------------------
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

    def to_rdkit(self, embed=False):
        """Create a sanitized RDKit molecule of the (modified) protein.

        Unlike the generic ``Compound.to_rdkit``, this export carries
        the chemistry the package knows. It writes the formal charges of
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

    def save_pdb(self, filename, overwrite=False):
        """Write a prepared PDB file for downstream residue-template loaders.

        The file carries residue names, real PDB residue numbers, chain
        identifiers, HETATM records for residues mBuild built, a TER
        after each chain, and CONECT records for exactly the bonds that
        residue adjacency cannot imply. A strict residue-template reader
        fails on a CONECT its definitions cannot explain, so peptide
        bonds are left implied rather than written.

        Parameters
        ----------
        filename : str
            Path of the file to write.
        overwrite : bool, optional, default=False
            Overwrite an existing file.
        """
        write_pdb(self, filename, overwrite=overwrite)

    def visualize(self, show_box=False):  # pragma: no cover
        """Show the protein in an NGLView widget.

        The standard residues draw as a cartoon. Every ``HETATM``
        residue, which is where an attached fragment lives, draws as
        licorice, and so does every protein residue that carries a
        recorded inter-residue bond, on top of its cartoon. The bond
        between a modified lysine and its fragment is therefore drawn,
        which it would not be if the licorice stopped at the fragment:
        NGLView draws a bond only when both of its atoms are in the
        representation's selection. The widget reads the same PDB text
        that ``save_pdb`` writes, so the bonds it shows are the bonds the
        file carries.

        Parameters
        ----------
        show_box : bool, optional, default=False
            Draw the unit cell, when the protein has a box. The box of
            a loaded structure is the crystal cell of the input file,
            which is rarely what a reader wants to look at, so it is
            off unless asked for.

        ``Compound.visualize`` is not used here. Its arguments (ports,
        bead sizes) do not apply to a protein, and its backends rebuild
        the structure through a converter that drops the residue
        information this view is organised around.

        Returns
        -------
        nglview.NGLWidget
            The widget. Return it from a notebook cell to display it.

        Raises
        ------
        ImportError
            If nglview is not installed.
        """
        nglview = import_("nglview")

        widget = nglview.NGLWidget(nglview.TextStructure(pdb_text(self), ext="pdb"))
        widget.clear_representations()
        widget.add_representation("cartoon", selection="protein")
        widget.add_representation(
            "licorice", selection=self._licorice_selection(), radius=0.25
        )
        if show_box and self.box is not None:
            widget.add_representation("unitcell")
        return widget

    def _licorice_selection(self):
        """NGL selection for the fragments and the residues bonded to them.

        ``hetero`` covers every attached fragment. Each residue that a
        recorded inter-residue bond touches is added by number and
        chain, in NGL's ``<resnum>:<chain>`` form, so a modified lysine
        or a disulfide-linked cysteine shows its side chain and the
        bond that leaves it.
        """
        terms = ["hetero"]
        for bond in self.cross_bonds:
            for residue in (bond.residue1, bond.residue2):
                if residue.hetatm:
                    continue
                chain_id = _chain_of(residue).chain_id
                term = f"{residue.resnum}{residue.icode}"
                terms.append(f"{term}:{chain_id}" if chain_id else term)
        return " or ".join(dict.fromkeys(terms))

    def bond_records(self):
        """Return one plain dict per recorded inter-residue bond.

        Each dict describes a covalent modification completely: which
        residues bond through which atoms, which leaving atoms were
        removed on each side, and the bond order. Downstream tools
        format these records into their own vocabulary (residue
        definitions, crosslink declarations, templates).

        A record names the atoms that the bond itself displaces. A
        residue that was prepared in another protonation state carries
        that state in its template, and not in the record of a bond at
        the same atom. The two are separate properties of the product.

        The residue numbers alone do not address a residue. A number
        repeats across chains, and an insertion code splits one number
        into several residues. ``chain_ids`` and ``icodes`` complete
        the address, so a reader finds each residue with the same three
        fields that ``get_residue`` takes.

        The records are sorted by the address of the first residue,
        then by the address of the second: chain identifier, residue
        number, insertion code, and atom name. The order of
        ``cross_bonds`` follows the order of the calls that made the
        bonds, and a reload makes them in file order, so the two orders
        differ. A sorted list compares equal across a save and a load.

        Returns
        -------
        list of dict
            One dict per record, with the keys ``residue_names``,
            ``residue_numbers``, ``chain_ids``, ``icodes``,
            ``atom_names``, ``leaving_atoms`` (one list per side), and
            ``bond_order``. Every key but ``bond_order`` holds one pair,
            in the order (residue 1, residue 2).
        """
        records = [
            {
                "residue_names": (bond.residue1.name, bond.residue2.name),
                "residue_numbers": (bond.residue1.resnum, bond.residue2.resnum),
                "chain_ids": (
                    _chain_of(bond.residue1).chain_id,
                    _chain_of(bond.residue2).chain_id,
                ),
                "icodes": (bond.residue1.icode, bond.residue2.icode),
                "atom_names": (bond.atom1_name, bond.atom2_name),
                "leaving_atoms": (list(bond.leaving1), list(bond.leaving2)),
                "bond_order": bond.order,
            }
            for bond in self.cross_bonds
        ]
        return sorted(
            records,
            key=lambda record: tuple(
                record[key][side]
                for side in (0, 1)
                for key in ("chain_ids", "residue_numbers", "icodes", "atom_names")
            ),
        )

    def relax_fragments(
        self,
        residues=None,
        n_steps=0,
        tolerance=50.0,
        platform="CPU",
    ):
        """Relax attached fragments while the protein stays fixed.

        Runs an energy minimization with mBuild's generic
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
        n_steps : int, optional, default=0
            Maximum minimization iterations. It reaches OpenMM as
            ``maxIterations``. ``0`` has OpenMM's meaning: the
            minimizer runs until it meets ``tolerance``, with no
            iteration limit. A positive value caps the iterations
            instead. The generic force field here is not a
            production force field, so this relaxation only removes
            bad geometry. The user must still run an energy
            minimization with a real force field before a simulation.
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

    def add_port_at(
        self,
        resnum,
        atom_name,
        chain_id=None,
        icode="",
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
        port = self._port_along_hydrogens(self, atom, hydrogens)
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
        relax=True,
        leaving_atom_names=None,
        fragment_leaving_atom_names=None,
    ):
        """Bond a fragment Compound onto a residue of this protein.

        One leaving group leaves each side per unit of bond order, so a
        double bond removes two from each atom. By default the leaving
        groups are hydrogens bonded to the named protein atom and to the
        named fragment atom. ``leaving_atom_names`` and
        ``fragment_leaving_atom_names`` name other atoms; a heavy atom
        takes the whole branch beyond its bond with it.

        Ports along the broken bonds align the fragment
        (``force_overlap``). A bond with ``bond_order`` then forms
        between the two named atoms. The new inter-residue bond is
        recorded in ``cross_bonds``, together with the names of every
        atom that left. The record holds everything a downstream tool
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
            Residue compounds and therefore gets wrapped. The name
            takes three characters or fewer; a longer name raises a
            ValueError.
        bond_order : int, optional, default=1
            Order of the new bond.
        leaving_atom_names : str or sequence of str, optional
            Names of the atoms that leave the protein atom. One name
            per unit of bond order, each bonded to the protein atom.
            By default the leaving atoms are hydrogens, taken in
            alphabetical order. The hydrogens on one atom are
            chemically equivalent, so the choice does not change the
            chemistry; a downstream residue library describes the
            product by naming the atom that is absent, so pass the name
            that library expects to make the written file match it.
            A name may also be a heavy atom. Then the whole substituent
            on the far side of that bond leaves with it, so naming the
            oxygen of a hydroxyl group removes the oxygen and its
            hydrogen together. A residue that this empties is dropped.
        fragment_leaving_atom_names : str or sequence of str, optional
            The same, for the fragment atom. A glycan built by a glycan
            builder ends in a hydroxyl residue on the anomeric carbon;
            naming its oxygen here removes that residue and opens the
            glycosidic bond site.
        relax : bool, optional, default=True
            When the placed fragment overlaps existing atoms, run an
            energy minimization that moves only the fragment
            (see ``relax_fragments``). The minimization runs until it
            converges. When a build must return in bounded time, pass
            ``relax=False`` and call ``relax_fragments`` with a
            positive ``n_steps``.

        Returns
        -------
        InterResidueBond
            The recorded bond, as appended to ``cross_bonds``.
        """
        # The name is checked first, before the attachment site is
        # read and before any warning is logged. The check ran inside
        # _as_residues before, so the charge warning of a valid site
        # reached the user ahead of the error about the name.
        _check_resname(fragment_resname)
        bond_order = int(bond_order)
        site_residue, site_atom, site_hydrogens = self._attachment_site(
            resnum, atom_name, chain_id, icode, bond_order, leaving_atom_names
        )
        self._warn_on_kept_charge(site_residue, resnum, atom_name)
        added, frag_residues = _as_residues(clone(fragment), fragment_resname)
        frag_atom, frag_residue, frag_hydrogens = self._fragment_site(
            frag_residues,
            fragment_atom_name,
            fragment_resnum,
            bond_order,
            fragment_leaving_atom_names,
        )

        # One leaving group per bond order unit on each side, a hydrogen
        # by default (the polymer.add_monomer convention); the port
        # points along the sum of the broken-bond vectors. Both ports
        # are opened while
        # the fragment is still detached, so that each removal runs on
        # its own compound.
        site_port = self._port_along_hydrogens(self, site_atom, site_hydrogens)
        site_residue.add(site_port, label="attach_site")
        frag_port = self._port_along_hydrogens(added, frag_atom, frag_hydrogens)
        added.add(frag_port, label="attach_frag")
        # A heavy leaving atom takes its substituent with it. When that
        # substituent was a residue of its own (the hydroxyl residue a
        # glycan builder puts on the anomeric carbon), ``Compound.remove``
        # has already detached the emptied residue from the fragment.
        # It is dropped from the list here too, so it is not numbered
        # and never reaches the chain.
        frag_residues[:] = [r for r in frag_residues if r is added or r.root is added]

        self._adopt_fragment(added, frag_residues, site_residue)
        self._align_on_ports(added, frag_port, site_port, bond_order)

        # The record is appended before the relaxation step, so the
        # protein state stays complete and consistent when relaxation
        # fails: the fragment is already bonded at this point.
        #
        # The site side reads the leaving-atom ledger, which names the
        # hydrogens that ports at this atom removed. A proton that
        # deprotonate() removed is not in the ledger, so the record
        # names only the atoms that this bond displaces.
        record = InterResidueBond(
            residue1=site_residue,
            residue2=frag_residue,
            atom1_name=site_atom.name,
            atom2_name=frag_atom.name,
            order=bond_order,
            leaving1=self._leaving_atoms[site_atom],
            leaving2=self._leaving_atoms[frag_atom],
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
            f"{_pdb_label(residue)} atom {atom_name} has formal charge "
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

    def _attachment_site(
        self, resnum, atom_name, chain_id, icode, bond_order, leaving_names=None
    ):
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
            Order of the new bond. One leaving group leaves per unit.

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
        hydrogens = self._bonded_hydrogens(
            atom, residue.name, bond_order, leaving_names
        )
        return residue, atom, hydrogens

    @staticmethod
    def _fragment_site(
        frag_residues, atom_name, fragment_resnum, bond_order, leaving_names=None
    ):
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
            Order of the new bond. One leaving group leaves per unit.

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
        hydrogens = Protein._bonded_hydrogens(
            atom, residue.name, bond_order, leaving_names
        )
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
        # The user asked for the relaxation and got it, so the result is
        # reported rather than warned about. Ordinary van der Waals
        # contacts near 2 A are expected after a minimization; only a
        # contact that stays well inside that means the minimizer failed.
        closest = self._closest_contact(added, site_atom, frag_atom)
        if closest is None:
            return
        if closest < 0.15:
            logger.warning(
                f"Relaxation left a fragment atom {closest * 10:.2f} A from an "
                "existing atom. Inspect the site; the minimizer may not have "
                "converged."
            )
        else:
            logger.info(
                f"Fragment relaxed; closest contact with existing atoms is now "
                f"{closest * 10:.2f} A."
            )

    def _closest_contact(self, added, site_atom, frag_atom):
        """Return the smallest distance (nm) from a fragment atom to any other atom.

        The new bond pair is left out. Returns None when there is
        nothing to compare against.
        """
        from scipy.spatial import cKDTree

        added_particles = [p for p in added.particles() if p is not frag_atom]
        added_set = set(added.particles()) | {site_atom}
        others = [p for p in self.particles() if p not in added_set]
        if not others or not added_particles:
            return None
        distances, _ = cKDTree([p.pos for p in others]).query(
            [p.pos for p in added_particles]
        )
        return float(distances.min())

    def _port_along_hydrogens(self, root, atom, hydrogens):
        """Remove the leaving groups and return a Port pointing along them.

        ``hydrogens`` are the atoms whose bond to ``atom`` breaks; each
        heavy one stands for its whole branch (``_substituent``). The
        port points along the sum of the broken-bond vectors, or along
        the first one when the sum is degenerate.
        ``Compound.remove`` leaves one auto-generated port on the atom
        per severed bond. Those ports are removed here, the same cleanup
        that ``Polymer.add_monomer`` does, so the returned Port is the
        only open port at the atom.

        This method is the only writer of the ``_leaving_atoms``
        ledger. The removed names are written under the anchor atom,
        which is where ``record_bond`` reads its default leaving-atom
        lists. A second port at the same atom adds its removed names to
        the entry, so the entry always names every hydrogen that a bond
        displaced at that atom.
        """
        orientation = sum(h.pos - atom.pos for h in hydrogens)
        if np.linalg.norm(orientation) < 1e-8:
            orientation = hydrogens[0].pos - atom.pos
        leaving = []
        for anchor in hydrogens:
            leaving.extend(self._substituent(atom, anchor))
        _remove_particles_and_ports(root, atom, leaving)
        self._leaving_atoms[atom] = tuple(
            sorted(self._leaving_atoms.get(atom, ()) + tuple(p.name for p in leaving))
        )
        return Port(
            anchor=atom, orientation=orientation, separation=_PORT_SEPARATION / 2
        )

    @staticmethod
    def _substituent(atom, anchor):
        """Return the atoms that leave when the ``atom``-``anchor`` bond breaks.

        For a hydrogen that is the hydrogen alone. For a heavy atom it
        is every atom reachable from ``anchor`` without passing through
        ``atom``: the whole group on the far side of the bond. A ring
        that contains both atoms has no far side, so that case raises
        instead of removing the rest of the molecule.
        """
        group = [anchor]
        seen = {atom, anchor}
        queue = deque([anchor])
        while queue:
            for neighbor in queue.popleft().direct_bonds():
                if neighbor in seen:
                    continue
                if atom in neighbor.direct_bonds():
                    raise MBuildError(
                        f"Leaving atom {anchor.name} is in a ring with "
                        f"{atom.name}, so there is no group on the far side "
                        "of the bond to remove. Name a hydrogen or an atom "
                        "outside the ring."
                    )
                seen.add(neighbor)
                group.append(neighbor)
                queue.append(neighbor)
        return group

    @staticmethod
    def _bonded_hydrogens(atom, residue_name, count, names=None):
        """Return ``count`` leaving atoms bonded to the atom.

        One leaving atom per unit of bond order. Without ``names`` they
        are hydrogens, taken in alphabetical order. That order is
        arbitrary as chemistry: the hydrogens on one atom are
        equivalent, so any of them may leave. It is not arbitrary to a
        downstream residue library, which describes the product by
        naming the atom that is absent. Pass ``names`` to choose the
        hydrogens that leave, so the written file matches such a
        description.

        A name may also be a heavy atom bonded to ``atom``. The bond to
        it breaks in place of a bond to a hydrogen, and
        ``_substituent`` takes the group beyond it along. That is how a
        hydroxyl leaves an anomeric carbon when a glycan is attached.

        Parameters
        ----------
        atom : mbuild.Compound
            The anchor atom.
        residue_name : str
            Name of the residue holding the anchor, used in errors.
        count : int
            How many hydrogens leave; one per unit of bond order.
        names : str or sequence of str, optional
            Names of the atoms that leave. Exactly ``count`` names are
            required, and each must name an atom bonded to ``atom``.

        Returns
        -------
        list of mbuild.Compound
            The atoms whose bond to ``atom`` breaks. A heavy atom in
            the list stands for its whole substituent.
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
        if names is None:
            if len(hydrogens) < count:
                raise MBuildError(
                    f"Atom {atom.name} of residue {residue_name} has "
                    f"{len(hydrogens)} bonded hydrogens, but a bond of order "
                    f"{count} must replace {count}. Pick an atom with enough "
                    "hydrogens."
                )
            return hydrogens[:count]

        if isinstance(names, str):
            names = [names]
        names = list(names)
        available = {particle.name: particle for particle in atom.direct_bonds()}
        if len(names) != count:
            raise MBuildError(
                f"A bond of order {count} replaces {count} bonds of "
                f"atom {atom.name} of residue {residue_name}, but "
                f"{len(names)} leaving-atom names were given: {names}."
            )
        if len(set(names)) != len(names):
            raise MBuildError(
                f"The leaving-atom names for atom {atom.name} of residue "
                f"{residue_name} repeat: {names}. Each name must be a "
                "different atom."
            )
        missing = [name for name in names if name not in available]
        if missing:
            raise MBuildError(
                f"Atom {atom.name} of residue {residue_name} has no bonded "
                f"atom named {missing[0]!r}. Its bonded atoms are "
                f"{sorted(available)}."
            )
        # Each leaving group frees one valence unit of the link atom, and
        # the new bond uses one unit per leaving group. A leaving atom
        # held by a double or triple bond would free more than that and
        # leave the link atom short, which nothing downstream reports:
        # RDKit sanitizes an under-valent atom as a radical. Refuse it.
        graph = atom.root.bond_graph
        for name in names:
            order = graph.edges[atom, available[name]].get("bond_order", 1.0)
            if order not in (1.0, 0.0):
                raise MBuildError(
                    f"Leaving atom {name} is joined to {atom.name} of residue "
                    f"{residue_name} by a bond of order {order:g}. Only an "
                    "atom held by a single bond can leave, because the new "
                    "bond replaces one bond on each side."
                )
        return [available[name] for name in names]

    def _warn_on_clashes(self, added, site_atom, frag_atom, cutoff=0.2):
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
