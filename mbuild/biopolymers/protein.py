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

from mbuild.biopolymers.ccd import CCDLibrary
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
from mbuild.utils.io import import_

logger = logging.getLogger(__name__)


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
        #: the disulfides and other crosslinks found at load time.
        self.cross_bonds = []
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

        self._build(groups, matches, conects)
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
        """Form the crosslink bonds and record them in ``cross_bonds``.

        A CONECT record carries no bond order, and the bridges the CCD
        templates describe are single bonds, so every crosslink takes
        the order 1.
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
            _, other_match, other_residue, other_record = expecting[partner_serial]
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
    # Hierarchy queries
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
