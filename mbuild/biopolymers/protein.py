"""Protein recipe: load protonated protein PDB files by template matching.

The loader matches residues against chemical templates, in the same
way modern residue-template PDB readers do, so a protein loaded (and
later modified) with mBuild round-trips through such tools. (The
matching model is verified compatible with the OpenFF Pablo reader by
a parity test that runs wherever openff-pablo is installed.)

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
from dataclasses import dataclass, field, replace

import numpy as np

from mbuild import clone
from mbuild.biopolymers.ccd import CCDLibrary
from mbuild.box import Box
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError
from mbuild.port import Port

logger = logging.getLogger(__name__)

__all__ = [
    "Protein",
    "Chain",
    "Residue",
    "fragment_from_pdb",
    "fragment_from_sdf",
    "prepare_fragment",
]


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
        #: Sparse map of atom name -> integer formal charge, filled by
        #: the loader (from the matched template) and fragment loaders.
        self.atom_formal_charges = {}
        #: Map of site label -> atom name for the fragment's covalent
        #: bond sites, set by attachment points in the SMILES (* or
        #: [*:n]) or by particle tags. attach() uses a lone entry when
        #: no fragment atom name is given; attach_multi() maps each
        #: label to a protein site.
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


def prepare_fragment(compound, resname):
    """Return a fragment as a named Residue with final atom names.

    ``attach()`` wraps and renames fragments internally, so a caller who
    passes a plain Compound cannot know the atom names in advance. This
    helper applies the same wrapping and renaming up front and returns
    the Residue, so the caller can (1) read the names to pick the
    attachment atom, (2) pass the same object to ``attach()``, and
    (3) reuse the names when building an external residue definition
    (e.g. a residue template for a downstream loader) for the fragment.

    Parameters
    ----------
    compound : mbuild.Compound
        The fragment. Cloned; the input is not changed.
    resname : str
        The residue name (up to 3 characters, e.g. "MYR").

    Returns
    -------
    Residue
        A detached residue with unique, stable atom names.
    """
    charges = None
    link_index = None
    if isinstance(compound, str):
        # A SMILES string: load it and keep its formal charges, which
        # an mbuild Compound cannot store. One dummy atom (*) marks the
        # attachment site: it is replaced by a hydrogen (the leaving
        # atom), and its neighbor becomes the fragment's link atom.
        from rdkit import Chem

        from mbuild.conversion import from_rdkit

        parsed = Chem.MolFromSmiles(compound)
        if parsed is None:
            raise MBuildError(f"Could not parse SMILES {compound!r}.")
        editable = Chem.RWMol(parsed)
        dummies = [atom for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
        link_index = {}
        for dummy in dummies:
            label = str(dummy.GetAtomMapNum() or 1)
            if label in link_index:
                raise MBuildError(
                    "Attachment points must carry distinct labels: write "
                    "them as [*:1], [*:2], ... when a fragment has more "
                    "than one."
                )
            neighbors = dummy.GetNeighbors()
            if len(neighbors) != 1:
                raise MBuildError("An attachment point (*) must bond exactly one atom.")
            link_index[label] = neighbors[0].GetIdx()
            dummy.SetAtomicNum(1)
        mol = editable.GetMol()
        Chem.SanitizeMol(mol)
        explicit = Chem.AddHs(mol)
        charges = [atom.GetFormalCharge() for atom in explicit.GetAtoms()]
        elements = [atom.GetSymbol() for atom in explicit.GetAtoms()]
        copied = from_rdkit(rdkit_mol=mol)
    else:
        copied = clone(compound)
    if isinstance(copied, Residue):
        residue = copied
        residue.name = (resname or residue.name)[:3].upper()
    else:
        residue = Protein._wrap_in_residue(copied, resname)
    Protein._ensure_unique_atom_names(residue)
    if charges is not None:
        particles = list(residue.particles())
        symbols = [particle.element.symbol for particle in particles]
        if len(particles) != len(charges) or symbols != elements:
            raise MBuildError(
                "Atom order of the loaded fragment does not match the "
                "SMILES, so formal charges cannot be mapped onto atoms. "
                "This is a bug in the loading path; please report it."
            )
        residue.atom_formal_charges = {
            particle.name: charge
            for particle, charge in zip(particles, charges)
            if charge
        }
        residue.formal_charge = sum(charges)
        residue.link_atoms = {
            label: particles[index].name for label, index in link_index.items()
        }
    if not residue.link_atoms:
        # mBuild's tagged-SMILES idiom: particle tags mark the sites.
        for particle in residue.particles():
            if particle.particle_tag:
                residue.link_atoms[str(particle.particle_tag)] = particle.name
    return residue


def fragment_from_pdb(filename, bond_orders=None):
    """Load a fragment PDB file into Residue compounds for ``attach()``.

    Unlike ``Protein``, this loader matches no templates and stamps no
    chemistry: it is for fragment files (e.g. GLYCAM glycans) whose
    residue codes are not in the CCD. Atom names, residue names, and
    residue numbers come from the records; **bonds come only from the
    file's CONECT records**, so the file must list them (no bonds are
    guessed from distances). Elements come from the element column.

    CONECT records carry no bond orders, and no order is guessed:
    every bond defaults to a single bond, and multiple bonds (e.g. the
    C=O of an N-acetyl sugar) must be declared through ``bond_orders``.

    Parameters
    ----------
    filename : str
        Path of the fragment PDB file.
    bond_orders : dict, optional
        Bond orders for the bonds that are not single. Keys are pairs
        of (residue number, atom name) tuples; values are the orders.
        Example: ``{((2, "C2N"), (2, "O2N")): 2}``. An entry whose
        atoms match no CONECT bond raises an error.

    Returns
    -------
    mbuild.Compound
        A compound whose children are ``Residue`` objects (hetatm=True),
        ready to pass to ``Protein.attach``.
    """
    with open(filename) as handle:
        text = handle.read()
    groups, conects, _ = _parse_pdb(text)
    if not conects:
        raise MBuildError(
            f"{filename} has no CONECT records. fragment_from_pdb takes "
            "connectivity only from CONECT records; add them or load the "
            "fragment from SMILES instead."
        )
    orders = {}
    for key, order in (bond_orders or {}).items():
        orders[frozenset(key)] = float(order)

    fragment = Compound(name="fragment")
    serial_to_particle = {}
    serial_key = {}
    for group in groups:
        residue = Residue(
            resname=group.resname,
            resnum=group.resnum,
            icode=group.icode,
            hetatm=True,
        )
        particles = []
        for record in group.records:
            if not record.element:
                raise MBuildError(
                    f"Atom {record.name!r} of {group.label} has no element "
                    "column; fragment_from_pdb needs elements."
                )
            particle = Compound(
                name=record.name,
                element=record.element.capitalize(),
                pos=record.pos,
            )
            particles.append(particle)
            serial_to_particle[record.serial] = particle
            serial_key[record.serial] = (group.resnum, record.name)
        residue.add(particles)
        fragment.add(residue)
    used_orders = set()
    for pair in conects:
        serials = tuple(pair)
        if len(serials) != 2:
            continue
        particles = [serial_to_particle.get(serial) for serial in serials]
        if None in particles:
            raise MBuildError(
                f"CONECT record references unknown atom serial in {serials}."
            )
        if not fragment.bond_graph.has_edge(*particles):
            key = frozenset(serial_key[serial] for serial in serials)
            order = orders.get(key, 1.0)
            if key in orders:
                used_orders.add(key)
            fragment.add_bond(particles, bond_order=order)
    unused = set(orders) - used_orders
    if unused:
        raise MBuildError(
            "bond_orders entries match no CONECT bond: "
            f"{sorted(tuple(sorted(key)) for key in unused)}."
        )
    if not orders:
        logger.info(
            f"All CONECT bonds of {filename} default to single bonds. "
            "Pass bond_orders={...} to declare multiple bonds."
        )
    return fragment


def fragment_from_sdf(filename, resname):
    """Load one molecule from an SDF file as a named Residue fragment.

    SDF is the preferred rich fragment format: unlike PDB, it encodes
    explicit bond orders and formal charges, together with coordinates.
    Prefer it (or SMILES) over ``fragment_from_pdb`` when you control
    the fragment source. Atom names are assigned as element+index
    (the SDF format has no atom names); read them from the returned
    residue. Formal charges from the SDF are kept on the residue's
    ``atom_formal_charges`` map, so exports carry them; the external
    residue definition is still best built from the same file.

    Parameters
    ----------
    filename : str
        Path of an SDF file holding exactly one molecule with explicit
        hydrogens and coordinates.
    resname : str
        The residue name (up to 3 characters).

    Returns
    -------
    Residue
        A detached residue ready to pass to ``Protein.attach``.
    """
    from mbuild.utils.io import import_

    import_("rdkit")
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(filename), removeHs=False, sanitize=True)
    molecules = [molecule for molecule in supplier if molecule is not None]
    if len(molecules) != 1:
        raise MBuildError(
            f"{filename} holds {len(molecules)} readable molecules; "
            "fragment_from_sdf takes exactly one."
        )
    molecule = molecules[0]
    if molecule.GetNumConformers() == 0:
        raise MBuildError(f"{filename} has no coordinates.")
    if any(atom.GetNumImplicitHs() for atom in molecule.GetAtoms()):
        raise MBuildError(
            f"{filename} has implicit hydrogens; write the SDF with all "
            "hydrogens explicit."
        )
    orders = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
        Chem.BondType.AROMATIC: 1.5,
    }
    conformer = molecule.GetConformer()
    residue = Residue(resname=(resname or "LIG")[:3].upper(), hetatm=True)
    particles = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        particles.append(
            Compound(
                name=atom.GetSymbol(),
                element=atom.GetSymbol(),
                pos=np.array([position.x, position.y, position.z]) / 10.0,
            )
        )
    residue.add(particles)
    for bond in molecule.GetBonds():
        order = orders.get(bond.GetBondType())
        if order is None:
            raise MBuildError(
                f"Unsupported SDF bond type {bond.GetBondType()} in {filename}."
            )
        residue.add_bond(
            (particles[bond.GetBeginAtomIdx()], particles[bond.GetEndAtomIdx()]),
            bond_order=order,
        )
    Protein._ensure_unique_atom_names(residue)
    residue.atom_formal_charges = {
        particles[atom.GetIdx()].name: atom.GetFormalCharge()
        for atom in molecule.GetAtoms()
        if atom.GetFormalCharge()
    }
    residue.formal_charge = sum(residue.atom_formal_charges.values())
    return residue


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
            residue.atom_formal_charges = {
                atom.name: atom.formal_charge
                for atom in match.record_atoms.values()
                if atom.formal_charge
            }
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
    # Canonical Compound verbs, routed to residue-aware behavior
    # ------------------------------------------------------------------
    def save(self, filename, **kwargs):
        """Save the protein; ``.pdb`` files route to ``save_pdb``.

        The generic ParmEd writer cannot express residue numbers, chain
        identifiers, or CONECT records, so a plain ``save`` would write
        a file that silently loses the protein's identity.
        """
        if str(filename).lower().endswith(".pdb"):
            unexpected = set(kwargs) - {"overwrite"}
            if unexpected:
                raise MBuildError(
                    "Saving a Protein to .pdb uses save_pdb(), which takes "
                    f"only 'overwrite'; the arguments {sorted(unexpected)} "
                    "would be ignored."
                )
            return self.save_pdb(filename, overwrite=kwargs.get("overwrite", False))
        return super().save(filename, **kwargs)

    def to_parmed(self, **kwargs):
        """Create a ParmEd structure with residues taken from hierarchy.

        ``conversion.save`` passes ``residues=None`` explicitly, so the
        default must fill in whenever the value is None, not only when
        the key is absent — otherwise every ParmEd-routed format (mol2,
        psf, ...) collapses the protein into one residue.
        """
        if kwargs.get("residues") is None:
            kwargs["residues"] = sorted({residue.name for residue in self.residues()})
        return super().to_parmed(**kwargs)

    def to_rdkit(self):
        """Create a sanitized RDKit molecule of the (modified) protein.

        Unlike the generic ``Compound.to_rdkit``, this export carries
        the chemistry the recipe knows: formal charges from the matched
        templates and fragment records, bond orders, explicit hydrogens,
        one conformer, and PDB residue info on every atom. The result
        sanitizes, so it is directly usable by RDKit and by tools that
        consume RDKit molecules.
        """
        from rdkit import Chem

        editable = Chem.RWMol()
        particle_index = {}
        particle_residue = {}
        chain_of = {}
        for chain in self.chains:
            for residue in self.residues(chain.chain_id):
                for particle in residue.particles():
                    particle_residue[particle] = residue
                    chain_of[particle] = chain.chain_id
        particles = [p for p in self.particles() if not p.port_particle]
        orphans = [p for p in particles if p not in particle_residue]
        if orphans:
            raise MBuildError(
                "Every atom of a Protein must belong to a Residue, but "
                f"{[p.name for p in orphans[:5]]} "
                f"{'(and more) ' if len(orphans) > 5 else ''}do not. Add "
                "atoms through attach() or into a Residue, not directly "
                "onto the Protein."
            )
        for particle in particles:
            residue = particle_residue[particle]
            atom = Chem.Atom(particle.element.atomic_number)
            atom.SetFormalCharge(residue.atom_formal_charges.get(particle.name, 0))
            atom.SetNoImplicit(True)
            info = Chem.AtomPDBResidueInfo()
            info.SetName(
                particle.name.center(4)
                if len(particle.name) >= 4
                else f" {particle.name:<3s}"
            )
            info.SetResidueName(residue.name)
            info.SetResidueNumber(residue.resnum)
            info.SetChainId(chain_of[particle] or " ")
            info.SetInsertionCode(residue.icode or " ")
            info.SetIsHeteroAtom(residue.hetatm)
            atom.SetPDBResidueInfo(info)
            particle_index[particle] = editable.AddAtom(atom)

        aromatic_pairs = []
        for particle1, particle2, data in self.bonds(return_bond_order=True):
            if particle1 not in particle_index or particle2 not in particle_index:
                continue
            order = float(data["bond_order"])
            if order <= 0.0:
                raise MBuildError(
                    f"Bond {particle1.name}-{particle2.name} has no bond "
                    "order; cannot export chemistry to RDKit."
                )
            bond_type = {
                1.0: Chem.BondType.SINGLE,
                2.0: Chem.BondType.DOUBLE,
                3.0: Chem.BondType.TRIPLE,
                1.5: Chem.BondType.AROMATIC,
            }[order]
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
        overlap so a downstream simulation does not blow up. Every atom
        outside the given residues gets zero mass, which OpenMM treats
        as immobile, so the protein coordinates do not change.

        Parameters
        ----------
        residues : iterable of Residue, optional
            The residues allowed to move. Default: every HETATM
            residue (i.e. all attached fragments).
        n_steps : int, optional, default=500
            Maximum minimization iterations.
        tolerance : float, optional, default=50.0
            Energy tolerance in kJ/mol/nm.
        platform : str, optional, default="CPU"
            OpenMM platform name.
        """
        from mbuild.simulation import OpenMMSimulation

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

        This is the canonical-mBuild escape hatch under ``attach()``:
        the named atom loses ``bond_order`` hydrogens, and a ``Port``
        pointing along the removed hydrogens is added to the residue.
        Use it with ``force_overlap`` for placements ``attach()`` does
        not cover. Note that bonds formed this way are not recorded in
        ``cross_bonds``.
        """
        residue = self.get_residue(resnum, chain_id=chain_id, icode=icode)
        atom = self.get_atom(resnum, atom_name, chain_id=chain_id, icode=icode)
        hydrogens = self._bonded_hydrogens(atom, residue.name, int(bond_order))
        orientation = sum(h.pos - atom.pos for h in hydrogens)
        if np.linalg.norm(orientation) < 1e-8:
            orientation = hydrogens[0].pos - atom.pos
        for hydrogen in hydrogens:
            self.remove(hydrogen)
        port = Port(anchor=atom, orientation=orientation, separation=separation / 2)
        residue.add(port, label="port[$]")
        return port

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

        ``bond_order`` hydrogens leave each side: hydrogens bonded to
        the named protein atom, and hydrogens bonded to the named
        fragment atom (one per unit of bond order, so a double bond
        removes two from each atom).
        Ports along the removed-hydrogen vectors align the fragment
        (``force_overlap``), a bond with ``bond_order`` forms between
        the two named atoms, and the new inter-residue bond is recorded
        in ``cross_bonds`` together with the removed (leaving) hydrogen
        names — everything a downstream tool needs to describe the
        modification (see ``bond_records``).

        The fragment is cloned; the original is not changed. Fragment
        residues keep their identity: a fragment whose children are
        ``Residue`` compounds (a single PTM residue, a linear polymer,
        or a branched glycan) is added residue-per-residue metadata
        intact; any other Compound is wrapped into one new ``Residue``.
        To build branched, multiply-linked structures, call ``attach``
        repeatedly — an attached residue is addressable like any other,
        so a later call can target it. Every call records its bond, so
        residues may carry any number of links inside mBuild.

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

        Returns
        -------
        InterResidueBond
            The recorded bond, as appended to ``cross_bonds``.
        """
        bond_order = int(bond_order)
        site_residue = self.get_residue(resnum, chain_id=chain_id, icode=icode)
        site_atom = self.get_atom(resnum, atom_name, chain_id=chain_id, icode=icode)
        site_hydrogens = self._bonded_hydrogens(
            site_atom, site_residue.name, bond_order
        )

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

        if fragment_atom_name is None:
            linked = [
                (label, residue)
                for residue in frag_residues
                for label in residue.link_atoms
            ]
            if len(linked) != 1:
                raise MBuildError(
                    "attach() bonds one site, but the fragment carries "
                    f"{len(linked)} attachment points. Pass "
                    "fragment_atom_name, mark one site (* in the SMILES), "
                    "or use attach_multi() for several labeled sites."
                )
            label, link_residue = linked[0]
            fragment_atom_name = link_residue.link_atoms[label]
            fragment_resnum = link_residue.resnum

        frag_atom, frag_residue = self._find_fragment_atom(
            frag_residues, fragment_atom_name, fragment_resnum
        )
        frag_hydrogens = self._bonded_hydrogens(
            frag_atom, frag_residue.name, bond_order
        )

        # One hydrogen leaves per bond order unit on each side (the
        # polymer.add_monomer convention); the port points along the sum
        # of the removed-hydrogen vectors.
        site_orientation = sum(h.pos - site_atom.pos for h in site_hydrogens)
        frag_orientation = sum(h.pos - frag_atom.pos for h in frag_hydrogens)
        if np.linalg.norm(site_orientation) < 1e-8:
            site_orientation = site_hydrogens[0].pos - site_atom.pos
        if np.linalg.norm(frag_orientation) < 1e-8:
            frag_orientation = frag_hydrogens[0].pos - frag_atom.pos

        for hydrogen in site_hydrogens:
            self.remove(hydrogen)
        frag_root = added if added.parent is None else added.root
        for hydrogen in frag_hydrogens:
            frag_root.remove(hydrogen)

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

        clashes = self._warn_on_clashes(added, site_atom, frag_atom)
        if clashes and relax:
            logger.info("Relaxing the placed fragment with the protein held fixed.")
            self.relax_fragments(residues=frag_residues)
            clashes = self._warn_on_clashes(added, site_atom, frag_atom)
            if not clashes:
                logger.info("Fragment overlaps resolved by relaxation.")

        record = InterResidueBond(
            residue1=site_residue,
            residue2=frag_residue,
            atom1_name=site_atom.name,
            atom2_name=frag_atom.name,
            order=bond_order,
            leaving1=tuple(sorted(h.name for h in site_hydrogens)),
            leaving2=tuple(sorted(h.name for h in frag_hydrogens)),
        )
        self.cross_bonds.append(record)
        return record

    def attach_multi(
        self, fragment, sites, fragment_resname=None, separation=0.15, relax=True
    ):
        """Tether one fragment to several protein sites at once.

        ``sites`` maps each attachment-point label of the fragment
        (``[*:1]``, ``[*:2]`` in its SMILES, or particle tags) to the
        protein site that label bonds, given as keyword arguments::

            protein.attach_multi(
                "[*:1]OCCOCCOCC[*:2]",
                sites={
                    "1": dict(resnum=63, atom_name="NZ", chain_id="A"),
                    "2": dict(resnum=27, atom_name="NZ", chain_id="A"),
                },
                fragment_resname="PEG",
            )

        The first site is placed by rigid port alignment, exactly like
        ``attach``. Each further tether is placed geometrically: the
        fragment rotates about its first bond and shears along its own
        axis until the link atom reaches the site, then a short
        protein-fixed minimization (``relax_fragments``) removes the
        strain. A warning appears only when the fragment cannot span
        its sites. All bonds are recorded in ``cross_bonds``.

        Returns
        -------
        list of InterResidueBond
            One record per site, in ``sites`` order.
        """
        if isinstance(fragment, str):
            fragment = prepare_fragment(fragment, fragment_resname or "LIG")
        probe_residues = (
            [fragment]
            if isinstance(fragment, Residue)
            else [c for c in fragment.successors() if isinstance(c, Residue)]
        )
        available = {
            label: (residue.resnum, residue.link_atoms[label])
            for residue in probe_residues
            for label in residue.link_atoms
        }
        missing = set(sites) - set(available)
        if missing:
            raise MBuildError(
                f"The fragment has no attachment points labeled "
                f"{sorted(missing)}; it has {sorted(available)}."
            )

        labels = list(sites)
        first_resnum, first_atom = available[labels[0]]
        records = [
            self.attach(
                fragment,
                first_atom,
                fragment_resnum=first_resnum,
                fragment_resname=fragment_resname,
                separation=separation,
                relax=False,  # relax once, after every tether is formed
                **sites[labels[0]],
            )
        ]
        for label in labels[1:]:
            candidates = [
                residue
                for residue in self.residues()
                if label in residue.link_atoms and residue.hetatm
            ]
            if len(candidates) != 1:
                raise MBuildError(
                    f"Attachment label {label!r} matches {len(candidates)} "
                    "residues in the protein; attach_multi supports one "
                    "fragment instance per label at a time."
                )
            site = dict(sites[label])
            bond_order = int(site.pop("bond_order", 1))
            site_residue = self.get_residue(
                site["resnum"],
                chain_id=site.get("chain_id"),
                icode=site.get("icode", ""),
            )
            site_atom = self.get_atom(
                site["resnum"],
                site["atom_name"],
                chain_id=site.get("chain_id"),
                icode=site.get("icode", ""),
            )
            frag_residue = candidates[0]
            frag_atom = _atom_in_residue(frag_residue, frag_residue.link_atoms[label])
            # Rigidly rotate the fragment about its first-bond atom so
            # this link atom points at its site: geometry is preserved,
            # and the remaining gap is only the slack of the fragment,
            # which minimization can close.
            pivot = _atom_in_residue(records[0].residue2, records[0].atom2_name)
            fragment_particles = [
                particle
                for residue in self.residues()
                if residue.hetatm and set(residue.link_atoms) & set(available)
                for particle in residue.particles()
            ]
            self._rotate_about(
                fragment_particles,
                pivot.pos,
                frag_atom.pos - pivot.pos,
                site_atom.pos - pivot.pos,
            )
            # Then shear the fragment along the pivot->link axis so the
            # link atom reaches the site. Bonds stretch a little
            # everywhere (local strain), which minimization fixes; it
            # cannot fix a fragment that has to travel.
            approach = frag_atom.pos - site_atom.pos
            approach_norm = float(np.linalg.norm(approach))
            if approach_norm > 1e-8:
                target = site_atom.pos + separation * approach / approach_norm
            else:
                target = site_atom.pos + np.array([separation, 0.0, 0.0])
            self._stretch_along(fragment_particles, pivot.pos, frag_atom, target)
            site_hydrogens = self._bonded_hydrogens(
                site_atom, site_residue.name, bond_order
            )
            frag_hydrogens = self._bonded_hydrogens(
                frag_atom, frag_residue.name, bond_order
            )
            for hydrogen in (*site_hydrogens, *frag_hydrogens):
                self.remove(hydrogen)
            self.add_bond((site_atom, frag_atom), bond_order=float(bond_order))
            distance = float(np.linalg.norm(site_atom.pos - frag_atom.pos))
            message = (
                f"Tether {label!r} formed at {distance * 10:.1f} A between "
                f"{site_residue.name} {site_residue.resnum} {site_atom.name} "
                f"and {frag_residue.name} {frag_residue.resnum} "
                f"{frag_atom.name}."
            )
            if distance > 0.2:
                logger.warning(
                    message + " The fragment cannot reach this site with "
                    "realistic geometry; relax and inspect the structure."
                )
            else:
                logger.info(message)
            record = InterResidueBond(
                residue1=site_residue,
                residue2=frag_residue,
                atom1_name=site_atom.name,
                atom2_name=frag_atom.name,
                order=bond_order,
                leaving1=tuple(sorted(h.name for h in site_hydrogens)),
                leaving2=tuple(sorted(h.name for h in frag_hydrogens)),
            )
            self.cross_bonds.append(record)
            records.append(record)
        if relax:
            logger.info("Relaxing tethered fragments with the protein held fixed.")
            self.relax_fragments(n_steps=2000)
            for label, record in zip(labels[1:], records[1:]):
                atom1 = _atom_in_residue(record.residue1, record.atom1_name)
                atom2 = _atom_in_residue(record.residue2, record.atom2_name)
                distance = float(np.linalg.norm(atom1.pos - atom2.pos))
                logger.info(
                    f"Tether {label!r} after relaxation: {distance * 10:.1f} A."
                )
        return records

    @staticmethod
    def _rotate_about(particles, pivot, from_vector, to_vector):
        """Rigidly rotate particles about a pivot, aligning two vectors."""
        norm_from = np.linalg.norm(from_vector)
        norm_to = np.linalg.norm(to_vector)
        if norm_from < 1e-8 or norm_to < 1e-8:
            return
        unit_from = from_vector / norm_from
        unit_to = to_vector / norm_to
        axis = np.cross(unit_from, unit_to)
        sine = np.linalg.norm(axis)
        cosine = float(np.dot(unit_from, unit_to))
        if sine < 1e-8:
            return  # already aligned (or exactly opposite: leave as is)
        axis = axis / sine
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
        for particle in particles:
            particle.pos = pivot + rotation @ (particle.pos - pivot)

    @staticmethod
    def _stretch_along(particles, pivot, link_atom, target):
        """Shear particles along pivot->link so the link atom hits target.

        Every particle moves by a fraction of the needed displacement,
        proportional to its projection onto the pivot->link axis, so the
        strain spreads over the whole fragment.
        """
        axis = link_atom.pos - pivot
        length = float(np.linalg.norm(axis))
        if length < 1e-8:
            return
        unit = axis / length
        displacement = target - link_atom.pos
        for particle in particles:
            weight = float(np.dot(particle.pos - pivot, unit)) / length
            particle.pos = particle.pos + np.clip(weight, 0.0, 1.0) * displacement

    @staticmethod
    def _bonded_hydrogens(atom, residue_name, count):
        """Return ``count`` hydrogens bonded to the atom (sorted by name).

        One hydrogen leaves per unit of bond order. Reactions that
        remove other leaving groups (e.g. condensations) are the domain
        of future reaction recipes built on top of ``attach``.
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
        atom (excluding the new bond pair) and warns below ``cutoff`` nm,
        so the user knows to relax the structure before simulating.
        """
        from scipy.spatial import cKDTree

        added_particles = [p for p in added.particles() if not p.port_particle]
        added_set = set(added_particles) | {site_atom}
        others = [
            p for p in self.particles() if p not in added_set and not p.port_particle
        ]
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

        The PDB export and template matching need atom names that are
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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_pdb(self, filename, overwrite=False):
        """Write a prepared PDB file for downstream residue-template loaders.

        Bonds made through ``add_port_at`` + ``force_overlap`` get
        CONECT records but no ``cross_bonds`` record, so downstream
        loaders need residue definitions for them from you.

        mBuild's generic PDB writer (ParmEd via ``save``) cannot express
        residue numbers, chain identifiers, HETATM records, or a
        selective CONECT policy, so this recipe has its own writer. The
        conventions follow the RCSB standard and residue-template
        readers: ``ATOM`` for residues loaded
        from ATOM records, ``HETATM`` for attached fragments and
        heteroatoms, ``TER`` after every chain, ``CRYST1`` when a box is
        set, and ``CONECT`` records **only** for bonds between
        non-adjacent residues (disulfides, attached fragments, branch
        links) — peptide bonds are implied by residue adjacency, and a
        CONECT that the residue templates cannot explain makes strict
        loaders fail.

        Parameters
        ----------
        filename : str
            Path of the PDB file to write.
        overwrite : bool, optional, default=False
            Overwrite the file if it exists.
        """
        import os

        if os.path.exists(filename) and not overwrite:
            raise IOError(f"{filename} exists; not overwriting")

        lines = []
        if self.box is not None:
            a, b, c = (length * 10.0 for length in self.box.lengths)
            alpha, beta, gamma = self.box.angles
            lines.append(
                f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
                f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1"
            )

        serial = 0
        particle_serial = {}
        particle_residue = {}
        residue_order = {}
        for chain in self.chains:
            residue = None
            # Residues are written sorted by number: template readers form
            # polymer links only between record-adjacent residues, so
            # backbone order in the file must follow residue numbers,
            # not attachment order.
            for residue in sorted(
                self.residues(chain.chain_id),
                key=lambda res: (res.resnum, res.icode),
            ):
                residue_order[id(residue)] = len(residue_order)
                if residue.resnum > 9999:
                    raise MBuildError(
                        "PDB residue numbers larger than 9999 are not supported."
                    )
                for particle in residue.particles():
                    serial += 1
                    if serial > 99999:
                        raise MBuildError(
                            "PDB atom serials larger than 99999 are not supported."
                        )
                    particle_serial[particle] = serial
                    particle_residue[particle] = residue
                    lines.append(
                        self._pdb_atom_line(serial, particle, residue, chain.chain_id)
                    )
            if residue is not None:
                serial += 1
                lines.append(
                    f"TER   {serial:5d}      {residue.name:<3s} "
                    f"{chain.chain_id or ' ':1s}{residue.resnum:4d}"
                    f"{residue.icode or ' ':1s}"
                )

        for line in self._conect_lines(
            particle_serial, particle_residue, residue_order
        ):
            lines.append(line)
        lines.append("END")

        with open(filename, "w") as handle:
            handle.write("\n".join(lines) + "\n")

    @staticmethod
    def _pdb_atom_line(serial, particle, residue, chain_id):
        record = "HETATM" if residue.hetatm else "ATOM  "
        name = particle.name
        # PDB alignment: names shorter than 4 characters are right-shifted
        # by one column (element starts in column 14).
        name_field = name.center(4) if len(name) >= 4 else f" {name:<3s}"
        x, y, z = particle.pos * 10.0
        element = particle.element.symbol.upper() if particle.element else ""
        return (
            f"{record}{serial:5d} {name_field[:4]} {residue.name:<3.3s} "
            f"{chain_id or ' ':1.1s}{residue.resnum:4d}{residue.icode or ' ':1.1s}"
            f"   {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"          {element:>2.2s}"
        )

    def _conect_lines(self, particle_serial, particle_residue, residue_order):
        """Yield CONECT lines, following the RCSB convention.

        Every bond that touches a HETATM residue is listed, because PDB
        viewers (e.g. PyMOL) treat CONECT records as the complete bond
        list for HETATM atoms and skip distance-based perception for
        them. Bonds between different ATOM residues are listed too
        (disulfides), except peptide bonds, which residue adjacency
        implies. Strict template readers accept these records because
        their residue definitions predict all of them.
        """
        partners = {}
        for particle1, particle2 in self.bonds():
            residue1 = particle_residue.get(particle1)
            residue2 = particle_residue.get(particle2)
            if residue1 is None or residue2 is None:
                continue
            if residue1 is residue2:
                if not residue1.hetatm:
                    continue
            elif not (residue1.hetatm or residue2.hetatm):
                adjacent = (
                    abs(residue_order[id(residue1)] - residue_order[id(residue2)]) == 1
                )
                if adjacent and {particle1.name, particle2.name} == {"C", "N"}:
                    continue  # implied peptide bond
            serial1 = particle_serial[particle1]
            serial2 = particle_serial[particle2]
            partners.setdefault(serial1, []).append(serial2)
            partners.setdefault(serial2, []).append(serial1)
        for serial in sorted(partners):
            bonded = sorted(partners[serial])
            for start in range(0, len(bonded), 4):
                chunk = bonded[start : start + 4]
                yield (
                    "CONECT"
                    + f"{serial:5d}"
                    + "".join(f"{other:5d}" for other in chunk)
                )

    def bond_records(self):
        """Return one plain dict per recorded inter-residue bond.

        Each dict describes a covalent modification completely: which
        residues bond through which atoms, which leaving atoms were
        removed on each side, and the bond order. Downstream tools
        format these records into their own vocabulary (residue
        definitions, crosslink declarations, templates).

        Keys per record: ``residue_names``, ``residue_numbers``,
        ``atom_names``, ``leaving_atoms`` (one list per side), and
        ``bond_order``.
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
