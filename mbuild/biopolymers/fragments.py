"""Fragment preparation for covalent attachment to biopolymers.

A fragment is any Compound organized as Residue objects with unique
atom names, chemistry-complete bonds, and (optionally) labeled
attachment sites. These helpers build such fragments from SMILES
strings (with * attachment points) and SDF files.
"""

import itertools
import logging

import numpy as np

from mbuild import clone
from mbuild.biopolymers.protein import Residue, _rdkit_bond_orders
from mbuild.bond_graph import BondGraph
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError

logger = logging.getLogger(__name__)

__all__ = ["fragment_from_sdf", "fragment_from_smiles", "prepare_fragment"]


#: Longest residue name a caller may give a fragment. The wwPDB Format
#: Guide v3.30, section 9 (Coordinate Section, ATOM), declares the
#: residue name in columns 18-20 and column 21 blank. The reader of
#: this recipe also accepts a four-character name, because a membrane
#: builder writes lipid names that fill column 21. A fragment name is
#: generated here, so it stays inside the three declared columns, and
#: every PDB reader accepts it.
_MAX_RESNAME_LENGTH = 3


def _check_resname(resname):
    """Raise if a caller-supplied fragment residue name is too long.

    A longer name was cut to three characters without a message. Two
    different names then produced the same residue name, which is the
    collision that a longer name was picked to avoid.

    Parameters
    ----------
    resname : str or None
        The name the caller gave. None and the empty string pass, as
        the caller asked for a generated name.

    Raises
    ------
    ValueError
        If the name is longer than three characters.
    """
    if resname and len(resname) > _MAX_RESNAME_LENGTH:
        raise ValueError(
            f"Fragment residue name {resname!r} is {len(resname)} "
            f"characters; the limit is {_MAX_RESNAME_LENGTH}. The wwPDB "
            "Format Guide declares the residue name in columns 18-20, so "
            "a longer name does not fit. Pick a name of three characters "
            "or fewer that does not collide with an assigned CCD "
            "component code."
        )


def _wrap_in_residue(compound, fragment_resname):
    """Wrap a detached Compound into a single flat Residue.

    The compound's particles become direct children of the new
    Residue, and each bond is added again with its bond order. The
    residue must be flat because some exports (for example the GMSO
    converter) derive a particle's residue from its direct parent, so
    a particle nested below a wrapper Compound would get the wrong
    residue. Ports of the compound are carried over.

    ``Compound.flatten`` does the same move on one compound, but it is
    not reused here, because it loses the bond orders.
    ``Compound.flatten`` collects each bond as a pair of particles and
    adds it again as ``add_bond(pair)``, and ``Compound.add_bond``
    turns the absent ``bond_order`` into 0.0. The recipe needs a real
    bond order on every bond: ``Protein.to_rdkit`` raises on a bond
    whose order is 0.0. The loop below therefore reads ``bond_order``
    from each bond and writes the same value back.
    """
    # A caller-supplied name is checked by _check_resname, so only a
    # generated fallback name can be too long here. The compound name
    # is cut to the limit; see _MAX_RESNAME_LENGTH for the reason.
    resname = (fragment_resname or compound.name or "LIG")[:_MAX_RESNAME_LENGTH].upper()
    if not resname.isalnum():
        resname = "LIG"
    residue = Residue(resname=resname, resnum=1, hetatm=True)
    if not compound.children:
        # The compound is itself a particle; add it directly.
        residue.add(compound)
    else:
        particles = list(compound.particles())
        bonds = [
            (particle1, particle2, data["bond_order"])
            for particle1, particle2, data in compound.bonds(return_bond_order=True)
        ]
        ports = list(compound.all_ports())
        for part in particles + ports:
            part.parent.children.remove(part)
            part.parent = None
        for particle in particles:
            # Compound.add removes the parent from the bond graph only
            # when the added child carries a graph. Give each detached
            # particle the single-node graph a standalone particle has,
            # so the Residue itself does not stay in the graph as a
            # spurious particle node.
            particle.bond_graph = BondGraph()
            particle.bond_graph.add_node(particle)
        residue.add(particles)
        for port in ports:
            residue.add(port)
        for particle1, particle2, order in bonds:
            residue.add_bond((particle1, particle2), bond_order=order)
    logger.info(f"Fragment {compound.name!r} wrapped into residue {resname!r}.")
    return residue


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


def fragment_from_smiles(smiles, resname):
    """Load a SMILES string as a named Residue fragment.

    The residue keeps the formal charges from the SMILES, which an
    mbuild Compound cannot store. Each dummy atom (``*`` or ``[*:n]``)
    marks an attachment site: it is replaced by a hydrogen (the leaving
    atom), and its neighbor is recorded in ``link_atoms``.

    Parameters
    ----------
    smiles : str
        The fragment as a SMILES string.
    resname : str
        The residue name, of three characters or fewer (e.g. "MYR").

    Returns
    -------
    Residue
        A detached residue with unique, stable atom names.

    Raises
    ------
    ValueError
        If ``resname`` is longer than three characters.
    """
    _check_resname(resname)

    from rdkit import Chem

    from mbuild.conversion import from_rdkit

    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise MBuildError(f"Could not parse SMILES {smiles!r}.")
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
    # Pass the explicit-hydrogen molecule. from_rdkit calls AddHs on
    # its input; on a molecule whose hydrogens are already explicit
    # that call adds no atoms and keeps the atom order. The particle
    # order and the charges list then come from one AddHs result, so
    # the positional mapping below is exact.
    copied = from_rdkit(rdkit_mol=explicit)
    residue = _wrap_in_residue(copied, resname)
    _ensure_unique_atom_names(residue)
    particles = list(residue.particles())
    symbols = [particle.element.symbol for particle in particles]
    if symbols != elements:
        raise MBuildError(
            "Atom order of the loaded fragment does not match the "
            "SMILES, so formal charges cannot be mapped onto atoms. "
            "This is a bug in the loading path; please report it."
        )
    residue.atom_formal_charges = {
        particle.name: charge for particle, charge in zip(particles, charges) if charge
    }
    residue.formal_charge = sum(charges)
    residue.link_atoms = {
        label: particles[index].name for label, index in link_index.items()
    }
    return residue


def _as_residues(compound, resname):
    """Return a fragment compound and its residues, wrapping if needed.

    A Residue input is its own single residue. A Compound that holds
    Residue children keeps those residues. Any other Compound is
    wrapped into one new flat Residue named after ``resname``. Atom
    names are made unique inside each residue, not across the fragment,
    so every residue keeps its own ``atom_formal_charges`` and
    ``link_atoms`` keys.

    The compound is changed in place, so callers clone first.

    Parameters
    ----------
    compound : mbuild.Compound
        The fragment.
    resname : str or None
        Residue name for a wrapped compound.

    Returns
    -------
    compound : mbuild.Compound
        The input compound, or the new Residue that wraps it. A wrapped
        input is replaced, so callers must use the returned object.
    residues : list of Residue
        The residues of the returned compound.

    Notes
    -----
    ``resname`` is not checked here. Each caller checks the name
    before it starts, so the error comes before any other work.
    """
    if isinstance(compound, Residue):
        # successors() does not yield the compound itself, so a Residue
        # input needs its own arm.
        residues = [compound]
    else:
        residues = [
            child for child in compound.successors() if isinstance(child, Residue)
        ]
        if not residues:
            compound = _wrap_in_residue(compound, resname)
            residues = [compound]
    for residue in residues:
        _ensure_unique_atom_names(residue)
    return compound, residues


def prepare_fragment(compound, resname):
    """Return a fragment as a named Residue with final atom names.

    ``attach()`` wraps and renames fragments internally, so a caller who
    passes a plain Compound cannot know the atom names in advance. This
    helper applies the same wrapping and renaming up front and returns
    the Residue, so the caller can (1) read the names to pick the
    attachment atom, (2) pass the same object to ``attach()``, and
    (3) reuse the names when building an external residue definition
    (e.g. a residue template for a downstream loader) for the fragment.

    A SMILES string is also accepted and is loaded through
    ``fragment_from_smiles``.

    Residue detection follows ``attach()``: a Residue input and a
    Compound that holds Residue children keep their residues; only a
    residue-less Compound is wrapped into one new Residue. Atom names
    are made unique within each residue, not across the fragment, so
    each residue keeps its ``atom_formal_charges`` and ``link_atoms``
    keys.

    Parameters
    ----------
    compound : mbuild.Compound or str
        The fragment, or a SMILES string for it. A Compound is cloned;
        the input is not changed.
    resname : str
        The residue name, of three characters or fewer (e.g. "MYR").
        Applied only when the fragment is wrapped or is itself a
        Residue; existing Residue children keep their names.

    Returns
    -------
    Residue or mbuild.Compound
        A detached fragment with unique, stable atom names. A Compound
        with Residue children is returned as the Compound that holds
        them; other inputs return a Residue.

    Raises
    ------
    ValueError
        If ``resname`` is longer than three characters. A Residue
        input that gets no ``resname`` keeps its own name, and that
        name has the same limit.
    """
    _check_resname(resname)
    if isinstance(compound, str):
        return fragment_from_smiles(compound, resname)
    copied = clone(compound)
    if isinstance(copied, Residue):
        # A Residue input takes the requested name. Without one it
        # keeps the name the caller built it with. Both names come
        # from the caller, so the used name is checked here and a name
        # that is too long raises. Residue children of a Compound keep
        # the names they came with.
        chosen = resname or copied.name
        _check_resname(chosen)
        copied.name = chosen.upper()
    copied, residues = _as_residues(copied, resname)
    for residue in residues:
        if not residue.link_atoms:
            # mBuild's tagged-SMILES idiom: particle tags mark the sites.
            for particle in residue.particles():
                if particle.particle_tag:
                    residue.link_atoms[str(particle.particle_tag)] = particle.name
    return copied


def fragment_from_sdf(filename, resname):
    """Load one molecule from an SDF file as a named Residue fragment.

    SDF is the preferred rich fragment format: unlike PDB, it encodes
    explicit bond orders and formal charges, together with coordinates.
    Atom names are assigned as element+index (the SDF format has no
    atom names); read them from the returned residue. Formal charges
    from the SDF are kept on the residue's ``atom_formal_charges`` map,
    so exports carry them; an external residue definition for a
    downstream loader is still best built from the same file.

    Parameters
    ----------
    filename : str
        Path of an SDF file holding exactly one molecule with explicit
        hydrogens and coordinates.
    resname : str
        The residue name, of three characters or fewer.

    Returns
    -------
    Residue
        A detached residue ready to pass to ``Protein.attach``.

    Raises
    ------
    ValueError
        If ``resname`` is longer than three characters.
    """
    from mbuild.utils.io import import_

    _check_resname(resname)
    import_("rdkit")
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(filename), removeHs=False, sanitize=True)
    # Two entries are enough to decide the count; do not parse the rest.
    molecules = [
        molecule for molecule in itertools.islice(supplier, 2) if molecule is not None
    ]
    if len(molecules) != 1:
        raise MBuildError(
            f"{filename} does not hold exactly one readable molecule; "
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
    orders = {bond_type: order for order, bond_type in _rdkit_bond_orders().items()}
    conformer = molecule.GetConformer()
    # Only the fallback name can be too long; see _MAX_RESNAME_LENGTH.
    residue = Residue(
        resname=(resname or "LIG")[:_MAX_RESNAME_LENGTH].upper(), hetatm=True
    )
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
    _ensure_unique_atom_names(residue)
    residue.atom_formal_charges = {
        particles[atom.GetIdx()].name: atom.GetFormalCharge()
        for atom in molecule.GetAtoms()
        if atom.GetFormalCharge()
    }
    residue.formal_charge = sum(residue.atom_formal_charges.values())
    return residue
