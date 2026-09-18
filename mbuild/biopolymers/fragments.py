"""Fragment preparation for covalent attachment to biopolymers.

A fragment is any Compound organized as Residue objects with unique
atom names, chemistry-complete bonds, and (optionally) labeled
attachment sites. These helpers build such fragments from SMILES
strings (with * attachment points) and from Compounds the caller
already has.
"""

import logging
from pathlib import Path

from mbuild import clone
from mbuild.biopolymers.ccd import CCDLibrary, _cif_category_rows, _parse_cif_blocks
from mbuild.biopolymers.protein_pdb_io import _parse_pdb
from mbuild.biopolymers.residue import Residue
from mbuild.bond_graph import BondGraph
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError

logger = logging.getLogger(__name__)

__all__ = [
    "fragment_from_ccd",
    "fragment_from_pdb",
    "fragment_from_smiles",
    "prepare_fragment",
]


#: Address of the wwPDB Format Guide section that the residue name
#: limit comes from. ``_check_resname`` puts it in its error message.
_WWPDB_COORDINATE_SECTION_URL = (
    "https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html"
)

#: Longest residue name a caller may give a fragment. The wwPDB Format
#: Guide v3.30, section 9 (Coordinate Section, ATOM), declares the
#: residue name in columns 18-20 and column 21 blank:
#: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
#: The reader of this package also accepts a four-character name,
#: because a membrane builder writes lipid names that fill column 21.
#: A fragment name is generated here, so it stays inside the three
#: declared columns, and every PDB reader accepts it.
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
            f"component code. {_WWPDB_COORDINATE_SECTION_URL}"
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
    turns the absent ``bond_order`` into 0.0. This package needs a real
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
        _move_into_residue(compound, residue)
    logger.info(f"Fragment {compound.name!r} wrapped into residue {resname!r}.")
    return residue


def _move_into_residue(compound, residue):
    """Move the particles, ports and bonds of a compound into a residue.

    The particles become direct children of ``residue`` and every bond
    is added again with its bond order. ``compound`` is left empty.
    ``_wrap_in_residue`` uses this to flatten a fragment into a new
    residue, and ``Protein.mutate`` uses it to move a placed side chain
    into the residue it now belongs to.

    Parameters
    ----------
    compound : mbuild.Compound
        The compound to empty. It must be detached from any Protein.
    residue : Residue
        The residue that receives the particles.
    """
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


def fragment_from_ccd(code, link_atom, library=None):
    """Build a fragment Residue from a Chemical Component Dictionary entry.

    The CCD component supplies everything the fragment needs: atom
    names, elements, formal charges, bonds with orders, and the ideal
    coordinates that the wwPDB publishes with every component. Because
    the atoms carry the names the CCD gives them, a downstream residue
    library that knows the component recognises the written file with
    no hand-written definition.

    Parameters
    ----------
    code : str
        The component's CCD code, for example ``"KUT"``.
    link_atom : str
        Name of the atom that will bond to the protein. It is recorded
        in ``link_atoms``, so ``Protein.attach`` needs no
        ``fragment_atom_name``.
    library : CCDLibrary, optional
        The library to read from. The default downloads any component
        that is not among the shipped ones.

    Returns
    -------
    Residue
        A detached residue at the component's ideal coordinates.

    Raises
    ------
    KeyError
        If the code is not a CCD component, or ``link_atom`` is not one
        of its atoms.
    """
    library = library or CCDLibrary(download=True)
    template = library[code][0]
    positions = _ideal_coordinates(code, library)

    residue = Residue(resname=code.upper(), resnum=1, hetatm=True)
    particles = {}
    for atom in template.atoms:
        particle = Compound(
            name=atom.name, element=atom.element, pos=positions[atom.name]
        )
        particles[atom.name] = particle
        residue.add(particle)
    for bond in template.bonds:
        residue.add_bond(
            (particles[bond.atom1], particles[bond.atom2]),
            bond_order=float(bond.order),
        )
    residue.atom_formal_charges = {
        atom.name: atom.formal_charge for atom in template.atoms if atom.formal_charge
    }
    residue.formal_charge = sum(residue.atom_formal_charges.values())
    if link_atom not in particles:
        raise KeyError(
            f"{code} has no atom named {link_atom!r}. Its atoms are {sorted(particles)}."
        )
    residue.link_atoms = {"1": link_atom}
    return residue


def _ideal_coordinates(code, library):
    """Return ``{atom name: position in nm}`` from a component's CIF file.

    ``CCDLibrary`` is a matcher and keeps no coordinates, but the file it
    caches carries them in the ``pdbx_model_Cartn_*_ideal`` columns.
    """
    library[code]  # fills the cache, and raises for an unknown code
    for directory in library._paths:
        path = directory / f"{code.upper()}.cif"
        if path.exists():
            break
    else:
        raise KeyError(f"No cached CCD file for {code!r}.")
    keys, loops = _parse_cif_blocks(path.read_text())
    rows = _cif_category_rows(keys, loops, "_chem_comp_atom")
    return {
        row["atom_id"]: tuple(
            float(row[f"pdbx_model_Cartn_{axis}_ideal"]) / 10.0 for axis in "xyz"
        )
        for row in rows
    }


def fragment_from_pdb(filename, charge=0):
    """Load a PDB file with CONECT records as a fragment of Residues.

    This is the loader for fragments that no residue template library
    describes, such as a glycan written by a glycan builder with its
    own residue names. Every residue in the file becomes one
    ``Residue`` that keeps its name, number and atom names, so a
    downstream force field that assigns parameters by residue name
    still recognises it after ``Protein.attach``.

    A PDB file carries no bond orders, so they are perceived from the
    connectivity and the coordinates with RDKit, for the net charge the
    caller gives. Formal charges that the perception assigns are stored
    on each residue. The file must list every bond in ``CONECT``
    records; nothing is inferred from distances.

    Parameters
    ----------
    filename : str or path-like
        The PDB file. All of its ``ATOM`` and ``HETATM`` records form
        one fragment.
    charge : int, optional, default=0
        Net formal charge of the fragment, for the bond order
        perception.

    Returns
    -------
    mbuild.Compound
        A Compound whose children are the residues of the file, in
        file order, each flagged as ``HETATM``.

    Raises
    ------
    MBuildError
        If the file has no ``CONECT`` records, if a ``CONECT`` names an
        atom that is not in the file, or if RDKit cannot perceive bond
        orders for the given charge.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    with open(filename) as handle:
        groups, conects, _ = _parse_pdb(handle.read())
    if not conects:
        raise MBuildError(
            f"{filename} has no CONECT records. fragment_from_pdb reads bonds "
            "from CONECT records only, so every bond of the fragment must be "
            "listed. Glycan builders and most modelling programs write them."
        )
    records = [record for group in groups for record in group.records]
    index_of = {}
    for index, record in enumerate(records):
        if record.serial in index_of:
            raise MBuildError(
                f"{filename} repeats atom serial {record.serial}, so its "
                "CONECT records are ambiguous."
            )
        index_of[record.serial] = index
        if not record.element:
            raise MBuildError(
                f"{filename}: atom {record.name} of residue {record.resname} "
                "has no element symbol in columns 77-78."
            )
    pairs = set()
    for conect in conects:
        serial1, serial2 = sorted(conect)
        for serial in (serial1, serial2):
            if serial not in index_of:
                raise MBuildError(
                    f"{filename}: CONECT record names atom serial {serial}, "
                    "which is not in the file."
                )
        pairs.add((index_of[serial1], index_of[serial2]))

    editable = Chem.RWMol()
    for record in records:
        rdkit_atom = Chem.Atom(record.element.capitalize())
        rdkit_atom.SetNoImplicit(True)
        editable.AddAtom(rdkit_atom)
    for index1, index2 in sorted(pairs):
        editable.AddBond(index1, index2, Chem.BondType.SINGLE)
    conformer = Chem.Conformer(len(records))
    for index, record in enumerate(records):
        conformer.SetAtomPosition(index, [float(x) * 10.0 for x in record.pos])
    editable.AddConformer(conformer)
    mol = editable.GetMol()
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
        Chem.SanitizeMol(mol)
    except Exception as error:
        raise MBuildError(
            f"{filename}: RDKit could not perceive bond orders for a net "
            f"charge of {charge:+d} ({error}). Check the CONECT records and "
            "the charge."
        ) from error

    fragment = Compound(name=Path(filename).stem[:_MAX_RESNAME_LENGTH].upper())
    particles = []
    for group in groups:
        residue = Residue(
            resname=group.resname, resnum=group.resnum, icode=group.icode, hetatm=True
        )
        for record in group.records:
            particle = Compound(
                name=record.name, element=record.element, pos=record.pos
            )
            residue.add(particle)
            particles.append(particle)
        fragment.add(residue)
        residue_indices = range(len(particles) - len(group.records), len(particles))
        residue.atom_formal_charges = {
            records[index].name: mol.GetAtomWithIdx(index).GetFormalCharge()
            for index in residue_indices
            if mol.GetAtomWithIdx(index).GetFormalCharge()
        }
        residue.formal_charge = sum(residue.atom_formal_charges.values())
    for bond in mol.GetBonds():
        fragment.add_bond(
            (particles[bond.GetBeginAtomIdx()], particles[bond.GetEndAtomIdx()]),
            bond_order=bond.GetBondTypeAsDouble(),
        )
    logger.info(
        f"Loaded fragment {filename} with {len(groups)} residues, "
        f"{len(particles)} atoms and {mol.GetNumBonds()} bonds."
    )
    return fragment


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
    This function does not validate ``resname``. The public callers,
    ``prepare_fragment`` and ``Protein.attach``, call ``_check_resname``
    as their first step, so a name that is too long raises before any
    object is cloned or changed. A new caller must do the same.
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
