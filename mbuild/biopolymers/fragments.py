"""Fragment preparation for covalent attachment to biopolymers.

A fragment is any Compound organized as Residue objects with unique
atom names, chemistry-complete bonds, and (optionally) labeled
attachment sites. These helpers build such fragments from SMILES
strings (with * attachment points) and SDF files.
"""

import logging

import numpy as np

from mbuild import clone
from mbuild.biopolymers.protein import Protein, Residue
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError

logger = logging.getLogger(__name__)

__all__ = ["fragment_from_sdf", "prepare_fragment"]


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


def fragment_from_sdf(filename, resname):
    """Load one molecule from an SDF file as a named Residue fragment.

    SDF is the preferred rich fragment format: unlike PDB, it encodes
    explicit bond orders and formal charges, together with coordinates.
    Prefer it (or SMILES) over ``fragment_from_pdb`` when you control
    the fragment source. Atom names are assigned as element+index
    (the SDF format has no atom names); read them from the returned
    residue. Formal charges from the SDF are kept on the residue's
    ``atom_formal_charges`` map, so exports carry them; the external
    (Pablo) residue definition is still best built from the same file.

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
