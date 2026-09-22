"""Chain, Residue, and InterResidueBond: the residue data model of a biopolymer.

The module also holds the helpers that read or edit one residue, which
the other modules of the package share: naming a residue the way the
loader does, finding an atom by name, assigning a template, removing
atoms with the ports that ``Compound.remove`` opens, and building the
RDKit molecule that every export and match in the package starts from.
"""

from dataclasses import dataclass
from functools import lru_cache

from mbuild.biopolymers.protein_pdb_io import _format_residue_label
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError
from mbuild.port import Port
from mbuild.utils.io import import_

__all__ = ["Chain", "InterResidueBond", "Residue"]


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
    #: The reaction string that made the bond, or None for a plain
    #: substitution of leaving atoms. See ``mbuild.biopolymers.reactions``.
    reaction: str = None


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


def _rdkit_mol(compound, particles):
    """Return an editable RDKit molecule of ``particles`` and a particle-to-index map.

    Every RDKit export and match in this package starts from this
    molecule: ``Protein.to_rdkit``, the template matching of
    ``reactions``, the conformer sampling of ``relax``, and
    ``draw_fragment``. Each atom carries its element, the formal charge
    that its residue's ``atom_formal_charges`` gives it, and no
    implicit hydrogens. Each bond between two of the particles carries
    the bond order that ``compound`` holds; an aromatic bond and its
    atoms are flagged aromatic, which ``SanitizeMol`` requires. A bond
    with an order outside ``_rdkit_bond_orders`` raises, because this
    package needs a real order on every bond. The molecule is not
    sanitized and holds no conformer; each caller finishes it.

    Parameters
    ----------
    compound : mbuild.Compound
        The compound that owns the bond graph, for example a Protein
        or a detached fragment.
    particles : iterable of mbuild.Compound
        The particles to export, each a direct child of its residue.

    Returns
    -------
    rdkit.Chem.RWMol, dict
        The molecule, and a map of particle -> atom index.
    """
    orders = _rdkit_bond_orders()
    from rdkit import Chem

    editable = Chem.RWMol()
    index = {}
    for particle in particles:
        charges = getattr(particle.parent, "atom_formal_charges", {})
        atom = Chem.Atom(particle.element.atomic_number)
        atom.SetFormalCharge(charges.get(particle.name, 0))
        atom.SetNoImplicit(True)
        index[particle] = editable.AddAtom(atom)
    for particle1, particle2, data in compound.bonds(return_bond_order=True):
        if particle1 not in index or particle2 not in index:
            continue
        order = float(data["bond_order"])
        if order not in orders:
            raise MBuildError(
                f"Bond {particle1.name}-{particle2.name} has bond order "
                f"{order:g}, so its chemistry cannot be exported to RDKit; "
                "every bond needs the order 1, 1.5, 2 or 3."
            )
        count = editable.AddBond(index[particle1], index[particle2], orders[order])
        if order == 1.5:
            bond = editable.GetBondWithIdx(count - 1)
            bond.SetIsAromatic(True)
            bond.GetBeginAtom().SetIsAromatic(True)
            bond.GetEndAtom().SetIsAromatic(True)
    return editable, index
