"""Reaction strings for ``Protein.attach``.

A reaction string is an RDKit reaction SMARTS with two reactant
templates and one product template, such as::

    [S:1][H:2].[C:3]=[C:4]>>[S:1][C:3][C:4][H:2]

``attach`` reads the string as a rule and never runs it on the protein.
The first template is matched around the protein atom the caller names,
the second on the fragment (``attach`` tries the other order when only
that one fits), and the difference between the reactant templates and
the product template says what changes:

- A reactant atom with no map number, or a mapped atom absent from the
  product, leaves, together with everything beyond it. This is the same
  rule the leaving-atom keywords follow.
- A product bond between mapped atoms that no reactant template holds
  forms, with the product's bond order. A ring closure is two such bonds.
- A bond whose order differs between reactant and product takes the
  product's order, and a bond absent from the product is broken.
- A mapped atom whose product template states a formal charge takes it.
- A product hydrogen with no map number is created on its mapped
  neighbour. A product heavy atom with no map number is refused, because
  nothing gives it coordinates.

The edits are then applied to the Compound with the machinery
``attach`` already has, so the protein keeps its coordinates and its
residue hierarchy. ``REACTIONS`` holds the strings for the common
conjugations, keyed by name; ``attach`` accepts either the name or a
string.
"""

from dataclasses import dataclass, field

import numpy as np

from mbuild.exceptions import MBuildError

__all__ = ["REACTIONS"]

#: Named reaction strings that ``attach`` accepts in place of a SMARTS.
#: The protein side is written first. Every string forms the bond the
#: leaving-atom keywords cannot describe, or several bonds at once.
REACTIONS = {
    # One hydrogen leaves each side; the default behaviour of attach.
    "hydrogen substitution": "[*:1][H].[*:2][H]>>[*:1][*:2]",
    # Amine plus carboxylic acid, losing water. Deprotonate a lysine
    # first, so the amine is the neutral species.
    "amide coupling": "[N:1]([H:2])[H].[C:3](=[O:4])[O][H]>>[N:1]([H:2])[C:3]=[O:4]",
    # Thiol-Michael addition to a maleimide; the thiol hydrogen moves to
    # the other alkene carbon. The whole imide ring is in the template,
    # so a dye's other double bonds do not match.
    "thiol-maleimide": (
        "[S:1][H:2].[O:5]=[C:6]1[N:7][C:8](=[O:9])[C:3]=[C:4]1"
        ">>[S:1][C:3]1[C:8](=[O:9])[N:7][C:6](=[O:5])[C:4]1[H:2]"
    ),
    # Azide plus alkyne to a 1,2,3-triazole (a click reaction). Both new
    # ring bonds form, and the azide charges vanish.
    "azide-alkyne triazole": (
        "[N:1]=[N+:2]=[N-:3].[C:4]#[C:5]>>[N:1]1[N+0:2]=[N+0:3][C:4]=[C:5]1"
    ),
    # Asparagine amide nitrogen to the anomeric carbon of a glycan whose
    # hydroxyl leaves. Name the anomeric carbon with fragment_atom_name.
    "N-glycosylation": "[N:1]([H:2])[H].[C:3][O][H]>>[N:1]([H:2])[C:3]",
    # Serine or threonine hydroxyl oxygen to the anomeric carbon.
    "O-glycosylation": "[O:1][H].[C:2][O][H]>>[O:1][C:2]",
}

_BOND_ORDERS = {"SINGLE": 1.0, "DOUBLE": 2.0, "TRIPLE": 3.0, "AROMATIC": 1.5}


@dataclass
class ReactionPlan:
    """The edits a reaction string makes, as particles of the two compounds.

    Attributes
    ----------
    leaving : list of (mbuild.Compound, mbuild.Compound)
        ``(kept atom, leaving atom)`` pairs. The leaving atom and every
        atom beyond it, away from the kept atom, are removed.
    formed : list of (mbuild.Compound, mbuild.Compound, float)
        Bonds to add, with their order.
    removed : list of (mbuild.Compound, mbuild.Compound)
        Bonds to break between atoms that both stay.
    reordered : list of (mbuild.Compound, mbuild.Compound, float)
        Bonds that stay and take a new order.
    charges : dict of mbuild.Compound -> int
        Formal charges the product template states.
    new_hydrogens : list of mbuild.Compound
        Atoms that gain a hydrogen the product creates.
    smarts : str
        The reaction string, resolved from its name.
    """

    leaving: list = field(default_factory=list)
    formed: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    reordered: list = field(default_factory=list)
    charges: dict = field(default_factory=dict)
    new_hydrogens: list = field(default_factory=list)
    smarts: str = ""

    def cross_bonds(self, side_particles):
        """Return the formed bonds with one atom in ``side_particles``."""
        side = set(side_particles)
        return [bond for bond in self.formed if (bond[0] in side) != (bond[1] in side)]


def _query_mol(compound, particles):
    """Return an RDKit molecule of ``particles`` and its index-to-particle map.

    The molecule is for substructure matching only. It carries the
    elements, the bond orders and the formal charges, and it is not
    sanitized: a residue cut from its chain has open valences that
    sanitization would reject, and matching does not need it.
    """
    from rdkit import Chem

    orders = {value: getattr(Chem.BondType, key) for key, value in _BOND_ORDERS.items()}
    residues = {}
    for particle in particles:
        residues.setdefault(id(particle.parent), particle.parent)
    charges = {}
    for residue in residues.values():
        for name, charge in getattr(residue, "atom_formal_charges", {}).items():
            for particle in residue.particles_by_name(name):
                charges[particle] = charge
    editable = Chem.RWMol()
    index = {}
    for particle in particles:
        atom = Chem.Atom(particle.element.symbol)
        atom.SetFormalCharge(charges.get(particle, 0))
        atom.SetNoImplicit(True)
        index[particle] = editable.AddAtom(atom)
    for particle1, particle2, data in compound.bonds(return_bond_order=True):
        if particle1 in index and particle2 in index:
            order = orders.get(float(data["bond_order"]), Chem.BondType.SINGLE)
            editable.AddBond(index[particle1], index[particle2], order)
    mol = editable.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol, {i: particle for particle, i in index.items()}


def _match(template, mol, particle_of, anchor, what):
    """Return the one match of ``template`` on ``mol`` as template index -> particle.

    A match must contain ``anchor`` when one is given. Matches that
    cover the same atoms in another order, which a symmetric template
    produces, count once. Zero matches return None; several raise.
    """
    matches = mol.GetSubstructMatches(
        template, uniquify=False, useChirality=False, maxMatches=10000
    )
    if anchor is not None:
        matches = [m for m in matches if any(particle_of[i] is anchor for i in m)]
    unique = {}
    for match in matches:
        unique.setdefault(frozenset(match), match)
    if not unique:
        return None

    # Matches that differ only in which hydrogens of one heavy atom they
    # take are the same chemistry, as with the three protons of an
    # amine. The one with the alphabetically first hydrogen names is
    # used, the rule the leaving-atom keywords follow. Matches that
    # differ in a heavy atom are different sites, and the caller must
    # name the atom that reacts.
    def heavy(match):
        return frozenset(i for i in match if particle_of[i].element.symbol != "H")

    def hydrogen_names(match):
        return sorted(
            particle_of[i].name for i in match if particle_of[i].element.symbol == "H"
        )

    if len({heavy(match) for match in unique}) > 1:
        names = sorted(
            " ".join(particle_of[i].name for i in match) for match in unique.values()
        )
        raise MBuildError(
            f"The {what} template of the reaction matches {len(unique)} groups "
            f"of atoms ({'; '.join(names)}). Name the atom that reacts to choose."
        )
    match = min(unique.values(), key=hydrogen_names)
    return {index: particle_of[atom] for index, atom in enumerate(match)}


def _formal_charge(atom):
    """Return the charge a template atom states, or None when it states none."""
    for line in atom.DescribeQuery().splitlines():
        line = line.strip()
        if line.startswith("AtomFormalCharge"):
            return int(line.split()[1])
    return None


def plan_reaction(
    reaction, site, site_particles, site_anchor, fragment, frag_particles, frag_anchor
):
    """Read a reaction string into the edits it makes on two compounds.

    Parameters
    ----------
    reaction : str
        A key of ``REACTIONS`` or a reaction SMARTS with two reactant
        templates and one product template.
    site, fragment : mbuild.Compound
        The compounds that own the bond graphs of the two sides.
    site_particles, frag_particles : list of mbuild.Compound
        The particles the templates may match on each side.
    site_anchor, frag_anchor : mbuild.Compound or None
        An atom the match on that side must contain.

    Returns
    -------
    ReactionPlan
    """
    from rdkit.Chem import AllChem

    smarts = REACTIONS.get(reaction, reaction)
    rxn = AllChem.ReactionFromSmarts(smarts)
    if rxn.GetNumReactantTemplates() != 2 or rxn.GetNumProductTemplates() != 1:
        raise MBuildError(
            "A reaction string for attach() has two reactant templates, the "
            "protein side then the fragment, and one product template. "
            f"{smarts!r} has {rxn.GetNumReactantTemplates()} and "
            f"{rxn.GetNumProductTemplates()}."
        )
    site_mol, site_of = _query_mol(site, site_particles)
    frag_mol, frag_of = _query_mol(fragment, frag_particles)
    templates = [rxn.GetReactantTemplate(0), rxn.GetReactantTemplate(1)]
    matched = None
    for order in ((0, 1), (1, 0)):
        site_match = _match(
            templates[order[0]], site_mol, site_of, site_anchor, "protein"
        )
        frag_match = _match(
            templates[order[1]], frag_mol, frag_of, frag_anchor, "fragment"
        )
        if site_match is not None and frag_match is not None:
            matched = {order[0]: site_match, order[1]: frag_match}
            break
    if matched is None:
        raise MBuildError(
            f"The reaction {smarts!r} does not match the protein site and the "
            "fragment. The protein template must include the named atom, and "
            "each side must hold the atoms the template asks for, hydrogens "
            "included."
        )

    plan = ReactionPlan(smarts=smarts)
    by_map = {}
    reactant_bonds = {}
    for index, template in enumerate(templates):
        particles = matched[index]
        for atom in template.GetAtoms():
            if atom.GetAtomMapNum():
                by_map[atom.GetAtomMapNum()] = particles[atom.GetIdx()]
        for bond in template.GetBonds():
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            if a.GetAtomMapNum() and b.GetAtomMapNum():
                key = frozenset((a.GetAtomMapNum(), b.GetAtomMapNum()))
                reactant_bonds[key] = _BOND_ORDERS.get(str(bond.GetBondType()), 1.0)
    product = rxn.GetProductTemplate(0)
    product_maps = {
        atom.GetAtomMapNum() for atom in product.GetAtoms() if atom.GetAtomMapNum()
    }

    # Leaving atoms: unmapped reactant atoms, and mapped ones the product
    # drops. Only the leaving atom bonded to a kept atom is listed; the
    # rest of its group, such as the hydrogen of a leaving hydroxyl,
    # goes with it as the substituent beyond that bond.
    for index, template in enumerate(templates):
        particles = matched[index]
        for atom in template.GetAtoms():
            number = atom.GetAtomMapNum()
            if number and number in product_maps:
                continue
            kept = [n for n in atom.GetNeighbors() if n.GetAtomMapNum() in product_maps]
            if len(kept) > 1:
                raise MBuildError(
                    f"Leaving atom {particles[atom.GetIdx()].name} of the reaction "
                    f"{smarts!r} bonds {len(kept)} atoms that stay, so the group "
                    "that leaves with it is not defined."
                )
            if kept:
                plan.leaving.append(
                    (particles[kept[0].GetIdx()], particles[atom.GetIdx()])
                )

    # Bonds of the product between mapped atoms: formed or reordered.
    product_bonds = {}
    for bond in product.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomMapNum() and b.GetAtomMapNum():
            key = frozenset((a.GetAtomMapNum(), b.GetAtomMapNum()))
            order = _BOND_ORDERS.get(str(bond.GetBondType()), 1.0)
            product_bonds[key] = order
            pair = tuple(by_map[n] for n in sorted(key))
            if key not in reactant_bonds:
                plan.formed.append((*pair, order))
            elif reactant_bonds[key] != order:
                plan.reordered.append((*pair, order))
    for key in reactant_bonds:
        if key not in product_bonds:
            plan.removed.append(tuple(by_map[n] for n in sorted(key)))

    # Charges the product states, and atoms the product creates.
    for atom in product.GetAtoms():
        number = atom.GetAtomMapNum()
        if number:
            charge = _formal_charge(atom)
            if charge is not None:
                plan.charges[by_map[number]] = charge
            continue
        neighbours = [n for n in atom.GetNeighbors() if n.GetAtomMapNum()]
        if atom.GetAtomicNum() != 1 or len(neighbours) != 1:
            raise MBuildError(
                f"The product of {smarts!r} creates an atom that no reactant "
                "supplies. Only a hydrogen on one mapped atom can be created, "
                "because nothing else has coordinates."
            )
        plan.new_hydrogens.append(by_map[neighbours[0].GetAtomMapNum()])
    return plan


def open_direction(atom):
    """Return a unit vector into the open coordination site of an atom.

    It points away from the mean of the unit vectors along the atom's
    bonds. Unit vectors, not bond vectors, so that a long and a short
    bond on a linear atom do not leave a spurious direction along the
    axis. When that mean vanishes, the atom is linear or trigonal
    planar, and the open site is perpendicular: to the plane of a
    planar atom, which is where an addition to an alkene carbon goes,
    or to the axis of a linear atom such as an alkyne carbon. It is
    where a new bond or a new hydrogen goes when no leaving atom shows
    the way.
    """
    vectors = []
    for neighbour in atom.direct_bonds():
        vector = neighbour.pos - atom.pos
        vectors.append(vector / np.linalg.norm(vector))
    if not vectors:
        return np.array([1.0, 0.0, 0.0])
    total = -np.sum(vectors, axis=0)
    if np.linalg.norm(total) > 0.3:
        return total / np.linalg.norm(total)
    normal = np.cross(vectors[0], vectors[1]) if len(vectors) > 1 else np.zeros(3)
    if np.linalg.norm(normal) < 1e-3:
        # Linear: any perpendicular to the axis.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(vectors[0], axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        normal = np.cross(vectors[0], axis)
    return normal / np.linalg.norm(normal)
