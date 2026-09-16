"""Bond a fragment to a residue: the implementation behind ``Protein.attach``.

The steps are those of ``Polymer.add_monomer``: the leaving atoms go,
a ``Port`` opens along each broken bond, ``force_overlap`` aligns the
fragment on the two ports and adds the bond, and the fragment residues
join the chain. A reaction string (``mbuild.biopolymers.reactions``)
replaces the one-bond rule with the edits the string states, applied
here with the same machinery. Every function takes the ``Protein`` as
its first argument, so the class keeps only the public verbs.
"""

import logging
from collections import deque

import numpy as np

from mbuild import clone, force_overlap
from mbuild.biopolymers.fragments import (
    _as_residues,
    _check_resname,
    _detach_particles,
    _move_into_residue,
)
from mbuild.biopolymers.relax import (
    _PORT_SEPARATION,
    _PROTON_BOND_LENGTH,
    _fit_placement,
    _open_direction,
    _relax_if_clashing,
    _relax_until_bonded,
)
from mbuild.biopolymers.residue import (
    InterResidueBond,
    _atom_in_residue,
    _atom_of,
    _chain_of,
    _pdb_label,
    _remove_particles_and_ports,
)
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError
from mbuild.port import Port

logger = logging.getLogger(__name__)


def _attach(
    protein,
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
    reaction=None,
    merge=False,
):
    # The name is checked first, before the attachment site is
    # read and before any warning is logged. The check ran inside
    # _as_residues before, so the charge warning of a valid site
    # reached the user ahead of the error about the name.
    """Bond a fragment onto a residue; see ``Protein.attach``."""
    _check_resname(fragment_resname)
    if reaction is not None:
        return _attach_by_reaction(
            protein,
            fragment,
            reaction,
            fragment_atom_name,
            resnum,
            atom_name,
            chain_id,
            icode,
            fragment_resnum,
            fragment_resname,
            relax,
            merge,
        )
    bond_order = int(bond_order)
    site_residue, site_atom, site_hydrogens = _attachment_site(
        protein, resnum, atom_name, chain_id, icode, bond_order, leaving_atom_names
    )
    _warn_on_kept_charge(site_residue, resnum, atom_name)
    added, frag_residues = _as_residues(clone(fragment), fragment_resname)
    frag_atom, frag_residue, frag_hydrogens = _fragment_site(
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
    site_port = _port_along_hydrogens(protein, protein, site_atom, site_hydrogens)
    site_residue.add(site_port, label="attach_site")
    frag_port = _port_along_hydrogens(protein, added, frag_atom, frag_hydrogens)
    added.add(frag_port, label="attach_frag")
    # A heavy leaving atom takes its substituent with it. When that
    # substituent was a residue of its own (the hydroxyl residue a
    # glycan builder puts on the anomeric carbon), ``Compound.remove``
    # has already detached the emptied residue from the fragment.
    # It is dropped from the list here too, so it is not numbered
    # and never reaches the chain.
    frag_residues[:] = [r for r in frag_residues if r is added or r.root is added]

    placed = list(added.particles())
    _place_fragment(
        protein,
        added,
        frag_residues,
        site_residue,
        site_port,
        frag_port,
        bond_order,
        merge,
    )
    if merge:
        _relax_if_clashing(protein, placed, site_atom, frag_atom, relax)
        return site_residue

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
        leaving1=protein._leaving_atoms[site_atom],
        leaving2=protein._leaving_atoms[frag_atom],
    )
    protein.cross_bonds.append(record)

    _relax_if_clashing(protein, placed, site_atom, frag_atom, relax)
    return record


def _attach_by_reaction(
    protein,
    fragment,
    reaction,
    fragment_atom_name,
    resnum,
    atom_name,
    chain_id,
    icode,
    fragment_resnum,
    fragment_resname,
    relax,
    merge,
):
    """Bond a fragment by the rule a reaction string states.

    ``reactions.plan_reaction`` reads the string into leaving atoms,
    formed, broken and reordered bonds, charges and new hydrogens.
    This method applies them in that order with the same steps as
    the plain ``attach``: leaving groups go, ports open at the two
    atoms of the bond that joins the sides, the fragment is
    aligned and bonded through ``force_overlap``, and the remaining
    edits follow. A residue whose bonds or charges the reaction
    changed keeps no template, because no CCD variant describes it;
    its bond record does.
    """
    from mbuild.biopolymers.reactions import plan_reaction

    site_residue = protein.get_residue(resnum, chain_id=chain_id, icode=icode)
    site_anchor = _atom_of(site_residue, atom_name)
    added, frag_residues = _as_residues(clone(fragment), fragment_resname)
    frag_anchor = None
    if fragment_atom_name is not None:
        frag_anchor, _ = _find_fragment_atom(
            frag_residues, fragment_atom_name, fragment_resnum
        )
    else:
        linked = [(r, name) for r in frag_residues for name in r.link_atoms.values()]
        if len(linked) == 1:
            frag_anchor = _atom_in_residue(*linked[0])
    plan = plan_reaction(
        reaction,
        protein,
        list(site_residue.particles()),
        site_anchor,
        added,
        list(added.particles()),
        frag_anchor,
    )
    in_fragment = set(added.particles())
    # A hydrogen that changes partners forms a bond too, but it is not
    # a bond between the sides: it follows the fragment. Only bonds
    # between atoms that stay put place the fragment and are recorded.
    moving = {
        atom
        for pair in plan.removed
        for atom in pair
        if atom.element.symbol == "H"
        and len([q for q in plan.removed if atom in q])
        == len(list(atom.direct_bonds()))
    }
    cross = [
        b
        for b in plan.cross_bonds(in_fragment)
        if not (b[0] in moving or b[1] in moving)
    ]
    if not cross:
        raise MBuildError(
            f"The reaction {plan.smarts!r} forms no bond between the protein "
            "and the fragment, so there is nothing to attach."
        )
    primary = next((b for b in cross if site_anchor in b[:2]), cross[0])
    site_atom, frag_atom = (
        (primary[0], primary[1])
        if primary[1] in in_fragment
        else (primary[1], primary[0])
    )

    # 1. Leaving groups. The port at each bonding atom points along
    #    its leaving group, or into its open site when nothing leaves.
    def direction(atom):
        vectors = [
            leaving.pos - atom.pos for kept, leaving in plan.leaving if kept is atom
        ]
        total = np.sum(vectors, axis=0) if vectors else np.zeros(3)
        return total if np.linalg.norm(total) > 1e-8 else _open_direction(atom)

    site_direction, frag_direction = direction(site_atom), direction(frag_atom)
    for kept, leaving in plan.leaving:
        root = added if leaving in in_fragment else protein
        group = _substituent(kept, leaving)
        _remove_particles_and_ports(root, kept, group)
        protein._leaving_atoms[kept] = tuple(
            sorted(protein._leaving_atoms.get(kept, ()) + tuple(p.name for p in group))
        )
    frag_residues[:] = [r for r in frag_residues if r is added or r.root is added]

    # 2. Place the fragment along the bond that joins the sides.
    site_port = Port(
        anchor=site_atom,
        orientation=site_direction,
        separation=_PORT_SEPARATION / 2,
    )
    frag_port = Port(
        anchor=frag_atom,
        orientation=frag_direction,
        separation=_PORT_SEPARATION / 2,
    )
    site_residue.add(site_port, label="attach_site")
    added.add(frag_port, label="attach_frag")
    placed = list(added.particles())
    _place_fragment(
        protein,
        added,
        frag_residues,
        site_residue,
        site_port,
        frag_port,
        primary[2],
        merge,
    )
    if len(cross) > 1:
        _fit_placement(protein, placed, cross)

    # 3. The other edits. Bonds are broken before any hydrogen that
    #    moves is placed, so the open site it moves to is current.
    ports_before = set(protein.all_ports())
    touched = set()
    for atom1, atom2 in plan.removed:
        protein.remove_bond((atom1, atom2))
        touched.update((atom1.parent, atom2.parent))
    migrated = []
    for atom1, atom2, order in plan.formed:
        if {atom1, atom2} == {site_atom, frag_atom}:
            continue
        for hydrogen, partner in ((atom1, atom2), (atom2, atom1)):
            if hydrogen.element.symbol != "H" or list(hydrogen.direct_bonds()):
                continue
            hydrogen.pos = partner.pos + _open_direction(partner) * _PROTON_BOND_LENGTH
            if partner in in_fragment and hydrogen not in in_fragment:
                migrated.append(hydrogen)
            if hydrogen.parent is not partner.parent:
                # From the file's point of view the hydrogen has
                # left the atom it was on, so the record names it
                # there, as a residue library expects.
                left = next(
                    other
                    for pair in plan.removed
                    for other in pair
                    if hydrogen in pair and other is not hydrogen
                )
                protein._leaving_atoms[left] = tuple(
                    sorted(protein._leaving_atoms.get(left, ()) + (hydrogen.name,))
                )
            _move_hydrogen_to(hydrogen, partner.parent)
        protein.add_bond((atom1, atom2), bond_order=order)
        touched.update((atom1.parent, atom2.parent))
    for atom1, atom2, order in plan.reordered:
        protein.add_bond((atom1, atom2), bond_order=order)
        touched.update((atom1.parent, atom2.parent))
    for heavy in plan.new_hydrogens:
        new = Compound(
            name=_free_hydrogen_name(heavy.parent),
            element="H",
            pos=heavy.pos + _open_direction(heavy) * _PROTON_BOND_LENGTH,
        )
        heavy.parent.add(new)
        protein.add_bond((heavy, new), bond_order=1.0)
        touched.add(heavy.parent)
    new_ports = [port for port in protein.all_ports() if port not in ports_before]
    if new_ports:
        protein.remove(new_ports)
    # A hydrogen that crossed to the fragment now moves with it, in
    # the relaxation and in any conformer the relaxation tries.
    placed.extend(migrated)
    for particle, charge in plan.charges.items():
        residue = particle.parent
        if charge:
            residue.atom_formal_charges[particle.name] = charge
        else:
            residue.atom_formal_charges.pop(particle.name, None)
        residue.formal_charge = sum(residue.atom_formal_charges.values())
        touched.add(residue)
    for residue in touched:
        if getattr(residue, "template", None) is not None:
            logger.info(
                f"The reaction changed the chemistry of {_pdb_label(residue)}, "
                "so it keeps no CCD definition; its bond record describes it."
            )
            residue.template = None

    # 4. Records, one per bond between the sides, the reaction on each.
    #    A merged fragment leaves no bond between residues to record.
    result = site_residue if merge else None
    for atom1, atom2, order in () if merge else cross:
        protein_atom, fragment_atom = (
            (atom1, atom2) if atom2 in in_fragment else (atom2, atom1)
        )
        record = InterResidueBond(
            residue1=protein_atom.parent,
            residue2=fragment_atom.parent,
            atom1_name=protein_atom.name,
            atom2_name=fragment_atom.name,
            order=int(order) if float(order).is_integer() else order,
            leaving1=protein._leaving_atoms.get(protein_atom, ()),
            leaving2=protein._leaving_atoms.get(fragment_atom, ()),
            reaction=plan.smarts,
        )
        protein.cross_bonds.append(record)
        if protein_atom is site_atom:
            result = record
    # A ring closure leaves its bonds long after the rigid placement,
    # so the fragment is relaxed whether or not it clashes, and the
    # bond lengths are checked afterwards. The minimizer now and
    # then returns without moving an atom, so the check repeats
    # the relaxation a few times before it gives up.
    if relax and len(cross) > 1:
        try:
            import mbuild.simulation  # noqa: F401
        except ImportError as error:
            logger.warning(
                "Cannot relax the placed fragment: mbuild.simulation is not "
                f"importable ({error}). The ring-closing bonds keep their "
                "rigid-placement lengths until relax_fragments() runs."
            )
        else:
            _relax_until_bonded(protein, placed, cross)
    else:
        _relax_if_clashing(protein, placed, site_atom, frag_atom, relax)
    return result


def _attachment_site(
    protein, resnum, atom_name, chain_id, icode, bond_order, leaving_names=None
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
    residue = protein.get_residue(resnum, chain_id=chain_id, icode=icode)
    atom = _atom_of(residue, atom_name)
    hydrogens = _bonded_hydrogens(atom, residue.name, bond_order, leaving_names)
    return residue, atom, hydrogens


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

    atom, residue = _find_fragment_atom(frag_residues, atom_name, fragment_resnum)
    hydrogens = _bonded_hydrogens(atom, residue.name, bond_order, leaving_names)
    return atom, residue, hydrogens


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


def _port_along_hydrogens(protein, root, atom, hydrogens):
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
        leaving.extend(_substituent(atom, anchor))
    _remove_particles_and_ports(root, atom, leaving)
    protein._leaving_atoms[atom] = tuple(
        sorted(protein._leaving_atoms.get(atom, ()) + tuple(p.name for p in leaving))
    )
    return Port(anchor=atom, orientation=orientation, separation=_PORT_SEPARATION / 2)


def _place_fragment(
    protein, added, frag_residues, site_residue, site_port, frag_port, order, merge
):
    """Place the fragment and bond it, as its own residues or merged.

    Without ``merge`` the fragment residues join the chain and
    ``force_overlap`` aligns the ports and adds the bond. With
    ``merge`` the fragment is aligned while still detached, its one
    residue's atoms move into the site residue, and the bond is
    added there. The site residue takes the fragment's formal
    charges and drops its template.
    """
    if not merge:
        _adopt_fragment(protein, added, frag_residues, site_residue)
        _align_on_ports(added, frag_port, site_port, order)
        return
    if len(frag_residues) != 1:
        raise MBuildError(
            f"merge=True puts one residue into the site residue, but the "
            f"fragment holds {len(frag_residues)}."
        )
    fragment = frag_residues[0]
    force_overlap(
        move_this=added,
        from_positions=frag_port,
        to_positions=site_port,
        add_bond=False,
    )
    site_atom, frag_atom = site_port.anchor, frag_port.anchor
    protein.remove([site_port])
    added.remove([frag_port])
    _rename_clashing_atoms(fragment, site_residue)
    charges = dict(fragment.atom_formal_charges)
    _move_into_residue(added, site_residue)
    protein.add_bond((site_atom, frag_atom), bond_order=float(order))
    site_residue.atom_formal_charges.update(charges)
    site_residue.formal_charge = sum(site_residue.atom_formal_charges.values())
    if site_residue.template is not None:
        logger.info(
            f"{_pdb_label(site_residue)} now holds the fragment's atoms, so it "
            "keeps no CCD definition; rename it for the residue library that "
            "reads the file."
        )
        site_residue.template = None


def _adopt_fragment(protein, added, frag_residues, site_residue):
    """Renumber the fragment residues and add them to the chain.

    The fragment residues continue the residue numbering of the
    site's chain and are marked as HETATM records, so that a
    written PDB separates them from the standard residues.
    """
    chain = _chain_of(site_residue)
    next_resnum = max(r.resnum for r in protein.residues(chain.chain_id)) + 1
    for offset, residue in enumerate(frag_residues):
        residue.resnum = next_resnum + offset
        residue.hetatm = True
    chain.add(added)


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


def _rename_clashing_atoms(frag, residue):
    """Give fragment atoms names that the residue does not use yet.

    A fragment can carry any atom name, so a fragment atom named
    like an atom of the residue it joins is renamed element plus
    index, skipping names in use, and the charge and bond-site maps
    of the fragment follow.
    """
    taken = {particle.name for particle in residue.particles()}
    renamed = {}
    counters = {}
    for particle in frag.particles():
        if particle.name not in taken:
            taken.add(particle.name)
            continue
        symbol = particle.element.symbol.upper()
        while True:
            counters[symbol] = counters.get(symbol, 0) + 1
            candidate = f"{symbol}{counters[symbol]}"
            if candidate not in taken:
                break
        renamed[particle.name] = candidate
        particle.name = candidate
        taken.add(candidate)
    if renamed:
        frag.atom_formal_charges = {
            renamed.get(name, name): charge
            for name, charge in frag.atom_formal_charges.items()
        }
        frag.link_atoms = {
            label: renamed.get(name, name) for label, name in frag.link_atoms.items()
        }


def _move_hydrogen_to(hydrogen, residue):
    """Move a hydrogen that changed partners into its new residue.

    A hydrogen belongs to the residue of the atom it bonds, so one
    that a reaction moves across the sides, as the thiol hydrogen
    of a thiol-Michael addition does, changes residue. The written
    file then lists it under the right residue. It keeps its name
    unless that residue already uses it.
    """
    if hydrogen.parent is residue:
        return
    _detach_particles([hydrogen])
    if any(p.name == hydrogen.name for p in residue.particles()):
        hydrogen.name = _free_hydrogen_name(residue)
    residue.add(hydrogen)


def _free_hydrogen_name(residue):
    """Return a hydrogen name that no atom of the residue uses."""
    taken = {particle.name for particle in residue.particles()}
    index = 1
    while f"H{index}" in taken:
        index += 1
    return f"H{index}"


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
