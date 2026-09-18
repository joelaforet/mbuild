"""Replace the side chain of one residue: the implementation behind ``Protein.mutate``.

Everything beyond the bond from ``CA`` to its one heavy non-backbone
neighbour leaves. The new side chain comes from a CCD component, cut at
the same bond, or from a star-marked fragment; it is aligned along the
old bond with the ``Port`` machinery that ``attach`` uses, moved into
the residue, and the residue is re-matched to a definition. Each
function takes the ``Protein`` as its first argument, and the class
keeps only the public verb.
"""

import numpy as np

from mbuild import clone, force_overlap
from mbuild.biopolymers.attach import (
    _bonded_hydrogens,
    _rename_clashing_atoms,
    _substituent,
)
from mbuild.biopolymers.fragments import (
    _as_residues,
    _check_resname,
    _move_into_residue,
    fragment_from_ccd,
)
from mbuild.biopolymers.matching import _leaving_expectations
from mbuild.biopolymers.relax import _PORT_SEPARATION, _relax_if_clashing, _unit
from mbuild.biopolymers.residue import (
    _assign_template,
    _atom_in_residue,
    _pdb_label,
    _remove_particles_and_ports,
)
from mbuild.compound import Compound
from mbuild.exceptions import MBuildError
from mbuild.port import Port

#: The 20 residue names that a PDB file writes as ATOM records. Every
#: other residue, including a non-canonical amino acid from the CCD, is
#: a HETATM residue. ``mutate`` sets the flag of the residue it renames.
_STANDARD_RESNAMES = frozenset(
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]
)


#: Length of a C-H bond, in nm, for the alpha hydrogen that a mutation
#: to glycine adds.
_ALPHA_H_BOND = 0.109


def _alpha_handedness(n, ca, c, cb):
    """Return ``"L"`` or ``"D"`` for an alpha carbon from four positions.

    The sign of the triple product ``(N - CA) x (C - CA) . (CB - CA)``
    is positive for every L amino acid and negative for every D amino
    acid. It is a geometric statement about the four atoms, and it does
    not depend on the side chain, so unlike a CIP label it reads the
    same for cysteine (whose L form is *R*) as for the other residues.
    For a glycine, pass the position of one alpha hydrogen as ``cb``
    to read the handedness a side chain in that place would have.

    Parameters
    ----------
    n, ca, c, cb : numpy.ndarray
        Positions of the backbone nitrogen, the alpha carbon, the
        carbonyl carbon, and the beta carbon (or an alpha hydrogen).

    Returns
    -------
    str
        ``"L"`` or ``"D"``.
    """
    volume = np.dot(np.cross(n - ca, c - ca), cb - ca)
    return "L" if volume > 0 else "D"


def _beta_atom(ca, backbone):
    """Return the one heavy atom on CA that is not a backbone atom, or None.

    The atom is ``CB`` in the canonical residues, but a CCD component
    may name it otherwise: ``4AF`` (p-acetyl-L-phenylalanine) calls it
    ``C3``. Bonding decides, not the name. None means a glycine-like
    alpha carbon with two hydrogens. Two heavy atoms, as in
    2-aminoisobutyric acid, raise, because such a residue has no
    single side chain to swap.

    Parameters
    ----------
    ca : mbuild.Compound
        The alpha carbon.
    backbone : iterable of mbuild.Compound
        Its backbone neighbours, ``N`` and ``C``.
    """
    heavy = [
        atom
        for atom in ca.direct_bonds()
        if atom not in backbone and atom.element.symbol.upper() != "H"
    ]
    if len(heavy) > 1:
        raise MBuildError(
            f"CA carries {len(heavy)} heavy atoms besides N and C, so there "
            "is no single side chain to replace."
        )
    return heavy[0] if heavy else None


def _mutate(
    protein,
    resnum,
    to,
    *,
    chain_id=None,
    icode="",
    resname=None,
    stereo=None,
    relax=True,
):
    """Replace the side chain of one residue; see ``Protein.mutate``."""
    if stereo is not None:
        stereo = str(stereo).upper()
        if stereo not in ("L", "D"):
            raise ValueError(f"stereo must be 'L', 'D' or None, not {stereo!r}.")
    residue = protein.get_residue(resnum, chain_id=chain_id, icode=icode)
    label = _pdb_label(residue)
    n, ca, c = (_atom_in_residue(residue, name) for name in ("N", "CA", "C"))
    if n is None or ca is None or c is None:
        raise MBuildError(
            f"mutate() needs the backbone atoms N, CA and C of {label}, "
            "and at least one of them is absent."
        )
    frag, link, anchor, new_name, hetatm, component_stereo = _side_chain_fragment(
        protein, to, resname
    )
    if stereo is None:
        stereo = component_stereo

    # 1. The old side chain leaves, and the direction the new one
    #    takes is chosen. Flipping the handedness swaps the roles of
    #    the alpha hydrogen and the side chain around CA.
    leaving, orientation, moved = _old_side_chain(protein, residue, n, ca, c, stereo)
    _remove_particles_and_ports(protein, ca, leaving)

    # 2. The new side chain arrives. A glycine needs only a hydrogen.
    if frag is None:
        new_h = Compound(
            name="HA3",
            element="H",
            pos=ca.pos + _unit(orientation) * _ALPHA_H_BOND,
        )
        residue.add(new_h)
        protein.add_bond((ca, new_h), bond_order=1.0)
        _atom_in_residue(residue, "HA").name = "HA2"
        added = [new_h]
        link = new_h
    else:
        frag_leaving = _substituent(link, anchor)
        frag_orientation = anchor.pos - link.pos
        _remove_particles_and_ports(frag, link, frag_leaving)
        site_port = Port(
            anchor=ca, orientation=orientation, separation=_PORT_SEPARATION / 2
        )
        frag_port = Port(
            anchor=link,
            orientation=frag_orientation,
            separation=_PORT_SEPARATION / 2,
        )
        residue.add(site_port)
        frag.add(frag_port)
        force_overlap(
            move_this=frag,
            from_positions=frag_port,
            to_positions=site_port,
            add_bond=False,
        )
        protein.remove([site_port])
        frag.remove([frag_port])
        _rename_clashing_atoms(frag, residue)
        if (
            not isinstance(to, str)
            and link.name != "CB"
            and _atom_in_residue(frag, "CB") is None
        ):
            # A user fragment names its atoms by element and index.
            # The atom on CA is the beta carbon of the mutant, and
            # readers of the written file look for it under CB. A CCD
            # component keeps its own name for that atom.
            frag.atom_formal_charges = {
                ("CB" if name == link.name else name): charge
                for name, charge in frag.atom_formal_charges.items()
            }
            link.name = "CB"
        frag_charges = dict(frag.atom_formal_charges)
        added = list(frag.particles())
        _move_into_residue(frag, residue)
        protein.add_bond((ca, link), bond_order=1.0)

    # 3. The residue takes its new identity.
    residue.name = new_name
    residue.hetatm = hetatm
    if isinstance(to, str):
        _assign_template(residue, _variant_for(residue, protein.library[new_name]))
    else:
        # No CCD definition describes a user fragment. The backbone
        # keeps the charges its old definition gave the atoms that
        # are still present, and the side chain brings its own.
        charges = {
            name: charge
            for name, charge in residue.atom_formal_charges.items()
            if _atom_in_residue(residue, name) is not None
        }
        charges.update(frag_charges)
        residue.template = None
        residue.atom_formal_charges = charges
        residue.formal_charge = sum(charges.values())

    _relax_if_clashing(protein, added + moved, ca, link, relax)
    return residue


def _side_chain_fragment(protein, to, resname):
    """Return the side-chain source of ``mutate`` and how to cut it.

    Returns
    -------
    frag : Residue or None
        A detached residue that holds the new side chain, or None
        for a mutation to glycine.
    link : mbuild.Compound or None
        The atom of ``frag`` that bonds to ``CA``.
    anchor : mbuild.Compound or None
        The atom bonded to ``link`` whose branch leaves: the ``CA``
        of a CCD component, so the whole backbone goes, or one
        hydrogen of a user fragment.
    new_name : str
        The residue name after the mutation.
    hetatm : bool
        Whether the mutant is written as HETATM records.
    stereo : str or None
        The handedness of a CCD component, read from its ideal
        coordinates, or None when the source does not fix one.
    """
    if isinstance(to, str):
        code = to.upper()
        if code == "GLY":
            return None, None, None, code, False, None
        try:
            template = protein.library[code][0]
        except KeyError as error:
            raise MBuildError(
                f"{error.args[0]} Pass download=True to Protein() to "
                "fetch a CCD component that is not bundled."
            ) from error
        if template.linking != "peptide" or not {"N", "CA", "C"} <= (
            template.atom_names
        ):
            raise MBuildError(
                f"{code} is not an alpha amino acid, so it has no side "
                "chain to mutate to. Pass a side-chain fragment instead."
            )
        # The whole component is built, then cut at the bond from CA
        # to the side chain. The side-chain atom is the heavy
        # neighbour of CA that is not N or C, whatever its name.
        frag = fragment_from_ccd(code, "CA", library=protein.library)
        n, ca, c = (_atom_in_residue(frag, name) for name in ("N", "CA", "C"))
        link = _beta_atom(ca, (n, c))
        if link is None:
            raise MBuildError(
                f"{code} has no side chain on CA. Use 'GLY' for a glycine."
            )
        return (
            frag,
            link,
            ca,
            code,
            code not in _STANDARD_RESNAMES,
            _alpha_handedness(n.pos, ca.pos, c.pos, link.pos),
        )
    _check_resname(resname)
    frag, residues = _as_residues(clone(to), resname)
    if len(residues) != 1:
        raise MBuildError(
            f"A side chain is one residue, but the fragment holds {len(residues)}."
        )
    frag = residues[0]
    if resname:
        frag.name = resname.upper()
    sites = list(frag.link_atoms.values())
    if len(sites) != 1:
        raise MBuildError(
            "mutate() bonds one atom of the side chain to CA, but the "
            f"fragment marks {len(sites)} bond sites. Mark exactly one "
            "atom with * in the SMILES, as prepare_fragment does."
        )
    link = _atom_in_residue(frag, sites[0])
    anchor = _bonded_hydrogens(link, frag.name, 1)[0]
    return frag, link, anchor, frag.name, True, None


def _old_side_chain(protein, residue, n, ca, c, stereo):
    """Return the atoms that leave ``CA``, and the new side chain's direction.

    For a residue with a ``CB``, the side chain is every atom beyond
    the ``CA``-``CB`` bond and the direction is that bond, so the
    new ``CB`` lands where the old one was and the handedness is
    kept. When ``stereo`` asks for the other handedness, the alpha
    hydrogen moves onto the old ``CB`` direction and the side chain
    takes the old hydrogen direction. A glycine has two alpha
    hydrogens; the one whose place gives the wanted handedness
    leaves, and the other is renamed ``HA``.

    Returns
    -------
    leaving : list of mbuild.Compound
        The atoms to remove.
    orientation : numpy.ndarray
        The direction, from ``CA``, of the new side chain.
    moved : list of mbuild.Compound
        The atoms whose position this call changed, so that the
        relaxation may move them too.
    """
    label = _pdb_label(residue)
    try:
        cb = _beta_atom(ca, (n, c))
    except MBuildError as error:
        raise MBuildError(
            f"mutate() cannot replace the side chain of {label}: {error.args[0]}"
        ) from error
    if cb is None:
        alphas = [
            atom for atom in ca.direct_bonds() if atom.element.symbol.upper() == "H"
        ]
        if len(alphas) != 2:
            raise MBuildError(
                f"{label} has no side chain on CA and {len(alphas)} alpha "
                "hydrogens instead of two, so mutate() cannot tell where a "
                "side chain goes."
            )
        wanted = stereo or "L"
        leaving = next(
            atom
            for atom in alphas
            if _alpha_handedness(n.pos, ca.pos, c.pos, atom.pos) == wanted
        )
        kept = alphas[1] if leaving is alphas[0] else alphas[0]
        kept.name = "HA"
        return [leaving], leaving.pos - ca.pos, []
    try:
        leaving = _substituent(ca, cb)
    except MBuildError as error:
        raise MBuildError(
            f"mutate() cannot remove the side chain of {label}: "
            f"{error.args[0]} A side chain bonded to the backbone at "
            "both ends, as in proline, or joined to another residue by a "
            "crosslink, as a disulfide cysteine, cannot be replaced."
        ) from error
    orientation = cb.pos - ca.pos
    if stereo is None or stereo == _alpha_handedness(n.pos, ca.pos, c.pos, cb.pos):
        return leaving, orientation, []
    ha = _atom_in_residue(residue, "HA")
    if ha is None:
        raise MBuildError(
            f"Flipping the handedness of {label} needs its alpha hydrogen HA."
        )
    new_orientation = ha.pos - ca.pos
    ha.pos = ca.pos + _unit(orientation) * np.linalg.norm(new_orientation)
    return leaving, new_orientation, [ha]


def _variant_for(residue, variants):
    """Return the template variant that describes a residue's atoms.

    The atoms present must all belong to the variant, and every
    atom of the variant that is absent must be one that a peptide
    bond or a crosslink removes, as the loader's matcher requires.
    The variants are tried in library order, so the base
    protonation state wins when several fit.
    """
    present = {particle.name for particle in residue.particles()}
    for variant in variants:
        if not present <= variant.atom_names:
            continue
        missing = variant.atom_names - present
        explained = _leaving_expectations(variant, missing)[3]
        if missing == explained:
            return variant
    raise MBuildError(
        f"No {residue.name} definition describes the atoms of "
        f"{_pdb_label(residue)} after the mutation: {sorted(present)}."
    )
