"""Add and remove protons at a named atom: behind ``Protein.deprotonate`` and ``protonate``.

Both operations work through the residue definitions. ``deprotonate``
builds the new template variant from the old one, because a residue can
lose a proton that no library variant loses; ``protonate`` selects a
library variant, because the library holds every variant that carries
an added proton. Each function takes the ``Protein`` as its first
argument, and the class keeps only the public verbs.
"""

import logging
from itertools import combinations

from mbuild.biopolymers.ccd import _ACIDIC_PROTONS, _BASIC_ATOMS
from mbuild.biopolymers.matching import _bond_separation
from mbuild.biopolymers.relax import _proton_position
from mbuild.biopolymers.residue import (
    _assign_template,
    _atom_in_residue,
    _atom_of,
    _pdb_label,
    _remove_particles_and_ports,
)
from mbuild.compound import Compound

logger = logging.getLogger(__name__)

#: Largest separation, in bonds, at which two charged atoms of one
#: residue count as one functional group. Arginine's NH1 and NH2 are
#: two bonds apart through CZ, and their opposite charges come from the
#: template model. An N-terminal serine's N and OG are three bonds
#: apart, and their opposite charges are a real zwitterion.
_SPLIT_CHARGE_MAX_BONDS = 2


def _deprotonate(protein, resnum, atom_name, chain_id=None, icode=""):
    """Remove the acidic proton of one atom; see ``Protein.deprotonate``."""
    residue = protein.get_residue(resnum, chain_id=chain_id, icode=icode)
    atom = _atom_of(residue, atom_name)
    variant = residue.template
    protons = []
    if variant is not None:
        bonded = variant.bonded_names(atom_name)
        for name in _ACIDIC_PROTONS.get(residue.name, ()):
            if name in bonded and _atom_in_residue(residue, name) is not None:
                protons.append(name)
    if not protons:
        logger.warning(
            f"Atom {atom_name} of residue {_pdb_label(residue)} "
            "carries no acidic proton, so nothing changed. The atom is "
            "already deprotonated, or its protons are not acidic."
        )
        return
    proton_name = protons[0]
    _remove_particles_and_ports(protein, atom, [_atom_in_residue(residue, proton_name)])
    _assign_template(residue, variant.deprotonated_at(proton_name))
    _warn_if_variant_is_absent(protein, residue, atom_name, proton_name)
    _warn_on_split_charge(residue)


def _protonate(protein, resnum, atom_name, chain_id=None, icode=""):
    """Add a proton to one atom; see ``Protein.protonate``."""
    residue = protein.get_residue(resnum, chain_id=chain_id, icode=icode)
    atom = _atom_of(residue, atom_name)
    variant = residue.template
    base = protein.library[residue.name][0]
    protons = []
    if variant is not None and atom_name in base.atom_names:
        bonded = base.bonded_names(atom_name)
        protons = [
            name
            for name in _ACIDIC_PROTONS.get(residue.name, ())
            if name in bonded and name not in variant.atom_names
        ]
        protons += [
            proton
            for heavy, proton in _BASIC_ATOMS.get(residue.name, ())
            if heavy == atom_name and proton not in variant.atom_names
        ]
    # An atom bonded to a second residue, such as the N of an
    # internal residue or the SG of a disulfide, has no free
    # valence. Its template proton is the leaving atom of that
    # bond, so adding it back would over-coordinate the atom.
    if any(other.parent is not residue for other in atom.direct_bonds()):
        protons = []
    target = None
    for proton_name in protons:
        wanted = variant.atom_names | {proton_name}
        target = next(
            (
                other
                for other in protein.library[residue.name]
                if other.atom_names == wanted
            ),
            None,
        )
        if target is not None:
            break
    if target is None:
        logger.warning(
            f"Atom {atom_name} of residue {_pdb_label(residue)} "
            "has no protonation variant in the CCD template, so nothing "
            "changed. The atom is already protonated, it bonds to another "
            "residue, or the template does not protonate it."
        )
        return
    proton = Compound(name=proton_name, element="H", pos=_proton_position(atom))
    residue.add(proton)
    residue.add_bond((atom, proton), bond_order=1.0)
    _assign_template(residue, target)
    _warn_on_split_charge(residue)


def _warn_on_split_charge(residue):
    """Warn when two charged atoms lie in one functional group.

    ``ResidueTemplate.deprotonated_at`` decrements the charge of the
    heavy atom that held the proton. It changes no other atom, so a
    residue whose charge sat on a second atom now holds two charged
    atoms. Two charges within ``_SPLIT_CHARGE_MAX_BONDS`` bonds sit
    on one functional group, and they usually show that the template
    model split a delocalized charge across it. Two charges that are
    further apart sit on separate functional groups, where they can
    be a real zwitterion. The warning therefore names only the pairs
    of the first kind, and the call proceeds.

    Parameters
    ----------
    residue : Residue
        The residue, with its new template already assigned.
    """
    charges = {
        name: charge for name, charge in residue.atom_formal_charges.items() if charge
    }
    if len(charges) < 2:
        return
    variant = residue.template
    pairs = []
    for first, second in combinations(sorted(charges), 2):
        separation = _bond_separation(variant, first, second, _SPLIT_CHARGE_MAX_BONDS)
        if separation is None:
            continue
        pairs.append(
            f"{first} {charges[first]:+d} and {second} "
            f"{charges[second]:+d}, {separation} bonds apart"
        )
    if not pairs:
        return
    logger.warning(
        f"{_pdb_label(residue)} holds charged atoms within "
        f"{_SPLIT_CHARGE_MAX_BONDS} bonds after this call: "
        f"{'; '.join(pairs)}. Load the protein again and deprotonate "
        "another atom if one charged atom is correct for the chemistry "
        "you model."
    )


def _warn_if_variant_is_absent(protein, residue, atom_name, proton_name):
    """Warn when no library variant describes the deprotonated residue.

    ``deprotonate`` builds the new template variant from the old
    one. The library holds fewer variants than that construction can
    produce, so the result can be a residue that no library variant
    describes. The loader matches a file against the library
    variants, so a PDB written from such a residue does not reload.
    The warning names the consequence and the call proceeds.

    Parameters
    ----------
    residue : Residue
        The residue, with its new template already assigned.
    atom_name : str
        Name of the heavy atom that lost the proton.
    proton_name : str
        Name of the removed proton.
    """
    variant = residue.template
    library_variants = protein.library[residue.name]
    if any(other.atom_names == variant.atom_names for other in library_variants):
        return
    logger.warning(
        f"{_pdb_label(residue)} atom {atom_name} lost {proton_name}. "
        f"The template library holds no {residue.name} variant with the "
        f"atoms of {variant.description}. A PDB written from this protein "
        "does not reload with Protein(). Deprotonate another atom if the "
        "written file must reload."
    )
