"""Inline PDB text used by the biopolymers tests.

These builders return PDB text for small, hand-specified structures, so
the tests that use them need no fixture files. They are shared by the
matcher tests and the loader tests.
"""

import numpy as np

# Atoms of a glycine that is missing its OXT and HXT atoms, so it
# expects a peptide bond to the residue that follows it. The
# coordinates come from residue 1 of the openff-pablo polyglycines
# asset, in Angstrom, with the two extra amine hydrogens added.
_GLY_EXPECTS_POSTERIOR = [
    ("N", 7.341, 2.356, 3.392),
    ("H", 7.627, 3.312, 3.718),
    ("H2", 8.010, 1.800, 3.800),
    ("H3", 6.500, 2.000, 3.700),
    ("CA", 7.516, 2.347, 1.949),
    ("HA2", 8.439, 2.929, 1.738),
    ("HA3", 7.729, 1.293, 1.661),
    ("C", 6.355, 2.899, 1.216),
    ("O", 6.349, 2.964, -0.058),
]

# Atoms of a glycine that is missing its H2 atom, so it expects a
# peptide bond to the residue before it. The coordinates come from
# residue 2 of the same asset, with the OXT atom added.
_GLY_EXPECTS_PRIOR = [
    ("N", 5.181, 3.391, 1.857),
    ("H", 5.124, 3.366, 2.904),
    ("CA", 4.120, 3.904, 1.015),
    ("HA2", 3.779, 3.133, 0.287),
    ("HA3", 4.566, 4.715, 0.402),
    ("C", 2.970, 4.378, 1.803),
    ("O", 3.014, 4.297, 3.061),
    ("OXT", 1.900, 4.850, 1.200),
]


def _gly_gly_with_ter(resnum=2, icode=" ", offset=0.0, complete=False):
    """Return PDB text for a glycine pair with a TER between them.

    Both residues sit in chain A. The first is residue 1 and carries
    the TER record, which the OpenMM writer formats as
    ``TER   %5s      %3s %s%4s``.

    Parameters
    ----------
    resnum : int, optional, default=2
        Residue number of the second residue.
    icode : str, optional, default=" "
        Insertion code of the second residue.
    offset : float, optional, default=0.0
        Shift of the second residue along x, in Angstrom.
    complete : bool, optional, default=False
        Give both residues every leaving atom, so that neither residue
        expects a peptide bond.

    Returns
    -------
    str
        The PDB text.
    """
    first = list(_GLY_EXPECTS_POSTERIOR)
    second = list(_GLY_EXPECTS_PRIOR)
    if complete:
        first += [("OXT", 5.300, 3.400, 1.900), ("HXT", 5.400, 3.700, 2.800)]
        second += [("H2", 5.900, 4.000, 1.500)]
    lines = []
    serial = 1
    for number, code, atoms, shift in (
        (1, " ", first, 0.0),
        (resnum, icode, second, offset),
    ):
        for name, x, y, z in atoms:
            field = f" {name:<3s}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:5d} {field} GLY A{number:4d}{code}   "
                f"{x + shift:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {name[0]:>2s}"
            )
            serial += 1
        if atoms is first:
            lines.append(f"TER   {serial:5d}      GLY A{1:4d} ")
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _gly_gly_hexadecimal():
    """Return PDB text whose atom serials and residue numbers overflow.

    OpenMM writes an atom serial or a residue number that does not fit
    its columns as a shifted hexadecimal field, in
    ``openmm/app/pdbfile.py::_formatIndex``. It writes atom serial
    100000 as ``A0000`` and residue number 10001 as ``A001``. This
    function rewrites both fields of ``_gly_gly_with_ter`` into that
    form: the serials start at 100000 and the two residues are
    numbered 10001 and 10002.

    Returns
    -------
    str
        The PDB text.
    """
    lines = []
    serial = 0xA0000
    for line in _gly_gly_with_ter(complete=True).splitlines():
        if line.startswith(("ATOM  ", "TER   ")):
            resnum = 0xA000 + int(line[22:26])
            line = f"{line[:6]}{serial:5X}{line[11:22]}{resnum:4X}{line[26:]}"
            serial += 1
        lines.append(line)
    return "\n".join(lines) + "\n"


# Atoms of a cysteine whose HG atom is absent, so a disulfide bridge
# can take its place. The names, element symbols and coordinates are
# those of CYS 222 of the bundled 3cu9 asset, in Angstrom, plus an H2
# atom, which makes the residue a complete chain of one residue.
_CYS_BRIDGED_ATOMS = [
    ("N", "N", -25.649, 10.119, 5.071),
    ("CA", "C", -25.005, 11.177, 4.321),
    ("C", "C", -25.532, 11.323, 2.911),
    ("O", "O", -24.803, 11.847, 2.036),
    ("CB", "C", -23.463, 10.894, 4.313),
    ("SG", "S", -22.792, 10.598, 5.958),
    ("OXT", "O", -26.847, 10.869, 2.580),
    ("H", "H", -26.260, 10.369, 5.835),
    ("H2", "H", -26.150, 9.430, 4.550),
    ("HA", "H", -25.227, 12.126, 4.810),
    ("HB2", "H", -22.971, 11.782, 3.916),
    ("HB3", "H", -23.274, 10.015, 3.697),
    ("HXT", "H", -27.018, 11.033, 1.649),
]


def _cys_cys_cross_chain_disulfide():
    """Return PDB text for two cysteines of two chains, bridged.

    Chain A holds the atoms of ``_CYS_BRIDGED_ATOMS``, which are the
    atoms of CYS 222 of the 3cu9 asset. The second SG sits 2.05 A from
    the first one, which is the S-S bond length of a disulfide, along
    the SG 222 to SG 221 direction of that asset. The CB-SG-SG angle is
    then the measured angle of 3cu9. Chain B holds the residue of chain
    A turned by 180 degrees about an axis through the middle of the new
    bond. That turn is the C2 symmetry of a disulfide, and it keeps the
    L configuration of both residues. The axis stands 45 degrees out of
    the CB-SG-SG plane, which puts the CB-SG-SG-CB dihedral at -90
    degrees, the preferred value of a disulfide. A CONECT record joins
    the two SG atoms and a TER record ends each chain.

    Returns
    -------
    str
        The PDB text.
    """
    positions = {name: np.array([x, y, z]) for name, _, x, y, z in _CYS_BRIDGED_ATOMS}
    # The SG-SG direction of 3cu9, from SG of CYS 222 to SG of CYS 221.
    along = np.array([-22.908, 8.520, 6.302]) - positions["SG"]
    along /= np.linalg.norm(along)
    bond = 2.05 * along
    middle = positions["SG"] + bond / 2
    # Frame of the CB-SG-SG plane. A turn of 180 degrees about an axis
    # that stands at the angle phi out of that plane sets the
    # CB-SG-SG-CB dihedral to 2 * phi, so phi = -45 degrees gives -90.
    in_plane = positions["CB"] - positions["SG"]
    in_plane -= in_plane.dot(along) * along
    in_plane /= np.linalg.norm(in_plane)
    axis = in_plane - np.cross(along, in_plane)
    axis /= np.linalg.norm(axis)

    lines = []
    serial = 1
    sulfurs = []
    for chain_id, turned in (("A", False), ("B", True)):
        for name, element, x, y, z in _CYS_BRIDGED_ATOMS:
            position = np.array([x, y, z])
            if turned:
                offset = position - middle
                position = middle + 2 * axis.dot(offset) * axis - offset
            field = f" {name:<3s}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:5d} {field} CYS {chain_id}   1    "
                f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )
            if name == "SG":
                sulfurs.append(serial)
            serial += 1
        lines.append(f"TER   {serial:5d}      CYS {chain_id}   1")
        serial += 1
    lines.append(f"CONECT{sulfurs[0]:5d}{sulfurs[1]:5d}")
    lines.append(f"CONECT{sulfurs[1]:5d}{sulfurs[0]:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"
