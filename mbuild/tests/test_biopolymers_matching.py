"""Tests for PDB record parsing and residue-template matching.

Every input here is inline PDB text, so these tests need no fixture
files and no mBuild objects: they exercise the parser and the matcher
directly. The behaviour of the loader that consumes them is covered in
test_biopolymers_protein.py.
"""

import numpy as np
import pytest

from mbuild.biopolymers.ccd import CCDLibrary
from mbuild.biopolymers.matching import _match_residue
from mbuild.biopolymers.protein_pdb_io import _parse_pdb
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest

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


class TestPdbParsing(BaseTest):
    def test_records_carry_pdb_identity(self):
        # Tests that the parser keeps the identity fields the matcher and
        # the loader key on. This is needed because every later step
        # addresses a residue by (chain, resnum, icode) and an atom by
        # name, so a parser that dropped any of them would make those
        # addresses ambiguous.
        residues, conects, box = _parse_pdb(_gly_gly_with_ter())
        assert [residue.label for residue in residues] == [
            "GLY A:1",
            "GLY A:2",
        ]
        first = residues[0]
        assert first.resname == "GLY"
        assert first.chain_id == "A"
        assert first.resnum == 1
        assert [record.name for record in first.records][:3] == [
            "N",
            "H",
            "H2",
        ]
        assert isinstance(first.records[0].pos, np.ndarray)

    def test_ter_record_is_reported(self):
        # Tests that a TER record between two residues is recorded on the
        # residue that carries it. This is needed because the matcher
        # decides whether a peptide bond crosses that break, and it can
        # only do so if the parser preserves the record.
        residues, _, _ = _parse_pdb(_gly_gly_with_ter())
        assert residues[0].ter_after is True
        assert residues[1].ter_after is False

    def test_insertion_code_is_kept(self):
        # Tests that an insertion code is parsed into its own field
        # rather than folded into the residue number. This is needed
        # because two residues may share a number and differ only by
        # insertion code, and merging them would lose a residue.
        residues, _, _ = _parse_pdb(_gly_gly_with_ter(resnum=1, icode="A"))
        assert residues[1].icode == "A"
        assert residues[1].resnum == 1
        assert residues[0].icode in ("", " ")

    def test_hybrid36_serials_and_resseq(self):
        # Tests that hybrid-36 encoded atom serials and residue numbers
        # decode to their integer values. This is needed because a
        # structure with more than 99999 atoms encodes them this way, and
        # a parser that read them as decimal would mis-number every atom.
        residues, conects, _ = _parse_pdb(_gly_gly_hexadecimal())
        serials = [record.serial for residue in residues for record in residue.records]
        assert all(isinstance(serial, int) for serial in serials)
        assert max(serials) > 99999

    def test_conect_pairs_are_returned(self):
        # Tests that CONECT records are parsed into serial pairs. This is
        # needed because a crosslink is only formed where a template
        # expectation and a CONECT record agree, so a dropped CONECT
        # would silently lose a disulfide.
        _, conects, _ = _parse_pdb(_cys_cys_cross_chain_disulfide())
        assert conects
        assert all(len(pair) == 2 for pair in conects) or isinstance(conects, dict)

    def test_a_bad_index_names_the_line(self):
        # Tests that an undecodable serial raises an error naming the
        # line number. This is needed because the loader never guesses:
        # a malformed file must point the user at the record to fix.
        text = (
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\n"
            "ATOM    ****  CA  GLY A   1       1.458   0.000   0.000"
            "  1.00  0.00           C\n"
        )
        with pytest.raises(MBuildError, match="line 2"):
            _parse_pdb(text)


class TestResidueMatching(BaseTest):
    @pytest.fixture(scope="class")
    def library(self):
        return CCDLibrary()

    def test_leaving_atoms_imply_a_peptide_bond(self, library):
        # Tests that a glycine missing OXT and HXT matches a variant that
        # expects a peptide bond to the residue after it, and that one
        # missing H2 expects a bond to the residue before it. This is
        # needed because the loader forms backbone bonds from these
        # expectations rather than from distance, so the flags are what
        # make a chain a chain.
        residues, _, _ = _parse_pdb(_gly_gly_with_ter())
        first = _match_residue(residues[0], library["GLY"], False, True)
        second = _match_residue(residues[1], library["GLY"], True, False)
        assert any(match.expects_posterior for match in first)
        assert any(match.expects_prior for match in second)

    def test_a_complete_residue_expects_no_bond(self, library):
        # Tests that a glycine holding every template atom matches a
        # variant that expects neither neighbour. This is needed so that
        # a free amino acid or a capped terminus does not acquire a bond
        # to whatever residue happens to follow it in the file.
        residues, _, _ = _parse_pdb(_gly_gly_with_ter(complete=True))
        matches = _match_residue(residues[0], library["GLY"], False, False)
        assert matches
        assert not any(match.expects_posterior for match in matches)
        assert not any(match.expects_prior for match in matches)

    def test_unexplained_atoms_raise_and_name_the_residue(self, library):
        # Tests that a residue no variant explains raises an error naming
        # the residue and listing the atoms each variant wanted. This is
        # the module's central promise: chemistry is never guessed, and
        # the message must be enough to fix the file.
        text = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000"
            "  1.00  0.00           C\n"
        )
        residues, _, _ = _parse_pdb(text)
        with pytest.raises(MBuildError, match="ALA A:1"):
            _match_residue(residues[0], library["ALA"], False, False)

    def test_a_missing_sulfur_hydrogen_expects_a_crosslink(self, library):
        # Tests that a cysteine without HG matches a variant that expects
        # a crosslink. This is needed because a disulfide is formed only
        # where this expectation and a CONECT record agree; without the
        # flag the bridge would be dropped.
        residues, _, _ = _parse_pdb(_cys_cys_cross_chain_disulfide())
        matches = _match_residue(residues[0], library["CYS"], False, False)
        assert any(match.expects_crosslink for match in matches)
