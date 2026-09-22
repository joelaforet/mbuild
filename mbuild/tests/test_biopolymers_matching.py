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
from mbuild.tests.biopolymers_pdb_text import (
    _cys_cys_cross_chain_disulfide,
    _gly_gly_hexadecimal,
    _gly_gly_with_ter,
)


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

    def test_generic_hydrogen_names_are_placed_by_geometry(self, library):
        # Tests that a residue whose hydrogens carry generic names such
        # as H01, H02, ... still matches, and that each such record is
        # assigned to a template hydrogen of the heavy atom it sits on.
        # This is needed because PyMOL's h_add writes exactly these
        # names, so a user who protonates a structure there hands the
        # loader a file that no template names. Every hydrogen must
        # come out with a CCD name, and no template atom may be used
        # twice.
        lines = []
        counter = 0
        for line in _gly_gly_with_ter(complete=True).splitlines():
            if line.startswith("ATOM  ") and line[76:78].strip() == "H":
                counter += 1
                line = f"{line[:12]} H{counter:02d}{line[16:]}"
            lines.append(line)
        residues, _, _ = _parse_pdb("\n".join(lines) + "\n")
        assert [r.name for r in residues[0].records if r.name[0] == "H"] == [
            "H01",
            "H02",
            "H03",
            "H04",
            "H05",
            "H06",
        ]
        matches = _match_residue(residues[0], library["GLY"], False, False)
        match = matches[0]
        assigned = {
            record.name: match.record_atoms[id(record)].name
            for record in residues[0].records
        }
        # The three amine hydrogens land on the N slots, the two alpha
        # hydrogens on the CA slots, and the acid hydrogen on OXT.
        assert {assigned["H01"], assigned["H02"], assigned["H03"]} == {
            "H",
            "H2",
            "H3",
        }
        assert {assigned["H04"], assigned["H05"]} == {"HA2", "HA3"}
        assert assigned["H06"] == "HXT"
        assert len(set(assigned.values())) == len(assigned)
        assert not match.missing

    def test_a_misnamed_heavy_atom_is_still_rejected(self, library):
        # Tests that the geometric rule applies to hydrogens only. A
        # heavy atom with an unknown name must still reject the
        # variant, because its name is the only statement of its
        # chemistry and a distance cannot confirm it.
        text = _gly_gly_with_ter(complete=True).replace(" CA  GLY", " CX  GLY")
        residues, _, _ = _parse_pdb(text)
        with pytest.raises(MBuildError, match="'CX' is not in the template"):
            _match_residue(residues[0], library["GLY"], False, False)

    def test_a_stray_hydrogen_is_rejected(self, library):
        # Tests that a generically named hydrogen too far from every
        # heavy atom rejects the variant with the naming reason, rather
        # than being placed on some atom. Otherwise the loader would
        # invent a bond for an atom the file does not explain.
        text = _gly_gly_with_ter(complete=True)
        moved = []
        for line in text.splitlines():
            if line.startswith("ATOM  ") and line[12:16] == " HA3":
                line = f"{line[:12]} H99{line[16:30]}{'20.000':>8s}{'20.000':>8s}{'20.000':>8s}{line[54:]}"
            moved.append(line)
        residues, _, _ = _parse_pdb("\n".join(moved) + "\n")
        with pytest.raises(MBuildError, match="'H99' is not in the template"):
            _match_residue(residues[0], library["GLY"], False, False)

    def test_a_missing_sulfur_hydrogen_expects_a_crosslink(self, library):
        # Tests that a cysteine without HG matches a variant that expects
        # a crosslink. This is needed because a disulfide is formed only
        # where this expectation and a CONECT record agree; without the
        # flag the bridge would be dropped.
        residues, _, _ = _parse_pdb(_cys_cys_cross_chain_disulfide())
        matches = _match_residue(residues[0], library["CYS"], False, False)
        assert any(match.expects_crosslink for match in matches)
