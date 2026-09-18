"""Tests for loading a protonated protein PDB into a Compound tree."""

import logging
from pathlib import Path

import numpy as np
import pytest

import mbuild as mb
from mbuild.biopolymers import Protein
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest
from mbuild.tests.biopolymers_pdb_text import (
    _cys_cys_cross_chain_disulfide,
    _gly_gly_hexadecimal,
    _gly_gly_with_ter,
)
from mbuild.utils.io import get_fn


class TestProteinLoad(BaseTest):
    @pytest.fixture(scope="class")
    def protein_8ciq(self):
        # Small single-chain structure carrying three disulfides.
        return Protein(get_fn("8ciq.pdb"))

    @pytest.fixture(scope="class")
    def _protein_6m03_cached(self):
        """Load the protein once per class; tests clone it.

        The load is the slow part, and several tests mutate the protein,
        so each test gets its own clone of one cached load rather than
        sharing one object or re-reading the file.
        """
        return Protein(get_fn("6m03_protonated.pdb"))

    @pytest.fixture
    def protein_6m03(self, _protein_6m03_cached):
        return mb.clone(_protein_6m03_cached)

    def test_load_protonated_protein(self, protein_6m03):
        # Tests that a pdbfixer-protonated protein PDB loads by template
        # matching with full chemistry: hierarchy, per-residue formal
        # charges from the matched variants, charged termini, and a bond
        # order on every bond. This is needed because the whole package
        # rests on the loader stamping template chemistry instead of
        # guessing from the file. The test loads the bundled protonated
        # SARS-CoV-2 main protease (306 residues, net charge -4 at pH 7)
        # and checks structure and chemistry counts.
        protein = protein_6m03
        assert [chain.chain_id for chain in protein.chains] == ["A"]
        residues = list(protein.residues())
        assert len(residues) == 306
        assert protein.n_particles == 4682
        assert protein.net_formal_charge == -4

        first, last = residues[0], residues[-1]
        assert (first.name, first.resnum, first.formal_charge) == ("SER", 1, 1)
        assert last.formal_charge == -1
        assert any(p.name == "OXT" for p in last.particles())

        lysines = [r for r in residues if r.name == "LYS"]
        assert all(r.formal_charge == 1 for r in lysines)

        orders = {
            bond[2]["bond_order"] for bond in protein.bonds(return_bond_order=True)
        }
        assert orders == {1.0, 2.0}

    def test_load_strict_errors(self, tmp_path, monkeypatch):
        # Tests that the loader fails loudly, with the residue named in
        # the message, on a bad atom name and on an unknown residue code.
        # This is needed because the loader must never guess chemistry:
        # a file that does not match the templates has to be fixed by the
        # user, not silently misread. The test corrupts one atom name and
        # one residue name of the good asset and asserts on the errors.
        # It also asserts that the list of accepted names holds H3, the
        # extra proton of the N-terminal serine, which only one variant
        # carries. The user download cache points at an empty tmp
        # directory, so a definition downloaded in an earlier session
        # cannot make the corrupt residue name known.
        from mbuild.biopolymers import ccd

        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        text = open(get_fn("6m03_protonated.pdb")).read()

        bad_atom = Path("bad_atom.pdb")
        bad_atom.write_text(text.replace(" CB  SER A   1", " QQ  SER A   1", 1))
        with pytest.raises(MBuildError, match="SER A:1") as error:
            Protein(str(bad_atom))
        assert ", H3," in str(error.value)

        bad_residue = Path("bad_residue.pdb")
        bad_residue.write_text(text.replace("SER A   1", "XYZ A   1"))
        with pytest.raises(MBuildError, match="download=True"):
            Protein(str(bad_residue))

    def test_four_character_residue_name_is_read_whole(self, tmp_path, monkeypatch):
        # Tests that a four-character residue name reaches the template
        # lookup whole. This is needed because the reader took columns
        # 18-20 only. A name such as the lipid name DLPC became DLP. It
        # then matched a different component with no error at all: a
        # silent wrong answer. The test renames the N-terminal serine
        # of the good asset to a four-character name, which fills
        # column 21 too, and reads the name back from the
        # unknown-residue error. The user download cache points at an
        # empty tmp directory, so a definition downloaded in an earlier
        # session cannot make the name known.
        from mbuild.biopolymers import ccd

        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        wide = Path("wide_resname.pdb")
        wide.write_text(
            open(get_fn("6m03_protonated.pdb")).read().replace("SER A   1", "DLPCA   1")
        )
        with pytest.raises(MBuildError, match="'DLPC' is not in the CCD"):
            Protein(str(wide))

    def test_alternate_locations_keep_the_first_conformer(self, caplog):
        # Tests that a residue with two alternate locations loads with
        # the first conformer only and logs one warning that names the
        # residue. This is needed because a crystal structure gives a
        # disordered atom one record per conformer, and the loader
        # raised on every altLoc letter, so such a file could not load
        # at all. The test writes a copy of the good asset in which
        # three atoms of the N-terminal serine carry an altLoc 'A'
        # record and an altLoc 'B' record 1 A away, then checks the
        # log and the position of the CB atom, which must hold the 'A'
        # coordinates.
        lines = []
        for line in open(get_fn("6m03_protonated.pdb")).read().splitlines():
            if line[17:26] == "SER A   1" and line[12:16].strip() in (
                "CB",
                "OG",
                "HG",
            ):
                shifted = f"{float(line[30:38]) + 1.0:8.3f}"
                lines.append(line[:16] + "A" + line[17:])
                lines.append(line[:16] + "B" + line[17:30] + shifted + line[38:])
            else:
                lines.append(line)
        altloc = Path("altloc.pdb")
        altloc.write_text("\n".join(lines) + "\n")

        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein = Protein(str(altloc))
        assert len(caplog.records) == 1
        assert "SER A:1" in caplog.text
        assert protein.get_atom(1, "CB", chain_id="A").pos[0] == pytest.approx(-0.2645)

    def test_alternate_location_letters_differ_per_residue(self, caplog):
        # Tests that two residues whose disorder groups use different
        # altLoc letters both keep their first conformer. This is
        # needed because the reader chose one letter for the whole
        # file. A residue that used other letters lost every disordered
        # record, and it left the structure with no error. The test
        # writes a glycine pair. The CA atom carries an 'A' and a 'B'
        # record in residue 1, and a 'C' and a 'D' record in residue 2.
        # The test then checks the particle count, the x position of
        # each kept CA record, and the warning text.
        lines = []
        for line in _gly_gly_with_ter(complete=True).splitlines():
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                first, second = ("A", "B") if line[22:26] == "   1" else ("C", "D")
                shifted = f"{float(line[30:38]) + 1.0:8.3f}"
                lines.append(line[:16] + first + line[17:])
                lines.append(line[:16] + second + line[17:30] + shifted + line[38:])
            else:
                lines.append(line)
        letters = Path("altloc_letters.pdb")
        letters.write_text("\n".join(lines) + "\n")

        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein = Protein(str(letters))
        assert protein.n_particles == 20
        # The kept records are the 'A' record of residue 1 and the 'C'
        # record of residue 2, so both CA atoms hold the unshifted x.
        assert protein.get_atom(1, "CA", chain_id="A").pos[0] == pytest.approx(0.7516)
        assert protein.get_atom(2, "CA", chain_id="A").pos[0] == pytest.approx(0.4120)
        assert "GLY A:1 keeps 'A'" in caplog.text
        assert "GLY A:2 keeps 'C'" in caplog.text

    def test_monatomic_ion_templates(self):
        # Tests that a PDB which holds one ion of each bundled
        # monatomic component loads with the formal charge of every
        # ion. This is needed because a prepared MD system carries
        # counter-ions, and the library held sodium and chloride only.
        # The CIF reader also read the loop_ form of a category alone,
        # so even those two templates carried no atoms and no ion could
        # load. The test writes one HETATM record per ion, each in its
        # own residue, and compares the residue charges: five cations,
        # five anions and neutral xenon.
        ions = ["LI", "NA", "K", "RB", "CS", "F", "CL", "BR", "I", "IOD", "XE"]
        lines = []
        for serial, resname in enumerate(ions, start=1):
            atom = "I" if resname == "IOD" else resname
            lines.append(
                f"HETATM{serial:5d} {atom:<4s} {resname:<4s}"
                f"A{serial:4d}    {serial * 5.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
                f"  1.00  0.00          {atom:>2s}"
            )
        ion_file = Path("ions.pdb")
        ion_file.write_text("\n".join(lines) + "\nEND\n")

        protein = Protein(str(ion_file))
        assert (
            sorted(residue.formal_charge for residue in protein.residues())
            == [-1] * 5 + [0] + [1] * 5
        )

    def test_pymol_generic_hydrogen_names(self, tmp_path, caplog):
        # Tests that a file whose hydrogens carry the generic names a
        # preparation tool such as PyMOL's h_add writes (H01, H02, ...)
        # loads, that every hydrogen comes out with its CCD name, and
        # that one warning for the whole load names the renaming. This
        # is needed because such a file is the common case for a user
        # who protonated a structure in a viewer, and the templates
        # know no such names. The test rewrites the hydrogen names of
        # the bundled 8ciq asset per residue and compares the loaded
        # atom names against the load of the original file.
        original = Protein(get_fn("8ciq.pdb"))
        lines = []
        counters = {}
        for line in Path(get_fn("8ciq.pdb")).read_text().splitlines():
            if line.startswith("ATOM") and line[76:78].strip() == "H":
                key = line[17:27]
                counters[key] = counters.get(key, 0) + 1
                line = f"{line[:12]} H{counters[key]:02d}{line[16:]}"
            lines.append(line)
        renamed = tmp_path / "8ciq_pymol_names.pdb"
        renamed.write_text("\n".join(lines) + "\n")
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein = Protein(str(renamed))
        warnings = [r for r in caplog.records if "hydrogen names" in r.getMessage()]
        assert len(warnings) == 1
        assert "H01 of" in warnings[0].getMessage()
        assert protein.n_particles == original.n_particles
        assert protein.n_bonds == original.n_bonds
        for before, after in zip(original.residues(), protein.residues()):
            assert after.name == before.name
            assert after.template.description == before.template.description
            assert sorted(p.name for p in after.particles()) == sorted(
                p.name for p in before.particles()
            )
        assert not any(p.name.startswith("H0") for p in protein.particles())

    def test_amber_digit_first_hydrogen_names(self):
        # Tests that a capped arginine written with digit-first
        # hydrogen names (2HB, 1HH1) loads with the canonical names of
        # the CCD template (HB2, HH11). This is needed because
        # digit-first names are PDB format version 2 names that
        # Amber-style tools still write, and the first-hit name lookup
        # rejects every one of them, so such a file could not load at
        # all. The test loads the bundled capped-arginine asset, reads
        # back the residue names and the arginine atom names, and
        # checks the position of one renamed hydrogen: a swapped
        # assignment keeps the name set but moves the atoms.
        protein = Protein(get_fn("capped_arg_altresonance.pdb"))
        residues = list(protein.residues())
        assert [residue.name for residue in residues] == ["ACE", "ARG", "NME"]
        names = {particle.name for particle in residues[1].particles()}
        assert {"HB2", "HB3", "HH11", "HH12"} <= names
        # The record named 2HB sits at (0.336, -0.947, 2.140) angstrom
        # in the asset, so HB2 must carry that position in nanometers.
        hb2 = next(residues[1].particles_by_name("HB2"))
        assert np.allclose(hb2.pos, [0.0336, -0.0947, 0.2140])

    def test_v2_glycine_alpha_hydrogens(self):
        # Tests that a glycine written with the alpha-hydrogen names
        # HA1 and HA2 loads as the atoms HA2 and HA3. This is needed
        # because HA1/HA2 are the PDB format version 2 names of the
        # atoms that wwPDB version 3 calls HA2/HA3, and the CCD lists
        # HA1 as an alternative name of HA2 and HA2 as an alternative
        # name of HA3, so a first-hit lookup gives both records the
        # atom HA2 and the residue matches no variant. The test renames
        # the alpha hydrogens of every glycine in the bundled
        # protonated asset, loads the result, reads back the atom names
        # of a glycine, and checks the position of one of them: a
        # swapped assignment keeps the name set but moves the atoms.
        text = open(get_fn("6m03_protonated.pdb")).read()
        renamed = Path("6m03_v2_glycine.pdb")
        renamed.write_text(
            text.replace(" HA2 GLY", " HA1 GLY").replace(" HA3 GLY", " HA2 GLY")
        )

        protein = Protein(str(renamed))
        assert protein.n_particles == 4682
        glycine = next(r for r in protein.residues() if r.name == "GLY")
        assert {"HA2", "HA3"} <= {p.name for p in glycine.particles()}
        # The first glycine is GLY A 2. Its renamed HA1 record sits at
        # (-2.640, -4.080, -12.616) angstrom, and it supplies HA2.
        ha2 = next(glycine.particles_by_name("HA2"))
        assert np.allclose(ha2.pos, [-0.2640, -0.4080, -1.2616])

    def test_hexadecimal_serial_and_resseq(self):
        # Tests that a PDB whose atom serials and residue numbers are
        # written in hexadecimal loads with the decoded values. This is
        # needed because OpenMM switches both fields to hexadecimal
        # when they overflow their columns. Every system above 99999
        # atoms or 9999 residues carries such fields, and the
        # decimal-only reader raised ValueError on them. The test writes a
        # glycine pair whose serials read A0000 upward and whose
        # residue numbers read A001 and A002, then checks the particle
        # count and both residue numbers.
        overflowed = Path("gly_hex.pdb")
        overflowed.write_text(_gly_gly_hexadecimal())
        protein = Protein(str(overflowed))
        assert protein.n_particles == 20
        assert [r.resnum for r in protein.residues()] == [10001, 10002]

    def test_hexadecimal_conect_serials(self):
        # Tests that a CONECT record whose atom serials are written in
        # hexadecimal loads. This is needed because OpenMM applies the
        # same overflow encoding to CONECT serials as to ATOM serials.
        # The decimal-only reader raised ValueError on the CONECT
        # records of every system above 99999 atoms. The test appends a
        # CONECT record for the backbone N-CA bond of the first residue
        # of the same glycine pair, whose serials read A0000 and A0004,
        # then loads the file. The loader checks every CONECT record
        # against the bonds the templates predict, so a load that
        # succeeds proves both serials decoded to the two bonded atoms.
        with_conect = Path("gly_hex_conect.pdb")
        with_conect.write_text(_gly_gly_hexadecimal() + "CONECTA0000A0004\n")
        protein = Protein(str(with_conect))
        assert protein.n_particles == 20

    def test_index_decoding_error_names_the_line(self):
        # Tests that a residue number which is neither decimal nor
        # hexadecimal raises an error that names the line and the
        # field. This is needed because a prepared system holds
        # millions of records, and a message without the line number
        # leaves the user no way to find the bad record. The test
        # writes a glycine pair, corrupts the residue number columns of
        # the third record, and asserts on the error text.
        lines = _gly_gly_with_ter().splitlines()
        lines[2] = lines[2][:22] + "1Q2W" + lines[2][26:]
        corrupt = Path("bad_index.pdb")
        corrupt.write_text("\n".join(lines) + "\n")
        with pytest.raises(MBuildError, match=r"'1Q2W' on line 3"):
            Protein(str(corrupt))

    def test_extra_ter_inside_one_chain(self, caplog):
        # Tests that a TER record between two consecutive residues of
        # one chain keeps the peptide bond and logs a warning. This is
        # needed because a tool that writes a PDB file from a topology
        # puts a TER at the end of each topology chain, so a capped
        # structure carries a TER before its C-terminal cap; treating
        # every TER as a chain break rejected such a file, because the
        # SER before the TER has no OXT atom and the NME after it has
        # no H2 atom. The test loads such a file, checks the bond
        # between the SER C atom and the NME N atom, and reads the log.
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein = Protein(get_fn("capped_ser_extrater.pdb"))
        carbon = protein.get_atom(178, "C", chain_id="A")
        nitrogen = protein.get_atom(179, "N", chain_id="A")
        assert protein.bond_graph.has_edge(carbon, nitrogen)
        assert "SER A:178" in caplog.text and "NME A:179" in caplog.text

    def test_far_apart_pair_across_a_ter_does_not_bond(self):
        # Tests that the loader ends the chain at a TER when the two
        # residues are too far apart for a peptide bond, and that the
        # error names the TER. This is needed because the advisory-TER
        # rule read the chain identifier and the residue numbering
        # only, so two separate molecules that share one chain and
        # carry increasing numbers were bonded across 21 A. The test
        # writes a glycine pair whose second residue is shifted by
        # 20 A and asserts on the error text.
        far = Path("gly_far.pdb")
        far.write_text(_gly_gly_with_ter(offset=-20.0))
        with pytest.raises(MBuildError, match="too far for a peptide bond") as info:
            Protein(str(far))
        assert "A TER record separates GLY A:1 from GLY A:2" in str(info.value)

    def test_ter_warning_only_where_the_bond_is_made(self, caplog):
        # Tests that no TER warning is logged when the two residues do
        # not expect a peptide bond. This is needed because the warning
        # was logged from the adjacency list, which only reports that a
        # bond is possible, so the loader claimed to keep a bond that
        # the matched templates never asked for. The test writes a
        # glycine pair that carries every leaving atom, keeps the TER,
        # and checks both the missing bond and the empty log.
        complete = Path("gly_complete.pdb")
        complete.write_text(_gly_gly_with_ter(complete=True))
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein = Protein(str(complete))
        carbon = protein.get_atom(1, "C", chain_id="A")
        nitrogen = protein.get_atom(2, "N", chain_id="A")
        assert not protein.bond_graph.has_edge(carbon, nitrogen)
        assert "TER" not in caplog.text

    @pytest.mark.parametrize("resnum, icode", [(3, ""), (1, "A")])
    def test_advisory_ter_keeps_the_bond(self, resnum, icode):
        # Tests that a TER between two residues of one chain stays advisory
        # and keeps the peptide bond, both when the second residue carries a
        # numbering gap and when it carries an insertion code. This is
        # needed because OpenMM writes numbering gaps and insertion codes,
        # and the first advisory-TER rule read both as chain breaks. The
        # test writes a glycine pair numbered 1 and 3, and a pair numbered 1
        # and 1A, each with a TER between them, and checks the C to N bond.
        path = Path(f"gly_ter_{resnum}{icode}.pdb")
        path.write_text(_gly_gly_with_ter(resnum=resnum, icode=icode or " "))
        protein = Protein(str(path))
        carbon = protein.get_atom(1, "C", chain_id="A")
        nitrogen = protein.get_atom(resnum, "N", chain_id="A", icode=icode)
        assert protein.bond_graph.has_edge(carbon, nitrogen)

    def test_vicinal_disulfide_3cu9(self):
        # Tests that a disulfide between two adjacent cysteines loads as
        # one SG-SG cross bond with neutral residues. This is needed
        # because a bridged cysteine (HG absent) matches both the
        # crosslink variants and the thiolate variants, and before the
        # CONECT-aware filter the loader rejected every disulfide file
        # as ambiguous. The test loads the vicinal disulfide of 3cu9 and
        # checks the recorded bond, the leaving atoms, the charges, and
        # the bond graph edge.
        from mbuild.biopolymers.protein import Chain

        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        (record,) = protein.cross_bonds
        assert (record.atom1_name, record.atom2_name) == ("SG", "SG")
        assert record.leaving1 == ("HG",) and record.leaving2 == ("HG",)
        assert record.residue1.formal_charge == 0
        assert record.residue2.formal_charge == 0
        sg1 = next(record.residue1.particles_by_name("SG"))
        sg2 = next(record.residue2.particles_by_name("SG"))
        assert protein.bond_graph.has_edge(sg1, sg2)
        assert isinstance(record.residue1.parent, Chain)

    def test_disulfides_8ciq(self, protein_8ciq):
        # Tests that a protein with three disulfides loads all three
        # cross bonds. This is needed because a file with several
        # bridges shows that the crosslink filter pairs each CONECT
        # record with the right two cysteines. The test loads the 8ciq
        # asset and counts the cross bonds.
        assert len(protein_8ciq.cross_bonds) == 3

    def test_bare_thiolate_loads_without_conect(self):
        # Tests that a cysteine without HG and without an SS CONECT
        # loads as a deprotonated thiolate, not as an error. This is
        # needed because the CONECT-aware filter must reject only the
        # crosslink variants in this case and keep the thiolate variant,
        # which is the openff-pablo behavior. The test removes the
        # CONECT records from the 3cu9 asset (its cysteines already
        # carry no HG) and checks the charges and the empty cross-bond
        # list.
        text = open(get_fn("3cu9_vicinal_disulfide.pdb")).read()
        stripped = Path("thiolate.pdb")
        stripped.write_text(
            "\n".join(
                line for line in text.splitlines() if not line.startswith("CONECT")
            )
        )
        protein = Protein(str(stripped))
        assert protein.cross_bonds == []
        for residue in protein.residues():
            assert residue.atom_formal_charges["SG"] == -1

    def test_ss_conect_with_hg_present_errors(self):
        # Tests that an SS CONECT to a cysteine that still carries its
        # HG raises an error that names the disulfide conflict. This is
        # needed because the loader must never guess: the file claims a
        # disulfide through the CONECT record and denies it through the
        # present HG, and only the user can decide which one is true.
        # The test appends an HG atom to one 3cu9 cysteine, keeps the
        # CONECT records, and asserts on the error message.
        text = open(get_fn("3cu9_vicinal_disulfide.pdb")).read()
        hg_line = (
            "ATOM     24  HG  CYS A 222     -22.000  10.500   6.500"
            "  1.00 11.91           H"
        )
        lines = text.splitlines()
        insert_at = next(
            index for index, line in enumerate(lines) if line.startswith("CONECT")
        )
        lines.insert(insert_at, hg_line)
        bad = Path("hg_present.pdb")
        bad.write_text("\n".join(lines))
        with pytest.raises(MBuildError, match="signals a disulfide"):
            Protein(str(bad))

    def test_non_cys_bridge_conect_names_the_cys_limit(self):
        # Tests that a CONECT record between the sulfur atoms of two
        # residues that are not both CYS raises an error naming both
        # residues and the CYS-only limit. This is needed because only
        # the CYS template carries the SG-SG crosslink, and the old
        # message said that no template predicts the bond, which does
        # not say which residues mBuild bridges. The test appends a
        # CONECT between the SG of CYS 16 and the SD of MET 17 of the
        # bundled 6m03 asset and reads both residue labels and the
        # CYS token back from the error text.
        text = open(get_fn("6m03_protonated.pdb")).read()
        bad = Path("met_bridge.pdb")
        bad.write_text(text.replace("END", "CONECT  235  249\nEND"))
        with pytest.raises(MBuildError) as error:
            Protein(str(bad))
        message = str(error.value)
        assert "CYS A:16 SG" in message and "MET A:17 SD" in message
        assert "CYS residues only" in message

    def test_bridged_residue_without_its_sulfur_hydrogen(self):
        # Tests that a residue which matches no template variant, and which
        # a CONECT record bridges through a sulfur atom that lost its
        # hydrogen, reports the CYS-only limit and asks for that hydrogen
        # back. This is needed because the match error alone tells the user
        # to protonate the file, and the new hydrogen breaks the bridge. The
        # test drops the HG and the HA of CYS 16 of the 6m03 asset and
        # bridges its SG to the SD of MET 17.
        text = open(get_fn("6m03_protonated.pdb")).read()
        lines = [
            line
            for line in text.splitlines()
            if "HG  CYS A  16" not in line and "HA  CYS A  16" not in line
        ]
        insert_at = next(
            index
            for index, line in enumerate(lines)
            if line.startswith(("CONECT", "END"))
        )
        lines.insert(insert_at, "CONECT  235  249")
        bad = Path("bridged_no_hg.pdb")
        bad.write_text("\n".join(lines) + "\n")
        with pytest.raises(MBuildError) as error:
            Protein(str(bad))
        message = str(error.value)
        assert "CYS A:16 SG" in message and "MET A:17 SD" in message
        assert "CYS residues only" in message
        # The other bridge error asks for the CONECT record alone. This
        # one must also name the hydrogen that the file must regain.
        assert "add the hydrogen HG" in message

    def test_cross_chain_disulfide(self):
        # Tests that a disulfide between a cysteine of chain A and a
        # cysteine of chain B loads as one cross bond, and that both
        # residues matched the bridged CYS variant. This is needed because
        # the crosslink filter reads the global atom serials of the CONECT
        # record, so a filter keyed on chain-local state would miss such a
        # bond. The test writes two bridged cysteines, one per chain, then
        # reads the labels, the atom names and the leaving atom of the bond.
        path = Path("cross_chain_ss.pdb")
        path.write_text(_cys_cys_cross_chain_disulfide())
        protein = Protein(str(path))
        (record,) = protein.cross_bonds
        assert (record.atom1_name, record.atom2_name) == ("SG", "SG")
        labels = [
            (residue.parent.chain_id, residue.resnum)
            for residue in (record.residue1, record.residue2)
        ]
        assert labels == [("A", 1), ("B", 1)]
        # The bridged variant is the one whose SG gives up an HG atom.
        # A thiolate match would leave SG with no leaving atom.
        assert record.leaving1 == ("HG",) and record.leaving2 == ("HG",)

    def test_get_atom(self, protein_6m03):
        # Tests that residues and atoms are addressable by residue number
        # and atom name. This is needed because functionalization
        # workflows pick attachment sites this way (e.g. lysine NZ). The
        # test fetches a known atom and asserts the not-found error names
        # the residue's atoms.
        protein = protein_6m03
        nz = protein.get_atom(90, "NZ", chain_id="A")
        assert nz.name == "NZ" and nz.element.symbol == "N"
        with pytest.raises(MBuildError, match="no atom"):
            protein.get_atom(90, "XX", chain_id="A")

    def test_get_residue_ambiguity_error_names_chains(self):
        # Tests that get_residue raises an MBuildError that lists the
        # candidate chains when a residue number repeats across chains, even
        # when one residue sits under a wrapper Compound. This is needed
        # because the old error path read residue.parent.chain_id, and a
        # wrapped fragment residue's parent is the wrapper, so the error
        # report itself crashed. The test adds a wrapped residue with a
        # duplicate number in a second chain.
        from mbuild.biopolymers.protein import Chain, Residue

        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        wrapper = mb.Compound(name="wrapper")
        wrapper.add(Residue(resname="LIG", resnum=221, hetatm=True))
        chain = Chain(chain_id="B")
        chain.add(wrapper)
        protein.add(chain)
        with pytest.raises(MBuildError, match=r"chains \['A', 'B'\]"):
            protein.get_residue(221)
