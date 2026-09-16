"""Tests for writing a prepared PDB file and exporting to RDKit."""

from pathlib import Path

import pytest

import mbuild as mb
from mbuild.biopolymers import Protein
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import get_fn, has_rdkit

try:
    import nglview  # noqa: F401

    has_nglview = True
except ImportError:
    has_nglview = False


class TestProteinExports(BaseTest):
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

    def test_pdb_text_is_what_save_pdb_writes(self, protein_6m03, tmp_path):
        from mbuild.biopolymers.protein_pdb_io import pdb_text

        written = tmp_path / "out.pdb"
        protein_6m03.save_pdb(written)
        assert written.read_text() == pdb_text(protein_6m03)

    def test_licorice_selection_names_the_bonded_residues(self):
        # Fragments are "hetero"; a protein residue that carries a
        # recorded bond joins the selection by number and chain, so the
        # bond between them is drawn. 3cu9 has one disulfide, between
        # cysteines 221 and 222 of chain A.
        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        assert len(protein.cross_bonds) == 1
        assert protein._licorice_selection() == "hetero or 221:A or 222:A"

    @pytest.mark.skipif(not has_nglview, reason="nglview is not installed")
    def test_visualize_returns_a_widget(self, protein_6m03):
        # The widget only renders inside a notebook, and its
        # representation list fills from the front end, so headless the
        # test can only check that the widget builds.
        import nglview

        assert isinstance(protein_6m03.visualize(), nglview.NGLWidget)
        assert isinstance(protein_6m03.visualize(show_box=True), nglview.NGLWidget)

    def test_save_pdb_roundtrip(self, protein_6m03):
        # Tests that a PDB loaded into mBuild, written out by save_pdb,
        # and read back into mBuild gives the same structure and
        # chemistry. This is needed because the written file is the
        # handoff artifact for residue-template readers, which apply
        # the same matching rules as Protein. The test writes and
        # reloads the protein, compares counts and net charge, checks
        # the fixed-column layout of one atom line, and checks that no
        # bond-records file was written.
        protein = protein_6m03
        out = Path("roundtrip.pdb")
        protein.save_pdb(str(out))
        # 6m03 holds no disulfide and no attached fragment, so there is
        # no bond record and no template to write. An empty file would
        # tell a user nothing and would look like a file to pass back.
        assert not Path("roundtrip.bondrecords.json").exists()
        reloaded = Protein(str(out))
        assert reloaded.n_particles == protein.n_particles
        assert reloaded.net_formal_charge == protein.net_formal_charge
        assert len(list(reloaded.residues())) == 306

        lines = out.read_text().splitlines()
        first = next(line for line in lines if line.startswith("ATOM"))
        assert first[6:11] == "    1"
        assert first[12:16].strip() == "N"
        assert first[17:20] == "SER"
        assert first[21] == "A"
        assert first[22:26] == "   1"
        assert first[76:78].strip() == "N"
        assert sum(line.startswith("TER") for line in lines) == 1

        with pytest.raises(IOError, match="not overwriting"):
            protein.save_pdb(str(out))

    def test_save_pdb_rejects_orphan_atoms(self, protein_6m03):
        # Tests that save_pdb raises on a particle outside a
        # Chain -> Residue path. This is needed because the writer walks
        # protein.chains, so such a particle and its bonds would be
        # silently absent from the file; to_rdkit already raises on the
        # same condition and the two exports must agree. The test adds
        # a bonded carbon directly onto the Protein and asserts the
        # orphan error.
        protein = protein_6m03
        carbon = mb.Compound(name="CX", element="C")
        protein.add(carbon)
        protein.add_bond((carbon, protein.get_atom(5, "NZ", chain_id="A")))
        with pytest.raises(MBuildError, match="must belong to a Residue"):
            protein.save_pdb("orphan.pdb")

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_reverse_nc_bond_gets_conect(self, protein_6m03):
        # Tests that a bond from N of a residue to C of the next residue
        # gets a CONECT record. This is needed because the writer used
        # to suppress every C/N bond between order-adjacent residues,
        # so this non-peptide bond was dropped and a reader
        # reconstructed the wrong molecule; only the true peptide link
        # (C of the earlier residue to N of the later one) is implied
        # by record adjacency. The test adds the reverse bond between
        # residues 10 and 11, writes the file, and asserts the CONECT
        # pair from both sides.
        protein = protein_6m03
        n10 = protein.get_atom(10, "N", chain_id="A")
        c11 = protein.get_atom(11, "C", chain_id="A")
        protein.add_bond((n10, c11))
        out = Path("reverse.pdb")
        protein.save_pdb(str(out))

        lines = out.read_text().splitlines()
        serial_of = {
            (int(line[22:26]), line[12:16].strip()): int(line[6:11])
            for line in lines
            if line.startswith("ATOM")
        }
        pairs = {
            (int(line[6:11]), partner)
            for line in lines
            if line.startswith("CONECT")
            for partner in map(int, line[11:].split())
        }
        expected = (serial_of[(10, "N")], serial_of[(11, "C")])
        assert expected in pairs and expected[::-1] in pairs

    def test_peptide_bonds_write_no_conect(self, protein_6m03):
        # Tests that an unmodified protein writes zero CONECT records.
        # This is needed because strict template readers fail on a
        # CONECT their residue definitions cannot explain, so the
        # peptide-bond suppression must still cover every backbone
        # link after the direction-aware fix. The test writes the
        # loaded fixture and counts CONECT lines.
        protein = protein_6m03
        out = Path("plain.pdb")
        protein.save_pdb(str(out))
        lines = out.read_text().splitlines()
        assert sum(line.startswith("CONECT") for line in lines) == 0

    def test_insertion_codes_2mum(self):
        # Tests that a PDB with insertion codes loads with the inserted
        # residues addressable through get_residue(resnum, icode=...)
        # and keeps the codes through a save_pdb round trip. This is
        # needed because numbering schemes for antibodies and proteases
        # insert residues under a shared number, the loader keys
        # residues on (resnum, icode), and the writer sorts by the same
        # pair. The test loads the prepared 2MUM structure, addresses
        # two residues that share number 28, and compares the insertion
        # codes and the residue count after a round trip.
        protein = Protein(get_fn("2MUM_icode.pdb"))
        residues = list(protein.residues())
        assert len(residues) == 50
        inserted = sorted((r.resnum, r.icode) for r in residues if r.icode)
        assert inserted == [(14, "A"), (28, "A"), (28, "B"), (34, "A"), (40, "A")]
        assert protein.get_residue(28, icode="A").name == "ASP"
        assert protein.get_residue(28, icode="B").name == "CYS"
        assert protein.get_residue(28).name == "TYR"

        protein.save_pdb("2mum_roundtrip.pdb")
        reloaded = Protein("2mum_roundtrip.pdb")
        assert len(list(reloaded.residues())) == 50
        assert (
            sorted((r.resnum, r.icode) for r in reloaded.residues() if r.icode)
            == inserted
        )

    def test_hexadecimal_indices_round_trip(self):
        # Tests that a structure whose residue numbers overflow their
        # four columns is written and read back unchanged, and that a
        # number above the encodable range raises. This is needed
        # because the writer raised on every residue number above 9999,
        # so a file that the reader accepts could not be written again.
        # The test loads the prepared 2MUM structure and adds 10000 to
        # every residue number. It writes the protein, reads it back,
        # and compares the particle count and the numbers with their
        # insertion codes. It then sets a number above the range of the
        # encoding and asserts on the error.
        protein = Protein(get_fn("2MUM_icode.pdb"))
        for residue in protein.residues():
            residue.resnum += 10000
        numbers = [(r.resnum, r.icode) for r in protein.residues()]
        assert numbers[0][0] == 10001

        protein.save_pdb("2mum_hex_roundtrip.pdb")
        reloaded = Protein("2mum_hex_roundtrip.pdb")
        assert reloaded.n_particles == protein.n_particles
        assert [(r.resnum, r.icode) for r in reloaded.residues()] == numbers

        # The encoding repeats above 10 ** 4 + 6 * 16 ** 3 = 34576, so
        # a residue number of 34576 could not be read back.
        next(iter(protein.residues())).resnum = 34576
        with pytest.raises(MBuildError, match="residue number 34576"):
            protein.save_pdb("2mum_hex_overflow.pdb")

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_to_rdkit_carries_chemistry(self, protein_6m03):
        # Tests that the RDKit export carries the chemistry the loader
        # derived: a sanitized Mol whose net formal charge equals the
        # protein's, every bond with a real order, and PDB residue
        # information on each atom. This is needed because the generic
        # Compound export has no formal-charge path and returns an
        # unsanitized RWMol, so an OpenFF or RDKit handoff through it
        # would silently mis-protonate the protein.
        from rdkit import Chem

        mol = protein_6m03.to_rdkit()
        assert Chem.GetFormalCharge(mol) == protein_6m03.net_formal_charge
        assert mol.GetNumAtoms() == protein_6m03.n_particles
        info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
        assert (info.GetResidueName(), info.GetResidueNumber()) == ("SER", 1)
        assert all(
            bond.GetBondType() != Chem.BondType.UNSPECIFIED for bond in mol.GetBonds()
        )
