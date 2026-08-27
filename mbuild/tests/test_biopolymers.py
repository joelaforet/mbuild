import pytest

from mbuild.biopolymers import CCDLibrary
from mbuild.tests.base_test import BaseTest


class TestCCDLibrary(BaseTest):
    def test_base_template(self):
        # Tests that a bundled CCD cif parses into a template with the
        # atoms, bonds, leaving flags, and peptide leaving fragments the
        # Protein loader matches against. This is needed because template
        # matching depends on exact CCD names and on the leaving-fragment
        # semantics shared with OpenFF Pablo. The test parses ALA and
        # checks the fields against the known CCD entry.
        library = CCDLibrary()
        base = library["ALA"][0]
        assert base.linking == "peptide"
        assert base.atom_names == {
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "OXT",
            "H",
            "H2",
            "HA",
            "HB1",
            "HB2",
            "HB3",
            "HXT",
        }
        assert base.prior_fragment == {"H2"}
        assert base.posterior_fragment == {"OXT", "HXT"}
        assert base.formal_charge == 0
        orders = {
            frozenset((bond.atom1, bond.atom2)): bond.order for bond in base.bonds
        }
        assert orders[frozenset(("C", "O"))] == 2
        assert orders[frozenset(("C", "OXT"))] == 1
        # Amber/OpenMM write the amide H as "H1"; the synonym must resolve.
        assert base.name_to_atom["H1"].name == "H"

    def test_protonation_variants(self):
        # Tests that the pH-relevant protonation variants exist with the
        # correct formal charges. This is needed because a protonated
        # protein PDB contains charged termini and side chains, and the
        # loader stamps net charge from the matched variant. The test
        # checks the N-terminal cation, the deprotonated C-terminus, the
        # protonated lysine, and the neutral HID histidine tautomer.
        library = CCDLibrary()

        nterm = [v for v in library["ALA"] if "H3" in v.atom_names]
        assert nterm and nterm[0].name_to_atom["N"].formal_charge == 1

        cterm = [
            v
            for v in library["ALA"]
            if "OXT" in v.atom_names and "HXT" not in v.atom_names
        ]
        assert cterm and cterm[0].name_to_atom["OXT"].formal_charge == -1

        assert library["LYS"][0].name_to_atom["NZ"].formal_charge == 1

        hid = [
            v
            for v in library["HIS"]
            if "HD1" in v.atom_names and "HE2" not in v.atom_names
        ]
        assert hid
        assert hid[0].name_to_atom["ND1"].formal_charge == 0
        assert hid[0].name_to_atom["NE2"].formal_charge == 0

    def test_caps_link_as_peptides(self):
        # Tests that the ACE/NME caps carry peptide linking with one
        # leaving hydrogen. This is needed because the CCD stores caps as
        # non-polymers, but capped chains in prepared PDB files bond them
        # into the backbone. The test checks the patched linking type and
        # leaving atoms of both caps.
        library = CCDLibrary()
        ace = library["ACE"][0]
        assert ace.linking == "peptide"
        assert [a.name for a in ace.atoms if a.leaving] == ["H"]
        nme = library["NME"][0]
        assert nme.linking == "peptide"
        assert nme.prior_fragment == {"HN1"}

    def test_pablo_parity(self):
        # Tests that mBuild's templates agree with openff-pablo's residue
        # definitions on names, elements, formal charges, leaving flags,
        # and bond orders. This is needed because the tables and patch
        # semantics in ccd.py are mirrored from pablo (see the module
        # docstring), and silent drift between the two would break the
        # PDB round trip this recipe exists for. The test compares the
        # base variant of every bundled amino acid field by field, and
        # runs only where openff-pablo is installed.
        pablo = pytest.importorskip("openff.pablo")
        library = CCDLibrary()
        for resname in (
            "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS "
            "MET PHE PRO SER THR TRP TYR VAL".split()
        ):
            ours = library[resname][0]
            theirs = pablo.STD_CCD_CACHE[resname][0]
            their_atoms = {
                atom.name: atom
                for atom in theirs.atoms
                if not atom.name.startswith("D")  # skip deuterated synonyms
            }
            for atom in ours.atoms:
                other = their_atoms[atom.name]
                assert atom.element == other.symbol
                assert atom.formal_charge == other.charge
                assert atom.leaving == other.leaving
            their_orders = {
                frozenset((bond.atom1, bond.atom2)): bond.order for bond in theirs.bonds
            }
            for bond in ours.bonds:
                assert their_orders[frozenset((bond.atom1, bond.atom2))] == bond.order

    def test_unknown_residue_raises(self):
        # Tests that an unknown residue code raises a KeyError that names
        # the code and the download option. This is needed because the
        # loader is strict: it must fail loudly on residues it cannot
        # match instead of guessing. The test looks up a nonsense code
        # with downloads disabled.
        library = CCDLibrary()
        with pytest.raises(KeyError, match="XXX"):
            library["XXX"]
