import urllib.request

import pytest

from mbuild.biopolymers import CCDLibrary, ccd
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import has_pycifrw

ATOM_LOOP = """loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
"""

BOND_LOOP = """loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
"""


@pytest.mark.skipif(not has_pycifrw, reason="pycifrw is not installed")
class TestCCDLibrary(BaseTest):
    @pytest.fixture(autouse=True)
    def user_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        return tmp_path

    @pytest.mark.parametrize(
        "path", sorted(ccd.CCD_CACHE_DIR.glob("*.cif")), ids=lambda p: p.stem
    )
    def test_bundled_templates_parse(self, path):
        variants = CCDLibrary()[path.stem]
        base = variants[0]
        assert base.name == path.stem
        assert base.atoms
        for bond in base.bonds:
            assert {bond.atom1, bond.atom2} <= base.atom_names
            assert bond.order in (1, 2, 3)

    def test_base_template(self):
        base = CCDLibrary()["ALA"][0]
        assert base.linking == "peptide"
        assert base.description == "ALANINE"
        names = "N CA C O CB OXT H H2 HA HB1 HB2 HB3 HXT"
        assert base.atom_names == set(names.split())
        assert base.prior_fragment == {"H2"}
        assert base.posterior_fragment == {"OXT", "HXT"}
        assert base.formal_charge == 0
        orders = {frozenset((b.atom1, b.atom2)): b.order for b in base.bonds}
        assert orders[frozenset(("C", "O"))] == 2
        assert orders[frozenset(("C", "OXT"))] == 1
        assert base.name_to_atom["H1"].name == "H"
        assert base.name_to_atom["1HB"].name == "HB1"

    def test_aromatic_flag(self):
        phe = CCDLibrary()["PHE"][0]
        aromatic = {atom.name for atom in phe.atoms if atom.aromatic}
        assert aromatic == {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}

    def test_atoms_named(self):
        gly = CCDLibrary()["GLY"][0]
        assert [atom.name for atom in gly.atoms_named("HA2")] == ["HA2", "HA3"]
        leu = CCDLibrary()["LEU"][0]
        assert [atom.name for atom in leu.atoms_named("2HB")] == ["HB3", "HB2"]
        assert leu.atoms_named("XX") == []

    def test_terminal_variants(self):
        variants = CCDLibrary()["ALA"]
        nterm = [v for v in variants if "H3" in v.atom_names]
        assert nterm and nterm[0].name_to_atom["N"].formal_charge == 1
        cterm = [
            v for v in variants if "OXT" in v.atom_names and "HXT" not in v.atom_names
        ]
        assert cterm and cterm[0].name_to_atom["OXT"].formal_charge == -1

    @pytest.mark.parametrize(
        "resname, proton, heavy, charge",
        [
            ("LYS", "HZ3", "NZ", 1),
            ("ASP", "HD2", "OD2", 0),
            ("GLU", "HE2", "OE2", 0),
            ("CYS", "HG", "SG", 0),
        ],
    )
    def test_side_chain_variants(self, resname, proton, heavy, charge):
        # charge is that of the heavy atom while it carries the proton.
        variants = CCDLibrary()[resname]
        present = {proton in v.atom_names for v in variants}
        assert present == {True, False}
        for variant in variants:
            expected = charge if proton in variant.atom_names else charge - 1
            assert variant.name_to_atom[heavy].formal_charge == expected

    def test_histidine_tautomers(self):
        side_chains = {}
        for variant in CCDLibrary()["HIS"]:
            names = variant.atom_names
            if {"H2", "HXT"} <= names and "H3" not in names:
                side_chains["HD1" in names, "HE2" in names] = variant
        assert side_chains[True, True].formal_charge == 1
        for key, double in (((True, False), "NE2"), ((False, True), "ND1")):
            variant = side_chains[key]
            assert variant.formal_charge == 0
            assert variant.name_to_atom["ND1"].formal_charge == 0
            assert variant.name_to_atom["NE2"].formal_charge == 0
            orders = {frozenset((b.atom1, b.atom2)): b.order for b in variant.bonds}
            assert orders[frozenset(("CE1", double))] == 2
        assert (False, False) not in side_chains

    def test_arginine_keeps_one_guanidinium_proton(self):
        for variant in CCDLibrary()["ARG"]:
            assert {"HH12", "HH22"} & variant.atom_names

    def test_disulfide(self):
        cys = CCDLibrary()["CYS"][0]
        assert cys.crosslink == ("SG", "SG")
        assert cys.leaving_fragment_of("SG") == {"HG"}

    def test_caps_link_as_peptides(self):
        library = CCDLibrary()
        ace = library["ACE"][0]
        assert ace.linking == "peptide"
        assert [a.name for a in ace.atoms if a.leaving] == ["H"]
        nme = library["NME"][0]
        assert nme.linking == "peptide"
        assert nme.prior_fragment == {"HN1"}
        assert nme.name_to_atom["CH3"].name == "C"

    def test_ion(self):
        na = CCDLibrary()["NA"]
        assert len(na) == 1
        assert na[0].atoms[0].element == "Na"
        assert na[0].formal_charge == 1
        assert CCDLibrary()["I"][0].formal_charge == -1

    def test_unknown_residue(self, user_cache):
        library = CCDLibrary()
        assert "XXX" not in library
        with pytest.raises(KeyError, match="XXX"):
            library["XXX"]

    def test_user_paths(self, tmp_path, user_cache):
        text = (ccd.CCD_CACHE_DIR / "ALA.cif").read_text()
        local = tmp_path / "local"
        local.mkdir()
        (local / "ZZZ.cif").write_text(text.replace("ALA", "ZZZ"))
        assert CCDLibrary(paths=[local])["zzz"][0].name == "ZZZ"
        assert "ZZZ" not in CCDLibrary()

    def test_download(self, user_cache, monkeypatch):
        payload = (ccd.CCD_CACHE_DIR / "ALA.cif").read_bytes().replace(b"ALA", b"ZZZ")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return payload

        monkeypatch.setattr(urllib.request, "urlopen", lambda url: Response())
        assert CCDLibrary(download=True)["ZZZ"][0].name == "ZZZ"
        assert (user_cache / "ZZZ.cif").read_bytes() == payload
        assert CCDLibrary()["ZZZ"][0].name == "ZZZ"

    def test_download_error(self, user_cache, monkeypatch):
        def urlopen(url):
            raise OSError("offline")

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        with pytest.raises(KeyError, match="offline"):
            CCDLibrary(download=True)["ZZZ"]

    def test_parse_cache_is_shared(self):
        assert CCDLibrary()["ALA"] is CCDLibrary()["ALA"]

    def test_cif_quoting_and_null_values(self, tmp_path):
        path = tmp_path / "ZZZ.cif"
        path.write_text(
            "data_ZZZ\n_chem_comp.id ZZZ\n_chem_comp.name\n;\nA LONG\nNAME\n;\n"
            + ATOM_LOOP
            + "ZZZ \"C1'\" ? C ?\nZZZ C2 . C .\nZZZ 'O5\"' O5X O -1\n"
            + BOND_LOOP
            + "ZZZ \"C1'\" C2 SING\nZZZ C2 'O5\"' DOUB\n"
        )
        template = ccd.parse_ccd_cif(path)
        assert template.description == "A LONG NAME"
        assert template.linking is None
        assert [a.name for a in template.atoms] == ["C1'", "C2", 'O5"']
        assert [a.formal_charge for a in template.atoms] == [0, 0, -1]
        assert [a.synonyms for a in template.atoms] == [(), (), ("O5X",)]
        assert [(b.atom1, b.atom2, b.order) for b in template.bonds] == [
            ("C1'", "C2", 1),
            ("C2", 'O5"', 2),
        ]

    def test_negative_nitrogen_in_definition_is_kept(self, user_cache):
        (user_cache / "ZZZ.cif").write_text(
            "data_ZZZ\n_chem_comp.id ZZZ\n"
            + ATOM_LOOP
            + "ZZZ NA NA N -1\nZZZ C1 C1 C 0\n"
            + BOND_LOOP
            + "ZZZ NA C1 SING\n"
        )
        variants = CCDLibrary()["ZZZ"]
        assert variants[0].name_to_atom["NA"].formal_charge == -1
