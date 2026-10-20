"""Tests for the CCD residue template library."""

import pytest

from mbuild.biopolymers import CCDLibrary
from mbuild.exceptions import MBuildError
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

    def test_unknown_residue_raises(self, tmp_path, monkeypatch):
        # Tests that an unknown residue code raises a KeyError that names
        # the code and the download option. This is needed because the
        # loader is strict: it must fail loudly on residues it cannot
        # match instead of guessing. The test points the user download
        # cache at an empty tmp directory. A definition downloaded in
        # an earlier session therefore cannot make the code known. The
        # test then looks up a nonsense code with downloads disabled.
        from mbuild.biopolymers import ccd

        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        library = CCDLibrary()
        with pytest.raises(KeyError, match="XXX"):
            library["XXX"]

    def test_downloaded_templates_load_from_user_cache(self, tmp_path, monkeypatch):
        # Tests that a CCD definition already present in the user
        # download cache loads without download=True. This is needed
        # because the library searched only the caller paths and the
        # bundled directory, so a template downloaded in one session
        # was invisible in the next session unless the user passed
        # download=True again. The test points the cache constant at a
        # tmp directory, places a renamed copy of the bundled ALA
        # definition there, and loads it with downloads disabled.
        from mbuild.biopolymers import ccd

        source = ccd.CCD_CACHE_DIR / "ALA.cif"
        (tmp_path / "ZZZ.cif").write_text(source.read_text().replace("ALA", "ZZZ"))
        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        library = CCDLibrary(download=False)
        variants = library["ZZZ"]
        assert variants and variants[0].name == "ZZZ"

    def test_download_writes_the_user_cache(self, tmp_path, monkeypatch):
        # Tests that a residue code found in no search path is fetched
        # from RCSB, parsed into templates, and written into the user
        # cache directory. This is needed because the download is the
        # only way to load a residue that mBuild does not bundle, and
        # no test covered it, so a change to the response handling or
        # to the cache location would pass unnoticed. The test replaces
        # urllib.request.urlopen with a stub that returns the bundled
        # ALA definition renamed to ZZZ, points the cache constant at a
        # tmp directory, and checks the template and the written file.
        import urllib.request

        from mbuild.biopolymers import ccd

        source = ccd.CCD_CACHE_DIR / "ALA.cif"
        payload = source.read_text().replace("ALA", "ZZZ").encode()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return payload

        monkeypatch.setattr(urllib.request, "urlopen", lambda url: Response())
        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        variants = CCDLibrary(download=True)["ZZZ"]
        assert variants[0].name == "ZZZ"
        assert (tmp_path / "ZZZ.cif").read_bytes() == payload

    def test_cif_parser_reads_adjacent_loops(self):
        # Tests that the CIF parser reads a loop_ block that follows
        # another loop_ block with no '#' or blank separator. This is
        # needed because the old parser consumed the second loop_ token
        # inside the first block and dropped the second block, so a
        # downloaded component file in that layout lost its bonds. The
        # test parses a minimal component with adjacent atom and bond
        # loops and checks that both survive.
        from mbuild.biopolymers.ccd import parse_ccd_cif

        text = "\n".join(
            (
                "data_ZZZ",
                "_chem_comp.id ZZZ",
                '_chem_comp.type "L-PEPTIDE LINKING"',
                "loop_",
                "_chem_comp_atom.comp_id",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "ZZZ C1 C 0",
                "ZZZ C2 C 0",
                "loop_",
                "_chem_comp_bond.comp_id",
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "ZZZ C1 C2 SING",
            )
        )
        template = parse_ccd_cif(text)
        assert {atom.name for atom in template.atoms} == {"C1", "C2"}
        assert len(template.bonds) == 1

    def test_cif_parser_reads_unknown_value_tokens(self):
        # Tests that the CIF parser reads the '?' unknown-value token and
        # the '.' inapplicable-value token as a formal charge of zero and as
        # no alternate name. This is needed because CCD files from RCSB
        # carry these tokens. The parser crashed on int('?'). It also kept
        # the token as a synonym, so an atom named '?' matched. The test
        # parses one component that carries both tokens in both columns and
        # checks the charges and the synonyms.
        from mbuild.biopolymers.ccd import parse_ccd_cif

        text = "\n".join(
            (
                "data_ZZZ",
                "_chem_comp.id ZZZ",
                "loop_",
                "_chem_comp_atom.comp_id",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.alt_atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "ZZZ C1 ? C ?",
                "ZZZ C2 . C .",
                "ZZZ C3 CB C 0",
            )
        )
        template = parse_ccd_cif(text)
        assert [atom.formal_charge for atom in template.atoms] == [0, 0, 0]
        assert [atom.synonyms for atom in template.atoms] == [(), (), ("CB",)]

    def test_negative_nitrogen_definition_keeps_its_variants(
        self, tmp_path, monkeypatch
    ):
        # Tests that a component whose definition carries a negatively
        # charged nitrogen keeps at least its base variant. This is
        # needed because the protonation filter dropped every variant
        # with a negative nitrogen. That rule is correct for a
        # histidine ring. It also deletes the negative pyrrole
        # nitrogens of a heme. The whole template then disappeared, and
        # heme proteins failed to load. The test writes a minimal
        # component with a negative nitrogen into the user cache
        # directory, loads it, and checks that the variant keeps the
        # charge.
        from mbuild.biopolymers import ccd

        text = "\n".join(
            (
                "data_ZZZ",
                "_chem_comp.id ZZZ",
                "loop_",
                "_chem_comp_atom.comp_id",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "ZZZ NA N -1",
                "ZZZ C1 C 0",
                "loop_",
                "_chem_comp_bond.comp_id",
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "ZZZ NA C1 SING",
            )
        )
        (tmp_path / "ZZZ.cif").write_text(text)
        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        variants = CCDLibrary()["ZZZ"]
        assert variants
        assert variants[0].name_to_atom["NA"].formal_charge == -1

    def test_empty_variant_list_names_the_residue(self, tmp_path, monkeypatch):
        # Tests that a residue left with no template variant raises an
        # MBuildError that names the residue. This is needed because
        # the loader indexes the first variant. An empty list then
        # raised a bare IndexError, which named neither the residue nor
        # the cause. The guard is defensive: the current rules always
        # keep the base variant. The test points the user cache at a
        # tmp copy of the bundled ALA definition and replaces the
        # variant generator with one that returns nothing.
        from mbuild.biopolymers import ccd

        source = ccd.CCD_CACHE_DIR / "ALA.cif"
        (tmp_path / "ZZZ.cif").write_text(source.read_text().replace("ALA", "ZZZ"))
        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        monkeypatch.setattr(ccd, "_protonation_variants", lambda template: [])
        with pytest.raises(MBuildError, match="ZZZ"):
            CCDLibrary()["ZZZ"]

    def test_default_libraries_share_parsed_templates(self):
        # Tests that two CCDLibrary instances share the parsed variant
        # list of one cif file. This is needed because every Protein()
        # builds its own default library, and re-parsing the bundled
        # files dominated the load time of every Protein after the
        # first. The test compares object identity across two default
        # instances, which only the class-level parse cache can give.
        assert CCDLibrary()["ALA"] is CCDLibrary()["ALA"]
