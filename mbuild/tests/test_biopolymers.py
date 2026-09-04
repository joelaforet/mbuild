import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import mbuild as mb
from mbuild.biopolymers import CCDLibrary, Protein
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import (
    get_fn,
    has_hoomd,
    has_openff_pablo,
    has_openmm,
    has_packmol,
    has_rdkit,
)

#: Atoms of a glycine that is missing its OXT and HXT atoms, so it
#: expects a peptide bond to the residue that follows it. The
#: coordinates come from residue 1 of the openff-pablo polyglycines
#: asset, in Angstrom, with the two extra amine hydrogens added.
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

#: Atoms of a glycine that is missing its H2 atom, so it expects a
#: peptide bond to the residue before it. The coordinates come from
#: residue 2 of the same asset, with the OXT atom added.
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


#: The RCSB chemical component definition of selenocysteine, from
#: https://files.rcsb.org/ligands/download/SEC.cif, downloaded
#: 2026-09-04, with the descriptor, identifier and audit blocks
#: removed. mBuild bundles no SEC definition, and the bridged-SEC test
#: needs a residue that bridges through selenium.
_SEC_CIF = """\
data_SEC
#
_chem_comp.id SEC
_chem_comp.name SELENOCYSTEINE
_chem_comp.type "L-PEPTIDE LINKING"
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
SEC N   N   N  0 1 N N N Y Y N 38.770 10.663 52.598 -0.783 1.676  -0.339 N   SEC 1
SEC CA  CA  C  0 1 N N R Y N N 38.352 11.626 51.586 -0.938 0.217  -0.405 CA  SEC 2
SEC CB  CB  C  0 1 N N N N N N 38.574 11.050 50.186 0.042  -0.445 0.565  CB  SEC 3
SEC SE  SE  SE 0 0 N N N N N N 38.291 12.371 48.883 1.879  -0.092 -0.020 SE  SEC 4
SEC C   C   C  0 1 N N N Y N Y 36.864 11.874 51.824 -2.349 -0.156 -0.027 C   SEC 5
SEC O   O   O  0 1 N N N Y N Y 36.018 11.106 51.371 -3.030 0.619  0.602  O   SEC 6
SEC OXT OXT O  0 1 N Y N Y N Y 36.557 12.878 52.638 -2.848 -1.348 -0.389 OXT SEC 7
SEC H   HN1 H  0 1 N N N Y Y N 38.621 11.049 53.508 -1.373 2.134  -1.017 H   SEC 8
SEC H2  HN2 H  0 1 N Y N Y Y N 38.235 9.824  52.500 -0.969 2.018  0.592  H2  SEC 9
SEC HA  HA  H  0 1 N N N Y N N 38.908 12.569 51.692 -0.732 -0.125 -1.419 HA  SEC 10
SEC HB2 HB1 H  0 1 N N N N N N 39.606 10.678 50.107 -0.105 -0.037 1.565  HB2 SEC 11
SEC HB3 HB2 H  0 1 N N N N N N 37.871 10.220 50.020 -0.134 -1.521 0.582  HB3 SEC 12
SEC HE  HE  H  0 1 N N N N N N 38.508 11.801 47.556 2.691  -0.839 1.084  HE  SEC 13
SEC HXT HXT H  0 1 N Y N Y N Y 35.620 12.886 52.792 -3.757 -1.542 -0.123 HXT SEC 14
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
SEC N   CA  SING N N 1
SEC N   H   SING N N 2
SEC N   H2  SING N N 3
SEC CA  CB  SING N N 4
SEC CA  C   SING N N 5
SEC CA  HA  SING N N 6
SEC CB  SE  SING N N 7
SEC CB  HB2 SING N N 8
SEC CB  HB3 SING N N 9
SEC SE  HE  SING N N 10
SEC C   O   DOUB N N 11
SEC C   OXT SING N N 12
SEC OXT HXT SING N N 13
#"""

#: Atoms of selenocysteine, with the element symbol and the ideal
#: coordinates of the definition above, in Angstrom. HE, the selenium
#: hydrogen, is absent: a diselenide bridge replaces it.
_SEC_BRIDGED_ATOMS = [
    ("N", "N", -0.783, 1.676, -0.339),
    ("CA", "C", -0.938, 0.217, -0.405),
    ("CB", "C", 0.042, -0.445, 0.565),
    ("SE", "SE", 1.879, -0.092, -0.020),
    ("C", "C", -2.349, -0.156, -0.027),
    ("O", "O", -3.030, 0.619, 0.602),
    ("OXT", "O", -2.848, -1.348, -0.389),
    ("H", "H", -1.373, 2.134, -1.017),
    ("H2", "H", -0.969, 2.018, 0.592),
    ("HA", "H", -0.732, -0.125, -1.419),
    ("HB2", "H", -0.105, -0.037, 1.565),
    ("HB3", "H", -0.134, -1.521, 0.582),
    ("HXT", "H", -3.757, -1.542, -0.123),
]


def _sec_diselenide():
    """Return PDB text for two selenocysteines joined by a diselenide.

    Each residue sits in its own chain and carries every atom of the
    CCD definition except HE. A CONECT record joins the two SE atoms.
    The second residue is the first one turned 180 degrees about the z
    axis and moved along x. The two SE atoms are then 2.33 A apart,
    which is the Se-Se bond length of a diselenide.

    Returns
    -------
    str
        The PDB text.
    """
    lines = []
    serial = 1
    selenium = []
    for chain_id, turned in (("A", False), ("B", True)):
        for name, element, x, y, z in _SEC_BRIDGED_ATOMS:
            if turned:
                x, y = 4.210 - x, -y
            field = f" {name:<3s}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:5d} {field} SEC {chain_id}   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {element:>2s}"
            )
            if name == "SE":
                selenium.append(serial)
            serial += 1
        lines.append(f"TER   {serial:5d}      SEC {chain_id}   1")
        serial += 1
    lines.append(f"CONECT{selenium[0]:5d}{selenium[1]:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


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

    @pytest.mark.skipif(not has_openff_pablo, reason="openff-pablo is not installed")
    def test_pablo_parity(self):
        # Tests that mBuild's templates agree with openff-pablo's residue
        # definitions on names, elements, formal charges, leaving flags,
        # and bond orders. This is needed because the tables and patch
        # semantics in ccd.py are mirrored from pablo (see the module
        # docstring), and silent drift between the two would break the
        # handoff this recipe exists for: mBuild builds the modified
        # coordinates, and a residue-template reader such as
        # openff-pablo reads the written PDB. The test compares the
        # base variant of every bundled amino acid field by field, and
        # runs only where openff-pablo is installed.
        import openff.pablo as pablo

        library = CCDLibrary()
        for resname in (
            "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS "
            "MET PHE PRO SER THR TRP TYR VAL "
            "ACE NME HOH NA CL".split()
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
            # The first-hit names stay a subset of pablo's names, in
            # any of its variants. The fallback matcher accepts a wider
            # set of input spellings than this, for example the
            # digit-first name 3HB, which pablo rejects. A file that
            # only the fallback reads still round-trips, because
            # save_pdb writes the canonical names that pablo reads.
            their_names = {
                name
                for variant in pablo.STD_CCD_CACHE[resname]
                for name in variant.name_to_atom
            }
            unmatched = set(ours.name_to_atom) - their_names
            assert not unmatched, f"{resname}: {unmatched}"

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

    def test_cif_parser_reads_unknown_charge_token(self):
        # Tests that the CIF parser treats the '?' unknown-value token
        # and the '.' inapplicable-value token in the charge column as
        # a formal charge of zero. This is needed because CCD component
        # files downloaded from RCSB can carry these tokens, and
        # int('?') crashed the parser. The test parses a minimal
        # component with both tokens and checks the charges.
        from mbuild.biopolymers.ccd import parse_ccd_cif

        text = "\n".join(
            (
                "data_ZZZ",
                "_chem_comp.id ZZZ",
                "loop_",
                "_chem_comp_atom.comp_id",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "ZZZ C1 C ?",
                "ZZZ C2 C .",
            )
        )
        template = parse_ccd_cif(text)
        assert [atom.formal_charge for atom in template.atoms] == [0, 0]

    def test_cif_parser_reads_unknown_alt_atom_name(self):
        # Tests that the CIF parser treats the '?' unknown-value token
        # and the '.' inapplicable-value token in the alternate atom
        # name column as no alternate name. This is needed because the
        # parser kept the token itself as a synonym. A PDB record whose
        # atom is named '?' then matched the atom, and the synonym
        # named nothing. The test parses a minimal component with both
        # tokens and checks the synonyms.
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
                "ZZZ C1 ? C",
                "ZZZ C2 . C",
                "ZZZ C3 CB C",
            )
        )
        template = parse_ccd_cif(text)
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

    def test_register_caller_built_template(self):
        # Tests that template_from_dict parses a caller-built template
        # and that register() stores it under its residue name in one
        # library only. This is needed because a reload of a modified
        # protein registers the template of a fragment that the CCD
        # does not define, and the parsed variant lists are shared
        # class-wide, so a write into one library would change the
        # templates of every other Protein. The test registers an
        # acetyl template, reads back its leaving fragment, checks a
        # second library, and checks the two-name error.
        from mbuild.biopolymers.ccd import template_from_dict

        template = template_from_dict(
            {
                "name": "OC8",
                "description": "OC8 as built by mBuild",
                "atoms": [
                    {"name": "C1", "element": "C", "formal_charge": 0},
                    {"name": "O1", "element": "O", "formal_charge": 0},
                    {"name": "H1", "element": "H", "leaving": True},
                ],
                "bonds": [
                    {"atom1": "C1", "atom2": "O1", "order": 2},
                    {"atom1": "C1", "atom2": "H1", "order": 1},
                ],
                "linking": None,
                "crosslink": None,
            }
        )
        # The leaving fragment is the property the loader reads: it
        # proves that the leaving flag and the bond both survived.
        assert template.leaving_fragment_of("C1") == {"H1"}
        library = CCDLibrary()
        library.register(template)
        assert library["OC8"] == [template]
        assert "OC8" not in CCDLibrary()
        with pytest.raises(MBuildError, match="variants of one residue"):
            library.register(template, replace(template, name="AC1"))


class TestProtein(BaseTest):
    def test_load_protonated_protein(self, protein_6m03):
        # Tests that a pdbfixer-protonated protein PDB loads by template
        # matching with full chemistry: hierarchy, per-residue formal
        # charges from the matched variants, charged termini, and a bond
        # order on every bond. This is needed because the whole recipe
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

    def test_multichain_no_ter_1p3q(self):
        # Tests that a four-chain PDB without TER records loads into
        # four chains and that no peptide bond crosses a chain
        # boundary. This is needed because without TER records the
        # loader must separate chains from the chain identifier column
        # alone, and a linker keyed only on record adjacency would bond
        # the last residue of one chain to the first residue of the
        # next. The test loads the prepared 1p3q structure, checks the
        # chain ids, checks each chain-boundary residue pair for bonds,
        # and round-trips the chain count through save_pdb.
        protein = Protein(get_fn("1p3q_noter.pdb"))
        chain_ids = [chain.chain_id for chain in protein.chains]
        assert chain_ids == ["A", "B", "C", "D"]
        for earlier, later in zip(chain_ids, chain_ids[1:]):
            last = set(list(protein.residues(chain_id=earlier))[-1].particles())
            first = set(next(iter(protein.residues(chain_id=later))).particles())
            assert not any(
                (a in last and b in first) or (a in first and b in last)
                for a, b in protein.bonds()
            )

        protein.save_pdb("1p3q_roundtrip.pdb")
        reloaded = Protein("1p3q_roundtrip.pdb")
        assert [chain.chain_id for chain in reloaded.chains] == chain_ids

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

    def test_numbering_gap_across_a_ter_loads(self):
        # Tests that a TER between residue 1 and residue 3 of one chain
        # stays advisory and keeps the peptide bond. This is needed
        # because OpenMM writes a numbering gap where a loop is
        # missing, and the first advisory-TER rule demanded strictly
        # consecutive numbers, so such a file could not load at all.
        # The test writes a glycine pair numbered 1 and 3 with a TER
        # between them and checks the C to N bond.
        gap = Path("gly_gap.pdb")
        gap.write_text(_gly_gly_with_ter(resnum=3))
        protein = Protein(str(gap))
        carbon = protein.get_atom(1, "C", chain_id="A")
        nitrogen = protein.get_atom(3, "N", chain_id="A")
        assert protein.bond_graph.has_edge(carbon, nitrogen)

    def test_insertion_code_across_a_ter_loads(self):
        # Tests that a TER between residue 1 and residue 1A of one
        # chain stays advisory and keeps the peptide bond. This is
        # needed because OpenMM writes insertion codes, and the first
        # advisory-TER rule compared residue numbers alone, so it read
        # such a pair as a chain break and the file could not load. The
        # test writes a glycine pair numbered 1 and 1A with a TER
        # between them and checks the C to N bond.
        inserted = Path("gly_icode.pdb")
        inserted.write_text(_gly_gly_with_ter(resnum=1, icode="A"))
        protein = Protein(str(inserted))
        carbon = protein.get_atom(1, "C", chain_id="A")
        nitrogen = protein.get_atom(1, "N", chain_id="A", icode="A")
        assert protein.bond_graph.has_edge(carbon, nitrogen)

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
        # candidate chains when a residue number repeats across chains,
        # even when one residue sits under a wrapper Compound inside
        # its chain. This is needed because the old error path read
        # residue.parent.chain_id, and a wrapped fragment residue's
        # parent is the wrapper, so the error report itself crashed
        # with AttributeError. The test adds a wrapped residue with a
        # duplicate number in a second chain and asserts on the
        # message.
        from mbuild.biopolymers.protein import Chain, Residue

        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        wrapper = mb.Compound(name="wrapper")
        wrapper.add(Residue(resname="LIG", resnum=221, hetatm=True))
        chain = Chain(chain_id="B")
        chain.add(wrapper)
        protein.add(chain)
        with pytest.raises(MBuildError, match=r"chains \['A', 'B'\]"):
            protein.get_residue(221)

    def test_deprotonate_neutralizes_the_site(self, protein_6m03, caplog):
        # Tests that deprotonate() removes the acidic proton of the
        # named atom and rewrites the residue chemistry: the matched
        # template variant, the residue formal charge, the sparse
        # per-atom charge map, and the protein net charge. This is
        # needed because acylation reacts the neutral amine, so a
        # charged site must reach the neutral form before attach()
        # bonds to it, and every consumer of the charge (to_rdkit,
        # net_formal_charge) must agree. The test deprotonates LYS 12
        # NZ of the bundled 6m03 asset, reads the state back, and calls
        # the method a second time to check that a rerun warns and
        # changes nothing.
        protein = protein_6m03
        residue = protein.get_residue(12, chain_id="A")
        nz = protein.get_atom(12, "NZ", chain_id="A")
        net_before = protein.net_formal_charge
        n_hydrogens = sum(1 for p in nz.direct_bonds() if p.element.symbol == "H")

        protein.deprotonate(12, "NZ", chain_id="A")

        hydrogens = sum(1 for p in nz.direct_bonds() if p.element.symbol == "H")
        assert hydrogens == n_hydrogens - 1
        assert residue.formal_charge == 0
        assert "NZ" not in residue.atom_formal_charges
        assert residue.template.description.endswith("-HZ3")
        assert protein.net_formal_charge == net_before - 1

        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.deprotonate(12, "NZ", chain_id="A")
        assert "no acidic proton" in caplog.text
        assert sum(1 for p in nz.direct_bonds() if p.element.symbol == "H") == hydrogens
        assert protein.net_formal_charge == net_before - 1

    def test_deprotonate_warns_when_no_library_variant_matches(
        self, protein_6m03, caplog
    ):
        # Tests that deprotonate() warns when the residue it builds
        # matches no variant of the template library, and stays silent
        # when a variant matches. This is needed because deprotonate()
        # constructs the new template instead of matching it, so some
        # sites give a residue that save_pdb writes and the loader
        # cannot read back. The test deprotonates TRP 31 NE1, which the
        # library does not hold. It then deprotonates CYS 16 SG, which
        # the library holds, and reads the log after each call.
        protein = protein_6m03
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.deprotonate(31, "NE1", chain_id="A")
        assert "holds no TRP variant" in caplog.text
        assert "does not reload" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.deprotonate(16, "SG", chain_id="A")
        assert "holds no CYS variant" not in caplog.text

    def test_deprotonate_warns_on_a_split_charge(self, protein_6m03, caplog):
        # Tests that deprotonate() warns for two charged atoms within
        # two bonds, and stays silent for two charged atoms further
        # apart. This is needed because arginine's two nitrogens are one
        # group split by the template model, while an N-terminal serine
        # is a real zwitterion. The test deprotonates ARG 4 NH1 and
        # SER 1 OG of the bundled 6m03 asset and reads the log and the
        # per-atom charges back.
        protein = protein_6m03
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.deprotonate(4, "NH1", chain_id="A")
        assert "NH1 -1 and NH2 +1, 2 bonds apart" in caplog.text
        assert protein.get_residue(4, chain_id="A").atom_formal_charges == {
            "NH1": -1,
            "NH2": 1,
        }

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.deprotonate(1, "OG", chain_id="A")
        assert "bonds apart" not in caplog.text
        assert protein.get_residue(1, chain_id="A").atom_formal_charges == {
            "N": 1,
            "OG": -1,
        }

    def test_protonate_restores_the_deprotonated_site(self, protein_6m03):
        # Tests that protonate() undoes deprotonate(): the atom count,
        # the residue charge, the matched variant and the net charge all
        # come back, and the new proton sits one bond length from the
        # heavy atom and clear of every other atom. This is needed
        # because protonate() selects the new variant from the library
        # and places the proton itself, so a wrong variant or a wrong
        # position gives a residue that no longer describes a lysine.
        # The test deprotonates LYS 12 NZ of the bundled 6m03 asset,
        # protonates it again, and reads the state and the geometry.
        protein = protein_6m03
        residue = protein.get_residue(12, chain_id="A")
        description = residue.template.description
        n_particles = protein.n_particles
        net_before = protein.net_formal_charge

        protein.deprotonate(12, "NZ", chain_id="A")
        protein.protonate(12, "NZ", chain_id="A")

        assert protein.n_particles == n_particles
        assert residue.formal_charge == 1
        assert residue.template.description == description
        assert protein.net_formal_charge == net_before
        nz = protein.get_atom(12, "NZ", chain_id="A")
        hz3 = protein.get_atom(12, "HZ3", chain_id="A")
        assert protein.bond_graph.has_edge(nz, hz3)
        assert 0.09 < np.linalg.norm(hz3.pos - nz.pos) < 0.11
        assert (
            min(
                np.linalg.norm(particle.pos - hz3.pos)
                for particle in protein.particles()
                if particle is not hz3 and particle is not nz
            )
            > 0.07
        )

    def test_protonate_neutralizes_an_aspartate(self, protein_6m03):
        # Tests that protonate() turns a charged aspartate into the
        # neutral acid: the residue charge goes to zero and a new HD2
        # bonds to OD2 at the O-H bond length. This is needed because a
        # site that the file left anionic is the second use of the
        # method, next to undoing deprotonate(), and the OD2 of a
        # carboxylate has one bonded neighbor, which is the placement
        # rule that tilts the proton off the bond axis. The test
        # protonates ASP 33 OD2 of the bundled 6m03 asset and reads the
        # charges, the bond and the bond length.
        protein = protein_6m03
        protein.protonate(33, "OD2", chain_id="A")
        residue = protein.get_residue(33, chain_id="A")
        od2 = protein.get_atom(33, "OD2", chain_id="A")
        hd2 = protein.get_atom(33, "HD2", chain_id="A")
        assert residue.formal_charge == 0
        assert "OD2" not in residue.atom_formal_charges
        assert protein.bond_graph.has_edge(od2, hd2)
        assert 0.09 < np.linalg.norm(hd2.pos - od2.pos) < 0.10

    def test_protonate_warns_on_an_ineligible_atom(self, protein_6m03, caplog):
        # Tests that protonate() warns and changes nothing for an atom
        # that takes no proton: a carbon, and a lysine nitrogen that
        # already carries three hydrogens. This is needed because the
        # method must run twice without an error, the way deprotonate()
        # does, and a silent no-op would hide a wrong atom name. The
        # test calls the method on the CB and on the charged NZ of
        # LYS 12 of the bundled 6m03 asset, and reads the log, the atom
        # count and the net charge back.
        protein = protein_6m03
        n_particles = protein.n_particles
        net_before = protein.net_formal_charge
        for atom_name in ("CB", "NZ"):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="mbuild"):
                protein.protonate(12, atom_name, chain_id="A")
            assert f"Atom {atom_name} of residue LYS 12" in caplog.text
            assert "no protonation variant" in caplog.text
        assert protein.n_particles == n_particles
        assert protein.net_formal_charge == net_before

    def test_protonated_protein_reloads(self, protein_6m03):
        # Tests that a protein written after protonate() loads again
        # with the same net formal charge. This is needed because
        # protonate() takes its new variant from the template library,
        # and that is what lets the loader match the residue again;
        # deprotonate() builds its variant instead and can write a
        # residue that does not reload. The test protonates ASP 33 OD2
        # of the bundled 6m03 asset, writes the file, loads it, and
        # compares the charges.
        protein = protein_6m03
        protein.protonate(33, "OD2", chain_id="A")
        path = Path("protonated.pdb")
        protein.save_pdb(str(path))
        reloaded = Protein(str(path))
        assert reloaded.net_formal_charge == protein.net_formal_charge
        assert reloaded.get_residue(33, chain_id="A").formal_charge == 0

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach(self, protein_6m03, acetone):
        # Tests that attach() substitutes one hydrogen on each side,
        # bonds the named atoms at the requested separation, adds the
        # fragment as its own HETATM residue, and records the bond with
        # its leaving hydrogens. This is needed because the recorded
        # bond is exactly what Pablo's with_crosslink needs to load the
        # modified protein. The test attaches an acetone-derived
        # fragment at LYS 5 NZ and checks topology, count, and record;
        # it also checks that an atom without hydrogens is rejected.
        import numpy as np

        protein = protein_6m03
        n_before = protein.n_particles
        fragment = acetone

        with pytest.raises(MBuildError, match="0 bonded hydrogens"):
            protein.attach(fragment, "C2", resnum=5, atom_name="NZ", chain_id="A")

        record = protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACT",
            relax=False,  # test the raw port placement deterministically
        )
        assert (record.atom1_name, record.atom2_name) == ("NZ", "C1")
        assert record.leaving1 == ("HZ1",) and record.leaving2 == ("H1",)
        assert protein.n_particles == n_before + fragment.n_particles - 2

        fragment_residue = protein.get_residue(307, chain_id="A")
        assert fragment_residue.name == "ACT" and fragment_residue.hetatm

        nz = protein.get_atom(5, "NZ", chain_id="A")
        carbon = protein.get_atom(307, "C1", chain_id="A")
        assert protein.bond_graph.has_edge(nz, carbon)
        assert np.isclose(np.linalg.norm(carbon.pos - nz.pos), 0.15, atol=1e-3)

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_after_deprotonate_records_one_leaving_atom(
        self, protein_6m03, acetone
    ):
        # Tests that a bond record names only the hydrogen that the bond
        # displaces, and not the proton that an earlier deprotonate()
        # call removed at the same atom. This is needed because a
        # downstream residue library permits one leaving atom per
        # linking atom, and the protonation state travels with the
        # residue in its template. The test deprotonates LYS 5 NZ,
        # attaches a fragment at that atom, and reads the leaving atoms
        # of the record and the template description back.
        protein = protein_6m03
        protein.deprotonate(5, "NZ", chain_id="A")
        record = protein.attach(
            acetone,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACT",
            relax=False,
        )
        assert record.leaving1 == ("HZ1",)
        assert protein.bond_records()[-1]["leaving_atoms"] == (["HZ1"], ["H1"])
        assert protein.get_residue(5, chain_id="A").template.description.endswith(
            "-HZ3"
        )

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_chained(self, protein_6m03, acetone):
        # Tests that a residue added by attach() can itself be a later
        # attachment site. This is needed because multi-residue and
        # branched structures (polymer chains, Y-shaped glycans) are
        # built by repeated attach() calls, and every link must be
        # recorded without any per-residue limit. The test attaches a
        # fragment to the protein and a second fragment to the first,
        # then checks both recorded bonds and the connectivity.
        protein = protein_6m03
        fragment = acetone

        protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="AC1",
            relax=False,
        )
        record = protein.attach(
            fragment,
            "C1",
            resnum=307,
            atom_name="C3",
            chain_id="A",
            fragment_resname="AC2",
            relax=False,
        )
        assert record.residue1.resnum == 307
        assert record.residue2.resnum == 308
        assert len(protein.cross_bonds) == 2
        first = protein.get_atom(307, "C3", chain_id="A")
        second = protein.get_atom(308, "C1", chain_id="A")
        assert protein.bond_graph.has_edge(first, second)

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_bond_order_removes_matching_hydrogens(self, protein_6m03):
        # Tests that a higher-order attach removes one hydrogen per bond
        # order unit on each side, and rejects invalid orders. This is
        # needed because removing a single hydrogen while writing a
        # double bond produces impossible valences (a pentavalent
        # nitrogen), which review reproduced. The test forms an
        # imine-like double bond at LYS 5 NZ and checks the hydrogen
        # count, recorded leaving atoms, and the order guard.
        protein = protein_6m03
        fragment = mb.load("CC=O", smiles=True)
        record = protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="IMN",
            bond_order=2,
            relax=False,
        )
        assert record.leaving1 == ("HZ1", "HZ2")
        assert len(record.leaving2) == 2
        nz = protein.get_atom(5, "NZ", chain_id="A")
        assert sorted(p.name for p in nz.direct_bonds()) == ["C1", "CE", "HZ3"]

        with pytest.raises(MBuildError, match="bond_order must be"):
            protein.attach(
                fragment,
                "C1",
                resnum=12,
                atom_name="NZ",
                chain_id="A",
                bond_order=0,
            )

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_warns_when_the_anchor_keeps_its_charge(
        self, protein_6m03, acetone, caplog
    ):
        # Tests that attach() warns when the anchor atom holds a formal
        # charge that the new bond does not change, and that it still
        # forms the bond. This is needed because one hydrogen leaves the
        # anchor and the new bond takes its place, so a lysine at +1
        # gives a protonated amide, which is not a real species; the
        # recipe stays permissive and names deprotonate() as the remedy
        # instead of blocking the call. The test attaches a fragment to
        # a charged LYS 5 NZ without deprotonating it, reads the log,
        # and checks the new bond.
        protein = protein_6m03
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.attach(
                acetone,
                "C1",
                resnum=5,
                atom_name="NZ",
                chain_id="A",
                fragment_resname="ACT",
                relax=False,
            )
        assert 'deprotonate(5, "NZ", chain_id="A")' in caplog.text
        nz = protein.get_atom(5, "NZ", chain_id="A")
        carbon = protein.get_atom(307, "C1", chain_id="A")
        assert protein.bond_graph.has_edge(nz, carbon)

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_names_no_remedy_for_a_negative_anchor(
        self, protein_6m03, acetone, caplog
    ):
        # Tests that attach() states the charge but names no remedy when
        # the anchor atom holds a negative charge. This is needed
        # because deprotonate() removes a proton, so it makes a negative
        # anchor more negative; the remedy fits a positive anchor only.
        # The test deprotonates ARG 4 NH1 to reach a negative anchor,
        # attaches a fragment to that atom, and reads the log.
        protein = protein_6m03
        protein.deprotonate(4, "NH1", chain_id="A")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.attach(
                acetone,
                "C1",
                resnum=4,
                atom_name="NH1",
                chain_id="A",
                fragment_resname="ACT",
                relax=False,
            )
        state = "NH1 has formal charge -1 before this bond and -1 after it"
        assert state in caplog.text
        assert "deprotonate(4" not in caplog.text

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_warns_on_clashes(self, protein_6m03, caplog):
        # Tests that attaching a bulky fragment into a crowded site logs
        # a clash warning. This is needed because port alignment is
        # rigid and a fragment placed inside the protein would otherwise
        # fail silently until the MD run fails. The test attaches
        # triphenylmethane at a buried lysine and checks the log.
        protein = protein_6m03
        bulky = mb.load("C(c1ccccc1)(c1ccccc1)c1ccccc1", smiles=True)
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.attach(
                bulky,
                "C1",
                resnum=5,
                atom_name="NZ",
                chain_id="A",
                fragment_resname="TPM",
                relax=False,
            )
        assert "Relax the structure" in caplog.text

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_without_simulation_support(self, protein_6m03, caplog, monkeypatch):
        # Tests that attach() returns a complete recorded bond and only
        # warns when mbuild.simulation cannot import, and that a direct
        # relax_fragments() call raises a clear error instead. This is
        # needed because mbuild.simulation imports hoomd, which base
        # installs do not have, and the old code crashed after the
        # fragment was already bonded but before the bond was recorded.
        # The test blocks the module in sys.modules, attaches a bulky
        # fragment that triggers the automatic relax path, and checks
        # the record, the warning, and the error.
        import sys

        protein = protein_6m03
        bulky = mb.load("C(c1ccccc1)(c1ccccc1)c1ccccc1", smiles=True)
        monkeypatch.setitem(sys.modules, "mbuild.simulation", None)
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            record = protein.attach(
                bulky,
                "C1",
                resnum=5,
                atom_name="NZ",
                chain_id="A",
                fragment_resname="TPM",
            )
        assert "Cannot relax" in caplog.text
        assert protein.cross_bonds == [record]
        assert (record.atom1_name, record.atom2_name) == ("NZ", "C1")
        with pytest.raises(MBuildError, match="not importable"):
            protein.relax_fragments()

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    @pytest.mark.skipif(
        not (has_hoomd and has_openmm),
        reason="relax_fragments needs mbuild.simulation (hoomd) and openmm",
    )
    def test_relax_fragments(self, protein_6m03):
        # Tests that relax_fragments() pulls a clashing attached
        # fragment out of steric overlap while the protein stays fixed.
        # This is needed because attach() places fragments rigidly, and
        # relax=False leaves any overlap in place for a later explicit
        # relax call; this is the only test of that call. The test
        # attaches a bulky fragment with relax=False, counts fragment
        # atoms within the 0.1 nm clash cutoff of protein atoms before
        # and after relax_fragments(), and asserts the count decreased
        # while the protein coordinates did not change.
        import numpy as np
        from scipy.spatial import cKDTree

        protein = protein_6m03
        bulky = mb.load("C(c1ccccc1)(c1ccccc1)c1ccccc1", smiles=True)
        protein.attach(
            bulky,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="TPM",
            relax=False,
        )
        fragment_atoms = set(protein.get_residue(307, chain_id="A").particles())
        others = [p for p in protein.particles() if p not in fragment_atoms]

        def clash_count():
            tree = cKDTree([p.pos for p in others])
            distances, _ = tree.query([p.pos for p in fragment_atoms])
            return int((distances < 0.1).sum())

        protein_positions = np.array([p.pos for p in others])
        before = clash_count()
        assert before > 0
        protein.relax_fragments(n_steps=50)
        assert clash_count() < before
        assert np.allclose([p.pos for p in others], protein_positions)

    def test_add_port_at(self, protein_6m03):
        # Tests add_port_at, the low-level alternative to attach(): it
        # removes bond_order hydrogens from the named atom and returns
        # a standard mBuild Port there, so a user can place a compound
        # with force_overlap directly when attach() does not cover the
        # placement. The test creates a port at LYS 12 NZ and checks
        # the anchor and the number of remaining hydrogens.
        from mbuild.port import Port

        protein = protein_6m03
        port = protein.add_port_at(12, "NZ", chain_id="A")
        assert isinstance(port, Port)
        assert port.anchor.name == "NZ"
        nz = protein.get_atom(12, "NZ", chain_id="A")
        hydrogens = [p for p in nz.direct_bonds() if p.element.symbol == "H"]
        assert len(hydrogens) == 2

        with pytest.raises(MBuildError, match="bond_order must be"):
            protein.add_port_at(5, "NZ", chain_id="A", bond_order=0)

        # Tests record_bond, which names the bond that force_overlap
        # formed. This is needed because that path writes no record, so
        # bond_records() reported nothing and a downstream loader could
        # not learn about the modification. The test bonds a methyl
        # fragment onto the port and reads back the record; its leaving
        # atom must be the hydrogen that add_port_at removed above.
        from mbuild.biopolymers.fragments import prepare_fragment
        from mbuild.coordinate_transform import force_overlap
        from mbuild.lib.moieties import CH3

        fragment = prepare_fragment(CH3(), "MET")
        fragment.resnum = 400
        next(iter(protein.chains)).add(fragment)
        force_overlap(
            move_this=fragment,
            from_positions=fragment.all_ports()[0],
            to_positions=port,
            add_bond=True,
        )
        protein.record_bond(nz, protein.get_atom(400, "C1", chain_id="A"))
        assert protein.bond_records()[-1] == {
            "residue_names": ("LYS", "MET"),
            "residue_numbers": (12, 400),
            "chain_ids": ("A", "A"),
            "icodes": ("", ""),
            "atom_names": ("NZ", "C1"),
            "leaving_atoms": (["HZ1"], []),
            "bond_order": 1,
        }

        # Tests that two ports at one atom accumulate in the
        # leaving-atom ledger. This is needed because a residue can
        # carry more than one modification at the same atom, and a
        # record that names only the last hydrogen tells a downstream
        # loader that the other hydrogen is still there. The test opens
        # a second port at the same NZ, bonds a second fragment to it,
        # and reads the leaving atoms of the second record.
        second_port = protein.add_port_at(12, "NZ", chain_id="A")
        second = prepare_fragment(CH3(), "ME2")
        second.resnum = 401
        next(iter(protein.chains)).add(second)
        force_overlap(
            move_this=second,
            from_positions=second.all_ports()[0],
            to_positions=second_port,
            add_bond=True,
        )
        protein.record_bond(nz, protein.get_atom(401, "C1", chain_id="A"))
        assert protein.bond_records()[-1]["leaving_atoms"] == (["HZ1", "HZ2"], [])

    def test_record_bond_rejects_bonds_it_cannot_describe(self, protein_6m03):
        # Tests the three conditions record_bond refuses: an atom that
        # is not in a residue of this protein, two atoms that are not
        # bonded, and two atoms of one residue. This is needed because
        # cross_bonds drives save_pdb and bond_records, so a record
        # that does not describe a real bond between two residues sends
        # a downstream tool a modification that is not there. The test
        # calls record_bond once per condition and reads the errors.
        protein = protein_6m03
        nz = protein.get_atom(5, "NZ", chain_id="A")

        outside = mb.Compound(name="CX", element="C")
        with pytest.raises(MBuildError, match="not in a residue"):
            protein.record_bond(nz, outside)

        with pytest.raises(MBuildError, match="are not bonded"):
            protein.record_bond(nz, protein.get_atom(12, "NZ", chain_id="A"))

        with pytest.raises(MBuildError, match="both in residue"):
            protein.record_bond(nz, protein.get_atom(5, "CE", chain_id="A"))

    def test_port_cleanup_leaves_consistent_state(self):
        # Tests that the port creation path removes the auto-generated
        # ports and keeps the atom's remaining bonds. This is needed
        # because the cleanup no longer goes through Compound.remove,
        # which rescanned every particle of the protein per call; the
        # direct removal must leave the same state. The test creates a
        # port at a CB atom and checks that the returned Port is the
        # only port and that the bond graph keeps the other neighbors.
        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        port = protein.add_port_at(221, "CB")
        assert list(protein.all_ports()) == [port]
        cb = protein.get_atom(221, "CB")
        assert sorted(p.name for p in cb.direct_bonds()) == ["CA", "HB3", "SG"]

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_clone_preserves_protein_state(self, protein_6m03, acetone):
        # Tests that clone() returns a working Protein whose library and
        # cross-bond records survive, with the records pointing at the
        # cloned residues. This is needed because packing and solvation
        # workflows clone their inputs, and a clone that loses these
        # attributes crashes later calls. The test clones a modified
        # protein and checks identity and remapping.
        protein = protein_6m03
        fragment = acetone
        protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACT",
            relax=False,
        )
        copy = mb.clone(protein)
        assert copy.library is protein.library
        assert len(copy.cross_bonds) == 1
        assert copy.cross_bonds[0].residue1 is not protein.cross_bonds[0].residue1
        assert copy.cross_bonds[0].residue1.resnum == 5
        assert copy.net_formal_charge == protein.net_formal_charge

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

    def test_disulfides_8ciq(self):
        # Tests that a protein with three disulfides loads all three
        # cross bonds and keeps them through a save_pdb round trip. This
        # is needed because the written PDB is the handoff artifact for
        # residue-template readers, and a lost CONECT record would make
        # the reloaded protein mis-protonate the bridged cysteines. The
        # test loads 8ciq, writes it back, reloads it, and compares the
        # cross-bond counts.
        protein = Protein(get_fn("8ciq.pdb"))
        assert len(protein.cross_bonds) == 3
        out = Path("8ciq_roundtrip.pdb")
        protein.save_pdb(str(out))
        reloaded = Protein(str(out))
        assert len(reloaded.cross_bonds) == 3

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
        # bundled 6m03 asset and reads the error text.
        text = open(get_fn("6m03_protonated.pdb")).read()
        bad = Path("met_bridge.pdb")
        bad.write_text(text.replace("END", "CONECT  235  249\nEND"))
        with pytest.raises(MBuildError) as error:
            Protein(str(bad))
        message = str(error.value)
        assert "CYS A:16 SG" in message and "MET A:17 SD" in message
        assert "forms disulfide bridges between CYS residues only" in message

    def test_bridged_sec_names_the_cys_limit(self, tmp_path, monkeypatch):
        # Tests that a diselenide between two selenocysteines raises the
        # CYS-only message and not the missing-atom message. This is
        # needed because a bridged SEC lacks the HE that every SEC
        # template variant holds, so the residue matches no variant and
        # the match error tells the user to protonate the file. A new HE
        # breaks the bridge, so that remedy is wrong here. The test puts
        # the RCSB SEC definition in the user download cache, loads two
        # bridged SEC residues, and reads the CYS-only text of the
        # error.
        from mbuild.biopolymers import ccd

        (tmp_path / "SEC.cif").write_text(_SEC_CIF)
        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        bridged = Path("sec_bridge.pdb")
        bridged.write_text(_sec_diselenide())
        with pytest.raises(MBuildError) as error:
            Protein(str(bridged))
        message = str(error.value)
        assert "SEC A:1 SE" in message and "SEC B:1 SE" in message
        assert "forms disulfide bridges between CYS residues only" in message

    @pytest.mark.skipif(not has_openff_pablo, reason="openff-pablo is not installed")
    def test_cross_chain_disulfide_2zuq(self):
        # Tests that a disulfide between two different chains loads,
        # which proves the crosslink filter works on global serials and
        # not on residue adjacency. This is needed because inter-chain
        # disulfides are common in multimeric proteins, and a filter
        # keyed on chain-local state would miss them. The test loads the
        # prepared 2zuq structure from the installed openff-pablo test
        # data and checks for a cross bond whose residues sit in
        # different chains.
        from importlib import resources

        from mbuild.biopolymers.protein import Chain

        data = (
            resources.files("openff.pablo._tests")
            / "data"
            / "prepared_pdbs"
            / "2zuq_prepared.pdb"
        )
        if not data.is_file():
            pytest.skip("openff-pablo test data is not installed")

        def chain_of(residue):
            return next(
                ancestor
                for ancestor in residue.ancestors()
                if isinstance(ancestor, Chain)
            ).chain_id

        with resources.as_file(data) as path:
            protein = Protein(str(path))
        cross_chain = {
            frozenset((chain_of(record.residue1), chain_of(record.residue2)))
            for record in protein.cross_bonds
            if chain_of(record.residue1) != chain_of(record.residue2)
        }
        assert frozenset(("A", "C")) in cross_chain


class TestProteinExports(BaseTest):
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

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_save_pdb_orders_residues_by_number(self, protein_6m03, acetone):
        # Tests that save_pdb writes residues sorted by residue number
        # inside each chain, even when attachments happened in a
        # different order. This is needed because Pablo forms polymer
        # links only between record-adjacent residues, so a fragment
        # chain built middle-first (like an NHS trimer reactive at the
        # middle monomer) must still export in backbone order. The test
        # attaches two fragments, swaps their numbers, and checks the
        # file order.
        protein = protein_6m03
        fragment = acetone
        first = protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="AC1",
            relax=False,
        ).residue2
        second = protein.attach(
            fragment,
            "C1",
            resnum=307,
            atom_name="C3",
            chain_id="A",
            fragment_resname="AC2",
            relax=False,
        ).residue2
        first.resnum, second.resnum = 308, 307

        out = Path("ordered.pdb")
        protein.save_pdb(str(out))
        hetero_resnames = []
        for line in out.read_text().splitlines():
            if line.startswith("HETATM"):
                name = line[17:20]
                if name not in hetero_resnames:
                    hetero_resnames.append(name)
        assert hetero_resnames == ["AC2", "AC1"]

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
    def test_modified_protein_needs_the_bond_records_file(
        self, protein_6m03, acetone, tmp_path, monkeypatch
    ):
        # Tests that a written modified protein fails to reload with the
        # library error for an unknown residue when the bond-records
        # file is not passed. This is needed because the fragment has no
        # CCD entry, so the loader has no other source for its
        # chemistry, and the error must name the residue instead of
        # guessing one. The test attaches a fragment, writes the file,
        # loads the PDB file alone, and asserts on the library message.
        # The user download cache points at an empty tmp directory, so
        # an ACT definition downloaded in an earlier session cannot make
        # the fragment residue known.
        from mbuild.biopolymers import ccd

        monkeypatch.setattr(ccd, "USER_CCD_CACHE_DIR", tmp_path)
        protein = protein_6m03
        protein.attach(
            acetone,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACT",
            relax=False,
        )
        path = Path("modified_reload.pdb")
        protein.save_pdb(str(path))
        with pytest.raises(MBuildError, match="is not in the CCD template library"):
            Protein(str(path))

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_modified_protein_export(self, protein_6m03, acetone):
        # Tests that an attached fragment exports as HETATM records with
        # a CONECT for the new bond, and that bond_records() describes
        # the modification completely and neutrally. This is needed
        # because strict template loaders require the cross-residue
        # CONECT, fail on unexplained CONECTs (so peptide bonds must
        # not get them), and downstream tools format the records into
        # their own vocabulary. A complete record also addresses each
        # residue by chain and insertion code, because a residue number
        # alone repeats across chains. The test attaches a fragment at
        # LYS 5 NZ, writes the file, and checks records.
        protein = protein_6m03
        fragment = acetone
        protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="XCT",
            relax=False,
        )
        out = Path("modified.pdb")
        protein.save_pdb(str(out))

        lines = out.read_text().splitlines()
        conects = [line for line in lines if line.startswith("CONECT")]
        hetatm_serials = {
            int(line[6:11]) for line in lines if line.startswith("HETATM")
        }
        assert len(hetatm_serials) == 9
        # RCSB convention: every HETATM atom lists its bonds in CONECT,
        # so viewers that trust CONECT for HETATM records draw the
        # fragment. The crosslink pair must appear from both sides.
        conect_owners = {int(line[6:11]) for line in conects}
        assert hetatm_serials <= conect_owners
        serial_of = {
            line[12:16].strip(): int(line[6:11])
            for line in lines
            if line.startswith(("ATOM", "HETATM"))
            and line[17:20] in ("LYS", "XCT")
            and int(line[22:26]) in (5, 307)
        }
        nz, c1 = serial_of["NZ"], serial_of["C1"]
        pairs = {
            (int(line[6:11]), partner)
            for line in conects
            for partner in map(int, line[11:].split())
        }
        assert (nz, c1) in pairs and (c1, nz) in pairs
        # Protein backbone bonds stay implied by adjacency: no CONECT
        # between two ATOM-record backbone atoms.
        atom_serials = {
            int(line[6:11])
            for line in lines
            if line.startswith("ATOM") and line[12:16].strip() in ("C", "N")
        }
        assert not any(
            owner in atom_serials and partner in atom_serials
            for owner, partner in pairs
        )

        assert protein.bond_records() == [
            {
                "residue_names": ("LYS", "XCT"),
                "residue_numbers": (5, 307),
                "chain_ids": ("A", "A"),
                "icodes": ("", ""),
                "atom_names": ("NZ", "C1"),
                "leaving_atoms": (["HZ1"], ["H1"]),
                "bond_order": 1,
            }
        ]

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_save_pdb_writes_the_bond_records_sidecar(self, protein_6m03, caplog):
        # Tests that save_pdb writes the bond-records file next to the
        # PDB file, that the file holds the records and a template for
        # the fragment residue, that the info log names the file and
        # the call that reads it back, and that a false
        # write_bond_records writes no such file. This is needed
        # because the file carries the only chemistry a loader can get
        # for a residue that the CCD does not define, and a user who
        # never sees the file cannot pass it back. The test acylates
        # LYS 5 NZ, saves twice, and reads the document, the log, and
        # the directory.
        from mbuild.biopolymers.fragments import prepare_fragment

        protein = protein_6m03
        protein.deprotonate(5, "NZ", chain_id="A")
        protein.attach(
            prepare_fragment("*C(=O)CCCCCCC", "OC8"),
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            relax=False,
        )
        with caplog.at_level(logging.INFO, logger="mbuild"):
            protein.save_pdb("sidecar.pdb")
        assert (
            'Protein("sidecar.pdb", bond_records="sidecar.bondrecords.json")'
            in caplog.text
        )
        document = json.loads(Path("sidecar.bondrecords.json").read_text())
        assert document["format"] == "mbuild.biopolymers.bondrecords"
        assert document["version"] == 1
        assert document["bond_records"] == [
            {
                "residue_names": ["LYS", "OC8"],
                "residue_numbers": [5, 307],
                "chain_ids": ["A", "A"],
                "icodes": ["", ""],
                "atom_names": ["NZ", "C1"],
                "leaving_atoms": [["HZ1"], ["H1"]],
                "bond_order": 1,
            }
        ]
        template = document["templates"]["OC8"]
        # H1 left the fragment when the bond formed, so the residue no
        # longer holds it. The template must carry it back, with its
        # bond to the link atom, because the loader reads the leaving
        # fragment of an atom from the template bonds.
        assert {
            "name": "H1",
            "element": "H",
            "formal_charge": 0,
            "leaving": True,
        } in template["atoms"]
        assert {"atom1": "C1", "atom2": "H1", "order": 1} in template["bonds"]
        assert {"C1", "O1"} <= {atom["name"] for atom in template["atoms"]}

        protein.save_pdb("plain.pdb", write_bond_records=False)
        assert not Path("plain.bondrecords.json").exists()

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_reload_from_the_bond_records_file(self, protein_6m03):
        # Tests that a protein with an acylated lysine writes and loads
        # again with the same particles, bonds, formal charges and bond
        # records, and that a second save and reload of the reloaded
        # protein gives the same result. This is needed because the
        # file is the only source of chemistry for the fragment residue
        # and for the hydrogen that the new bond displaced on the
        # lysine, so without it neither residue matches a template. The
        # reload registers that template in the library, so the second
        # save must still write it. The test acylates LYS 5 NZ, saves,
        # loads the pair of files, compares the two proteins, then
        # saves and loads the reloaded protein and compares again.
        from mbuild.biopolymers.fragments import prepare_fragment

        protein = protein_6m03
        protein.deprotonate(5, "NZ", chain_id="A")
        protein.attach(
            prepare_fragment("*C(=O)CCCCCCC", "OC8"),
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            relax=False,
        )
        protein.save_pdb("acylated.pdb")
        reloaded = Protein("acylated.pdb", bond_records="acylated.bondrecords.json")
        assert reloaded.n_particles == protein.n_particles
        assert reloaded.n_bonds == protein.n_bonds
        assert reloaded.net_formal_charge == protein.net_formal_charge
        assert reloaded.bond_records() == protein.bond_records()
        # The new bond took the place of a hydrogen of the neutral
        # amine, so the acylated lysine is a neutral amide. The charge
        # comes from the matched template, not from the file, and the
        # match must therefore pick the deprotonated variant.
        assert reloaded.get_residue(5, chain_id="A").formal_charge == 0

        reloaded.save_pdb("acylated_again.pdb")
        twice = Protein(
            "acylated_again.pdb", bond_records="acylated_again.bondrecords.json"
        )
        assert twice.n_particles == protein.n_particles
        assert twice.net_formal_charge == protein.net_formal_charge
        assert twice.bond_records() == protein.bond_records()

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_reload_separates_modified_and_free_lysines(self, protein_6m03):
        # Tests that a protein with two acylated lysines loads again
        # with those two bonded and every other lysine free. This is
        # needed because the bond-records file patches the crosslink
        # onto every LYS template variant, and a patch that forced the
        # bond would leave each free lysine without a match. The test
        # acylates LYS 5 and LYS 12, reloads, compares the two record
        # lists, and reads the charge and the side-chain hydrogens of
        # those two and of the free LYS 61. The lists are compared as
        # lists, so the record order must match as well.
        from mbuild.biopolymers.fragments import prepare_fragment

        protein = protein_6m03
        for resnum in (5, 12):
            protein.deprotonate(resnum, "NZ", chain_id="A")
            protein.attach(
                prepare_fragment("*C(=O)CCCCCCC", "OC8"),
                resnum=resnum,
                atom_name="NZ",
                chain_id="A",
                relax=False,
            )
        protein.save_pdb("two_sites.pdb")
        reloaded = Protein("two_sites.pdb", bond_records="two_sites.bondrecords.json")
        assert reloaded.bond_records() == protein.bond_records()
        for resnum in (5, 12):
            residue = reloaded.get_residue(resnum, chain_id="A")
            assert residue.formal_charge == 0
            assert "HZ1" not in {atom.name for atom in residue.particles()}
        free = reloaded.get_residue(61, chain_id="A")
        assert free.formal_charge == 1
        assert {"HZ1", "HZ2", "HZ3"} <= {atom.name for atom in free.particles()}

    def test_bond_records_file_of_another_kind_is_refused(self):
        # Tests that a bond-records file whose format marker or version
        # is not the one this mBuild writes raises an error that names
        # the file. This is needed because the file patches the residue
        # templates that the loader matches against, so a file from
        # another tool or another version would change the chemistry of
        # the loaded protein with no message. The test writes both kinds
        # of file and loads a protein with each.
        path = Path("foreign.bondrecords.json")
        path.write_text(json.dumps({"format": "other.tool", "version": 1}))
        with pytest.raises(MBuildError, match="foreign.bondrecords.json"):
            Protein(get_fn("6m03_protonated.pdb"), bond_records=str(path))

        path.write_text(
            json.dumps({"format": "mbuild.biopolymers.bondrecords", "version": 99})
        )
        with pytest.raises(MBuildError, match="version 99"):
            Protein(get_fn("6m03_protonated.pdb"), bond_records=str(path))

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_two_fragments_under_one_code_warn(self, protein_6m03, caplog):
        # Tests that save_pdb warns when two hetatm residues share a
        # residue name but hold different atoms. This is needed because
        # the bond-records file holds one template per residue name, so
        # the reload gives both residues the chemistry of the first,
        # and a user who reuses a code would get the wrong molecule
        # with no message. The test acylates LYS 5 and LYS 12 with two
        # fragments of different length under the code OC8, saves, and
        # reads the log.
        from mbuild.biopolymers.fragments import prepare_fragment

        protein = protein_6m03
        for resnum, smiles in ((5, "*C(=O)CCCCCCC"), (12, "*C(=O)C")):
            protein.deprotonate(resnum, "NZ", chain_id="A")
            protein.attach(
                prepare_fragment(smiles, "OC8"),
                resnum=resnum,
                atom_name="NZ",
                chain_id="A",
                relax=False,
            )
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.save_pdb("one_code.pdb")
        assert "Two residues named OC8 hold different atoms or bonds" in caplog.text

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_a_leaving_atom_keeps_its_element_across_saves(self, protein_6m03):
        # Tests that a save of a reloaded protein writes the element
        # that the template gives for a leaving atom. This is needed
        # because a leaving atom is absent from the protein, so the
        # writer has no particle to read the element from, and a bond
        # that displaced a heavier atom used to become a hydrogen on
        # every save. The test acylates LYS 5 NZ, saves, edits the
        # written file so that the acyl side loses a chlorine, which is
        # what an acid chloride loses, reloads, saves again, and reads
        # the element back.
        from mbuild.biopolymers.fragments import prepare_fragment

        protein = protein_6m03
        protein.deprotonate(5, "NZ", chain_id="A")
        protein.attach(
            prepare_fragment("*C(=O)CCCCCCC", "OC8"),
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            relax=False,
        )
        protein.save_pdb("acyl.pdb")
        path = Path("acyl.bondrecords.json")
        document = json.loads(path.read_text())
        document["bond_records"][0]["leaving_atoms"][1] = ["CL1"]
        template = document["templates"]["OC8"]
        template["atoms"] = [
            atom for atom in template["atoms"] if atom["name"] != "H1"
        ] + [{"name": "CL1", "element": "Cl", "formal_charge": 0, "leaving": True}]
        template["bonds"] = [
            bond
            for bond in template["bonds"]
            if "H1" not in (bond["atom1"], bond["atom2"])
        ] + [{"atom1": "C1", "atom2": "CL1", "order": 1}]
        path.write_text(json.dumps(document))

        reloaded = Protein("acyl.pdb", bond_records=str(path))
        reloaded.save_pdb("acyl_again.pdb")
        written = json.loads(Path("acyl_again.bondrecords.json").read_text())
        assert {
            "name": "CL1",
            "element": "Cl",
            "formal_charge": 0,
            "leaving": True,
        } in written["templates"]["OC8"]["atoms"]

    def test_bond_records_file_with_a_missing_key_is_refused(self):
        # Tests that a record without one of the keys the reader
        # indexes raises an error that names the file and the key. This
        # is needed because a plain KeyError names the key alone, and a
        # user cannot tell which file or which record holds the fault.
        # The test writes a real pair of files, drops leaving_atoms
        # from the record, and loads the pair again.
        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        protein.save_pdb("incomplete.pdb")
        path = Path("incomplete.bondrecords.json")
        document = json.loads(path.read_text())
        del document["bond_records"][0]["leaving_atoms"]
        path.write_text(json.dumps(document))
        with pytest.raises(MBuildError, match="Record 0 of incomplete") as error:
            Protein("incomplete.pdb", bond_records=str(path))
        assert "'leaving_atoms'" in str(error.value)

    def test_the_writer_flag_is_refused_as_a_sidecar_path(self):
        # Tests that a bool given to the bond_records argument of
        # Protein raises a TypeError that names both arguments. This is
        # needed because the reader argument takes a path and the
        # writer flag of save_pdb takes a bool, and a caller who
        # confuses the two would otherwise reach open() with a file
        # descriptor. The test builds a Protein with bond_records=True.
        with pytest.raises(TypeError, match="write_bond_records"):
            Protein(get_fn("3cu9_vicinal_disulfide.pdb"), bond_records=True)

    def test_bond_record_of_another_protein_is_refused(self):
        # Tests that a record whose residue the PDB file does not hold
        # raises an error that names the bond-records file and the
        # record. This is needed because a record patches the templates
        # that the loader matches against, so a file of another protein
        # or of an earlier save would change the chemistry of the load
        # with no message. The test writes a real pair of files, edits
        # the residue number of the record, and loads the pair again.
        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        protein.save_pdb("edited.pdb")
        path = Path("edited.bondrecords.json")
        document = json.loads(path.read_text())
        record = document["bond_records"][0]
        record["residue_numbers"] = [999, record["residue_numbers"][1]]
        path.write_text(json.dumps(document))
        with pytest.raises(MBuildError, match="holds no residue CYS A:999") as error:
            Protein("edited.pdb", bond_records=str(path))
        assert "edited.bondrecords.json" in str(error.value)

    def test_save_pdb_warns_about_a_sidecar_it_did_not_write(self, caplog):
        # Tests that save_pdb warns when it writes no bond-records file
        # and one of an earlier save sits next to the PDB file. This is
        # needed because that file describes the earlier protein, so a
        # reader who passes it back gets a load that does not match the
        # new PDB file. The test saves a disulfide protein to one path
        # twice, the second time without the records, and reads the log.
        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        protein.save_pdb("stale.pdb")
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.save_pdb("stale.pdb", overwrite=True, write_bond_records=False)
        assert "stale.bondrecords.json exists and was not written again" in caplog.text

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

    def test_save_routes_pdb(self, protein_6m03):
        # Tests that the canonical save() verb writes a correct PDB via
        # save_pdb, and that to_parmed keeps the residue partitioning.
        # This is needed because the generic ParmEd path silently wrote
        # one residue named RES with no chains, which loses the protein's
        # identity without any warning. The test saves through save()
        # and checks residue fields, then counts ParmEd residues.
        protein = protein_6m03
        out = Path("routed.pdb")
        protein.save(str(out))
        first = next(
            line for line in out.read_text().splitlines() if line.startswith("ATOM")
        )
        assert first[17:20] == "SER" and first[21] == "A"
        assert len(protein.to_parmed().residues) == 306

        with pytest.raises(MBuildError, match="would be ignored"):
            protein.save(str(out), overwrite=True, residues=["FOO"])

        # conversion.save passes residues=None explicitly, so the mol2
        # route must still carry the residue partitioning.
        import parmed

        mol2 = Path("routed.mol2")
        protein.save(str(mol2), overwrite=True)
        assert len(parmed.load_file(str(mol2), structure=True).residues) == 306

    def test_module_save_refuses_to_overwrite(self):
        # Tests that a second GMSO-routed save to one path raises
        # IOError, and that overwrite=True writes the file. This is
        # needed because the module function checks the path itself,
        # before it builds the topology, so nothing else enforces the
        # rule for the GMSO extensions. The test writes one .gro file
        # three times.
        protein = Protein(get_fn("8ciq.pdb"))
        box = mb.Box([9.0, 9.0, 9.0])
        mb.biopolymers.save(protein, "once.gro", box=box)
        with pytest.raises(IOError, match="not overwriting"):
            mb.biopolymers.save(protein, "once.gro", box=box)
        mb.biopolymers.save(protein, "once.gro", box=box, overwrite=True)

    def test_residue_labels_without_residues(self):
        # Tests that residue_labels returns an empty map for a compound
        # that holds no Residue. This is needed because the function is
        # public and the GMSO export calls it on a packed system, so a
        # solvent-only compound must return a map instead of raising on
        # the empty span of residue numbers. The test labels one water
        # molecule.
        from mbuild.lib.molecules.water import WaterSPC

        assert mb.biopolymers.residue_labels(WaterSPC()) == {}

    def test_to_gmso_rejects_a_topology_it_cannot_align(self, monkeypatch):
        # Tests that the GMSO export raises when the sites of the
        # topology do not line up with the particles of the compound.
        # This is needed because the export rewrites each site's residue
        # by position in the two lists, so a converter that reordered
        # the sites would write the wrong residue onto every atom and
        # say nothing. The test patches Compound.to_gmso to convert a
        # moved copy of the protein, which keeps the site count and the
        # site names but changes every position.
        from mbuild.compound import Compound

        protein = Protein(get_fn("8ciq.pdb"))
        moved = mb.clone(protein)
        moved.translate([1.0, 0.0, 0.0])
        original = Compound.to_gmso
        monkeypatch.setattr(
            Compound, "to_gmso", lambda self, **kwargs: original(moved, **kwargs)
        )
        with pytest.raises(MBuildError, match="Site order"):
            mb.biopolymers.to_gmso(protein)

    def test_typed_only_extensions_name_the_missing_force_field(self):
        # Tests that a .data, .mcf or .top write raises an MBuildError
        # that names the extension when the topology carries no
        # force-field parameters. This is needed because each of the
        # three GMSO writers failed deep inside GMSO instead: the top
        # writer asserted "System not fully typed", the data writer
        # raised an AttributeError that carried a 400-character bond
        # repr, and the mcf writer raised a pydantic ValidationError.
        # The test saves an untyped protein to each of the three
        # extensions.
        protein = Protein(get_fn("8ciq.pdb"))
        box = mb.Box([9.0, 9.0, 9.0])
        for extension in (".data", ".mcf", ".top"):
            with pytest.raises(MBuildError, match=f"A \\{extension} file"):
                mb.biopolymers.save(protein, f"untyped{extension}", box=box)

    def test_module_save_routes_pdb_to_the_protein_writer(self):
        # Tests that the module-level save writes a Protein .pdb file
        # through save_pdb. This is needed because the module function
        # handed .pdb to Compound.save, so the file came from the
        # generic ParmEd writer: it carried a blank chain column and no
        # CONECT records, and a residue-template reader then lost the
        # chain and the disulfides. The test saves the three-disulfide
        # 8ciq structure through the module function and reads the
        # chain column and the CONECT count.
        protein = Protein(get_fn("8ciq.pdb"))
        out = Path("module_routed.pdb")
        mb.biopolymers.save(protein, str(out))
        lines = out.read_text().splitlines()
        atoms = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
        assert {line[21] for line in atoms} == {"A"}
        assert sum(line.startswith("CONECT") for line in lines) == 6

    def test_gmso_routed_save_and_trajectory_keep_residues(self, protein_6m03):
        # Tests that a GMSO-routed save (.gro) and to_trajectory keep
        # the per-residue partitioning. This is needed because
        # conversion.save calls the module-level to_gmso and the generic
        # to_trajectory assigns one default residue, so both silently
        # collapsed the protein into a single residue. The test saves a
        # .gro file and checks the residue columns, then builds an
        # mdtraj topology and counts its residues.
        protein = protein_6m03
        out = Path("identity.gro")
        protein.save(str(out))
        atom_lines = out.read_text().splitlines()[2 : 2 + protein.n_particles]
        first = atom_lines[0]
        assert (first[:5].strip(), first[5:10].strip()) == ("1", "SER")
        assert len({line[:10] for line in atom_lines}) == 306

        topology = protein.to_trajectory().topology
        assert len(list(topology.residues)) == 306

    def test_gmso_save_without_a_box_warns(self, caplog):
        # Tests that a .gro write with no box available logs a warning
        # that names the box argument, and that a write with a box logs
        # nothing. This is needed because mb.solvate and mb.fill_box
        # leave the packed system without a box, so the writer recorded
        # the bounding box of the atoms, which for one packed system
        # measured 4.18 x 5.06 x 4.52 nm instead of the 6 x 6 x 6 nm
        # packing box. The test saves a protein whose file carries no
        # CRYST1 record, once without a box and once with one, and
        # reads the log each time.
        protein = Protein(get_fn("8ciq.pdb"))
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            mb.biopolymers.save(protein, "no_box.gro")
        assert "box=mb.Box(...)" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            mb.biopolymers.save(protein, "boxed.gro", box=mb.Box([6.0, 6.0, 6.0]))
        assert "box=mb.Box(...)" not in caplog.text

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_to_gmso_keeps_residue_identity(self, protein_6m03):
        # Tests that the GMSO export keeps real PDB residue numbers and
        # the residue names of attached fragments. This is needed
        # because the generic converter renumbers residues per name and
        # misses fragment residues nested below a wrapper Compound, and
        # GMSO-side workflows (per-molecule force field application,
        # template mapping) key on this metadata. The test attaches a
        # fragment, exports, and checks numbers, the fragment residue,
        # and the chain label.
        from mbuild.biopolymers import prepare_fragment

        protein = protein_6m03
        fragment = prepare_fragment("*C(=O)C", "ACY")
        protein.attach(fragment, resnum=5, atom_name="NZ", chain_id="A", relax=False)

        topology = protein.to_gmso()
        assert topology.n_sites == protein.n_particles
        residues = {(site.residue.name, site.residue.number) for site in topology.sites}
        assert ("LYS", 5) in residues and ("ACY", 307) in residues
        assert ("SER", 1) in residues  # real numbering, not per-name 0..N
        first = next(iter(topology.sites))
        assert first.molecule.name == "Chain_A"

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_to_rdkit_carries_chemistry(self, protein_6m03):
        # Tests that Protein.to_rdkit returns a sanitized molecule with
        # the correct net formal charge and PDB residue info, including
        # charges of an attached zwitterionic fragment. This is needed
        # because the generic Compound export drops formal charges, so
        # RDKit/OpenFF handoff would silently mis-protonate. The test
        # exports before and after attaching a sulfobetaine and checks
        # net charge and per-atom info.
        from rdkit import Chem

        from mbuild.biopolymers import prepare_fragment

        protein = protein_6m03
        mol = protein.to_rdkit()
        assert Chem.GetFormalCharge(mol) == protein.net_formal_charge == -4
        info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
        assert (info.GetResidueName(), info.GetResidueNumber()) == ("SER", 1)

        fragment = prepare_fragment("C[N+](C)(C)CCS(=O)(=O)[O-]", "SBM")
        assert fragment.formal_charge == 0
        assert len(fragment.atom_formal_charges) == 2
        protein.attach(
            fragment, "C1", resnum=5, atom_name="NZ", chain_id="A", relax=False
        )
        assert Chem.GetFormalCharge(protein.to_rdkit()) == -4

        # Atoms outside any Residue must fail loudly, not vanish.
        protein.add(mb.Compound(name="XX", element="C"))
        with pytest.raises(MBuildError, match="belong to a Residue"):
            protein.to_rdkit()

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_volume_runs_through_to_rdkit(self):
        # Tests that Compound.volume() works on a Protein and returns a
        # positive float. This is needed because volume() calls
        # to_rdkit(embed=True), and the Protein override did not accept
        # the embed keyword, so volume() raised TypeError. The override
        # must accept and ignore embed, because the export already
        # carries the real coordinates. The test computes the volume of
        # a small disulfide peptide. Compound.volume references
        # Chem.AllChem without importing the submodule, so the test
        # imports it first; a fix there belongs to core mBuild, which
        # this recipe does not modify.
        from rdkit.Chem import AllChem  # noqa: F401

        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        volume = protein.volume()
        assert isinstance(volume, float) and volume > 0.0

    def test_to_parmed_rejects_residue_atom_name_collision(self):
        # Tests that to_parmed raises a clear error when a residue name
        # equals an atom name present in the protein. This is needed
        # because the generic converter matches each atom's own name
        # against the residue list before its ancestors, so a residue
        # named like an atom (an ion residue CA next to alpha-carbon
        # atoms CA) silently splits those atoms into spurious residues;
        # the converter itself is core code that this recipe does not
        # modify. The test renames one residue to CA and asserts on the
        # error.
        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        next(protein.residues()).name = "CA"
        with pytest.raises(MBuildError, match=r"\['CA'\].*spurious"):
            protein.to_parmed()

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    @pytest.mark.skipif(not has_openff_pablo, reason="openff-pablo is not installed")
    def test_pablo_pipeline_integration(self, protein_6m03):
        # Tests the full handoff: attach a CCD-named fragment, write the
        # prepared PDB, and load it through openff-pablo with the
        # crosslink spec mBuild recorded. This is needed because the
        # whole recipe exists so that Pablo can parameterize the
        # modified protein; the test proves the artifact and spec are
        # sufficient, with no adapter code in between. It builds an
        # acetaldehyde fragment with CCD ACE atom names, bonds it to
        # LYS 5 NZ, and asserts pablo returns one whole molecule with
        # the expected charge and connectivity. Runs only where a pablo
        # version with with_crosslink (>= 0.2) is installed.
        import openff.pablo as pablo

        if not hasattr(pablo, "STD_CCD_CACHE"):
            pytest.skip("openff-pablo >= 0.2 is required")

        fragment = mb.load("CC=O", smiles=True)
        carbons = list(fragment.particles_by_element("C"))
        oxygen = next(fragment.particles_by_element("O"))
        carbonyl = [c for c in carbons if oxygen in c.direct_bonds()][0]
        methyl = [c for c in carbons if c is not carbonyl][0]
        carbonyl.name, methyl.name, oxygen.name = "C", "CH3", "O"
        aldehyde_h = [h for h in carbonyl.direct_bonds() if h.element.symbol == "H"][0]
        aldehyde_h.name = "H"
        methyl_hydrogens = [p for p in methyl.direct_bonds() if p.element.symbol == "H"]
        for index, hydrogen in enumerate(methyl_hydrogens, 1):
            hydrogen.name = f"H{index}"

        protein = protein_6m03
        protein.attach(
            fragment,
            "CH3",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACE",
            relax=False,
        )
        out = Path("acetylated.pdb")
        protein.save_pdb(str(out))

        # bond_records() reports each attachment as residue names,
        # atom names, removed (leaving) atoms, and bond order. The
        # test reformats one record into the arguments of pablo's
        # with_crosslink: the step a user performs to load the
        # modified PDB with its full chemistry.
        (record,) = protein.bond_records()
        spec = {
            "residues": list(record["residue_names"]),
            "linking_atoms": list(record["atom_names"]),
            "leaving_atoms": [list(side) for side in record["leaving_atoms"]],
            "bond_order": record["bond_order"],
        }
        topology = pablo.topology_from_pdb(
            str(out),
            residue_library=pablo.STD_CCD_CACHE.with_crosslink(**spec),
        )
        assert topology.n_molecules == 1
        molecule = topology.molecule(0)
        assert molecule.n_atoms == protein.n_particles
        assert molecule.total_charge.m == protein.net_formal_charge
        nz = [
            atom
            for atom in molecule.atoms
            if atom.name == "NZ" and atom.metadata.get("residue_number") == 5
        ][0]
        assert "CH3" in {neighbor.name for neighbor in nz.bonded_atoms}


class TestProteinSolvation(BaseTest):
    @pytest.fixture
    def protein_8ciq(self):
        return Protein(get_fn("8ciq.pdb"))

    @pytest.fixture
    def water_sol(self):
        from mbuild.lib.molecules.water import WaterSPC

        water = WaterSPC()
        water.name = "SOL"
        return water

    def test_gro_residue_ids_unique_across_chains(self):
        # Tests that a .gro file written from a four-chain protein
        # carries one residue id for every residue. This is needed
        # because GMSO stores a site residue by value, so residues
        # that repeat a (name, number) pair across chains became one
        # GMSO residue and the file described 151 residues instead of
        # 228. The test loads the four-chain 1p3q structure, saves a
        # .gro file, and counts the distinct residue columns of the
        # atom lines.
        protein = Protein(get_fn("1p3q_noter.pdb"))
        out = Path("1p3q.gro")
        protein.save(str(out))
        atom_lines = out.read_text().splitlines()[2 : 2 + protein.n_particles]
        assert (
            len({line[:10] for line in atom_lines})
            == len(list(protein.residues()))
            == 228
        )

    @pytest.mark.skipif(not has_packmol, reason="PACKMOL is not installed")
    def test_solvate_keeps_chain_residue_hierarchy(self, protein_8ciq, water_sol):
        # Tests that the solute of a solvated system is still a
        # Protein that carries its chain, its residues and its bond
        # records, and that PACKMOL moved it as a rigid body. This is
        # needed because the recipe workflow attaches and relaxes
        # before it packs, so save_pdb, bond_records and the residue
        # accessors must still work on system.children[0]. The test
        # solvates 8ciq in ten waters and compares the solute against
        # the protein it was built from.
        before = protein_8ciq.xyz.copy()
        system = mb.solvate(protein_8ciq, water_sol, 10, mb.Box([6.0, 6.0, 6.0]))
        solute = system.children[0]
        assert isinstance(solute, Protein)
        assert [chain.chain_id for chain in solute.chains] == ["A"]
        labels = [(residue.name, residue.resnum) for residue in solute.residues()]
        assert labels[0] == ("ALA", 1) and labels[-1] == ("VAL", 35)
        assert len(solute.bond_records()) == 3
        # PACKMOL holds the solute with its "fixed" restraint, so the
        # only change is the shift to the centre of the box.
        shift = solute.xyz - before
        assert np.allclose(shift, shift[0], atol=1e-6)

    @pytest.mark.skipif(not has_packmol, reason="PACKMOL is not installed")
    def test_fill_box_fixed_orientation_does_not_rotate(self, protein_8ciq, water_sol):
        # Tests that mb.fill_box holds a protein rigid, and keeps its
        # chain and residue hierarchy, when the caller fixes the
        # orientation of the solute. This is needed because fill_box
        # rotates every compound by default, and a rotation scrambles a
        # protein: with fix_orientation=[False, False] the centred
        # coordinates of 8ciq move by up to 1.5 nm. The test packs 8ciq
        # with ten waters, then compares the centred coordinates of the
        # solute against the centred coordinates of the protein it was
        # built from.
        xyz = protein_8ciq.xyz
        before = xyz - xyz.mean(axis=0)
        system = mb.fill_box(
            [protein_8ciq, water_sol],
            [1, 10],
            box=mb.Box([6.0, 6.0, 6.0]),
            fix_orientation=[True, False],
        )
        solute = system.children[0]
        assert isinstance(solute, Protein)
        after = solute.xyz - solute.xyz.mean(axis=0)
        assert np.allclose(after, before, atol=1e-6)
        assert [chain.chain_id for chain in solute.chains] == ["A"]
        labels = [(residue.name, residue.resnum) for residue in solute.residues()]
        assert labels[0] == ("ALA", 1) and labels[-1] == ("VAL", 35)

    @pytest.mark.skipif(not has_packmol, reason="PACKMOL is not installed")
    def test_solvated_gro_keeps_residue_names(self, protein_8ciq, water_sol):
        # Tests that mb.biopolymers.save writes the protein residue
        # names of a packed system to a .gro file, and that the plain
        # Compound.save writes the chain name instead. This is needed
        # because conversion.save_in_gmso calls the module-level GMSO
        # converter, which no method on Protein can reach once the
        # protein is only a child of the packed system: every protein
        # atom then landed in one residue named "Chain". The test
        # solvates 8ciq in ten waters and reads the residue column of
        # both files.
        system = mb.solvate(protein_8ciq, water_sol, 10, mb.Box([6.0, 6.0, 6.0]))

        def residue_names(path):
            lines = path.read_text().splitlines()
            atom_lines = lines[2 : 2 + int(lines[1])]
            names = []
            for earlier, line in zip([None] + atom_lines, atom_lines):
                if earlier is None or line[:10] != earlier[:10]:
                    names.append(line[5:10].strip())
            return names

        plain = Path("plain.gro")
        system.save(str(plain))
        assert residue_names(plain) == ["Chain"] + ["SOL"] * 10

        routed = Path("routed.gro")
        mb.biopolymers.save(system, str(routed))
        assert (
            residue_names(routed)
            == [residue.name for residue in system.children[0].residues()]
            + ["SOL"] * 10
        )


class TestFragments(BaseTest):
    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_prepare_fragment(self, protein_6m03, acetone):
        # Tests that prepare_fragment returns a named Residue whose atom
        # names are final: the same names appear in the protein after
        # attach(). This is needed because callers must know the names
        # to pick the attachment atom and to build an external (Pablo)
        # residue definition, and attach() would otherwise rename atoms
        # invisibly inside its clone. The test prepares a fragment,
        # attaches it, and compares the name lists.
        from mbuild.biopolymers import prepare_fragment

        fragment = prepare_fragment(acetone, "ACT")
        names = [particle.name for particle in fragment.particles()]
        assert fragment.name == "ACT"
        assert len(set(names)) == len(names)
        assert "C1" in names

        protein = protein_6m03
        protein.attach(
            fragment, "C1", resnum=5, atom_name="NZ", chain_id="A", relax=False
        )
        attached = protein.get_residue(307, chain_id="A")
        assert [p.name for p in attached.particles()] == [
            name
            for name in names
            if name != "H1"  # the leaving hydrogen
        ]

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_prepare_fragment_keeps_multiple_residues(self):
        # Tests that prepare_fragment keeps the Residue children of a
        # multi-residue Compound, with their charges and link atoms.
        # This is needed because the old code wrapped the whole Compound
        # into one new Residue, which dropped the inner residues'
        # atom_formal_charges and link_atoms and renamed atoms across
        # residue boundaries. The test builds a Compound from two
        # prepared residues (one charged, one star-sited), runs
        # prepare_fragment, and checks that both residues survive with
        # their metadata mapped to real atom names.
        from mbuild.biopolymers import (
            Residue,
            fragment_from_smiles,
            prepare_fragment,
        )

        charged = fragment_from_smiles("C[NH3+]", "AMM")
        sited = fragment_from_smiles("*CC", "ETH")
        fragment = mb.Compound(name="LNK")
        fragment.add(charged)
        fragment.add(sited)

        prepared = prepare_fragment(fragment, "LNK")
        residues = {
            child.name: child
            for child in prepared.successors()
            if isinstance(child, Residue)
        }
        assert set(residues) == {"AMM", "ETH"}
        amm = residues["AMM"]
        assert amm.formal_charge == 1
        atom_names = {particle.name for particle in amm.particles()}
        assert set(amm.atom_formal_charges) <= atom_names
        assert sum(amm.atom_formal_charges.values()) == 1
        eth = residues["ETH"]
        eth_names = {particle.name for particle in eth.particles()}
        assert eth.link_atoms and set(eth.link_atoms.values()) <= eth_names

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_star_sited_fragment(self, protein_6m03):
        # Tests that a SMILES attachment point (*) marks the fragment's
        # bond site, so attach() needs no fragment atom name, and that
        # unlabeled multiple stars are rejected. This is needed because
        # star-sited fragments are the standard way chemists write
        # "link here", and reading auto-generated atom names is the
        # main UX friction otherwise. The test attaches an octanoyl
        # fragment written with a star and checks the recorded bond.
        from mbuild.biopolymers import prepare_fragment

        fragment = prepare_fragment("*C(=O)CCCCCCC", "OCT")
        assert fragment.link_atoms == {"1": "C1"}

        protein = protein_6m03
        record = protein.attach(
            fragment, resnum=5, atom_name="NZ", chain_id="A", relax=False
        )
        assert record.atom2_name == "C1"

        with pytest.raises(MBuildError, match="distinct labels"):
            prepare_fragment("*CC*", "BAD")
        two_sites = prepare_fragment("[*:1]CC[*:2]", "TWO")
        with pytest.raises(MBuildError, match="attachment points"):
            protein.attach(two_sites, resnum=90, atom_name="NZ", chain_id="A")

    def test_long_fragment_resname_is_rejected(self, protein_6m03):
        # Tests that a fragment residue name of more than three
        # characters raises a ValueError. The message names the limit
        # and the collision risk. Every entry point that takes such a
        # name is checked, and so is the name of a Residue the caller
        # built. This is needed because the name was cut to three
        # characters with no message, so "OCTL" and "OCTYL" both
        # became "OCT", the assigned CCD code for n-octane. The test
        # passes a name that is too long to each entry point and reads
        # back the message.
        from mbuild.biopolymers import fragment_from_smiles, prepare_fragment
        from mbuild.biopolymers.protein import Residue
        from mbuild.lib.moieties import CH3

        with pytest.raises(ValueError, match="OCTL") as error:
            prepare_fragment(CH3(), "OCTL")
        assert "the limit is 3" in str(error.value)
        assert "CCD component code" in str(error.value)

        with pytest.raises(ValueError, match="OCTL"):
            fragment_from_smiles("*C(=O)CCCCCCC", "OCTL")
        with pytest.raises(ValueError, match="OCTL"):
            protein_6m03.attach(
                CH3(),
                resnum=5,
                atom_name="NZ",
                chain_id="A",
                fragment_resname="OCTL",
                relax=False,
            )

        # A Residue the caller built carries its own name. Without a
        # resname argument that name is used, so it gets the check too.
        # A name that fits is kept, and upper-cased as before.
        built = Residue(resname="OCTYL")
        built.add(CH3())
        with pytest.raises(ValueError, match="OCTYL"):
            prepare_fragment(built, None)
        kept = Residue(resname="oct")
        kept.add(CH3())
        assert prepare_fragment(kept, None).name == "OCT"
