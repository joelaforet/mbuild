from pathlib import Path

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
    has_rdkit,
)


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
            # Every name our matcher accepts, pablo's must accept too
            # (in any of its variants); otherwise a file mBuild loads
            # could fail downstream.
            their_names = {
                name
                for variant in pablo.STD_CCD_CACHE[resname]
                for name in variant.name_to_atom
            }
            unmatched = set(ours.name_to_atom) - their_names
            assert not unmatched, f"{resname}: {unmatched}"

    def test_unknown_residue_raises(self):
        # Tests that an unknown residue code raises a KeyError that names
        # the code and the download option. This is needed because the
        # loader is strict: it must fail loudly on residues it cannot
        # match instead of guessing. The test looks up a nonsense code
        # with downloads disabled.
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

    def test_default_libraries_share_parsed_templates(self):
        # Tests that two CCDLibrary instances share the parsed variant
        # list of one cif file. This is needed because every Protein()
        # builds its own default library, and re-parsing the bundled
        # files dominated the load time of every Protein after the
        # first. The test compares object identity across two default
        # instances, which only the class-level parse cache can give.
        assert CCDLibrary()["ALA"] is CCDLibrary()["ALA"]


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

    def test_load_strict_errors(self):
        # Tests that the loader fails loudly, with the residue named in
        # the message, on a bad atom name and on an unknown residue code.
        # This is needed because the loader must never guess chemistry:
        # a file that does not match the templates has to be fixed by the
        # user, not silently misread. The test corrupts one atom name and
        # one residue name of the good asset and asserts on the errors.
        text = open(get_fn("6m03_protonated.pdb")).read()

        bad_atom = Path("bad_atom.pdb")
        bad_atom.write_text(text.replace(" CB  SER A   1", " QQ  SER A   1", 1))
        with pytest.raises(MBuildError, match="SER A:1"):
            Protein(str(bad_atom))

        bad_residue = Path("bad_residue.pdb")
        bad_residue.write_text(text.replace("SER A   1", "XYZ A   1"))
        with pytest.raises(MBuildError, match="download=True"):
            Protein(str(bad_residue))

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
    def test_attach_warns_on_clashes(self, protein_6m03, caplog):
        # Tests that attaching a bulky fragment into a crowded site logs
        # a clash warning. This is needed because port alignment is
        # rigid and a fragment placed inside the protein would otherwise
        # fail silently until the MD run fails. The test attaches
        # triphenylmethane at a buried lysine and checks the log.
        import logging

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
    @pytest.mark.skipif(
        not (has_hoomd and has_openmm),
        reason="relax_fragments needs mbuild.simulation (hoomd) and openmm",
    )

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
        import logging
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
        # reloads the protein, compares counts and net charge, and
        # checks the fixed-column layout of one atom line.
        protein = protein_6m03
        out = Path("roundtrip.pdb")
        protein.save_pdb(str(out))
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
    def test_modified_protein_export(self, protein_6m03, acetone):
        # Tests that an attached fragment exports as HETATM records with
        # a CONECT for the new bond, and that bond_records() describes
        # the modification completely and neutrally. This is needed
        # because strict template loaders require the cross-residue
        # CONECT, fail on unexplained CONECTs (so peptide bonds must
        # not get them), and downstream tools format the records into
        # their own vocabulary. The test attaches a fragment at LYS 5
        # NZ, writes the file, and checks records.
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
                "atom_names": ("NZ", "C1"),
                "leaving_atoms": (["HZ1"], ["H1"]),
                "bond_order": 1,
            }
        ]

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
    def test_fragment_from_sdf(self):
        # Tests that an SDF fragment loads with explicit bond orders,
        # elements, and coordinates, as a named Residue. This is needed
        # because SDF is the rich fragment format that, unlike PDB,
        # carries bond orders and formal charges, and workflows (e.g.
        # PolyzyMD) hand fragments off as charged SDF files. The test
        # writes an acetone SDF via RDKit and checks the loaded residue.
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from mbuild.biopolymers import fragment_from_sdf

        rdmol = Chem.AddHs(Chem.MolFromSmiles("CC(C)=O"))
        AllChem.EmbedMolecule(rdmol, randomSeed=3)
        path = Path("acetone.sdf")
        Chem.SDWriter(str(path)).write(rdmol)

        residue = fragment_from_sdf(str(path), "ACT")
        assert residue.name == "ACT" and residue.hetatm
        assert residue.n_particles == 10
        orders = {
            bond[2]["bond_order"] for bond in residue.bonds(return_bond_order=True)
        }
        assert orders == {1.0, 2.0}
        names = [p.name for p in residue.particles()]
        assert len(set(names)) == len(names) and "O1" in names

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
