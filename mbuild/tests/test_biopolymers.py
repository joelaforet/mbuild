import pytest

import mbuild as mb
from mbuild.biopolymers import CCDLibrary, Protein
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import get_fn, has_rdkit


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

    def test_load_protonated_protein(self):
        # Tests that a pdbfixer-protonated protein PDB loads by template
        # matching with full chemistry: hierarchy, per-residue formal
        # charges from the matched variants, charged termini, and a bond
        # order on every bond. This is needed because the whole recipe
        # rests on the loader stamping template chemistry instead of
        # guessing from the file. The test loads the bundled protonated
        # SARS-CoV-2 main protease (306 residues, net charge -4 at pH 7)
        # and checks structure and chemistry counts.
        protein = Protein(get_fn("6m03_protonated.pdb"))
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

    def test_load_strict_errors(self, tmp_path):
        # Tests that the loader fails loudly, with the residue named in
        # the message, on a bad atom name and on an unknown residue code.
        # This is needed because the loader must never guess chemistry:
        # a file that does not match the templates has to be fixed by the
        # user, not silently misread. The test corrupts one atom name and
        # one residue name of the good asset and asserts on the errors.
        text = open(get_fn("6m03_protonated.pdb")).read()

        bad_atom = tmp_path / "bad_atom.pdb"
        bad_atom.write_text(text.replace(" CB  SER A   1", " QQ  SER A   1", 1))
        with pytest.raises(MBuildError, match="SER A:1"):
            Protein(str(bad_atom))

        bad_residue = tmp_path / "bad_residue.pdb"
        bad_residue.write_text(text.replace("SER A   1", "XYZ A   1"))
        with pytest.raises(MBuildError, match="download=True"):
            Protein(str(bad_residue))

    def test_get_atom(self):
        # Tests that residues and atoms are addressable by residue number
        # and atom name. This is needed because functionalization
        # workflows pick attachment sites this way (e.g. lysine NZ). The
        # test fetches a known atom and asserts the not-found error names
        # the residue's atoms.
        protein = Protein(get_fn("6m03_protonated.pdb"))
        nz = protein.get_atom(90, "NZ", chain_id="A")
        assert nz.name == "NZ" and nz.element.symbol == "N"
        with pytest.raises(MBuildError, match="no atom"):
            protein.get_atom(90, "XX", chain_id="A")

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach(self):
        # Tests that attach() substitutes one hydrogen on each side,
        # bonds the named atoms at the requested separation, adds the
        # fragment as its own HETATM residue, and records the bond with
        # its leaving hydrogens. This is needed because the recorded
        # bond is exactly what Pablo's with_crosslink needs to load the
        # modified protein. The test attaches an acetone-derived
        # fragment at LYS 5 NZ and checks topology, count, and record;
        # it also checks that an atom without hydrogens is rejected.
        import numpy as np

        protein = Protein(get_fn("6m03_protonated.pdb"))
        n_before = protein.n_particles
        fragment = mb.load("CC(C)=O", smiles=True)

        with pytest.raises(MBuildError, match="no bonded hydrogen"):
            protein.attach(fragment, "C2", resnum=5, atom_name="NZ", chain_id="A")

        record = protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACT",
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
    def test_attach_chained(self):
        # Tests that a residue added by attach() can itself be a later
        # attachment site. This is needed because multi-residue and
        # branched structures (polymer chains, Y-shaped glycans) are
        # built by repeated attach() calls, and every link must be
        # recorded without any per-residue limit. The test attaches a
        # fragment to the protein and a second fragment to the first,
        # then checks both recorded bonds and the connectivity.
        protein = Protein(get_fn("6m03_protonated.pdb"))
        fragment = mb.load("CC(C)=O", smiles=True)

        protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="AC1",
        )
        record = protein.attach(
            fragment,
            "C1",
            resnum=307,
            atom_name="C3",
            chain_id="A",
            fragment_resname="AC2",
        )
        assert record.residue1.resnum == 307
        assert record.residue2.resnum == 308
        assert len(protein.cross_bonds) == 2
        first = protein.get_atom(307, "C3", chain_id="A")
        second = protein.get_atom(308, "C1", chain_id="A")
        assert protein.bond_graph.has_edge(first, second)

    def test_save_pdb_roundtrip(self, tmp_path):
        # Tests that save_pdb writes a PDB that the strict loader reads
        # back with identical structure and chemistry. This is needed
        # because the written file is the handoff artifact to OpenFF
        # Pablo, whose reader shares this loader's matching rules. The
        # test writes and reloads the protein, compares counts and net
        # charge, and checks the fixed-column layout of one atom line.
        protein = Protein(get_fn("6m03_protonated.pdb"))
        out = tmp_path / "roundtrip.pdb"
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
    def test_modified_protein_export(self, tmp_path):
        # Tests that an attached fragment exports as HETATM records with
        # exactly one CONECT pair for the new bond, and that
        # crosslink_specs() returns the with_crosslink kwargs for it.
        # This is needed because Pablo requires the crosslink CONECT,
        # fails on unexplained CONECTs (so peptide bonds must not get
        # them), and takes the spec verbatim. The test attaches a
        # fragment at LYS 5 NZ, writes the file, and checks records and
        # spec.
        protein = Protein(get_fn("6m03_protonated.pdb"))
        fragment = mb.load("CC(C)=O", smiles=True)
        protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="XCT",
        )
        out = tmp_path / "modified.pdb"
        protein.save_pdb(str(out))

        lines = out.read_text().splitlines()
        conects = [line for line in lines if line.startswith("CONECT")]
        assert len(conects) == 2  # one bond, written from both atoms
        serials = {int(part) for line in conects for part in line[6:].split()}
        named = {
            line[12:16].strip()
            for line in lines
            if line.startswith(("ATOM", "HETATM")) and int(line[6:11]) in serials
        }
        assert named == {"NZ", "C1"}
        assert sum(line.startswith("HETATM") for line in lines) == 9

        assert protein.crosslink_specs() == [
            {
                "residues": ["LYS", "XCT"],
                "linking_atoms": ["NZ", "C1"],
                "leaving_atoms": [["HZ1"], ["H1"]],
                "bond_order": 1,
            }
        ]

    def test_crosslink_specs_symmetric_and_multilink(self, caplog):
        # Tests that a symmetric bond (disulfide-like) collapses to the
        # one-element homodimer form of with_crosslink, and that a
        # residue with two recorded bonds logs the Pablo one-crosslink
        # limit warning. This is needed because Pablo's homodimer API
        # takes 1-tuples, and silently emitting specs Pablo cannot load
        # would break the handoff. The test appends two records over
        # real cysteine residues and inspects specs and the log.
        import logging

        from mbuild.biopolymers.protein import InterResidueBond

        protein = Protein(get_fn("6m03_protonated.pdb"))
        cysteines = [r for r in protein.residues() if r.name == "CYS"][:2]
        protein.cross_bonds.append(
            InterResidueBond(
                residue1=cysteines[0],
                residue2=cysteines[1],
                atom1_name="SG",
                atom2_name="SG",
                leaving1=("HG",),
                leaving2=("HG",),
            )
        )
        assert protein.crosslink_specs() == [
            {
                "residues": ["CYS"],
                "linking_atoms": ["SG"],
                "leaving_atoms": [["HG"]],
                "bond_order": 1,
            }
        ]

        protein.cross_bonds.append(
            InterResidueBond(
                residue1=cysteines[0],
                residue2=cysteines[1],
                atom1_name="CB",
                atom2_name="CB",
                leaving1=("HB2",),
                leaving2=("HB2",),
            )
        )
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.crosslink_specs()
        assert "one crosslink per residue definition" in caplog.text

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_pablo_pipeline_integration(self, tmp_path):
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
        pablo = pytest.importorskip("openff.pablo")
        if not hasattr(pablo, "STD_CCD_CACHE"):
            pytest.skip("openff-pablo >= 0.2 is required")

        fragment = mb.load("CC=O", smiles=True)
        carbons = [p for p in fragment.particles() if p.element.symbol == "C"]
        oxygen = [p for p in fragment.particles() if p.element.symbol == "O"][0]
        carbonyl = [c for c in carbons if oxygen in c.direct_bonds()][0]
        methyl = [c for c in carbons if c is not carbonyl][0]
        carbonyl.name, methyl.name, oxygen.name = "C", "CH3", "O"
        aldehyde_h = [h for h in carbonyl.direct_bonds() if h.element.symbol == "H"][0]
        aldehyde_h.name = "H"
        methyl_hydrogens = [p for p in methyl.direct_bonds() if p.element.symbol == "H"]
        for index, hydrogen in enumerate(methyl_hydrogens, 1):
            hydrogen.name = f"H{index}"

        protein = Protein(get_fn("6m03_protonated.pdb"))
        protein.attach(
            fragment,
            "CH3",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACE",
        )
        out = tmp_path / "acetylated.pdb"
        protein.save_pdb(str(out))

        (spec,) = protein.crosslink_specs()
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

    def test_unknown_residue_raises(self):
        # Tests that an unknown residue code raises a KeyError that names
        # the code and the download option. This is needed because the
        # loader is strict: it must fail loudly on residues it cannot
        # match instead of guessing. The test looks up a nonsense code
        # with downloads disabled.
        library = CCDLibrary()
        with pytest.raises(KeyError, match="XXX"):
            library["XXX"]
