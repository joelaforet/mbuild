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
        # a CONECT for the new bond, and that bond_records() describes
        # the modification completely and neutrally. This is needed
        # because strict template loaders require the cross-residue
        # CONECT, fail on unexplained CONECTs (so peptide bonds must
        # not get them), and downstream tools format the records into
        # their own vocabulary. The test attaches a fragment at LYS 5
        # NZ, writes the file, and checks records.
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

        # Downstream glue: format the neutral record into pablo's
        # with_crosslink vocabulary (this formatting lives outside
        # mBuild by design).
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

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_prepare_fragment(self):
        # Tests that prepare_fragment returns a named Residue whose atom
        # names are final: the same names appear in the protein after
        # attach(). This is needed because callers must know the names
        # to pick the attachment atom and to build an external (Pablo)
        # residue definition, and attach() would otherwise rename atoms
        # invisibly inside its clone. The test prepares a fragment,
        # attaches it, and compares the name lists.
        from mbuild.biopolymers import prepare_fragment

        fragment = prepare_fragment(mb.load("CC(C)=O", smiles=True), "ACT")
        names = [particle.name for particle in fragment.particles()]
        assert fragment.name == "ACT"
        assert len(set(names)) == len(names)
        assert "C1" in names

        protein = Protein(get_fn("6m03_protonated.pdb"))
        protein.attach(fragment, "C1", resnum=5, atom_name="NZ", chain_id="A")
        attached = protein.get_residue(307, chain_id="A")
        assert [p.name for p in attached.particles()] == [
            name
            for name in names
            if name != "H1"  # the leaving hydrogen
        ]

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_fragment_from_sdf(self, tmp_path):
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
        path = tmp_path / "acetone.sdf"
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
    def test_save_pdb_orders_residues_by_number(self, tmp_path):
        # Tests that save_pdb writes residues sorted by residue number
        # inside each chain, even when attachments happened in a
        # different order. This is needed because Pablo forms polymer
        # links only between record-adjacent residues, so a fragment
        # chain built middle-first (like an NHS trimer reactive at the
        # middle monomer) must still export in backbone order. The test
        # attaches two fragments, swaps their numbers, and checks the
        # file order.
        protein = Protein(get_fn("6m03_protonated.pdb"))
        fragment = mb.load("CC(C)=O", smiles=True)
        first = protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="AC1",
        ).residue2
        second = protein.attach(
            fragment,
            "C1",
            resnum=307,
            atom_name="C3",
            chain_id="A",
            fragment_resname="AC2",
        ).residue2
        first.resnum, second.resnum = 308, 307

        out = tmp_path / "ordered.pdb"
        protein.save_pdb(str(out))
        hetero_resnames = []
        for line in out.read_text().splitlines():
            if line.startswith("HETATM"):
                name = line[17:20]
                if name not in hetero_resnames:
                    hetero_resnames.append(name)
        assert hetero_resnames == ["AC2", "AC1"]

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_bond_order_removes_matching_hydrogens(self):
        # Tests that a higher-order attach removes one hydrogen per bond
        # order unit on each side, and rejects invalid orders. This is
        # needed because removing a single hydrogen while writing a
        # double bond produces impossible valences (a pentavalent
        # nitrogen), which review reproduced. The test forms an
        # imine-like double bond at LYS 5 NZ and checks the hydrogen
        # count, recorded leaving atoms, and the order guard.
        protein = Protein(get_fn("6m03_protonated.pdb"))
        fragment = mb.load("CC=O", smiles=True)
        record = protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="IMN",
            bond_order=2,
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
    def test_clone_preserves_protein_state(self):
        # Tests that clone() returns a working Protein whose library and
        # cross-bond records survive, with the records pointing at the
        # cloned residues. This is needed because packing and solvation
        # workflows clone their inputs, and a clone that loses these
        # attributes crashes later calls. The test clones a modified
        # protein and checks identity and remapping.
        protein = Protein(get_fn("6m03_protonated.pdb"))
        fragment = mb.load("CC(C)=O", smiles=True)
        protein.attach(
            fragment,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACT",
        )
        copy = mb.clone(protein)
        assert copy.library is protein.library
        assert len(copy.cross_bonds) == 1
        assert copy.cross_bonds[0].residue1 is not protein.cross_bonds[0].residue1
        assert copy.cross_bonds[0].residue1.resnum == 5
        assert copy.net_formal_charge == protein.net_formal_charge

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_to_rdkit_carries_chemistry(self):
        # Tests that Protein.to_rdkit returns a sanitized molecule with
        # the correct net formal charge and PDB residue info, including
        # charges of an attached zwitterionic fragment. This is needed
        # because the generic Compound export drops formal charges, so
        # RDKit/OpenFF handoff would silently mis-protonate. The test
        # exports before and after attaching a sulfobetaine and checks
        # net charge and per-atom info.
        from rdkit import Chem

        from mbuild.biopolymers import prepare_fragment

        protein = Protein(get_fn("6m03_protonated.pdb"))
        mol = protein.to_rdkit()
        assert Chem.GetFormalCharge(mol) == protein.net_formal_charge == -4
        info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
        assert (info.GetResidueName(), info.GetResidueNumber()) == ("SER", 1)

        fragment = prepare_fragment("C[N+](C)(C)CCS(=O)(=O)[O-]", "SBM")
        assert fragment.formal_charge == 0
        assert len(fragment.atom_formal_charges) == 2
        protein.attach(fragment, "C1", resnum=5, atom_name="NZ", chain_id="A")
        assert Chem.GetFormalCharge(protein.to_rdkit()) == -4

        # Atoms outside any Residue must fail loudly, not vanish.
        protein.add(mb.Compound(name="XX", element="C"))
        with pytest.raises(MBuildError, match="belong to a Residue"):
            protein.to_rdkit()

    def test_save_routes_pdb(self, tmp_path):
        # Tests that the canonical save() verb writes a correct PDB via
        # save_pdb, and that to_parmed keeps the residue partitioning.
        # This is needed because the generic ParmEd path silently wrote
        # one residue named RES with no chains, which loses the protein's
        # identity without any warning. The test saves through save()
        # and checks residue fields, then counts ParmEd residues.
        protein = Protein(get_fn("6m03_protonated.pdb"))
        out = tmp_path / "routed.pdb"
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

        mol2 = tmp_path / "routed.mol2"
        protein.save(str(mol2), overwrite=True)
        assert len(parmed.load_file(str(mol2), structure=True).residues) == 306

    def test_add_port_at(self):
        # Tests the canonical-port escape hatch: a real Port anchored at
        # the named atom, pointing along the removed hydrogen. This is
        # needed so power users can run force_overlap themselves for
        # placements attach() does not cover, keeping the recipe on
        # mBuild's standard linking machinery. The test creates a port
        # at LYS 12 NZ and checks anchor and hydrogen accounting.
        from mbuild.port import Port

        protein = Protein(get_fn("6m03_protonated.pdb"))
        port = protein.add_port_at(12, "NZ", chain_id="A")
        assert isinstance(port, Port)
        assert port.anchor.name == "NZ"
        nz = protein.get_atom(12, "NZ", chain_id="A")
        hydrogens = [p for p in nz.direct_bonds() if p.element.symbol == "H"]
        assert len(hydrogens) == 2

        with pytest.raises(MBuildError, match="bond_order must be"):
            protein.add_port_at(5, "NZ", chain_id="A", bond_order=0)

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_warns_on_clashes(self, caplog):
        # Tests that attaching a bulky fragment into a crowded site logs
        # a clash warning. This is needed because port alignment is
        # rigid and a fragment placed inside the protein would otherwise
        # fail silently until the MD run fails. The test attaches
        # triphenylmethane at a buried lysine and checks the log.
        import logging

        protein = Protein(get_fn("6m03_protonated.pdb"))
        bulky = mb.load("C(c1ccccc1)(c1ccccc1)c1ccccc1", smiles=True)
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.attach(
                bulky,
                "C1",
                resnum=5,
                atom_name="NZ",
                chain_id="A",
                fragment_resname="TPM",
            )
        assert "Relax the structure" in caplog.text

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_star_sited_fragment(self):
        # Tests that a SMILES attachment point (*) marks the fragment's
        # bond site, so attach() needs no fragment atom name, and that
        # unlabeled multiple stars are rejected. This is needed because
        # star-sited fragments are the standard way chemists write
        # "link here", and reading auto-generated atom names is the
        # main UX friction otherwise. The test attaches an octanoyl
        # fragment written with a star and checks the bond and record.
        from mbuild.biopolymers import prepare_fragment

        fragment = prepare_fragment("*C(=O)CCCCCCC", "OCT")
        assert fragment.link_atoms == {"1": "C1"}

        protein = Protein(get_fn("6m03_protonated.pdb"))
        record = protein.attach(fragment, resnum=5, atom_name="NZ", chain_id="A")
        assert record.atom2_name == "C1"
        nz = protein.get_atom(5, "NZ", chain_id="A")
        assert "C1" in {p.name for p in nz.direct_bonds()}

        with pytest.raises(MBuildError, match="distinct labels"):
            prepare_fragment("*CC*", "BAD")
        two_sites = prepare_fragment("[*:1]CC[*:2]", "TWO")
        with pytest.raises(MBuildError, match="attachment points"):
            protein.attach(two_sites, resnum=90, atom_name="NZ", chain_id="A")

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_to_gmso_keeps_residue_identity(self):
        # Tests that the GMSO export keeps real PDB residue numbers and
        # the residue names of attached fragments. This is needed
        # because the generic converter renumbers residues per name and
        # misses fragment residues nested below a wrapper Compound, and
        # GMSO-side workflows (per-molecule force field application,
        # template mapping) key on this metadata. The test attaches a
        # fragment, exports, and checks numbers, the fragment residue,
        # and the chain label.
        from mbuild.biopolymers import prepare_fragment

        protein = Protein(get_fn("6m03_protonated.pdb"))
        fragment = prepare_fragment("*C(=O)C", "ACY")
        protein.attach(fragment, resnum=5, atom_name="NZ", chain_id="A", relax=False)

        topology = protein.to_gmso()
        assert topology.n_sites == protein.n_particles
        residues = {(site.residue.name, site.residue.number) for site in topology.sites}
        assert ("LYS", 5) in residues and ("ACY", 307) in residues
        assert ("SER", 1) in residues  # real numbering, not per-name 0..N
        first = next(iter(topology.sites))
        assert first.molecule.name == "Chain_A"

    def test_unknown_residue_raises(self):
        # Tests that an unknown residue code raises a KeyError that names
        # the code and the download option. This is needed because the
        # loader is strict: it must fail loudly on residues it cannot
        # match instead of guessing. The test looks up a nonsense code
        # with downloads disabled.
        library = CCDLibrary()
        with pytest.raises(KeyError, match="XXX"):
            library["XXX"]
