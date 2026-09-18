"""Tests for attaching a fragment by a reaction string."""

from pathlib import Path

import numpy as np
import pytest

import mbuild as mb
from mbuild.biopolymers import REACTIONS, CCDLibrary, Protein, prepare_fragment
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import get_fn, has_openmm, has_rdkit

pytestmark = pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")


def _bond_order(protein, atom1, atom2):
    return protein.root.bond_graph.edges[atom1, atom2]["bond_order"]


def _length(atom1, atom2):
    return np.linalg.norm(atom1.pos - atom2.pos)


class TestReactionStrings(BaseTest):
    @pytest.fixture(scope="class")
    def _azf_cached(self):
        """A small protein with p-azido-L-phenylalanine (4II) at residue 5.

        The fixture was written by ``Protein.mutate`` from the 8ciq asset.
        The library reads the 4II definition from the bundled CIF file
        next to it, so the test needs no network.
        """
        library = CCDLibrary(paths=[Path(get_fn("8ciq_azf.pdb")).parent])
        return Protein(get_fn("8ciq_azf.pdb"), library=library)

    @pytest.fixture
    def azf_protein(self, _azf_cached):
        return mb.clone(_azf_cached)

    @pytest.fixture(scope="class")
    def _protein_6m03_cached(self):
        return Protein(get_fn("6m03_protonated.pdb"))

    @pytest.fixture
    def protein_6m03(self, _protein_6m03_cached):
        return mb.clone(_protein_6m03_cached)

    def test_named_reactions_parse(self):
        # Tests that every shipped reaction string is a reaction SMARTS
        # with two reactant templates and one product template, which
        # is the shape attach() requires. This is needed so a typo in
        # the library fails here and not in a user's session.
        from rdkit.Chem import AllChem

        assert len(REACTIONS) == 6
        for name, smarts in REACTIONS.items():
            reaction = AllChem.ReactionFromSmarts(smarts)
            assert reaction.GetNumReactantTemplates() == 2, name
            assert reaction.GetNumProductTemplates() == 1, name

    def test_click_reaction_closes_a_triazole(self, azf_protein):
        # Tests that the azide-alkyne reaction forms both ring bonds
        # between the azide of 4II and an alkyne fragment, sets the
        # bond orders and clears the azide charges as the product
        # template states, records both bonds with the reaction, and
        # exports through to_rdkit as a five-membered ring. This is
        # the reaction the leaving-atom keywords cannot express, and
        # the one a click-chemistry labeling needs. The rigid placement
        # cannot close a ring, so the test allows the bond lengths a
        # relaxation leaves when the simulation dependencies are
        # present, and only checks topology without them.
        protein = azf_protein
        azf = protein.get_residue(5, chain_id="A")
        assert azf.atom_formal_charges == {"N2": 1, "N3": -1}
        n_records = len(protein.cross_bonds)

        record = protein.attach(
            prepare_fragment("CC#C", "PRG"),
            resnum=5,
            atom_name="N3",
            chain_id="A",
            reaction="azide-alkyne triazole",
        )
        made = [b for b in protein.cross_bonds if b.reaction]
        assert len(made) == 2
        assert len(protein.cross_bonds) == n_records + 2
        assert record in made and record.atom1_name == "N3"
        assert record.reaction == REACTIONS["azide-alkyne triazole"]
        assert azf.atom_formal_charges == {}
        assert azf.formal_charge == 0
        assert azf.template is None
        atoms = {p.name: p for p in azf.particles()}
        assert _bond_order(protein, atoms["N1"], atoms["N2"]) == 1.0
        assert _bond_order(protein, atoms["N2"], atoms["N3"]) == 2.0
        fragment = record.residue2
        carbons = {p.name: p for p in fragment.particles() if p.element.symbol == "C"}
        alkyne = [b for b in made if b.atom2_name in carbons]
        assert len(alkyne) == 2
        pair = [carbons[b.atom2_name] for b in alkyne]
        assert _bond_order(protein, pair[0], pair[1]) == 2.0
        if has_openmm:
            for bond in made:
                atom1 = next(bond.residue1.particles_by_name(bond.atom1_name))
                atom2 = next(bond.residue2.particles_by_name(bond.atom2_name))
                assert _length(atom1, atom2) < 0.18
        mol = protein.to_rdkit()
        ring_sizes = {
            len(ring)
            for ring in mol.GetRingInfo().AtomRings()
            if any(
                mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueName() == "PRG"
                for i in ring
            )
        }
        assert 5 in ring_sizes
        records = [r for r in protein.bond_records() if "reaction" in r]
        assert len(records) == 2
        assert all(r["residue_names"] == ("4II", "PRG") for r in records)

    def test_reaction_written_fragment_first_still_matches(self, azf_protein):
        # Tests that a reaction whose first template is the fragment is
        # accepted, because attach() tries the two orders. This is
        # needed so that one shipped string serves both a protein acid
        # with a fragment amine and a protein amine with a fragment
        # acid.
        swapped = "[C:4]#[C:5].[N:1]=[N+:2]=[N-:3]>>[N:1]1[N+0:2]=[N+0:3][C:4]=[C:5]1"
        record = azf_protein.attach(
            prepare_fragment("CC#C", "PRG"),
            resnum=5,
            atom_name="N3",
            chain_id="A",
            reaction=swapped,
            relax=False,
        )
        assert record.residue1.name == "4II"
        assert len([b for b in azf_protein.cross_bonds if b.reaction]) == 2

    def test_thiol_maleimide_moves_the_thiol_hydrogen(self, protein_6m03):
        # Tests that the thiol-maleimide reaction bonds SG to one alkene
        # carbon, turns the C=C into a single bond, and moves the thiol
        # hydrogen onto the other carbon, which then carries two
        # hydrogens. This is needed because the product is a neutral
        # thioether with a saturated succinimide, and a hydrogen that
        # changes place is the case the leaving-atom keywords cannot
        # describe.
        protein = protein_6m03
        cys = next(
            r
            for r in protein.residues()
            if r.name == "CYS" and any(p.name == "HG" for p in r.particles())
        )
        sg = protein.get_atom(cys.resnum, "SG", chain_id="A")
        n_before = protein.n_particles
        record = protein.attach(
            prepare_fragment("C1=CC(=O)N(C)C1=O", "MAL"),
            resnum=cys.resnum,
            atom_name="SG",
            chain_id="A",
            reaction="thiol-maleimide",
            relax=False,
        )
        # No atom left the structure: the hydrogen moved instead. The
        # record still names it as leaving the cysteine, because that is
        # what a residue library reading the file sees.
        assert protein.n_particles == n_before + 13
        assert record.leaving1 == ("HG",) and record.leaving2 == ()
        # One bond joins the sides; the moved hydrogen's new bond is not
        # a bond between residues and is not recorded.
        assert [b for b in protein.cross_bonds if b.reaction] == [record]
        bonded_carbon = next(
            p for p in sg.direct_bonds() if p.parent is record.residue2
        )
        other_carbon = next(
            p
            for p in bonded_carbon.direct_bonds()
            if p.element.symbol == "C"
            and p.parent is record.residue2
            and len([h for h in p.direct_bonds() if h.element.symbol == "H"]) == 2
        )
        assert _bond_order(protein, bonded_carbon, other_carbon) == 1.0
        hydrogens = [h for h in other_carbon.direct_bonds() if h.element.symbol == "H"]
        assert {h.name for h in hydrogens} >= {"HG"}
        moved = next(h for h in hydrogens if h.name == "HG")
        assert _length(moved, other_carbon) == pytest.approx(0.1, abs=0.005)
        # The moved hydrogen now belongs to the fragment residue.
        assert moved.parent is record.residue2
        assert not any(p.name == "HG" for p in cys.particles())
        protein.to_rdkit()

    def test_amide_coupling_removes_water(self, protein_6m03):
        # Tests that the amide coupling removes one amine hydrogen and
        # the acid's hydroxyl, records them as the leaving atoms, and
        # bonds the nitrogen to the carbonyl carbon. Equivalent amine
        # hydrogens are one match, taken in alphabetical order, so a
        # lysine with three protons does not count as ambiguous.
        protein = protein_6m03
        lys = next(r for r in protein.residues() if r.name == "LYS")
        record = protein.attach(
            prepare_fragment("CC(=O)O", "ACT"),
            resnum=lys.resnum,
            atom_name="NZ",
            chain_id="A",
            reaction="amide coupling",
            relax=False,
        )
        assert len(record.leaving1) == 1 and record.leaving1[0].startswith("HZ")
        assert sorted(p[0] for p in record.leaving2) == ["H", "O"]
        assert record.residue2.n_particles == 6  # CH3-C(=O)-
        nz = protein.get_atom(lys.resnum, "NZ", chain_id="A")
        carbonyl = next(p for p in nz.direct_bonds() if p.parent is record.residue2)
        assert carbonyl.element.symbol == "C"
        assert protein.bond_records()[-1]["reaction"] == REACTIONS["amide coupling"]

    def test_ambiguous_heavy_atom_match_is_refused(self, protein_6m03):
        # Tests that a fragment template matching two different heavy
        # atoms raises and asks for the atom, while hydrogens on one
        # heavy atom do not. A glycan has many hydroxyl carbons, and
        # the anomeric one must be named.
        protein = protein_6m03
        ser = next(r for r in protein.residues() if r.name == "SER")
        glycol = prepare_fragment("OCCO", "EGL")
        with pytest.raises(MBuildError, match="Name the atom"):
            protein.attach(
                glycol,
                resnum=ser.resnum,
                atom_name="OG",
                chain_id="A",
                reaction="O-glycosylation",
                relax=False,
            )
        record = protein.attach(
            glycol,
            "C1",
            resnum=ser.resnum,
            atom_name="OG",
            chain_id="A",
            reaction="O-glycosylation",
            relax=False,
        )
        assert record.atom2_name == "C1"
        assert record.leaving1 == ("HG",)

    def test_reaction_that_forms_no_cross_bond_is_refused(self, protein_6m03):
        lys = next(r for r in protein_6m03.residues() if r.name == "LYS")
        with pytest.raises(MBuildError, match="forms no bond"):
            protein_6m03.attach(
                prepare_fragment("CC(=O)O", "ACT"),
                resnum=lys.resnum,
                atom_name="NZ",
                chain_id="A",
                reaction="[N:1][H:5].[C:2](=[O:3])[O:4][H:6]>>([N:1][H:5].[C:2](=[O:3])[O:4][H:6])",
            )

    def test_created_heavy_atom_is_refused(self, protein_6m03):
        # Tests that a product atom no reactant supplies is refused
        # when it is not a hydrogen, because it would have no
        # coordinates.
        lys = next(r for r in protein_6m03.residues() if r.name == "LYS")
        with pytest.raises(MBuildError, match="creates an atom"):
            protein_6m03.attach(
                prepare_fragment("CC(=O)O", "ACT"),
                resnum=lys.resnum,
                atom_name="NZ",
                chain_id="A",
                reaction="[N:1][H].[C:2](=O)O[H]>>[N:1][C:2](=O)O[Cl]",
            )

    def test_wrong_shape_and_no_match_name_the_problem(self, protein_6m03):
        lys = next(r for r in protein_6m03.residues() if r.name == "LYS")
        with pytest.raises(MBuildError, match="two reactant templates"):
            protein_6m03.attach(
                prepare_fragment("CC(=O)O", "ACT"),
                resnum=lys.resnum,
                atom_name="NZ",
                chain_id="A",
                reaction="[N:1][H]>>[N:1]",
            )
        with pytest.raises(MBuildError, match="does not match"):
            protein_6m03.attach(
                prepare_fragment("CC(=O)O", "ACT"),
                resnum=lys.resnum,
                atom_name="NZ",
                chain_id="A",
                reaction="thiol-maleimide",
            )
