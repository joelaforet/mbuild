"""Tests for planning a modification before applying it."""

import logging
from pathlib import Path

import numpy as np
import pytest

import mbuild as mb
from mbuild.biopolymers import (
    CCDLibrary,
    ModificationPlan,
    Protein,
    prepare_fragment,
)
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import get_fn, has_rdkit

pytestmark = pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")


def _positions(protein):
    return np.array([particle.pos for particle in protein.particles()])


class TestModificationPlan(BaseTest):
    @pytest.fixture(scope="class")
    def _protein_6m03_cached(self):
        return Protein(get_fn("6m03_protonated.pdb"))

    @pytest.fixture
    def protein(self, _protein_6m03_cached):
        return mb.clone(_protein_6m03_cached)

    @pytest.fixture
    def octanoyl(self):
        return prepare_fragment("*C(=O)CCCCCCC", "OC8")

    def test_plan_attach_changes_nothing_and_names_the_edits(self, protein, octanoyl):
        # Tests that a plan is read without touching the protein or the
        # fragment, and that it names the site atom, the fragment atom,
        # the leaving hydrogens and the bond, in its fields and its text.
        n_atoms, n_bonds, before = (
            protein.n_particles,
            protein.n_bonds,
            _positions(protein),
        )
        n_fragment = octanoyl.n_particles
        plan = protein.plan_attach(octanoyl, resnum=5, atom_name="NZ", chain_id="A")
        assert isinstance(plan, ModificationPlan)
        assert plan.kind == "attach" and plan.reaction is None and not plan.applied
        assert (protein.n_particles, protein.n_bonds) == (n_atoms, n_bonds)
        assert np.array_equal(_positions(protein), before)
        assert octanoyl.n_particles == n_fragment
        assert plan.fragment is not octanoyl
        assert plan.site is protein.get_residue(5, chain_id="A")
        assert (plan.site_atom.name, plan.fragment_atom.name) == ("NZ", "C1")
        assert [(k.name, a.name) for k, a in plan.leaving] == [
            ("NZ", "HZ1"),
            ("C1", "H1"),
        ]
        assert plan.formed == [(plan.site_atom, plan.fragment_atom, 1.0)]
        text = str(plan)
        assert text.startswith("attach: LYS A:5 NZ + OC8 C1")
        assert "leaves    protein: HZ1 | fragment: H1" in text
        assert "forms     NZ-C1 (1)" in text
        assert "record the bond" in text
        assert (
            plan.summary()
            == "attach: LYS A:5 NZ + OC8 C1; leaves HZ1, H1; forms NZ-C1 (1)"
        )

    def test_apply_equals_the_one_call_verb(self, protein, octanoyl):
        # Tests that apply(plan_attach(...)) builds the structure that
        # attach(...) builds, atom for atom, and returns the same record.
        other = mb.clone(protein)
        plan = protein.plan_attach(
            octanoyl, resnum=5, atom_name="NZ", chain_id="A", relax=False
        )
        record = protein.apply(plan)
        reference = other.attach(
            octanoyl, resnum=5, atom_name="NZ", chain_id="A", relax=False
        )
        assert plan.applied
        for name in ("atom1_name", "atom2_name", "order", "leaving1", "leaving2"):
            assert getattr(record, name) == getattr(reference, name)
        assert protein.n_particles == other.n_particles
        assert np.allclose(_positions(protein), _positions(other))

    def test_plan_is_applied_once_and_only_to_its_protein(self, protein, octanoyl):
        plan = protein.plan_attach(
            octanoyl, resnum=5, atom_name="NZ", chain_id="A", relax=False
        )
        protein.apply(plan)
        with pytest.raises(MBuildError, match="already been applied"):
            protein.apply(plan)
        other = mb.clone(protein)
        plan = protein.plan_deprotonate(12, "NZ", chain_id="A")
        with pytest.raises(MBuildError, match="another Protein"):
            other.apply(plan)

    def test_stale_plan_is_refused(self, protein, octanoyl):
        # Tests that a plan whose atoms another modification removed is
        # refused rather than applied to the wrong atoms.
        plan = protein.plan_deprotonate(5, "NZ", chain_id="A")
        protein.deprotonate(5, "NZ", chain_id="A")
        with pytest.raises(MBuildError, match="changed since"):
            protein.apply(plan)
        plan = protein.plan_attach(
            octanoyl, resnum=12, atom_name="NZ", chain_id="A", relax=False
        )
        protein.mutate(12, "ALA", chain_id="A", relax=False)
        with pytest.raises(MBuildError, match="changed since"):
            protein.apply(plan)

    def test_reaction_refuses_arguments_it_overrides(self, protein, octanoyl):
        # Tests that bond_order and the leaving-atom names raise with a
        # reaction string instead of being ignored.
        for kwargs in (
            {"bond_order": 2},
            {"leaving_atom_names": "HZ1"},
            {"fragment_leaving_atom_names": "H1"},
        ):
            with pytest.raises(MBuildError, match="cannot be given with reaction="):
                protein.plan_attach(
                    octanoyl,
                    resnum=5,
                    atom_name="NZ",
                    chain_id="A",
                    reaction="hydrogen substitution",
                    **kwargs,
                )

    def test_reaction_plan_warns_on_a_kept_charge(self, protein, caplog):
        # Tests that the kept-charge warning of attach also runs for a
        # reaction that leaves the anchor's charge alone, and that the
        # plan lists the group that leaves with a heavy atom.
        acid = prepare_fragment("OC(=O)CCCC", "ACD")
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            plan = protein.plan_attach(
                acid, resnum=5, atom_name="NZ", chain_id="A", reaction="amide coupling"
            )
        assert "formal charge +1 before this bond" in caplog.text
        assert plan.reaction.startswith("[N:1]([H:2])[H]")
        assert 'attach by reaction "amide coupling": LYS A:5 NZ + ACD C1' in str(plan)
        assert [a.name for _, a in plan.leaving] == ["HZ2", "O1"]
        assert plan.groups[plan.leaving[1][1]] == ("O1", "H1")
        assert "fragment: O1 (+H1)" in str(plan)

    def test_thiol_maleimide_plan_tells_the_hydrogen_that_moves(self, protein):
        maleimide = prepare_fragment("O=C1C=CC(=O)N1CC", "MAL")
        plan = protein.plan_attach(
            maleimide,
            resnum=16,
            atom_name="SG",
            chain_id="A",
            reaction="thiol-maleimide",
            relax=False,
        )
        assert not plan.leaving
        assert [(a.name, b.name) for a, b in plan.removed] == [("SG", "HG")]
        assert len(plan.reordered) == 1 and plan.reordered[0][2] == 1.0
        assert any(note.startswith("HG moves from SG to C") for note in plan.notes)
        text = str(plan)
        assert "breaks    SG-HG" in text and "reorders" in text
        assert "rigid fit" not in text  # one bond joins the sides
        record = protein.apply(plan)
        assert record.reaction == plan.reaction
        assert record.atom1_name == "SG"

    def test_click_plan_states_the_charges_and_keeps_quiet(self, caplog):
        # Tests that a reaction which sets the anchor's charge gives no
        # kept-charge warning, and that a ring closure is planned as
        # two bonds with a rigid fit.
        library = CCDLibrary(paths=[Path(get_fn("8ciq_azf.pdb")).parent])
        azf = Protein(get_fn("8ciq_azf.pdb"), library=library)
        alkyne = prepare_fragment("C#CCC(=O)N", "ALK")
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            plan = azf.plan_attach(
                alkyne,
                resnum=5,
                atom_name="N3",
                chain_id="A",
                reaction="azide-alkyne triazole",
                merge=True,
            )
        assert "formal charge" not in caplog.text
        assert {atom.name: charge for atom, charge in plan.charges.items()} == {
            "N2": 0,
            "N3": 0,
        }
        assert len(plan.formed) == 2
        text = str(plan)
        assert "charges   N2: +1 -> +0, N3: -1 -> +0" in text
        assert "rigid fit to every formed bond" in text
        assert "merge the atoms into the site residue" in text

    def test_plan_mutate_names_both_side_chains(self, protein):
        plan = protein.plan_mutate(7, "PHE", chain_id="A")
        assert plan.kind == "mutate"
        assert plan.site.name == "ALA"  # nothing changed yet
        assert plan.site_atom.name == "CA"
        assert plan.groups[plan.leaving[0][1]][0] == "CB"
        assert plan.fragment_atom.name == "CB"
        assert [(a.name, b.name, o) for a, b, o in plan.formed] == [("CA", "CB", 1.0)]
        assert "ALA A:7 becomes PHE" in plan.notes
        assert "L handedness" in plan.notes
        assert str(plan).startswith("mutate: ALA A:7 -> PHE")
        residue = protein.apply(plan)
        assert residue is plan.site and residue.name == "PHE"

    def test_plan_mutate_to_glycine_and_to_proline(self, protein):
        glycine = protein.plan_mutate(7, "GLY", chain_id="A")
        assert glycine.fragment is None
        assert glycine.new_hydrogens == [glycine.site_atom]
        assert "adds H    on CA" in str(glycine)
        proline = protein.plan_mutate(8, "PRO", chain_id="A")
        assert [(a.name, b.name) for a, b, _ in proline.formed] == [
            ("CA", "CB"),
            ("N", "CD"),
        ]
        assert any("closes a ring on N" in note for note in proline.notes)
        assert "superpose the component" in str(proline)
        flipped = protein.plan_mutate(7, "PHE", chain_id="A", stereo="D")
        assert "D handedness, swapping the alpha hydrogen and the side chain" in (
            flipped.notes
        )

    def test_plan_protonation_both_ways(self, protein, caplog):
        # Tests the deprotonate and protonate plans, including the plan
        # to do nothing, which applies as the warning the verb logs.
        plan = protein.plan_deprotonate(5, "NZ", chain_id="A")
        assert plan.kind == "deprotonate"
        assert plan.leaving[0][1].name == "HZ3"
        assert plan.charges[plan.site_atom] == 0
        assert "charges   NZ: +1 -> +0" in str(plan)
        protein.apply(plan)
        assert protein.get_residue(5, chain_id="A").formal_charge == 0
        again = protein.plan_deprotonate(5, "NZ", chain_id="A")
        assert not again.leaving and "nothing" in str(again)
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.apply(again)
        assert "no acidic proton" in caplog.text
        back = protein.plan_protonate(5, "NZ", chain_id="A")
        assert back.kind == "protonate"
        assert back.new_hydrogens == [back.site_atom]
        assert back.charges[back.site_atom] == 1
        protein.apply(back)
        assert protein.get_residue(5, chain_id="A").formal_charge == 1

    def test_one_call_verbs_log_the_plan(self, protein, caplog):
        with caplog.at_level(logging.INFO, logger="mbuild"):
            protein.deprotonate(5, "NZ", chain_id="A")
        assert "deprotonate: LYS A:5 NZ; leaves HZ3" in caplog.text

    def test_draw_renders_every_kind(self, protein, octanoyl, tmp_path):
        # Tests that every kind of plan draws, as SVG and as a file.
        maleimide = prepare_fragment("O=C1C=CC(=O)N1CC", "MAL")
        plans = [
            protein.plan_attach(octanoyl, resnum=5, atom_name="NZ", chain_id="A"),
            protein.plan_attach(
                maleimide,
                resnum=16,
                atom_name="SG",
                chain_id="A",
                reaction="thiol-maleimide",
            ),
            protein.plan_mutate(7, "PHE", chain_id="A"),
            protein.plan_mutate(7, "GLY", chain_id="A"),
            protein.plan_mutate(8, "PRO", chain_id="A"),
            protein.plan_deprotonate(5, "NZ", chain_id="A"),
            protein.plan_protonate(5, "NZ", chain_id="A"),  # nothing to do
        ]
        for plan in plans:
            svg = plan._repr_svg_()
            assert "<svg" in svg and "</svg>" in svg
            assert "<svg" in plan.draw(size=(600, 200), hydrogens=True).data
        plans[0].save(tmp_path / "plan.svg")
        assert "<svg" in (tmp_path / "plan.svg").read_text()
        plans[0].save(tmp_path / "plan.png")
        assert (tmp_path / "plan.png").read_bytes()[:4] == b"\x89PNG"

    def test_drawings_hide_hydrogens_without_radicals(self, protein, octanoyl):
        # Tests that hiding hydrogens in a panel leaves the heavy atoms
        # their hydrogen counts, so the drawing shows no radicals that
        # drawing every hydrogen would not show.
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        plan = protein.plan_attach(octanoyl, resnum=5, atom_name="NZ", chain_id="A")
        particles = list(plan.site.particles())
        counts = []
        for hydrogens in (False, True):
            mol, _, _ = plan._panel(protein, particles, set(), set(), hydrogens)
            Chem.SanitizeMol(mol)
            counts.append(Descriptors.NumRadicalElectrons(mol))
        assert counts[0] == counts[1]
        product, _ = plan._product(False)
        Chem.SanitizeMol(product)
        assert Descriptors.NumRadicalElectrons(product) == counts[1]

    def test_failed_apply_marks_the_plan_applied(self, protein, monkeypatch):
        # Tests that a plan whose apply raised midway is not applied
        # again: the first attempt changed the protein already.
        from mbuild.biopolymers import protonation

        def boom(residue):
            raise RuntimeError("stop")

        plan = protein.plan_deprotonate(5, "NZ", chain_id="A")
        monkeypatch.setattr(protonation, "_warn_on_split_charge", boom)
        with pytest.raises(RuntimeError):
            protein.apply(plan)
        assert plan.applied
        with pytest.raises(MBuildError, match="already been applied"):
            protein.apply(plan)

    def test_plan_mutate_from_proline(self, protein):
        plan = protein.plan_mutate(9, "ALA", chain_id="A", platform="CPU")
        assert plan.site.name == "PRO"
        group = plan.groups[plan.leaving[0][1]]
        assert group[0] == "CB" and {"CG", "CD"} <= set(group)
        assert [atom.name for atom in plan.new_hydrogens] == ["N"]
        assert any("ring on N opens" in note for note in plan.notes)
        assert "<svg" in plan._repr_svg_()
        assert protein.apply(plan).name == "ALA"

    def test_text_keeps_the_sides_after_apply(self, protein, octanoyl):
        plan = protein.plan_attach(
            octanoyl, resnum=5, atom_name="NZ", chain_id="A", relax=False
        )
        protein.apply(plan)
        assert "leaves    protein: HZ1 | fragment: H1" in str(plan)
        assert plan.summary().startswith("attach: LYS A:5 NZ + OC8 C1")

    def test_merge_on_the_one_bond_path(self, protein):
        acetone = mb.load("CC(C)=O", smiles=True)
        plan = protein.plan_attach(
            acetone,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            fragment_resname="ACE",
            merge=True,
            relax=False,
        )
        assert "merge the atoms into the site residue" in str(plan)
        residue = protein.apply(plan)
        assert residue is plan.site and residue.template is None

    def test_glycine_plan_follows_the_handedness(self, protein):
        protein.mutate(7, "GLY", chain_id="A", relax=False)
        left = protein.plan_mutate(7, "PHE", chain_id="A", stereo="L")
        right = protein.plan_mutate(7, "PHE", chain_id="A", stereo="D")
        assert left.leaving[0][1] is not right.leaving[0][1]
        assert {left.leaving[0][1].name, right.leaving[0][1].name} == {"HA2", "HA3"}

    def test_heavy_leaving_atoms_list_their_groups(self, protein, octanoyl):
        plan = protein.plan_attach(
            octanoyl, resnum=5, atom_name="CE", chain_id="A", leaving_atom_names="NZ"
        )
        assert plan.groups[plan.leaving[0][1]] == ("NZ", "HZ1", "HZ2", "HZ3")
        assert "protein: NZ (+HZ1 HZ2 HZ3)" in str(plan)
        plan = protein.plan_attach(
            octanoyl, resnum=5, atom_name="CA", chain_id="A", leaving_atom_names="CB"
        )
        assert "protein: CB (+" in str(plan) and "atoms)" in str(plan)

    def test_no_op_protonation_plan_goes_stale_with_the_template(self, protein):
        plan = protein.plan_protonate(5, "NZ", chain_id="A")
        assert "nothing" in str(plan)
        protein.deprotonate(5, "NZ", chain_id="A")
        with pytest.raises(MBuildError, match="changed since"):
            protein.apply(plan)
