"""Tests for changing the protonation state of a residue atom."""

import logging
from pathlib import Path

import numpy as np
import pytest

import mbuild as mb
from mbuild.biopolymers import Protein
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import get_fn, has_rdkit


class TestProteinProtonation(BaseTest):
    @pytest.fixture(scope="class")
    def _protein_6m03_cached(self):
        """Load the protein once per class; tests clone it."""
        return Protein(get_fn("6m03_protonated.pdb"))

    @pytest.fixture
    def protein_6m03(self, _protein_6m03_cached):
        return mb.clone(_protein_6m03_cached)

    @pytest.fixture
    def acetone(self):
        return mb.load("CC(C)=O", smiles=True)

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
        assert "NH1" in caplog.text and "NH2" in caplog.text
        assert "bonds apart" in caplog.text
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
            assert f"Atom {atom_name} of residue LYS A:12" in caplog.text
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
    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    @pytest.mark.parametrize(
        "deprotonate_first, resnum, atom_name, label, names_remedy",
        [(False, 5, "NZ", "LYS A:5", True), (True, 4, "NH1", "ARG A:4", False)],
    )
    def test_attach_reports_the_anchor_charge(
        self,
        protein_6m03,
        acetone,
        caplog,
        deprotonate_first,
        resnum,
        atom_name,
        label,
        names_remedy,
    ):
        # Tests that attach() warns when the anchor atom holds a formal
        # charge that the new bond does not change, and that it names
        # deprotonate() as the remedy for a positive anchor only. This is
        # needed because a lysine at +1 gives a protonated amide, which is
        # not a real species. deprotonate() removes a proton, so the remedy
        # fits a positive anchor only. The test attaches a fragment to the
        # charged LYS 5 NZ and to a negative ARG 4 NH1.
        protein = protein_6m03
        if deprotonate_first:
            protein.deprotonate(resnum, atom_name, chain_id="A")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            protein.attach(
                acetone,
                "C1",
                resnum=resnum,
                atom_name=atom_name,
                chain_id="A",
                fragment_resname="ACT",
                relax=False,
            )
        assert label in caplog.text
        remedy = f'deprotonate({resnum}, "{atom_name}", chain_id="A")'
        assert (remedy in caplog.text) is names_remedy

        anchor = protein.get_atom(resnum, atom_name, chain_id="A")
        carbon = protein.get_atom(307, "C1", chain_id="A")
        assert protein.bond_graph.has_edge(anchor, carbon)

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
