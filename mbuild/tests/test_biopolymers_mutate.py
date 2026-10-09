"""Tests for replacing the side chain of a residue in place."""

import logging
from pathlib import Path

import numpy as np
import pytest

import mbuild as mb
from mbuild.biopolymers import CCDLibrary, Protein, prepare_fragment
from mbuild.biopolymers.mutate import _alpha_handedness
from mbuild.biopolymers.relax import _closest_contact
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import get_fn, has_openmm, has_rdkit

BACKBONE = ("N", "CA", "C", "O", "H")


def _handedness(residue):
    """L or D of a residue that has a CB, read from its geometry."""
    positions = {particle.name: particle.pos for particle in residue.particles()}
    return _alpha_handedness(
        positions["N"], positions["CA"], positions["C"], positions["CB"]
    )


class TestProteinMutate(BaseTest):
    @pytest.fixture(scope="class")
    def _protein_6m03_cached(self):
        """Load the protein once per class; tests clone it."""
        return Protein(get_fn("6m03_protonated.pdb"))

    @pytest.fixture
    def protein(self, _protein_6m03_cached):
        return mb.clone(_protein_6m03_cached)

    def test_mutate_to_a_canonical_residue(self, protein):
        # Tests that mutating alanine 7 to phenylalanine keeps the
        # backbone coordinates, replaces the side chain atoms with the
        # CCD atoms of PHE, renames the residue, re-matches its
        # definition, and records no inter-residue bond. This is the
        # contract of a point mutation: the same residue, in the same
        # place in the chain, with new chemistry. The test compares
        # the backbone positions, the atom name set, the residue and
        # particle counts, and the definition before and after.
        residue = protein.get_residue(7, chain_id="A")
        assert residue.name == "ALA"
        before = {
            name: protein.get_atom(7, name, chain_id="A").pos.copy()
            for name in BACKBONE + ("HA",)
        }
        n_residues = len(list(protein.residues()))
        n_particles = protein.n_particles

        mutated = protein.mutate(7, "PHE", chain_id="A", relax=False)

        assert mutated is residue
        assert residue.name == "PHE"
        assert residue.hetatm is False
        assert residue.template.name == "PHE"
        assert residue.formal_charge == 0
        names = {particle.name for particle in residue.particles()}
        # PHE in a chain: every template atom but the peptide-bond
        # leaving atoms H2, OXT and HXT.
        assert names == residue.template.atom_names - {"H2", "OXT", "HXT"}
        assert len(list(protein.residues())) == n_residues
        # ALA has 10 atoms in the chain, PHE has 20.
        assert protein.n_particles == n_particles + 10
        assert protein.cross_bonds == []
        for name, pos in before.items():
            assert np.allclose(protein.get_atom(7, name, chain_id="A").pos, pos)
        cb = protein.get_atom(7, "CB", chain_id="A")
        ca = protein.get_atom(7, "CA", chain_id="A")
        assert np.linalg.norm(cb.pos - ca.pos) == pytest.approx(0.15, abs=0.01)
        assert _handedness(residue) == "L"
        assert cb in ca.direct_bonds()

    def test_mutant_round_trips_through_a_pdb_file(self, protein, tmp_path):
        # Tests that a mutated protein writes to a PDB file that loads
        # again as the mutant. This is needed because the new side chain
        # must carry CCD names and sit inside the residue's records;
        # otherwise the loader would refuse the file or read the old
        # residue. The test writes the file, reloads it, and compares.
        protein.mutate(7, "TRP", chain_id="A", relax=False)
        path = tmp_path / "mutant.pdb"
        protein.save_pdb(str(path))
        again = Protein(str(path))
        residue = again.get_residue(7, chain_id="A")
        assert residue.name == "TRP"
        assert residue.template.name == "TRP"
        assert again.n_particles == protein.n_particles
        assert again.n_bonds == protein.n_bonds

    def test_stereo_flips_the_alpha_carbon(self, protein):
        # Tests that stereo="D" builds the side chain on the other
        # face of CA and moves HA to the old CB direction, so the
        # handedness reads D afterwards, and that stereo="L" on an L
        # residue changes nothing. This is needed for D-amino acid
        # constructs, and the geometric test is what the loader's
        # exports will see. The test reads the handedness from the
        # four backbone positions before and after.
        residue = protein.get_residue(7, chain_id="A")
        assert _handedness(residue) == "L"
        ca = protein.get_atom(7, "CA", chain_id="A").pos.copy()
        old_cb = protein.get_atom(7, "CB", chain_id="A").pos.copy()
        old_ha = protein.get_atom(7, "HA", chain_id="A").pos.copy()

        protein.mutate(7, "PHE", chain_id="A", stereo="D", relax=False)
        assert _handedness(residue) == "D"
        new_cb = protein.get_atom(7, "CB", chain_id="A").pos
        new_ha = protein.get_atom(7, "HA", chain_id="A").pos
        # The side chain took the old HA direction, HA the old CB direction.
        assert np.dot(new_cb - ca, old_ha - ca) > 0
        assert np.dot(new_ha - ca, old_cb - ca) > 0

        protein.mutate(7, "PHE", chain_id="A", stereo="L", relax=False)
        assert _handedness(residue) == "L"
        protein.mutate(7, "PHE", chain_id="A", stereo="L", relax=False)
        assert _handedness(residue) == "L"

    def test_ccd_component_sets_the_default_handedness(self, protein):
        # Tests that a CCD code mutates to the handedness of the
        # component, whatever the residue had. A D residue mutated to
        # PHE comes back L, because the CCD entry PHE is
        # L-phenylalanine; a fragment, which fixes no handedness,
        # keeps the residue's. This is needed so that a D component
        # such as DAL gives a D residue by default, and the name and
        # the geometry of the mutant agree.
        residue = protein.get_residue(7, chain_id="A")
        protein.mutate(7, "ALA", chain_id="A", stereo="D", relax=False)
        assert _handedness(residue) == "D"
        protein.mutate(7, "PHE", chain_id="A", relax=False)
        assert _handedness(residue) == "L"
        protein.mutate(7, "ALA", chain_id="A", stereo="D", relax=False)
        protein.mutate(7, prepare_fragment("*C", "ALA"), chain_id="A", relax=False)
        assert _handedness(residue) == "D"

    def test_component_whose_beta_carbon_is_not_named_cb(self):
        # Tests that a CCD component whose side-chain carbon on CA is
        # not named CB, p-acetyl-L-phenylalanine 4AF with its C3, is
        # both a valid target and a valid source of a mutation. This is
        # needed because CCD atom names outside the canonical residues
        # follow no convention, so the side chain must be found by
        # bonding to CA and not by name. The test reads 4AF from a
        # bundled CIF file through a library that searches that
        # directory, mutates to it and back, and checks the names.
        library = CCDLibrary(paths=[Path(get_fn("4AF.cif")).parent])
        protein = Protein(get_fn("8ciq.pdb"), library=library)
        residue = protein.get_residue(3, chain_id="A")
        protein.mutate(3, "4AF", chain_id="A", relax=False)
        assert residue.name == "4AF"
        assert residue.hetatm is True
        names = {particle.name for particle in residue.particles()}
        assert "C3" in names and "CB" not in names
        assert residue.template.name == "4AF"
        ca = protein.get_atom(3, "CA", chain_id="A")
        assert protein.get_atom(3, "C3", chain_id="A") in ca.direct_bonds()
        protein.mutate(3, "ALA", chain_id="A", relax=False)
        assert residue.name == "ALA"
        assert {p.name for p in residue.particles()} == {
            "N",
            "H",
            "CA",
            "HA",
            "CB",
            "HB1",
            "HB2",
            "HB3",
            "C",
            "O",
        }

    def test_stereo_rejects_other_values(self, protein):
        with pytest.raises(ValueError, match="stereo"):
            protein.mutate(7, "PHE", chain_id="A", stereo="R")

    def test_glycine_gets_the_wanted_handedness(self, protein):
        # Tests that a glycine, which has no handedness, is mutated to
        # an L residue by default and to a D residue on request, and
        # that the remaining alpha hydrogen is renamed HA. This is
        # needed because the two alpha hydrogens are equivalent in
        # glycine but not once one of them becomes a side chain: the
        # choice fixes the stereocenter.
        residue = protein.get_residue(11, chain_id="A")
        assert residue.name == "GLY"
        protein.mutate(11, "ALA", chain_id="A", relax=False)
        assert residue.name == "ALA"
        assert residue.template.name == "ALA"
        names = {particle.name for particle in residue.particles()}
        assert "HA" in names and not names & {"HA2", "HA3"}
        assert _handedness(residue) == "L"

        other = mb.clone(protein)
        # Mutating the alanine back to glycine and on to a D residue.
        other.mutate(11, "GLY", chain_id="A", relax=False)
        other.mutate(11, "ALA", chain_id="A", stereo="D", relax=False)
        assert _handedness(other.get_residue(11, chain_id="A")) == "D"

    def test_mutate_to_glycine(self, protein):
        # Tests that a mutation to glycine removes the side chain and
        # adds one alpha hydrogen in its place, so the residue matches
        # the GLY definition. Glycine has no CB, so it is the one
        # target that no CCD side chain can supply.
        residue = protein.get_residue(7, chain_id="A")
        protein.mutate(7, "GLY", chain_id="A", relax=False)
        assert residue.name == "GLY"
        assert residue.template.name == "GLY"
        names = {particle.name for particle in residue.particles()}
        assert names == {"N", "H", "CA", "HA2", "HA3", "C", "O"}
        ca = protein.get_atom(7, "CA", chain_id="A")
        assert len(list(ca.direct_bonds())) == 4

    def test_disulfide_cysteine_is_refused(self):
        # Tests that a cysteine in a disulfide cannot be mutated, with an
        # error that names the crosslink. Its side chain is joined to
        # another residue, so removing it would break a bond the file
        # records; the user must decide about that bond first.
        protein = Protein(get_fn("8ciq.pdb"))
        bridged = protein.cross_bonds[0].residue1
        with pytest.raises(MBuildError, match="crosslink"):
            protein.mutate(bridged.resnum, "ALA", chain_id="A")

    def test_mutate_to_proline(self, protein, caplog, tmp_path):
        # Tests that alanine 7 becomes a proline whose ring closes on
        # the backbone nitrogen, with the backbone in place, the amide
        # hydrogen gone, the PRO definition matched, and a file that
        # loads again. This is needed because proline's side chain bonds
        # N as well as CA, so no single bond can align it: mutate()
        # superposes the CCD component on the backbone instead. Residue
        # 7 sits at a phi of about -100 degrees, so the test also checks
        # the warning about the strained ring.
        before = {
            name: protein.get_atom(7, name, chain_id="A").pos.copy()
            for name in ("N", "CA", "C", "O", "HA")
        }
        with caplog.at_level(logging.WARNING, logger="mbuild"):
            residue = protein.mutate(7, "PRO", chain_id="A", platform="CPU")

        assert residue.name == "PRO"
        assert residue.hetatm is False
        assert residue.template.name == "PRO"
        assert residue.formal_charge == 0
        atoms = {particle.name: particle for particle in residue.particles()}
        assert set(atoms) == residue.template.atom_names - {"H", "OXT", "HXT"}
        for name, pos in before.items():
            assert np.allclose(atoms[name].pos, pos)
        assert atoms["CD"] in atoms["N"].direct_bonds()
        assert atoms["CB"] in atoms["CA"].direct_bonds()
        n_cd = np.linalg.norm(atoms["N"].pos - atoms["CD"].pos)
        assert n_cd == pytest.approx(0.147, abs=0.01)
        assert _handedness(residue) == "L"
        assert "phi of ALA A:7" in caplog.text

        protein.save_pdb(tmp_path / "pro.pdb")
        again = Protein(tmp_path / "pro.pdb")
        assert again.get_residue(7, chain_id="A").template.name == "PRO"

    @pytest.mark.skipif(not (has_rdkit and has_openmm), reason="needs RDKit and OpenMM")
    def test_minimize_false_defers_to_one_relaxation(self, protein, monkeypatch):
        # Tests that mutate(minimize=False) runs no minimization, and that
        # one relax_fragments call over the mutated residues relaxes their
        # side chains, closes a proline ring, and leaves every backbone
        # atom where it was. This is needed because a structure with many
        # mutations, or mutations and attached glycans, is built fastest
        # by placing everything first and minimizing once, and
        # relax_fragments must keep the promise mutate makes about the
        # backbone. The test mutates ALA 7 to PHE and TYR 37 to PRO.
        import mbuild.biopolymers.protein as protein_module
        from mbuild.biopolymers import relax

        runs = []
        original = relax._relax_particles

        def counting(*args, **kwargs):
            runs.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(relax, "_relax_particles", counting)
        monkeypatch.setattr(protein_module, "_relax_particles", counting)
        backbone = ("N", "CA", "C", "O", "HA")
        before = {
            (resnum, name): protein.get_atom(resnum, name, chain_id="A").pos.copy()
            for resnum in (7, 37)
            for name in backbone
        }
        phe = protein.mutate(7, "PHE", chain_id="A", minimize=False)
        pro = protein.mutate(37, "PRO", chain_id="A", minimize=False)
        assert runs == []

        protein.relax_fragments(residues=[phe, pro], side_chains=True, platform="CPU")

        assert runs == [1]
        for (resnum, name), pos in before.items():
            assert np.allclose(protein.get_atom(resnum, name, chain_id="A").pos, pos)
        n, cd = (protein.get_atom(37, name, chain_id="A") for name in ("N", "CD"))
        assert np.linalg.norm(n.pos - cd.pos) == pytest.approx(0.147, abs=0.01)

    def test_mutate_from_proline(self, protein, tmp_path):
        # Tests that proline 9 becomes an alanine: the ring opens at N,
        # N takes back an amide hydrogen in the plane of the peptide
        # bond, no port is left behind, the backbone stays in place, and
        # the file loads again. This is needed because the side chain
        # of proline has no far side of the CA-CB bond until its bond
        # to N is broken, and a mid-chain residue other than proline
        # carries a hydrogen on N that the ALA definition requires.
        residue = protein.get_residue(9, chain_id="A")
        assert residue.name == "PRO"
        before = {
            name: protein.get_atom(9, name, chain_id="A").pos.copy()
            for name in ("N", "CA", "C", "O", "HA")
        }

        protein.mutate(9, "ALA", chain_id="A", platform="CPU")

        assert residue.name == "ALA"
        assert residue.template.name == "ALA"
        atoms = {particle.name: particle for particle in residue.particles()}
        assert set(atoms) == residue.template.atom_names - {"H2", "OXT", "HXT"}
        assert not any(isinstance(child, mb.Port) for child in residue.children)
        for name, pos in before.items():
            assert np.allclose(atoms[name].pos, pos)
        n, h = atoms["N"], atoms["H"]
        assert np.linalg.norm(h.pos - n.pos) == pytest.approx(0.101, abs=0.005)
        previous = next(a for a in n.direct_bonds() if a.name == "C")
        normal = np.cross(previous.pos - n.pos, atoms["CA"].pos - n.pos)
        normal /= np.linalg.norm(normal)
        assert abs(np.dot(h.pos - n.pos, normal)) < 0.01

        protein.save_pdb(tmp_path / "ala.pdb")
        again = Protein(tmp_path / "ala.pdb")
        assert again.get_residue(9, chain_id="A").template.name == "ALA"

    def test_charged_default_and_deprotonate(self, protein):
        # Tests that a CCD side chain arrives in the component's default
        # protonation state, a charged lysine here, and that
        # deprotonate() then works on the mutant as on a loaded
        # residue. This is needed because a mutation must leave the
        # residue in a state every other method understands.
        residue = protein.mutate(7, "LYS", chain_id="A", relax=False)
        assert residue.formal_charge == 1
        assert residue.atom_formal_charges == {"NZ": 1}
        protein.deprotonate(7, "NZ", chain_id="A")
        assert residue.formal_charge == 0
        assert residue.template.description == "LYSINE -HZ3"

    def test_side_chain_from_a_fragment(self, protein):
        # Tests that a side chain given as a star-marked SMILES fragment
        # is bonded at the marked atom, that the marked atom is named CB,
        # that the fragment's formal charges reach the residue, and that
        # the residue is a HETATM residue with no CCD definition. This is
        # the path for a non-canonical amino acid the CCD does not hold.
        # The fragment is the p-azidophenyl side chain of AzF.
        azf = prepare_fragment("*Cc1ccc(N=[N+]=[N-])cc1", "AZF")
        n_particles = protein.n_particles
        residue = protein.mutate(7, azf, chain_id="A", relax=False)
        assert residue.name == "AZF"
        assert residue.hetatm is True
        assert residue.template is None
        assert residue.formal_charge == 0
        assert sorted(residue.atom_formal_charges.values()) == [-1, 1]
        cb = protein.get_atom(7, "CB", chain_id="A")
        assert cb.element.symbol == "C"
        assert protein.get_atom(7, "CA", chain_id="A") in cb.direct_bonds()
        # ALA loses CB and three HB (4 atoms). The AzF side chain has 16:
        # seven carbons, six hydrogens and the three azide nitrogens.
        assert protein.n_particles == n_particles - 4 + 16
        assert _handedness(residue) == "L"

    def test_fragment_needs_one_bond_site(self, protein):
        with pytest.raises(MBuildError, match="exactly one"):
            protein.mutate(7, prepare_fragment("[*:1]C[*:2]", "BAD"), chain_id="A")

    def test_not_an_amino_acid_is_refused(self, protein):
        # Water is a CCD component the library holds, and it has no
        # side chain.
        with pytest.raises(MBuildError, match="not an alpha amino acid"):
            protein.mutate(7, "HOH", chain_id="A")

    def test_unknown_code_points_at_download(self, protein):
        with pytest.raises(MBuildError, match="download=True"):
            protein.mutate(7, "XQZ", chain_id="A")

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_mutant_exports_to_rdkit_with_cip_labels(self, protein):
        # Tests that the mutant exports through to_rdkit and that RDKit
        # reads the alpha carbon of the mutant as S for the L residue
        # and R for the D residue. This ties the geometric handedness
        # rule to the CIP label a downstream tool reports.
        from rdkit import Chem

        def cip(structure):
            mol = structure.to_rdkit()
            Chem.AssignStereochemistryFrom3D(mol)
            for atom in mol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info.GetResidueNumber() == 7 and info.GetName().strip() == "CA":
                    return atom.GetPropsAsDict().get("_CIPCode")

        protein.mutate(7, "PHE", chain_id="A", relax=False)
        assert cip(protein) == "S"
        protein.mutate(7, "PHE", chain_id="A", stereo="D", relax=False)
        assert cip(protein) == "R"

    def test_relaxation_keeps_the_backbone_and_the_far_residues(self, protein):
        # Tests that the default relax=True clears the overlap a rigidly
        # placed bulky side chain makes, that no backbone atom moves,
        # and that the only other atoms that move are side-chain atoms
        # near the new side chain. This is needed because the point of
        # a mutation is to keep the experimental coordinates: the
        # backbone stays where the file put it, the neighbouring side
        # chains give way as they would in any minimization, and the
        # rest of the protein does not change.
        pytest.importorskip("openmm")
        pytest.importorskip("hoomd")
        residue = protein.get_residue(7, chain_id="A")
        before = {p: p.pos.copy() for p in protein.particles()}
        protein.mutate(7, "TRP", chain_id="A")
        side_chain = [
            p for p in residue.particles() if p.name not in BACKBONE + ("HA",)
        ]
        moved = [
            p
            for p, pos in before.items()
            if p.parent is not residue and not np.allclose(p.pos, pos)
        ]
        assert not [p for p in moved if p.name in ("N", "CA", "C", "O")]
        centre = np.mean([p.pos for p in side_chain], axis=0)
        assert all(np.linalg.norm(p.pos - centre) < 1.2 for p in moved)
        for name in BACKBONE + ("HA",):
            assert np.allclose(
                protein.get_atom(7, name, chain_id="A").pos,
                before[protein.get_atom(7, name, chain_id="A")],
            )
        ca = protein.get_atom(7, "CA", chain_id="A")
        closest = _closest_contact(
            protein, side_chain, ca, protein.get_atom(7, "CB", chain_id="A")
        )
        # Rigid placement of TRP on this site leaves atoms under 0.08 nm apart;
        # a relaxed side chain sits at ordinary contact distances.
        assert closest > 0.12
