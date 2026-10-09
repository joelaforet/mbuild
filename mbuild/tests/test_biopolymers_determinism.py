"""Tests that a build gives the same result in every process.

A Compound hashes by ``id``, so the iteration order of a set of
particles follows memory addresses, which change from one process to
the next. Code that lets that order choose a result builds a different
structure from the same input. Most tests here change the hash of every
Compound, which reorders every such set the way another memory layout
would, but inside one process and on every run; the last one runs fresh
processes.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mbuild as mb
from mbuild.biopolymers import CCDLibrary, Protein, prepare_fragment
from mbuild.biopolymers.relax import _placement_torsions
from mbuild.biopolymers.residue import _rdkit_mol
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import get_fn, has_openmm, has_rdkit

pytestmark = pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")


def _under_each_hash(monkeypatch, build, salts=(1, 2, 3)):
    """Return ``build()`` once per salt, with Compounds hashed by ``(salt, id)``.

    Everything the build uses must be made inside it: an object made
    under one hash is not found again in a set under another.
    """
    results = []
    for salt in salts:
        monkeypatch.setattr(
            mb.Compound, "__hash__", lambda self, salt=salt: hash((salt, id(self)))
        )
        results.append(build())
    monkeypatch.undo()
    return results


def _free_cysteine(protein):
    return next(
        r
        for r in protein.residues()
        if r.name == "CYS" and any(p.name == "HG" for p in r.particles())
    )


def _azf_protein():
    library = CCDLibrary(paths=[Path(get_fn("8ciq_azf.pdb")).parent])
    return Protein(get_fn("8ciq_azf.pdb"), library=library)


class TestDeterminism(BaseTest):
    def test_rdkit_mol_of_a_residue_keeps_its_order(self, monkeypatch):
        # Tests that the RDKit molecule of one residue lists its atoms
        # and bonds in the same order under every hash. Below the root,
        # Compound.bonds reads a networkx subgraph that iterates a set
        # of particles, so the bond order, and every substructure match
        # made on the molecule, followed memory addresses.
        def build():
            protein = Protein(get_fn("6m03_protonated.pdb"))
            residue = next(r for r in protein.residues() if r.name == "TRP")
            mol, index = _rdkit_mol(residue, list(residue.particles()))
            name = {i: particle.name for particle, i in index.items()}
            return [atom.GetAtomicNum() for atom in mol.GetAtoms()], [
                (
                    name[b.GetBeginAtomIdx()],
                    name[b.GetEndAtomIdx()],
                    str(b.GetBondType()),
                )
                for b in mol.GetBonds()
            ]

        first, *others = _under_each_hash(monkeypatch, build)
        assert len(first[1]) > 20
        assert all(other == first for other in others)

    def test_thiol_maleimide_places_the_same_way(self, monkeypatch):
        # Tests that a thiol-maleimide attach gives the same coordinates
        # to the last digit under every hash. The fragment approaches
        # the alkene carbon along the normal of its plane, which has two
        # senses; the sense came from the order of the carbon's bonds,
        # which Compound.direct_bonds returns as a set, so the fragment
        # landed on either face of the ring.
        def build():
            protein = Protein(get_fn("6m03_protonated.pdb"))
            protein.attach(
                prepare_fragment("C1=CC(=O)N(C)C1=O", "MAL"),
                resnum=_free_cysteine(protein).resnum,
                atom_name="SG",
                chain_id="A",
                reaction="thiol-maleimide",
                relax=False,
            )
            return protein.xyz

        # The normal has two senses, so enough hashes are tried that both
        # come up.
        first, *others = _under_each_hash(monkeypatch, build, salts=range(8))
        for other in others:
            np.testing.assert_array_equal(other, first)

    def test_click_forms_the_documented_regioisomer(self, monkeypatch):
        # Tests that the azide-alkyne reaction on an unsymmetric
        # cyclooctyne (the DBCO core) forms the same triazole under
        # every hash, and that it is the isomer REACTIONS documents:
        # the alkyne carbon written first in the SMILES bonds the
        # terminal azide nitrogen. The template matches the triple bond
        # both ways round, and the first match is kept, so the isomer
        # is fixed only as long as the fragment's atom order is.
        def build():
            protein = _azf_protein()
            record = protein.attach(
                prepare_fragment("CC(=O)N1Cc2ccccc2C#Cc2ccccc21", "DBC"),
                resnum=5,
                atom_name="N3",
                chain_id="A",
                reaction="azide-alkyne triazole",
                relax=False,
            )
            azf = protein.get_residue(5, chain_id="A")
            partners = {
                atom.name: [
                    other.name
                    for other in protein.root.bond_graph.adj[atom]
                    if other.parent is record.residue2
                ]
                for atom in azf.particles()
                if atom.name in ("N1", "N3")
            }
            return partners, protein.xyz

        first, *others = _under_each_hash(monkeypatch, build)
        # C10 and C11 are the alkyne carbons, in the order of the SMILES.
        assert first[0] == {"N1": ["C11"], "N3": ["C10"]}
        for other in others:
            assert other[0] == first[0]
            np.testing.assert_array_equal(other[1], first[1])

    def test_torsion_search_turns_the_same_bonds(self, monkeypatch):
        # Tests that the placement search picks the same fragment
        # torsions under every hash when more bonds qualify than it
        # turns and several carry as many atoms. Each arm of this
        # fragment offers three bonds that carry five, four and three
        # carbons; four are turned, so one of three equal bonds is
        # chosen. The candidates came from a subgraph view that iterated
        # a set, so the choice, and the pose, followed memory addresses.
        def build():
            protein = Protein(get_fn("6m03_protonated.pdb"))
            cysteine = _free_cysteine(protein)
            record = protein.attach(
                prepare_fragment("[*:1]C(CCCCC)(CCCCC)CCCCC", "TRI"),
                resnum=cysteine.resnum,
                atom_name="SG",
                chain_id="A",
                relax=False,
            )
            sg = protein.get_atom(cysteine.resnum, "SG", chain_id="A")
            carbon = next(
                atom
                for atom in protein.root.bond_graph.adj[sg]
                if atom.parent is record.residue2
            )
            torsions = _placement_torsions(protein.root.bond_graph, sg, carbon, 2, 4)
            return [(a.name, b.name, len(moved)) for a, b, moved in torsions]

        first, *others = _under_each_hash(monkeypatch, build, salts=(1, 2, 3, 4, 5))
        # Four fragment bonds, the bond to the site, and two side-chain
        # bonds (CB-SG and CA-CB).
        assert len(first) == 7
        assert all(other == first for other in others)

    @pytest.mark.skipif(not has_openmm, reason="OpenMM is not installed")
    def test_attach_with_relaxation_is_the_same_in_fresh_processes(self):
        # Tests the whole attach, placement search and minimization
        # included, in fresh processes that differ in their string hash
        # seed and in the objects allocated before the build. This is
        # how the difference showed up for users: two runs of the same
        # script gave structures up to several angstroms apart. The CPU
        # platform is used because a GPU minimization is not
        # reproducible to the last digit even in one process.
        script = (
            "import sys, hashlib\n"
            "padding = [object() for _ in range(int(sys.argv[1]))]\n"
            "from mbuild.biopolymers import Protein, prepare_fragment\n"
            "from mbuild.utils.io import get_fn\n"
            "protein = Protein(get_fn('6m03_protonated.pdb'))\n"
            "cysteine = next(r for r in protein.residues() if r.name == 'CYS'\n"
            "                and any(p.name == 'HG' for p in r.particles()))\n"
            "protein.attach(prepare_fragment('C1=CC(=O)N(C)C1=O', 'MAL'),\n"
            "               resnum=cysteine.resnum, atom_name='SG', chain_id='A',\n"
            "               reaction='thiol-maleimide', platform='CPU')\n"
            "print(hashlib.sha256(protein.xyz.tobytes()).hexdigest())\n"
        )
        digests = set()
        for seed, padding in (("0", 0), ("0", 50_000), ("1", 777)):
            environment = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run(
                [sys.executable, "-c", script, str(padding)],
                capture_output=True,
                text=True,
                env=environment,
                check=True,
            )
            digests.add(result.stdout.strip().splitlines()[-1])
        assert len(digests) == 1
