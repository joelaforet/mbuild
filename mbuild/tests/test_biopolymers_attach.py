"""Tests for building fragments and attaching them covalently."""

import logging
from pathlib import Path

import numpy as np
import pytest

import mbuild as mb
from mbuild.biopolymers import (
    Protein,
    fragment_from_ccd,
    fragment_from_pdb,
    prepare_fragment,
)
from mbuild.exceptions import MBuildError
from mbuild.tests.base_test import BaseTest
from mbuild.utils.io import get_fn, has_hoomd, has_openmm, has_rdkit


def _chain_id(residue):
    """Chain identifier of a residue, read from its Chain parent."""
    parent = residue.parent
    while parent is not None and not hasattr(parent, "chain_id"):
        parent = parent.parent
    return parent.chain_id


class TestProteinModify(BaseTest):
    @pytest.fixture(scope="class")
    def _protein_6m03_cached(self):
        """Load the protein once per class; tests clone it.

        The load is the slow part, and several tests mutate the protein,
        so each test gets its own clone of one cached load rather than
        sharing one object or re-reading the file.
        """
        return Protein(get_fn("6m03_protonated.pdb"))

    @pytest.fixture
    def protein_6m03(self, _protein_6m03_cached):
        return mb.clone(_protein_6m03_cached)

    @pytest.fixture
    def acetone(self):
        return mb.load("CC(C)=O", smiles=True)

    def test_attach_merge_joins_the_site_residue(self, protein_6m03, acetone):
        # Tests that merge=True adds the fragment's atoms to the site
        # residue instead of a new residue: the residue count is
        # unchanged, the atoms carry names unique in the residue, the
        # bond is made, no record is written, and the residue is
        # returned. This is needed for products that a residue library
        # must describe as one component. A fragment of several
        # residues is refused.
        protein = protein_6m03
        lys = protein.get_residue(5, chain_id="A")
        n_atoms, n_residues = lys.n_particles, len(list(protein.residues()))
        result = protein.attach(
            acetone,
            "C1",
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            merge=True,
            relax=False,
        )
        assert result is lys
        assert len(list(protein.residues())) == n_residues
        assert lys.n_particles == n_atoms - 1 + 9  # one H left, acetone lost one
        names = [p.name for p in lys.particles()]
        assert len(names) == len(set(names))
        assert lys.template is None
        assert not [b for b in protein.cross_bonds if b.residue1 is lys]
        nz = protein.get_atom(5, "NZ", chain_id="A")
        assert any(
            p.parent is lys
            and p.name != "CE"
            and p.element.symbol == "C"
            and p.name not in ("CA", "C", "CB", "CG", "CD")
            for p in nz.direct_bonds()
        )

    def test_attach(self, protein_6m03, acetone):
        # Tests that attach() substitutes one hydrogen on each side,
        # bonds the named atoms, adds the fragment as its own HETATM
        # residue, and records the bond with its leaving hydrogens. This is needed because the recorded
        # bond is exactly what Pablo's with_crosslink needs to load the
        # modified protein. The test attaches an acetone-derived
        # fragment at LYS 5 NZ and checks topology, count, and record;
        # it also checks that an atom without hydrogens is rejected.
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
    def test_attach_names_the_fragment_link_atom(self, protein_6m03):
        # Tests that fragment_atom_name selects which atom of the
        # fragment forms the bond, and that the bond record names that
        # atom. This is needed because the caller reads the fragment
        # names from prepare_fragment and then picks one, so a lookup
        # that took the first name instead would bond the wrong atom
        # and write a record that does not describe the structure. The
        # test prepares the fragment [*:1]CCO, whose C1 and O1 atoms
        # differ, attaches it at LYS 12 NZ through O1, and reads the
        # record and the new bond.
        from mbuild.biopolymers import prepare_fragment

        protein = protein_6m03
        fragment = prepare_fragment("[*:1]CCO", "ETH")
        record = protein.attach(
            fragment, "O1", resnum=12, atom_name="NZ", chain_id="A", relax=False
        )
        assert record.atom2_name == "O1"
        nz = protein.get_atom(12, "NZ", chain_id="A")
        oxygen = protein.get_atom(307, "O1", chain_id="A")
        assert protein.bond_graph.has_edge(nz, oxygen)

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_needs_fragment_resnum_for_a_repeated_name(self, protein_6m03):
        # Tests that attach() refuses a fragment atom name that more
        # than one residue of the fragment carries, and that
        # fragment_resnum then selects the residue. This is needed
        # because the same atom name repeats in a multi-residue
        # fragment, so a first-hit lookup would bond the wrong residue.
        # The test builds a fragment of
        # two ethyl residues, both of which hold an atom named C1, and
        # attaches it at LYS 12 NZ without and with fragment_resnum.
        from mbuild.biopolymers import fragment_from_smiles

        first = fragment_from_smiles("[*:1]CC", "ET1")
        second = fragment_from_smiles("[*:1]CC", "ET2")
        second.resnum = 2
        fragment = mb.Compound(name="LNK")
        fragment.add(first)
        fragment.add(second)

        protein = protein_6m03
        with pytest.raises(MBuildError, match="pass fragment_resnum"):
            protein.attach(
                fragment, "C1", resnum=12, atom_name="NZ", chain_id="A", relax=False
            )
        record = protein.attach(
            fragment,
            "C1",
            resnum=12,
            atom_name="NZ",
            chain_id="A",
            fragment_resnum=2,
            relax=False,
        )
        assert record.residue2.name == "ET2"

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
        # relax call; this is the only test of that call. It passes
        # platform="CPU" because a downstream consumer chooses the
        # OpenMM platform, and the name reaches
        # Platform.getPlatformByName. The test attaches a bulky fragment
        # with relax=False, counts fragment atoms within the 0.2 nm
        # clash cutoff of protein atoms before and after
        # relax_fragments(), and asserts the count decreased while the
        # protein coordinates did not change.
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
        protein.relax_fragments(n_steps=50, platform="CPU")
        assert clash_count() < before
        assert np.allclose([p.pos for p in others], protein_positions)

    def test_add_port_at(self, protein_6m03):
        # Tests add_port_at, the low-level alternative to attach(). It
        # removes bond_order hydrogens from the named atom and returns a
        # standard mBuild Port there, so a user can place a compound with
        # force_overlap. This is needed because a port at the wrong anchor,
        # or a wrong hydrogen count, gives a wrong molecule. The test
        # creates a port at LYS 12 NZ, checks the anchor and the remaining
        # hydrogens, and passes an invalid bond order.
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

    def test_record_bond_names_a_forced_bond(self, protein_6m03):
        # Tests record_bond, which names the bond that force_overlap
        # formed. This is needed because that path writes no record, so
        # bond_records() reported nothing and a downstream loader could
        # not learn about the modification. The test opens a port at
        # LYS 12 NZ, bonds a methyl fragment onto the port, and reads
        # back the record; its leaving atom must be the hydrogen that
        # add_port_at removed.
        from mbuild.biopolymers.fragments import prepare_fragment
        from mbuild.coordinate_transform import force_overlap
        from mbuild.lib.moieties import CH3

        protein = protein_6m03
        port = protein.add_port_at(12, "NZ", chain_id="A")
        nz = protein.get_atom(12, "NZ", chain_id="A")
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

    def test_two_ports_at_one_atom_accumulate(self, protein_6m03):
        # Tests that two ports at one atom accumulate in the
        # leaving-atom ledger. This is needed because a residue can
        # carry more than one modification at the same atom, and a
        # record that names only the last hydrogen tells a downstream
        # loader that the other hydrogen is still there. The test opens
        # two ports at LYS 12 NZ, bonds one methyl fragment to each,
        # and reads the leaving atoms of the second record.
        from mbuild.biopolymers.fragments import prepare_fragment
        from mbuild.coordinate_transform import force_overlap
        from mbuild.lib.moieties import CH3

        protein = protein_6m03
        nz = protein.get_atom(12, "NZ", chain_id="A")
        for resnum, resname in ((400, "MET"), (401, "ME2")):
            port = protein.add_port_at(12, "NZ", chain_id="A")
            fragment = prepare_fragment(CH3(), resname)
            fragment.resnum = resnum
            next(iter(protein.chains)).add(fragment)
            force_overlap(
                move_this=fragment,
                from_positions=fragment.all_ports()[0],
                to_positions=port,
                add_bond=True,
            )
            protein.record_bond(nz, protein.get_atom(resnum, "C1", chain_id="A"))
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
    def test_leaving_atom_names_choose_which_hydrogens_go(self, protein_6m03):
        # Tests that the caller can name the hydrogens that leave on
        # each side, and that the record reports those names. This is
        # needed because the hydrogens on one atom are chemically
        # equivalent, so the default alphabetical choice is arbitrary,
        # but a downstream residue library describes the product by
        # naming the atom that is absent. A file whose missing hydrogen
        # is not the one that library names does not load.
        fragment = prepare_fragment("*C(=O)C", "AC2")
        default = protein_6m03.attach(
            fragment, resnum=5, atom_name="NZ", chain_id="A", relax=False
        )
        assert default.leaving1 == ("HZ1",)

        chosen = Protein(get_fn("6m03_protonated.pdb")).attach(
            fragment,
            resnum=5,
            atom_name="NZ",
            chain_id="A",
            relax=False,
            leaving_atom_names="HZ2",
        )
        assert chosen.leaving1 == ("HZ2",)

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_leaving_atom_names_are_checked(self, protein_6m03):
        # Tests that a leaving-atom name that is not a bonded hydrogen,
        # and a count that does not match the bond order, both raise and
        # name what is available. This is needed because a silently
        # ignored name would write a file that the downstream library
        # rejects for a reason far from the call that caused it.
        with pytest.raises(MBuildError, match="no bonded atom named"):
            protein_6m03.attach(
                prepare_fragment("*C(=O)C", "AC2"),
                resnum=5,
                atom_name="NZ",
                chain_id="A",
                relax=False,
                leaving_atom_names="HZ9",
            )
        with pytest.raises(MBuildError, match="leaving-atom names were given"):
            protein_6m03.attach(
                prepare_fragment("*C(=O)C", "AC2"),
                resnum=5,
                atom_name="NZ",
                chain_id="A",
                relax=False,
                leaving_atom_names=["HZ1", "HZ2"],
            )


class TestFragments(BaseTest):
    @pytest.fixture(scope="class")
    def _protein_6m03_cached(self):
        """Load the protein once per class; tests clone it.

        The load is the slow part, and several tests mutate the protein,
        so each test gets its own clone of one cached load rather than
        sharing one object or re-reading the file.
        """
        return Protein(get_fn("6m03_protonated.pdb"))

    @pytest.fixture
    def protein_6m03(self, _protein_6m03_cached):
        return mb.clone(_protein_6m03_cached)

    @pytest.fixture
    def acetone(self):
        return mb.load("CC(C)=O", smiles=True)

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

    def test_fragment_from_ccd_uses_the_component_names_and_geometry(
        self, protein_6m03
    ):
        # Alanine is one of the shipped components, so no download. The
        # fragment must carry the CCD atom names, the template's bonds
        # with orders, ideal coordinates, and the link atom the caller
        # named. Attaching it needs no fragment_atom_name.
        from mbuild.biopolymers import CCDLibrary

        alanine = fragment_from_ccd("ALA", link_atom="N", library=CCDLibrary())
        assert alanine.name == "ALA" and alanine.hetatm
        names = {particle.name for particle in alanine.particles()}
        assert {"N", "CA", "C", "O", "OXT", "CB", "HB1"} <= names
        orders = sorted(
            d["bond_order"] for *_, d in alanine.bonds(return_bond_order=True)
        )
        assert orders.count(2.0) == 1
        assert alanine.link_atoms == {"1": "N"}
        assert any(abs(p.pos).max() > 0 for p in alanine.particles())
        with pytest.raises(KeyError, match="no atom named"):
            fragment_from_ccd("ALA", link_atom="XX", library=CCDLibrary())
        lys = next(r for r in protein_6m03.residues() if r.name == "LYS")
        record = protein_6m03.attach(
            alanine,
            resnum=lys.resnum,
            atom_name="NZ",
            chain_id=_chain_id(lys),
            relax=False,
        )
        assert record.atom2_name == "N"

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_fragment_from_pdb_keeps_residues_and_perceives_orders(self):
        # A GLYCAM-built disaccharide stub: a hydroxyl residue (ROH) on
        # the anomeric carbon of an N-acetyl sugar (0VA). No template
        # library knows these names, so the loader must keep them as
        # they are, read every bond from the CONECT records, and find
        # the one double bond (the acetamido C=O) from the geometry.
        glycan = fragment_from_pdb(get_fn("glycam_G57321FI.pdb"))
        residues = [(child.name, child.resnum) for child in glycan.children]
        assert residues == [("ROH", 1), ("0VA", 2)]
        assert glycan.n_particles == 30
        orders = sorted(
            data["bond_order"] for *_, data in glycan.bonds(return_bond_order=True)
        )
        assert orders.count(2.0) == 1 and orders.count(1.0) == 29
        assert all(residue.formal_charge == 0 for residue in glycan.children)
        names = {particle.name for particle in glycan.children[1].particles()}
        assert {"C1", "C2N", "O2N", "H1"} <= names

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_fragment_from_pdb_requires_conect_records(self, tmp_path):
        text = Path(get_fn("glycam_G57321FI.pdb")).read_text()
        stripped = tmp_path / "no_conect.pdb"
        stripped.write_text(
            "".join(
                line for line in text.splitlines(True) if not line.startswith("CONECT")
            )
        )
        with pytest.raises(MBuildError, match="no CONECT records"):
            fragment_from_pdb(stripped)

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_heavy_leaving_atom_removes_its_group(self, protein_6m03):
        # N-glycosylation: the glycan's anomeric carbon C1 loses the
        # hydroxyl residue ROH (O1 and HO1), and the asparagine ND2
        # loses one hydrogen. Naming O1 as the fragment leaving atom
        # must remove O1 and HO1 together, drop the emptied ROH
        # residue, keep the GLYCAM residue name of the sugar, and
        # record every displaced atom.
        glycan = fragment_from_pdb(get_fn("glycam_G57321FI.pdb"))
        asn = next(r for r in protein_6m03.residues() if r.name == "ASN")
        before = protein_6m03.n_particles
        last = max(r.resnum for r in protein_6m03.residues(_chain_id(asn)))
        record = protein_6m03.attach(
            glycan,
            fragment_atom_name="C1",
            fragment_resnum=2,
            resnum=asn.resnum,
            atom_name="ND2",
            chain_id=_chain_id(asn),
            leaving_atom_names="HD22",
            fragment_leaving_atom_names="O1",
            relax=False,
        )
        assert protein_6m03.n_particles == before + 30 - 3
        added = [r for r in protein_6m03.residues() if r.name in ("ROH", "0VA")]
        assert [(r.name, r.resnum) for r in added] == [("0VA", last + 1)]
        assert record.leaving1 == ("HD22",)
        assert record.leaving2 == ("HO1", "O1")
        c1 = protein_6m03.get_atom(last + 1, "C1", chain_id=_chain_id(asn))
        nd2 = protein_6m03.get_atom(asn.resnum, "ND2", chain_id=_chain_id(asn))
        assert nd2 in c1.direct_bonds()
        assert {p.name for p in nd2.direct_bonds() if p.element.symbol == "H"} == {
            "HD21"
        }

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_renamed_site_residue_writes_under_its_new_name(
        self, protein_6m03, tmp_path
    ):
        # GLYCAM06 parameterizes a glycosylated serine as OLS. The
        # rename is the user's one line after attach, and it has to
        # reach the written file and the bond record.
        glycan = fragment_from_pdb(get_fn("glycam_G57321FI.pdb"))
        ser = next(r for r in protein_6m03.residues() if r.name == "SER")
        protein_6m03.attach(
            glycan,
            fragment_atom_name="C1",
            fragment_resnum=2,
            resnum=ser.resnum,
            atom_name="OG",
            chain_id=_chain_id(ser),
            leaving_atom_names="HG",
            fragment_leaving_atom_names="O1",
            relax=False,
        )
        ser.name = "OLS"
        written = tmp_path / "o_glycosylated.pdb"
        protein_6m03.save_pdb(written)
        lines = written.read_text().splitlines()
        names = {
            line[17:20]
            for line in lines
            if line.startswith("ATOM") and int(line[22:26]) == ser.resnum
        }
        assert names == {"OLS"}
        assert protein_6m03.bond_records()[0]["residue_names"] == ("OLS", "0VA")

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_rejects_a_double_bonded_leaving_atom(self, protein_6m03):
        # Cutting the C=O of acetone would free two valence units of the
        # carbon while the new single bond uses one. RDKit accepts the
        # under-valent carbon as a radical, so attach has to refuse.
        acetone = prepare_fragment("CC(=O)C", "ACE")
        carbonyl_c = next(
            p
            for p in acetone.particles()
            if p.element.symbol == "C"
            and any(q.element.symbol == "O" for q in p.direct_bonds())
        )
        oxygen = next(q for q in carbonyl_c.direct_bonds() if q.element.symbol == "O")
        lys = next(r for r in protein_6m03.residues() if r.name == "LYS")
        with pytest.raises(MBuildError, match="bond of order 2"):
            protein_6m03.attach(
                acetone,
                fragment_atom_name=carbonyl_c.name,
                fragment_leaving_atom_names=oxygen.name,
                resnum=lys.resnum,
                atom_name="NZ",
                chain_id=_chain_id(lys),
                relax=False,
            )

    @pytest.mark.skipif(not has_rdkit, reason="RDKit is not installed")
    def test_attach_rejects_a_ring_atom_as_leaving_atom(self, protein_6m03):
        # O5 is in the pyranose ring with C1, so nothing lies on the far
        # side of the C1-O5 bond. The call must refuse instead of
        # removing the rest of the sugar.
        glycan = fragment_from_pdb(get_fn("glycam_G57321FI.pdb"))
        asn = next(r for r in protein_6m03.residues() if r.name == "ASN")
        with pytest.raises(MBuildError, match="in a ring with"):
            protein_6m03.attach(
                glycan,
                fragment_atom_name="C1",
                fragment_resnum=2,
                resnum=asn.resnum,
                atom_name="ND2",
                chain_id=_chain_id(asn),
                fragment_leaving_atom_names="O5",
                relax=False,
            )

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
        # this package does not modify.
        from rdkit.Chem import AllChem  # noqa: F401

        protein = Protein(get_fn("3cu9_vicinal_disulfide.pdb"))
        volume = protein.volume()
        assert isinstance(volume, float) and volume > 0.0
