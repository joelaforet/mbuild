"""Place atoms added to a protein and relax them without tearing bonds.

``Protein.attach`` and ``Protein.mutate`` align a fragment rigidly along
the bond it forms, which can leave it inside the protein. The functions
here move it clear: a rigid fit that keeps every formed bond at bond
length (``_fit_placement``), other conformers when the fragment has no
clear rigid pose (``_try_conformers``), and a minimization with mBuild's
generic force field in which only the placed atoms and the side chains
near them move (``_relax_particles``). The minimization never returns a
torn fragment: any bond it stretched puts the positions back.
"""

import logging
from functools import lru_cache

import numpy as np

from mbuild.biopolymers.residue import _rdkit_mol
from mbuild.box import Box

logger = logging.getLogger(__name__)

#: Length, in nm, of the bond that a new Port forms. It is a rounded
#: value near the single-bond lengths that ``attach`` and
#: ``add_port_at`` form; mBuild's ``Polymer.from_big_smiles`` uses
#: 0.145 nm for a C-C bond. Each port of a bond takes half of it.
#: This follows the ``Polymer.add_monomer`` convention. Relaxation
#: corrects the length afterwards (see ``relax_fragments``).
_PORT_SEPARATION = 0.15


#: Length of a new X-H bond, in nm, keyed by the element symbol of the
#: heavy atom X. The values are the standard single-bond lengths: N-H
#: 1.01 A, O-H 0.96 A, S-H 1.34 A. Any other element gets
#: ``_PROTON_BOND_LENGTH``, which is a rounded value near the three
#: lengths.
_PROTON_BOND_LENGTHS = {"N": 0.101, "O": 0.096, "S": 0.134}
_PROTON_BOND_LENGTH = 0.100


def _unit(vector):
    """Return the vector scaled to length one, or unchanged if it is zero."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else vector


def _perpendicular(vector):
    """Return a unit vector perpendicular to ``vector``."""
    axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(_unit(vector), axis)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(vector, axis))


def _proton_position(atom):
    """Return a position for a proton added to ``atom``.

    The proton goes in the most open direction at the atom, which is
    the reverse of the sum of the unit vectors to its bonded neighbors
    (``_open_direction``).
    An atom with one neighbor has no such direction, because every
    direction around that one bond is equally open. The proton then
    goes at 109.47 degrees from the bond, which is the tetrahedral
    angle. The direction lies in the plane of the bond and one atom
    bonded to the neighbor. It points to the far side of the bond from
    that atom. A hydroxyl placed this way is anti to that atom.

    The result is a starting geometry. The bond length is correct and
    the angle is a standard value, but the position ignores every other
    atom. Run ``relax_fragments`` or an energy minimization to set the
    exact angle and torsion.

    This function computes the position. It does not use a ``Port``
    and ``force_overlap``. ``force_overlap`` superposes a port of a
    fragment on a port of the target and moves the whole fragment. The
    added proton is one particle, and it holds no port. No fragment
    moves, and no port is superposed. The direction comes from the
    atoms already bonded to ``atom``. This function reads those atoms
    directly.

    Parameters
    ----------
    atom : mbuild.Compound
        The heavy atom that takes the proton. It must carry at least
        one bond.

    Returns
    -------
    numpy.ndarray
        The position of the proton, in nm.
    """
    length = _PROTON_BOND_LENGTHS.get(atom.element.symbol, _PROTON_BOND_LENGTH)
    neighbors = sorted(atom.direct_bonds(), key=lambda particle: particle.name)
    if len(neighbors) == 1:
        bond = _unit(neighbors[0].pos - atom.pos)
        far = sorted(
            (
                particle
                for particle in neighbors[0].direct_bonds()
                if particle is not atom
            ),
            key=lambda particle: particle.name,
        )
        across = _perpendicular(bond)
        if far:
            reference = far[0].pos - neighbors[0].pos
            in_plane = reference - np.dot(reference, bond) * bond
            if np.linalg.norm(in_plane) > 1e-8:
                across = _unit(in_plane)
        # cos(109.47 degrees) = -1/3 and sin(109.47 degrees) = sqrt(8)/3.
        direction = -bond / 3.0 - across * np.sqrt(8.0) / 3.0
    else:
        direction = _open_direction(atom)
    return atom.pos + _unit(direction) * length


@lru_cache(maxsize=1)
def _default_platform():
    """Return ``"CUDA"`` when OpenMM can run on a GPU here, else ``"CPU"``.

    OpenMM lists the CUDA platform whenever its plugin loads, which
    says nothing about the driver, so the check builds a one-particle
    context on it. A minimization of a whole protein takes minutes on
    the CPU and seconds on a GPU, so the GPU is used whenever that
    context can be made. The answer is cached for the process.
    """
    import openmm

    try:
        system = openmm.System()
        system.addParticle(1.0)
        openmm.Context(
            system,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("CUDA"),
        )
    except Exception:  # noqa: BLE001 - OpenMM raises its own exception types
        return "CPU"
    return "CUDA"


def _relax_particles(protein, mobile, n_steps=0, tolerance=50.0, platform=None):
    """Minimize with every particle outside ``mobile`` held fixed.

    This is the minimization behind ``relax_fragments``, which
    moves whole residues, and behind ``attach`` with ``merge=True``
    and ``mutate``, which move atoms inside a residue whose backbone
    must not move. The parameters are those of ``relax_fragments``;
    ``platform`` None means ``_default_platform()``.
    """
    from mbuild.simulation import OpenMMSimulation

    platform = platform or _default_platform()
    mobile = set(mobile)
    bonds = [
        (a, b, np.linalg.norm(a.pos - b.pos))
        for a, b in protein.bonds()
        if a in mobile or b in mobile
    ]
    before = {particle: particle.pos.copy() for particle in mobile}
    # The box of a loaded protein is the crystal cell of the input
    # file, which is metadata, not a simulation box: a protein that
    # is longer than its cell has periodic images on top of itself.
    # The relaxation therefore runs in a box that holds the whole
    # structure with room for the cutoff on every side, so that no
    # image comes near any atom, and the cell is put back afterwards.
    # A box is kept, rather than none, because a system without one
    # has no cutoff and every force call visits every pair of atoms.
    box = protein.box
    particles = list(protein.particles())
    fixed = np.array([particle not in mobile for particle in particles])
    original = protein.xyz
    extent = original.max(axis=0) - original.min(axis=0)
    protein.box = Box(lengths=extent + 3.0)
    try:
        simulation = OpenMMSimulation(
            protein, forcefield=None, kick=False, platform=platform
        )
        for index, particle in enumerate(protein.particles()):
            if particle not in mobile:
                simulation.system.setParticleMass(index, 0.0)
        _descend_in_bounded_steps(simulation, mobile)
        simulation.minimize(n_steps=n_steps, tolerance=tolerance)
    finally:
        protein.box = box
    # The simulation writes every position back, and a GPU platform
    # returns them in single precision. The fixed atoms did not move, so
    # they keep the coordinates they had, to the last digit.
    relaxed = protein.xyz
    relaxed[fixed] = original[fixed]
    protein.xyz = relaxed
    # The generic force field can tear a molecule apart when the
    # start is bad enough: the repulsion between overlapping atoms
    # then outweighs every bond term. A torn fragment must never be
    # returned as if it were relaxed. Any bond that ended more than
    # half again its starting length puts the positions back and
    # warns, so the caller knows the site needs another look.
    torn = [
        (a.name, b.name, np.linalg.norm(a.pos - b.pos) * 10)
        for a, b, length in bonds
        if np.linalg.norm(a.pos - b.pos) > 1.5 * max(length, 0.1)
    ]
    if torn:
        for particle, position in before.items():
            particle.pos = position
        worst = max(torn, key=lambda t: t[2])
        logger.warning(
            f"Relaxation stretched {len(torn)} bonds beyond recognition "
            f"(worst {worst[0]}-{worst[1]} at {worst[2]:.1f} A), so the "
            "positions before it were kept. The placed atoms overlap the "
            "protein too badly for the generic force field; choose another "
            "site or fragment conformer, or relax with a real force field."
        )


def _descend_in_bounded_steps(simulation, mobile, max_step=0.005, rounds=200):
    """Walk the mobile atoms down the force with a capped step per round.

    A rigidly placed fragment can leave atoms a fraction of an
    angstrom from the protein, where the repulsion is enormous. A
    line-search minimizer started there takes a huge first step and
    stretches bonds instead of untangling atoms. This walk moves
    every mobile atom along its force by at most ``max_step`` nm per
    round, so overlapping atoms slide apart while their bonds hold,
    and stops as soon as the largest force is ordinary. The
    minimizer then finishes from a start it can handle.

    Parameters
    ----------
    simulation : mbuild.simulation.OpenMMSimulation
        The built simulation; its context is created here.
    mobile : set of mbuild.Compound
        The particles that may move.
    max_step : float, optional, default=0.005
        Largest displacement per atom per round, in nm.
    rounds : int, optional, default=200
        Most rounds to take.
    """
    import openmm
    import openmm.unit as u

    simulation._create_simulation(
        openmm.LangevinIntegrator(
            300 * u.kelvin, 1.0 / u.picosecond, 0.001 * u.picoseconds
        )
    )
    context = simulation.simulation.context
    particles = list(simulation.compound.particles())
    moving = np.array([particle in mobile for particle in particles])
    positions = np.array(
        context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(u.nanometer)
    )
    for _ in range(rounds):
        forces = np.array(
            context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(u.kilojoule_per_mole / u.nanometer)
        )
        forces[~moving] = 0.0
        largest = np.linalg.norm(forces, axis=1).max()
        if largest < 5000.0:
            break
        # Scale so that the most-pushed atom moves max_step; others less.
        positions = positions + forces * (max_step / largest)
        context.setPositions(positions * u.nanometer)
    simulation.positions = context.getState(getPositions=True).getPositions()


def _relax_until_bonded(protein, mobile, bonds, attempts=3, longest=0.18):
    """Relax the placed atoms until every formed bond is at bond length.

    Parameters
    ----------
    mobile : list of mbuild.Compound
        The particles that may move.
    bonds : list of (mbuild.Compound, mbuild.Compound, float)
        The formed bonds to check.
    attempts : int, optional, default=3
        How many relaxations to run before warning.
    longest : float, optional, default=0.18
        The bond length, in nm, above which a bond counts as open.
    """

    def open_bonds():
        return [
            (a.name, b.name, np.linalg.norm(a.pos - b.pos))
            for a, b, _ in bonds
            if np.linalg.norm(a.pos - b.pos) > longest
        ]

    # The side chains around the site move too. A fragment bonded to
    # a residue in a groove cannot clear the protein while every
    # protein atom stands still; a real minimization would move
    # those side chains, so this one does. The backbone keeps the
    # coordinates of the file.
    mobile = list(mobile) + _side_chains_near(protein, mobile)
    for _ in range(attempts):
        _relax_particles(protein, mobile, tolerance=1.0)
        if not open_bonds():
            return
    for name1, name2, length in open_bonds():
        logger.warning(
            f"The bond {name1}-{name2} formed by the reaction is "
            f"{length * 10:.2f} A long after relaxation. Inspect the site, "
            "or call relax_fragments() again."
        )


def _relax_if_clashing(protein, added, site_atom, frag_atom, relax):
    """Relax the placed atoms when they overlap other atoms.

    Port alignment is rigid, so a bulky fragment can land inside the
    protein. The relaxation moves only the placed atoms. It needs
    the simulation dependencies; without them the method warns and
    keeps the rigid placement.

    Parameters
    ----------
    added : list of mbuild.Compound
        The placed particles, which relaxation may move.
    site_atom : mbuild.Compound
        The protein atom of the new bond.
    frag_atom : mbuild.Compound
        The placed atom of the new bond.
    relax : bool
        False leaves the rigid placement in place.
    """
    clashes = _warn_on_clashes(protein, added, site_atom, frag_atom)
    if not (clashes and relax):
        return
    # Port alignment fixes the fragment up to a turn about the new
    # bond. Before the minimizer sees the overlap, turn and shift
    # the fragment rigidly to the pose that keeps the bond at length
    # and the fragment clear of the protein. A minimizer started from
    # atoms 0.3 A apart tears bonds; from a clear pose it converges.
    # A floppy fragment may have no clear rigid pose in the
    # conformer it arrived in, so other conformers are tried too.
    _fit_placement(protein, added, [(site_atom, frag_atom, 1.0)])
    if _closest_contact(protein, added, site_atom, frag_atom) < 0.1:
        _try_conformers(protein, added, site_atom, frag_atom)
    try:
        import mbuild.simulation  # noqa: F401
    except ImportError as error:
        # The automatic path only warns: the attachment itself
        # is complete, and the user can relax later on a system
        # with the simulation dependencies installed.
        logger.warning(
            "Cannot relax the placed fragment: mbuild.simulation "
            f"is not importable ({error}). Install the simulation "
            "dependencies (hoomd, openmm) or call "
            "relax_fragments() elsewhere. The fragment keeps its "
            "rigid placement."
        )
        return
    logger.info("Relaxing the placed atoms with the protein backbone held fixed.")
    _relax_until_bonded(protein, added, [(site_atom, frag_atom, 1.0)])
    # The user asked for the relaxation and got it, so the result is
    # reported rather than warned about. Ordinary van der Waals
    # contacts near 2 A are expected after a minimization; only a
    # contact that stays well inside that means the minimizer failed.
    closest = _closest_contact(protein, added, site_atom, frag_atom)
    if closest is None:
        return
    if closest < 0.15:
        logger.warning(
            f"Relaxation left a fragment atom {closest * 10:.2f} A from an "
            "existing atom. Inspect the site; the minimizer may not have "
            "converged."
        )
    else:
        logger.info(
            f"Fragment relaxed; closest contact with existing atoms is now "
            f"{closest * 10:.2f} A."
        )


def _fit_placement(protein, added, bonds):
    """Move the fragment rigidly so that every formed bond can close.

    Port alignment places the fragment for one bond. A reaction that
    forms several bonds between the sides, a ring closure, needs
    the fragment placed for all of them at once, and a terminal
    atom such as the outer nitrogen of an azide gives no bond
    direction that a single alignment could use. The fragment is
    therefore moved as a rigid body, starting from the aligned
    placement and from turns about the first bond, to the pose
    that brings every formed bond to bond length while keeping the
    fragment clear of the protein. Relaxation then refines the
    geometry.

    Parameters
    ----------
    added : list of mbuild.Compound
        The placed particles.
    bonds : list of (mbuild.Compound, mbuild.Compound, float)
        The formed bonds between the protein and the fragment.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    particles = list(added)
    fragment = set(particles)
    index = {particle: i for i, particle in enumerate(particles)}
    pairs = [(a, index[b]) if b in index else (b, index[a]) for a, b, _ in bonds]
    bonded = {fixed for fixed, _ in pairs}
    others = [p for p in protein.particles() if p not in fragment and p not in bonded]
    tree = cKDTree([p.pos for p in others]) if others else None
    start = np.array([particle.pos for particle in particles])
    centre = start.mean(axis=0)
    site_atom, frag_index = pairs[0]
    axis = start[frag_index] - site_atom.pos
    axis = axis / np.linalg.norm(axis)

    def posed(x):
        return Rotation.from_rotvec(x[:3]).apply(start - centre) + centre + x[3:]

    def cost(x):
        positions = posed(x)
        # The bond term weighs a hundred times the clash term: a pose
        # that trades bond length for clearance is not a placement,
        # since the minimizer that follows only removes overlap.
        total = 100 * sum(
            (np.linalg.norm(fixed.pos - positions[i]) - _PORT_SEPARATION) ** 2
            for fixed, i in pairs
        )
        if tree is not None:
            distances, _ = tree.query(positions)
            close = distances[distances < 0.25]
            total += ((0.25 - close) ** 2).sum()
        return total

    best = None
    for angle in np.radians(np.arange(0.0, 360.0, 30.0)):
        # Turn about the first bond, then correct the translation
        # that the turn about the centroid introduced.
        turned = Rotation.from_rotvec(axis * angle)
        shift = turned.apply(start[frag_index] - centre) + centre - start[frag_index]
        guess = np.concatenate([axis * angle, -shift])
        result = minimize(
            cost, guess, method="Powell", options={"xtol": 1e-4, "ftol": 1e-8}
        )
        if best is None or result.fun < best.fun:
            best = result
    for particle, position in zip(particles, posed(best.x)):
        particle.pos = position


def _try_conformers(protein, added, site_atom, frag_atom, count=8):
    """Re-embed the placed atoms in other conformers and keep the clearest.

    The conformer a fragment arrives in is one of many, and a long
    flexible one may cross the protein in every rigid pose. This
    builds an RDKit molecule from the placed atoms and their bonds,
    embeds ``count`` conformers, fits each rigidly with the bond
    kept at length, and keeps the pose whose closest contact with
    the protein is largest. The original pose competes on the same
    terms. Nothing is done when the molecule cannot be embedded.

    Parameters
    ----------
    protein : Protein
        The protein that owns the placed atoms.
    added : list of mbuild.Compound
        The placed particles.
    site_atom, frag_atom : mbuild.Compound
        The two atoms of the new bond.
    count : int, optional, default=8
        Conformers to try.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    particles = list(added)
    editable, index = _rdkit_mol(protein, particles)
    mol = editable.GetMol()
    try:
        Chem.SanitizeMol(mol)
        conformers = list(AllChem.EmbedMultipleConfs(mol, numConfs=count, randomSeed=0))
    except Exception:  # noqa: BLE001 - RDKit raises several unrelated types
        conformers = []
    if not conformers:
        return
    best = [particle.pos.copy() for particle in particles]
    best_clearance = _closest_contact(protein, added, site_atom, frag_atom)
    for conformer in conformers:
        positions = np.array(mol.GetConformer(conformer).GetPositions()) / 10.0
        # Put the bonding atom where it is now, then fit the rest.
        positions += frag_atom.pos - positions[index[frag_atom]]
        for particle, position in zip(particles, positions):
            particle.pos = position
        _fit_placement(protein, added, [(site_atom, frag_atom, 1.0)])
        clearance = _closest_contact(protein, added, site_atom, frag_atom)
        if clearance > best_clearance:
            best_clearance = clearance
            best = [particle.pos.copy() for particle in particles]
    for particle, position in zip(particles, best):
        particle.pos = position
    logger.info(
        f"Tried {len(conformers)} conformers of the placed fragment; the "
        f"clearest sits {best_clearance * 10:.2f} A from the protein."
    )


def _side_chains_near(protein, placed, radius=0.4):
    """Return the side-chain atoms of the protein within ``radius`` of placed atoms.

    Backbone atoms (``N``, ``CA``, ``C``, ``O`` and their hydrogens)
    are never returned, so a relaxation that frees these atoms keeps
    the backbone where the file put it.

    Parameters
    ----------
    placed : iterable of mbuild.Compound
        The atoms the relaxation is about.
    radius : float, optional, default=0.4
        Distance in nm.
    """
    from scipy.spatial import cKDTree

    placed = set(placed)
    backbone = {
        "N",
        "CA",
        "C",
        "O",
        "H",
        "H2",
        "H3",
        "HA",
        "HA2",
        "HA3",
        "OXT",
        "HXT",
    }
    others = [
        p for p in protein.particles() if p not in placed and p.name not in backbone
    ]
    if not others or not placed:
        return []
    tree = cKDTree([p.pos for p in others])
    near = set()
    for hits in tree.query_ball_point([p.pos for p in placed], radius):
        near.update(hits)
    return [others[i] for i in sorted(near)]


def _closest_contact(protein, added, site_atom, frag_atom):
    """Return the smallest distance (nm) from a placed atom to any other atom.

    ``added`` lists the placed particles. The new bond pair is left
    out. Returns None when there is nothing to compare against.
    """
    from scipy.spatial import cKDTree

    added_particles = [p for p in added if p is not frag_atom]
    added_set = set(added) | {site_atom}
    others = [p for p in protein.particles() if p not in added_set]
    if not others or not added_particles:
        return None
    distances, _ = cKDTree([p.pos for p in others]).query(
        [p.pos for p in added_particles]
    )
    return float(distances.min())


def _warn_on_clashes(protein, added, site_atom, frag_atom, cutoff=0.2):
    """Warn when placed atoms overlap the rest of the system.

    Port alignment is rigid; a bulky fragment can land inside the
    protein. The check compares every particle in ``added`` against
    every other atom, and it leaves out the new bond pair. It warns
    below ``cutoff`` nm, so the user knows to relax the structure
    before simulating.
    """
    from scipy.spatial import cKDTree

    added_particles = list(added)
    added_set = set(added_particles) | {site_atom}
    others = [p for p in protein.particles() if p not in added_set]
    placed = [p.pos for p in added_particles if p is not frag_atom]
    if not others or not placed:
        return 0
    tree = cKDTree([p.pos for p in others])
    distances, _ = tree.query(placed)
    n_clashes = int((distances < cutoff).sum())
    if n_clashes:
        logger.warning(
            f"{n_clashes} placed atoms sit within "
            f"{cutoff * 10:.1f} A of existing atoms (closest: "
            f"{distances.min() * 10:.2f} A). Relax the structure before "
            "simulating (e.g. relax_fragments(), which holds the "
            "protein fixed)."
        )
    return n_clashes


def _open_direction(atom):
    """Return a unit vector into the open coordination site of an atom.

    It points away from the mean of the unit vectors along the atom's
    bonds. Unit vectors, not bond vectors, so that a long and a short
    bond on a linear atom do not leave a spurious direction along the
    axis. When that mean vanishes, the atom is linear or trigonal
    planar, and the open site is perpendicular: to the plane of a
    planar atom, which is where an addition to an alkene carbon goes,
    or to the axis of a linear atom such as an alkyne carbon. It is
    where a new bond or a new hydrogen goes when no leaving atom shows
    the way.
    """
    vectors = []
    for neighbour in atom.direct_bonds():
        vector = neighbour.pos - atom.pos
        vectors.append(vector / np.linalg.norm(vector))
    if not vectors:
        return np.array([1.0, 0.0, 0.0])
    total = -np.sum(vectors, axis=0)
    if np.linalg.norm(total) > 0.3:
        return total / np.linalg.norm(total)
    normal = np.cross(vectors[0], vectors[1]) if len(vectors) > 1 else np.zeros(3)
    if np.linalg.norm(normal) < 1e-3:
        # Linear: any perpendicular to the axis.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(vectors[0], axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        normal = np.cross(vectors[0], axis)
    return normal / np.linalg.norm(normal)
