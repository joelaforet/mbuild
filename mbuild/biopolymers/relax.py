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

#: Nonbonded cutoff, in nm, of the relaxation's generic force field.
_RELAX_CUTOFF = 1.0
#: Distance, in nm, that a mobile atom may move in one relaxation
#: before the neighbourhood it was relaxed in is rebuilt around it.
_NEIGHBOURHOOD_MARGIN = 0.2
#: Most neighbourhoods one relaxation builds.
_NEIGHBOURHOOD_ROUNDS = 3


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


def _relax_particles(
    protein, mobile, n_steps=0, tolerance=50.0, platform=None, particles=None
):
    """Minimize with every particle outside ``mobile`` held fixed.

    This is the minimization behind ``relax_fragments``, which
    moves whole residues, and behind ``attach`` with ``merge=True``,
    which moves atoms that now sit inside a residue whose backbone
    must not move. The parameters are those of ``relax_fragments``;
    ``platform`` None means ``_default_platform()``.

    Only the mobile atoms and the atoms that can act on them enter the
    simulation (``_relax_neighbourhood``), so the cost follows the size
    of the site, not of the protein.

    ``particles`` is the list of the protein's particles. Walking the
    hierarchy for it costs time in proportion to the whole protein, and
    a relaxation moves atoms without adding or removing any, so a caller
    that runs several steps on one protein walks it once and passes the
    list to each (see ``_relax_if_clashing``). None walks it here.
    """
    platform = platform or _default_platform()
    mobile = set(mobile)
    if not mobile:
        return
    if particles is None:
        particles = list(protein.particles())
    # The bonds of the mobile atoms, read from each atom's neighbours
    # rather than from every bond of the protein.
    bonds = {}
    for atom in mobile:
        for neighbor in atom.direct_bonds():
            bonds.setdefault(frozenset((atom, neighbor)), (atom, neighbor))
    bonds = [(a, b, np.linalg.norm(a.pos - b.pos)) for a, b in bonds.values()]
    before = {particle: particle.pos.copy() for particle in mobile}
    # Each round minimizes in the neighbourhood of where the mobile
    # atoms start. A round that moves an atom further than the margin
    # may have brought it within the cutoff of an atom the
    # neighbourhood left out, so it runs again from where it ended.
    for _ in range(_NEIGHBOURHOOD_ROUNDS):
        start = {particle: particle.pos.copy() for particle in mobile}
        _relax_neighbourhood(protein, mobile, n_steps, tolerance, platform, particles)
        moved = max(np.linalg.norm(p.pos - start[p]) for p in mobile)
        if moved <= _NEIGHBOURHOOD_MARGIN:
            break
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


def _relax_neighbourhood(protein, mobile, n_steps, tolerance, platform, particles):
    """Minimize the mobile atoms in a copy of their neighbourhood.

    The copy holds every atom within the nonbonded cutoff of a mobile
    atom, widened by one bond and by ``_NEIGHBOURHOOD_MARGIN``, and the
    bonds among those atoms with their orders. The generic force field
    has bonded terms and a cut-off van der Waals term, and no
    electrostatics, so no atom outside the copy exerts a force on a
    mobile atom. An atom at the edge of the copy can lose a bonded
    partner, which changes the type the force field gives it, but such
    an atom is beyond the cutoff of every mobile atom, so its type
    changes no force that moves anything. Only the mobile atoms'
    positions are written back to the protein.

    The bounded descent and the minimization run in one OpenMM context.
    ``OpenMMSimulation.minimize`` would build a second context for the
    same system, which costs more than the minimization itself at this
    size.
    """
    import openmm
    import openmm.unit as u
    from scipy.spatial import cKDTree

    from mbuild.compound import Compound
    from mbuild.simulation import OpenMMSimulation

    xyz = np.array([particle.pos for particle in particles])
    index = {particle: i for i, particle in enumerate(particles)}
    radius = _RELAX_CUTOFF + _PORT_SEPARATION + _NEIGHBOURHOOD_MARGIN
    near = set()
    for hits in cKDTree(xyz).query_ball_point(
        xyz[[index[particle] for particle in mobile]], radius
    ):
        near.update(hits)
    copies = {}
    for i in sorted(near):
        particle = particles[i]
        copies[particle] = Compound(
            name=particle.name, element=particle.element, pos=xyz[i]
        )
    site = Compound(name="Neighbourhood")
    site.add(list(copies.values()))
    # Each bond is read from the adjacency of its first copied atom, so
    # only the bonds of the neighbourhood are visited.
    graph = protein.root.bond_graph
    for a in copies:
        for b, data in graph.adj[a].items():
            if b in copies and index[a] < index[b]:
                site.add_bond((copies[a], copies[b]), bond_order=data.get("bond_order"))
    # The simulation needs a box: without one there is no cutoff, and
    # every force call visits every pair of atoms. The box holds the
    # whole neighbourhood with room for the cutoff on every side, so
    # that no periodic image comes near any atom.
    extent = site.xyz.max(axis=0) - site.xyz.min(axis=0)
    site.box = Box(lengths=extent + 3.0)
    simulation = OpenMMSimulation(
        site, forcefield=None, kick=False, platform=platform, r_cut=_RELAX_CUTOFF
    )
    moving = {copies[particle] for particle in mobile}
    for i, copy in enumerate(site.particles()):
        if copy not in moving:
            simulation.system.setParticleMass(i, 0.0)
    _descend_in_bounded_steps(simulation, moving)
    context = simulation.simulation.context
    openmm.LocalEnergyMinimizer.minimize(
        context, tolerance * u.kilojoule_per_mole / u.nanometer, n_steps
    )
    positions = (
        context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(u.nanometer)
    )
    if np.isnan(positions).any():
        logger.warning("The relaxation produced NaN positions; none were kept.")
        return
    # Writing back only the mobile atoms also keeps the fixed atoms at
    # their coordinates to the last digit, which a GPU platform, with
    # its single-precision positions, would otherwise round.
    order = {copy: i for i, copy in enumerate(site.particles())}
    for particle in mobile:
        particle.pos = positions[order[copies[particle]]]


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


def _relax_until_bonded(
    protein, mobile, bonds, attempts=3, longest=0.18, platform=None, particles=None
):
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
    platform : str, optional
        OpenMM platform name; None means ``_default_platform()``.
    particles : list of mbuild.Compound, optional
        The protein's particles (see ``_relax_particles``).
    """
    if particles is None:
        particles = list(protein.particles())

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
    mobile = list(mobile) + _side_chains_near(protein, mobile, particles=particles)
    for _ in range(attempts):
        _relax_particles(
            protein, mobile, tolerance=1.0, platform=platform, particles=particles
        )
        if not open_bonds():
            return
    for name1, name2, length in open_bonds():
        logger.warning(
            f"The bond {name1}-{name2} formed by the reaction is "
            f"{length * 10:.2f} A long after relaxation. Inspect the site, "
            "or call relax_fragments() again."
        )


def _relax_if_clashing(protein, added, site_atom, frag_atom, relax, platform=None):
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
    platform : str, optional
        OpenMM platform name; None means ``_default_platform()``.
    """
    # The particles are listed once for every step below: the steps
    # move atoms but add or remove none.
    particles = list(protein.particles())
    clashes = _warn_on_clashes(
        protein, added, site_atom, frag_atom, particles=particles
    )
    if not (clashes and relax):
        return
    # Port alignment fixes the fragment up to a turn about the new
    # bond. Before the minimizer sees the overlap, turn and shift
    # the fragment rigidly to the pose that keeps the bond at length
    # and the fragment clear of the protein. A minimizer started from
    # atoms 0.3 A apart tears bonds; from a clear pose it converges.
    # A floppy fragment may have no clear rigid pose in the
    # conformer it arrived in, so other conformers are tried too.
    _fit_placement(protein, added, [(site_atom, frag_atom, 1.0)], particles=particles)
    if _closest_contact(protein, added, site_atom, frag_atom, particles) < 0.1:
        _try_conformers(protein, added, site_atom, frag_atom, particles=particles)
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
    _relax_until_bonded(
        protein,
        added,
        [(site_atom, frag_atom, 1.0)],
        platform=platform,
        particles=particles,
    )
    # The user asked for the relaxation and got it, so the result is
    # reported rather than warned about. Ordinary van der Waals
    # contacts near 2 A are expected after a minimization; only a
    # contact that stays well inside that means the minimizer failed.
    closest = _closest_contact(protein, added, site_atom, frag_atom, particles)
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


def _fit_placement(protein, added, bonds, particles=None):
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
    particles : list of mbuild.Compound, optional
        The protein's particles (see ``_relax_particles``).
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    everything = list(protein.particles()) if particles is None else particles
    particles = list(added)
    fragment = set(particles)
    index = {particle: i for i, particle in enumerate(particles)}
    pairs = [(a, index[b]) if b in index else (b, index[a]) for a, b, _ in bonds]
    bonded = {fixed for fixed, _ in pairs}
    others = [p for p in everything if p not in fragment and p not in bonded]
    tree = cKDTree([p.pos for p in others]) if others else None
    start = np.array([particle.pos for particle in particles])
    centre = start.mean(axis=0)
    site_atom, frag_index = pairs[0]
    axis = start[frag_index] - site_atom.pos
    axis = axis / np.linalg.norm(axis)

    anchors = np.array([fixed.pos for fixed, _ in pairs])
    bonding = np.array([i for _, i in pairs])

    def posed(x):
        return Rotation.from_rotvec(x[:3]).apply(start - centre) + centre + x[3:]

    def cost(x):
        positions = posed(x)
        # The bond term weighs a hundred times the clash term: a pose
        # that trades bond length for clearance is not a placement,
        # since the minimizer that follows only removes overlap.
        stretch = (
            np.linalg.norm(anchors - positions[bonding], axis=1) - _PORT_SEPARATION
        )
        total = 100 * stretch @ stretch
        if tree is not None:
            # Only contacts closer than 0.25 nm count, so the search
            # stops there; atoms with no such contact come back as inf.
            distances, _ = tree.query(positions, distance_upper_bound=0.25)
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
        # The cost is never negative, and it is zero for a pose with
        # every bond at length and no contact inside 0.25 nm. No other
        # start can improve on such a pose, so the search stops there.
        if best.fun < 1e-6:
            break
    for particle, position in zip(particles, posed(best.x)):
        particle.pos = position


def _try_conformers(protein, added, site_atom, frag_atom, count=8, particles=None):
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
    particles : list of mbuild.Compound, optional
        The protein's particles (see ``_relax_particles``).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    everything = list(protein.particles()) if particles is None else particles
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
    best_clearance = _closest_contact(protein, added, site_atom, frag_atom, everything)
    for conformer in conformers:
        positions = np.array(mol.GetConformer(conformer).GetPositions()) / 10.0
        # Put the bonding atom where it is now, then fit the rest.
        positions += frag_atom.pos - positions[index[frag_atom]]
        for particle, position in zip(particles, positions):
            particle.pos = position
        _fit_placement(
            protein, added, [(site_atom, frag_atom, 1.0)], particles=everything
        )
        clearance = _closest_contact(protein, added, site_atom, frag_atom, everything)
        if clearance > best_clearance:
            best_clearance = clearance
            best = [particle.pos.copy() for particle in particles]
    for particle, position in zip(particles, best):
        particle.pos = position
    logger.info(
        f"Tried {len(conformers)} conformers of the placed fragment; the "
        f"clearest sits {best_clearance * 10:.2f} A from the protein."
    )


def _side_chains_near(protein, placed, radius=0.4, particles=None):
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
    particles : list of mbuild.Compound, optional
        The protein's particles (see ``_relax_particles``).
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
    if particles is None:
        particles = list(protein.particles())
    others = [p for p in particles if p not in placed and p.name not in backbone]
    if not others or not placed:
        return []
    tree = cKDTree([p.pos for p in others])
    near = set()
    for hits in tree.query_ball_point([p.pos for p in placed], radius):
        near.update(hits)
    return [others[i] for i in sorted(near)]


def _closest_contact(protein, added, site_atom, frag_atom, particles=None):
    """Return the smallest distance (nm) from a placed atom to any other atom.

    ``added`` lists the placed particles. The new bond pair is left
    out. Returns None when there is nothing to compare against.
    ``particles`` is the protein's particle list (see
    ``_relax_particles``); None walks the protein for it.
    """
    from scipy.spatial import cKDTree

    added_particles = [p for p in added if p is not frag_atom]
    added_set = set(added) | {site_atom}
    if particles is None:
        particles = list(protein.particles())
    others = [p for p in particles if p not in added_set]
    if not others or not added_particles:
        return None
    distances, _ = cKDTree([p.pos for p in others]).query(
        [p.pos for p in added_particles]
    )
    return float(distances.min())


def _warn_on_clashes(protein, added, site_atom, frag_atom, cutoff=0.2, particles=None):
    """Warn when placed atoms overlap the rest of the system.

    Port alignment is rigid; a bulky fragment can land inside the
    protein. The check compares every particle in ``added`` against
    every other atom, and it leaves out the new bond pair. It warns
    below ``cutoff`` nm, so the user knows to relax the structure
    before simulating. ``particles`` is the protein's particle list
    (see ``_relax_particles``); None walks the protein for it.
    """
    from scipy.spatial import cKDTree

    added_particles = list(added)
    added_set = set(added_particles) | {site_atom}
    if particles is None:
        particles = list(protein.particles())
    others = [p for p in particles if p not in added_set]
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
