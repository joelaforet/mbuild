"""Place atoms added to a protein and relax them without tearing bonds.

``Protein.attach`` and ``Protein.mutate`` align a fragment rigidly along
the bond it forms, which can leave it inside the protein. The functions
here move it clear: a search over the torsions next to the new bond
that keeps every bond length and angle as it is (``_place_by_torsions``),
a rigid fit that keeps every formed bond at bond length when several
bonds form (``_fit_placement``), other conformers when the fragment has
no clear pose (``_try_conformers``), and a minimization with mBuild's
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

#: Distance, in nm, below which two atoms of a placed pose clash, and the
#: distance below which a contact adds to the soft placement penalty.
#: 0.17 nm is the van der Waals radius of carbon, the threshold the
#: GlycoShape Re-Glyco placement uses.
_CLASH_DISTANCE = 0.17
_CONTACT_DISTANCE = 0.22


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
    # half again its starting length puts back the positions of the
    # bonded group of mobile atoms it belongs to, and warns, so the
    # caller knows the site needs another look. Only that group goes
    # back: in a relaxation of many fragments at once, the others keep
    # their relaxed positions.
    torn = [
        (a, b, np.linalg.norm(a.pos - b.pos) * 10)
        for a, b, length in bonds
        if np.linalg.norm(a.pos - b.pos) > 1.5 * max(length, 0.1)
    ]
    if torn:
        import networkx as nx

        groups = nx.connected_components(protein.root.bond_graph.subgraph(mobile))
        hit = {atom for a, b, _ in torn for atom in (a, b)}
        restored = [group for group in groups if group & hit]
        for group in restored:
            for particle in group:
                particle.pos = before[particle]
        worst = max(torn, key=lambda t: t[2])
        logger.warning(
            f"Relaxation stretched {len(torn)} bonds beyond recognition "
            f"(worst {worst[0].name}-{worst[1].name} at {worst[2]:.1f} A), so "
            f"{len(restored)} group(s) of mobile atoms were put back where "
            "they started. Those atoms overlap the protein too badly for the "
            "generic force field; choose another site or fragment conformer, "
            "or relax with a real force field."
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


def _relax_if_clashing(
    protein, added, site_atom, frag_atom, relax, platform=None, minimize=True
):
    """Relax the placed atoms when they overlap other atoms.

    Port alignment is rigid, so a bulky fragment can land inside the
    protein. The fragment is first turned clear (``_place_by_torsions``),
    which may also turn the site residue's side chain about its chi
    bonds; the backbone never moves. The relaxation then moves only the
    placed atoms and the side chains near them. It needs the simulation
    dependencies; without them the method warns and keeps the placement.

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
    minimize : bool, optional, default=True
        False stops after the placement and leaves the minimization to
        a later ``relax_fragments`` call, which can then relax many
        fragments in one simulation.
    """
    # The particles are listed once for every step below: the steps
    # move atoms but add or remove none.
    particles = list(protein.particles())
    # The rigid placement is reported only when it is what the caller
    # keeps; otherwise the report waits for the placement below.
    clashes = _warn_on_clashes(
        protein, added, site_atom, frag_atom, particles=particles, warn=not relax
    )
    if not (clashes and relax):
        return
    # Port alignment fixes the fragment up to a turn about the new
    # bond. Before the minimizer sees the overlap, turn the fragment,
    # its rotatable bonds and the site side chain to a clear pose. A
    # minimizer started from atoms 0.3 A apart tears bonds; from a
    # clear pose it converges. Only when no turn clears the fragment
    # are other conformers of it tried.
    _place_by_torsions(protein, site_atom, frag_atom, particles=particles)
    if _closest_contact(protein, added, site_atom, frag_atom, particles) < 0.1:
        _try_conformers(protein, added, site_atom, frag_atom, particles=particles)
    if not minimize:
        _warn_on_clashes(protein, added, site_atom, frag_atom, particles=particles)
        return
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


def _place_by_torsions(
    protein, site_atom, frag_atom, particles=None, site_torsions=2, fragment_torsions=4
):
    """Turn the fragment and the site side chain about bonds until clear.

    Once a bond has formed at its length and angles, what is left to
    choose is torsions: the turn about the new bond itself, the turns
    about the side-chain bonds of the site residue that lead to it
    (chi angles), and the turns about the rotatable bonds of the
    fragment nearest the bond. Turning about a bond keeps every bond
    length and angle exactly as it was, so no pose this search makes
    needs a bond put back. The glycoprotein builders search the same
    space (GLYCAM-Web's Asn chi angles, the GlycoShape Re-Glyco phi and
    psi), as does covalent docking that treats the ligand as a side
    chain.

    Every candidate pose is scored at once: one KD-tree query against
    the atoms within reach, and the distances between the moved atoms
    that a turn can bring together, the latter for the best 64 poses of
    each round only. A contact inside
    ``_CONTACT_DISTANCE`` adds a soft penalty and one inside
    ``_CLASH_DISTANCE`` a far steeper one, so that a single deep overlap
    costs more than many shallow contacts. A grid over the new bond and
    the first chi angle comes first, then three rounds of random moves
    about the best poses, narrowing from 60 to 10 degrees. The best pose
    is kept only when it scores better than the start.

    Parameters
    ----------
    protein : Protein
        The protein that holds both atoms.
    site_atom, frag_atom : mbuild.Compound
        The two atoms of the new bond. Everything beyond ``frag_atom``
        is the fragment.
    particles : list of mbuild.Compound, optional
        The protein's particles (see ``_relax_particles``).
    site_torsions : int, optional, default=2
        Most side-chain bonds of the site residue to turn, counted from
        the site atom toward ``CA``. A bond whose far side holds more
        than this side chain and the fragment, as in a ring through the
        backbone or a crosslink, is never turned, nor is an amide C-N
        bond.
    fragment_torsions : int, optional, default=4
        Most rotatable bonds of the fragment to turn, nearest the new
        bond first. Only a bond outside every ring, between two atoms
        with other heavy neighbours, that carries at least three heavy
        atoms on its far side counts.
    """
    from scipy.spatial import cKDTree

    if particles is None:
        particles = list(protein.particles())
    graph = protein.root.bond_graph
    torsions = _placement_torsions(
        graph, site_atom, frag_atom, site_torsions, fragment_torsions
    )
    moving = list(dict.fromkeys(atom for _, _, atoms in torsions for atom in atoms))
    position_of = {atom: i for i, atom in enumerate(moving)}
    start = np.array([atom.pos for atom in moving])

    # The surroundings: every atom within reach of the fragment that does
    # not move. A moved atom's own partners one or two bonds away sit at
    # bonded distances in every pose, so those pairs, and only those, are
    # left out of the score.
    xyz = np.array([particle.pos for particle in particles])
    reach = np.linalg.norm(start - site_atom.pos, axis=1).max() + 0.6
    near = cKDTree(xyz).query_ball_point(site_atom.pos, reach)
    around = [i for i in near if particles[i] not in position_of]
    # The few surrounding atoms that are such a partner of some moved
    # atom are scored pair by pair with a fixed mask; the rest need no
    # mask and are scored with a nearest-neighbour query.
    partner_of = {}
    for atom in moving:
        for neighbor in graph.adj[atom]:
            for partner in [neighbor, *graph.adj[neighbor]]:
                if partner not in position_of:
                    partner_of.setdefault(partner, set()).add(position_of[atom])
    bonded = [i for i in around if particles[i] in partner_of]
    around = [i for i in around if particles[i] not in partner_of]
    bonded_xyz = xyz[bonded]
    scored = np.ones((len(moving), len(bonded)), dtype=bool)
    for k, i in enumerate(bonded):
        scored[sorted(partner_of[particles[i]]), k] = False
    tree = cKDTree(xyz[around]) if around else None

    # Each torsion as (fixed axis atom, moving axis atom, moved atoms),
    # applied innermost first: a turn nearer the protein then carries
    # the result of every turn beyond it.
    steps = [
        (a, position_of[b], np.array([position_of[atom] for atom in atoms]))
        for a, b, atoms in torsions
    ]

    def build(angles):
        poses = np.repeat(start[None], len(angles), axis=0)
        for t, (a, b, atoms) in enumerate(steps):
            theta = angles[:, t]
            if not theta.any():
                continue
            pivot = poses[:, b]
            base = poses[:, position_of[a]] if a in position_of else a.pos
            axis = pivot - base
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            axis = axis[:, None, :]
            arm = poses[:, atoms] - pivot[:, None]
            cos, sin = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
            # Rodrigues' rotation of every moved atom about the bond.
            arm = (
                arm * cos
                + np.cross(axis, arm) * sin
                + axis * (arm * axis).sum(-1, keepdims=True) * (1 - cos)
            )
            poses[:, atoms] = pivot[:, None] + arm
        return poses

    def penalty(distances):
        # A soft term for every contact inside _CONTACT_DISTANCE, and a
        # term a hundred times steeper inside _CLASH_DISTANCE, so that one
        # deep overlap costs more than many shallow contacts.
        soft = np.clip(_CONTACT_DISTANCE - distances, 0.0, None)
        hard = np.clip(_CLASH_DISTANCE - distances, 0.0, None)
        return (soft**2 + 100 * hard**2).sum(-1)

    # Pairs of moved atoms whose distance a turn can change: atoms moved
    # by different sets of turns, and not one or two bonds apart.
    moved_by = [frozenset() for _ in moving]
    for t, (_, _, atoms) in enumerate(steps):
        for i in atoms:
            moved_by[i] = moved_by[i] | {t}
    near_pairs = set()
    for atom in moving:
        for neighbor in graph.adj[atom]:
            for partner in [neighbor, *graph.adj[neighbor]]:
                if partner in position_of and partner is not atom:
                    near_pairs.add(frozenset((position_of[atom], position_of[partner])))
    pairs = np.array(
        [
            (i, j)
            for i in range(len(moving))
            for j in range(i + 1, len(moving))
            if moved_by[i] != moved_by[j] and frozenset((i, j)) not in near_pairs
        ],
        dtype=int,
    ).reshape(-1, 2)

    def outside(poses):
        """Penalty of each pose against the atoms that do not move."""
        total = np.zeros(len(poses))
        if tree is not None:
            distances, _ = tree.query(
                poses.reshape(-1, 3), distance_upper_bound=_CONTACT_DISTANCE
            )
            total += penalty(distances.reshape(poses.shape[:2]))
        if len(bonded):
            reach_out = np.linalg.norm(
                poses[:, :, None] - bonded_xyz[None, None], axis=-1
            )
            total += penalty(
                np.where(scored, reach_out, np.inf).reshape(len(poses), -1)
            )
        return total

    def score(poses, keep=8):
        """Return the indices and full penalties of the ``keep`` best poses.

        The surroundings are scored for every pose; the pairs within the
        moved atoms, which cost far more, only for the 64 best of those.
        """
        first = outside(poses)
        top = np.argsort(first)[:64]
        inner = np.linalg.norm(
            poses[top][:, pairs[:, 0]] - poses[top][:, pairs[:, 1]], axis=-1
        )
        full = first[top] + penalty(inner)
        order = np.argsort(full)[:keep]
        return top[order], full[order]

    n = len(steps)
    new_bond = next(
        t
        for t, (a, b, _) in enumerate(steps)
        if a is site_atom and moving[b] is frag_atom
    )
    turns = np.radians(np.arange(0.0, 360.0, 10.0))
    chi = np.radians(np.arange(0.0, 360.0, 30.0)) if new_bond + 1 < n else [0.0]
    grid = np.zeros((len(turns) * len(chi), n))
    grid[:, new_bond] = np.repeat(turns, len(chi))
    if new_bond + 1 < n:
        grid[:, new_bond + 1] = np.tile(chi, len(turns))
    picked, best_scores = score(build(grid))
    best = grid[picked]
    rng = np.random.default_rng(0)
    for spread in np.radians([60.0, 25.0, 10.0]):
        if best_scores[0] == 0:
            break
        moves = np.repeat(best, 128, axis=0) + rng.uniform(
            -spread, spread, (len(best) * 128, n)
        )
        candidates = np.vstack([best, moves])
        picked, best_scores = score(build(candidates))
        best = candidates[picked]
    pick = 0
    totals = best_scores
    current = score(start[None], keep=1)[1][0]
    if totals[pick] < current:
        for atom, position in zip(moving, build(best[pick : pick + 1])[0]):
            atom.pos = position


def _placement_torsions(graph, site_atom, frag_atom, site_torsions, fragment_torsions):
    """Return the torsions ``_place_by_torsions`` turns, innermost first.

    Each is ``(fixed atom, moving atom, atoms that move)``. The fragment
    is everything beyond ``frag_atom``; a site-residue bond is used only
    when its far side holds nothing but that residue's side chain and
    the fragment.
    """
    import networkx as nx

    fragment = _atoms_beyond(graph, site_atom, frag_atom)
    inside = set(fragment)
    links = graph.subgraph(inside)
    depth = nx.single_source_shortest_path_length(links, frag_atom)
    rotatable = []
    for u, v in nx.bridges(links):
        if not (_is_heavy(u) and _is_heavy(v)) or _is_amide(graph, u, v):
            continue
        if min(sum(_is_heavy(n) for n in graph.adj[x]) for x in (u, v)) < 2:
            continue
        near, far = (u, v) if depth[u] < depth[v] else (v, u)
        moved = _atoms_beyond(graph, near, far)
        if sum(_is_heavy(atom) for atom in moved) >= 3:
            rotatable.append((depth[near], near, far, moved))
    # The bonds that carry the most atoms come first: a glycosidic link
    # swings a whole branch, where an N-acetyl group turns four atoms.
    # The kept bonds are then turned innermost (smallest branch) first.
    rotatable.sort(key=lambda item: -sum(_is_heavy(atom) for atom in item[3]))
    kept = sorted(rotatable[:fragment_torsions], key=lambda item: len(item[3]))
    torsions = [(a, b, moved) for _, a, b, moved in kept]
    torsions.append((site_atom, frag_atom, fragment))

    residue = site_atom.parent
    allowed = inside | set(residue.children)
    ca = next((atom for atom in residue.children if atom.name == "CA"), None)
    if ca is None:
        return torsions
    # Bond distance from CA inside the residue picks, at each atom, the
    # neighbour that leads back toward the backbone (CB from CG, not OD1).
    toward = nx.single_source_shortest_path_length(
        graph.subgraph(set(residue.children) - inside), ca
    )
    child, used = site_atom, 0
    while used < site_torsions and child is not ca and child in toward:
        parents = [
            atom
            for atom in graph.adj[child]
            if atom in toward and toward[atom] == toward[child] - 1
        ]
        if not parents:
            break
        parent = parents[0]
        moved = _atoms_beyond(graph, parent, child)
        if not set(moved) <= allowed or any(
            atom.parent is residue and atom.name in ("N", "C", "O") for atom in moved
        ):
            break
        if not _is_amide(graph, parent, child):
            torsions.append((parent, child, moved))
            used += 1
        child = parent
    return torsions


def _atoms_beyond(graph, near, far):
    """Return ``far`` and every atom reachable from it without passing ``near``."""
    seen, stack, found = {near, far}, [far], [far]
    while stack:
        for neighbor in graph.adj[stack.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                found.append(neighbor)
                stack.append(neighbor)
    return found


def _is_heavy(atom):
    return atom.element.symbol.upper() != "H"


def _is_amide(graph, a, b):
    """Whether ``a``-``b`` is the C-N bond of an amide, which stays planar."""
    for n, c in ((a, b), (b, a)):
        if n.element.symbol.upper() == "N" and c.element.symbol.upper() == "C":
            for neighbor, data in graph.adj[c].items():
                if (
                    neighbor.element.symbol.upper() == "O"
                    and (data.get("bond_order") or 1) >= 2
                ):
                    return True
    return False


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
    """Re-embed the fragment in other conformers and keep the clearest.

    The conformer a fragment arrives in is one of many, and a long
    flexible one may cross the protein in every pose the torsion search
    reaches. This embeds ``count`` conformers of the fragment, the atoms
    beyond ``frag_atom``, with RDKit, places each one, and keeps the pose
    whose closest contact with the protein is largest. The original pose
    competes on the same terms. Nothing is done when the molecule cannot
    be embedded.

    The configuration of every stereocentre is read from the current
    coordinates and enforced in the embedding, and a conformer that comes
    out with another configuration is dropped, so a sugar keeps its
    identity. Each conformer is superposed on the current pose by
    ``frag_atom`` and its neighbours, which keeps the direction of the new
    bond and so the configuration at ``site_atom``, and is then turned
    clear with ``_place_by_torsions``.

    Parameters
    ----------
    protein : Protein
        The protein that owns the placed atoms.
    added : list of mbuild.Compound
        The placed particles; the clearance is measured for them.
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
    graph = protein.root.bond_graph
    fragment = _atoms_beyond(graph, site_atom, frag_atom)
    editable, index = _rdkit_mol(protein, fragment)
    # The bond to the protein is outside the fragment, so a hydrogen along
    # it stands in for the protein atom: without it the bonding atom would
    # be embedded with one neighbour too few, flat instead of tetrahedral,
    # and the configuration at an anomeric carbon would be lost.
    cap = Chem.Atom(1)
    cap.SetNoImplicit(True)
    cap = editable.AddAtom(cap)
    editable.AddBond(index[frag_atom], cap, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    current = np.array([atom.pos for atom in fragment])
    bond = site_atom.pos - frag_atom.pos
    capped = np.vstack([current, frag_atom.pos + bond / np.linalg.norm(bond) * 0.109])
    try:
        Chem.SanitizeMol(mol)
        start = Chem.Conformer(len(fragment) + 1)
        for i, position in enumerate(capped * 10.0):
            start.SetAtomPosition(i, position.tolist())
        mol.AddConformer(start, assignId=True)
        Chem.AssignStereochemistryFrom3D(mol)
        wanted = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
        conformers = list(
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=count,
                randomSeed=0,
                enforceChirality=True,
                clearConfs=False,
            )
        )
    except Exception:  # noqa: BLE001 - RDKit raises several unrelated types
        conformers = []
    if not conformers:
        return
    # The atoms that fix the frame of the new bond: the bonding atom, the
    # hydrogen along the bond, and the heavy neighbours of the bonding atom.
    # Its own hydrogens are left out, since equivalent ones may trade places
    # between the structure and a conformer.
    frame = [index[frag_atom], cap] + [
        index[n] for n in graph.adj[frag_atom] if n in index and _is_heavy(n)
    ]
    # The torsion search may turn the site residue's side chain too, so every
    # pose starts from, and is kept as, the positions of all of those atoms.
    movable = list(dict.fromkeys(fragment + list(site_atom.parent.particles())))
    start_pose = np.array([atom.pos for atom in movable])
    best = start_pose.copy()
    best_clearance = _closest_contact(protein, added, site_atom, frag_atom, everything)
    for conformer in conformers:
        check = Chem.Mol(mol, confId=conformer)
        check.RemoveAllConformers()
        check.AddConformer(mol.GetConformer(conformer), assignId=True)
        Chem.AssignStereochemistryFrom3D(check)
        found = Chem.FindMolChiralCenters(
            check, includeUnassigned=True, useLegacyImplementation=False
        )
        if found != wanted:
            continue
        for atom, position in zip(movable, start_pose):
            atom.pos = position
        positions = np.array(mol.GetConformer(conformer).GetPositions()) / 10.0
        rotation, shift = _kabsch(positions[frame], capped[frame])
        for atom, position in zip(fragment, positions[:-1] @ rotation.T + shift):
            atom.pos = position
        _place_by_torsions(protein, site_atom, frag_atom, particles=everything)
        clearance = _closest_contact(protein, added, site_atom, frag_atom, everything)
        if clearance > best_clearance:
            best_clearance = clearance
            best = np.array([atom.pos for atom in movable])
    for atom, position in zip(movable, best):
        atom.pos = position
    logger.info(
        f"Tried {len(conformers)} conformers of the placed fragment; the "
        f"clearest sits {best_clearance * 10:.2f} A from the protein."
    )


def _kabsch(source, target):
    """Return the rotation and shift that best map ``source`` onto ``target``.

    Apply it as ``points @ rotation.T + shift``. A reflection is never
    returned, since it would invert every stereocentre.
    """
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_centre).T @ (target - target_centre))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, target_centre - source_centre @ rotation.T


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


def _warn_on_clashes(
    protein, added, site_atom, frag_atom, cutoff=0.2, particles=None, warn=True
):
    """Warn when placed atoms overlap the rest of the system.

    Port alignment is rigid; a bulky fragment can land inside the
    protein. The check compares every particle in ``added`` against
    every other atom, and it leaves out the new bond pair. It warns
    below ``cutoff`` nm, so the user knows to relax the structure
    before simulating. ``particles`` is the protein's particle list
    (see ``_relax_particles``); None walks the protein for it. With
    ``warn=False`` it only counts.
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
    if n_clashes and warn:
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
