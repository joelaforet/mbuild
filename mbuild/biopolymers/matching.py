"""Match PDB residue records against CCD residue templates.

The matcher works on parsed PDB records and residue templates only; it
builds no mBuild objects. Given the records of one residue and the
template variants its residue name can take, it decides which variant
the file holds, which template atoms are missing, and which
inter-residue bonds those missing atoms imply: a peptide bond to the
residue before or after, or a crosslink such as a disulfide.

Chemistry is never guessed. A residue whose records no variant explains
raises an error that names the residue and the fix.
"""

import logging
from dataclasses import dataclass

import numpy as np

from mbuild.exceptions import MBuildError

logger = logging.getLogger(__name__)


#: Longest C to N distance, in nm, that the loader accepts as a peptide
#: bond across an advisory TER record. A peptide C-N bond is about
#: 0.133 nm long. The limit adds a margin of about 50 percent for a
#: strained or low-resolution structure. It still rejects two residues
#: that only share a chain identifier and increasing residue numbers.
_ADVISORY_TER_MAX_C_N = 0.2


#: Element symbols, in upper case, of the atoms that join two residues
#: through one covalent bond in a PDB entry. The CYS-CYS disulfide and
#: the SEC-SEC diselenide are the two common bridges.
_BRIDGING_ELEMENTS = frozenset(("S", "SE"))


@dataclass
class _Match:
    """The result of trying one template variant against one PDB residue.

    A residue name such as ``HIS`` maps to several template variants, one
    per protonation state, and the loader does not know in advance which
    one the file holds. It tries every variant and keeps one ``_Match``
    per variant that fits. ``_matches_agree`` then compares the survivors:
    if they describe the same chemistry, the first is used; if they
    disagree, the file is ambiguous and the loader raises.

    Attributes
    ----------
    variant : ResidueTemplate
        The template variant that was tried.
    record_atoms : dict
        Maps ``id(record)`` to the template atom that record is. See
        ``_assign_records`` and ``_assign_records_bipartite``.
    missing : set
        Names of template atoms that no record claimed. A fit requires
        every missing atom to be a leaving atom of an expected link.
    expects_prior, expects_posterior, expects_crosslink : bool
        Which links the missing atoms imply. A missing prior leaving
        fragment, ``H2`` on most residues and ``H`` on proline, means a
        peptide bond from the residue before (``expects_prior``).
        Missing ``OXT`` and ``HXT`` mean a peptide bond to the residue
        after (``expects_posterior``). A missing ``HG`` on cysteine
        means a disulfide bridge (``expects_crosslink``). ``_bond_backbone`` and
        ``_bond_crosslinks`` read these flags to form the inter-residue
        bonds, and ``_filter_crosslink_candidates`` uses the CONECT
        records to choose between a bridged and a free cysteine.

    Notes
    -----
    Name-based matching against residue templates with protonation
    variants is the usual design for a reader that needs formal charges
    from a PDB file. The comparison below is as of 2026-09-04.

    - openff-pablo uses the same design. Its ``ResidueMatch`` holds the
      same data. It reads old atom names as synonyms, but it raises when
      a synonym clashes with a canonical name, so it has no second pass:
      https://github.com/openforcefield/openff-pablo/blob/main/openff/pablo/_pdb_data.py
    - OpenMM ``ForceField`` matches a residue by bond graph, elements and
      connectivity, so it needs the bonds before the chemistry:
      https://docs.openmm.org/latest/userguide/application/06_creating_ffs.html#residue-templates
    - PDBFixer matches by residue name to its own template files and
      adds missing heavy atoms from them. It adds hydrogens with
      OpenMM's ``Modeller``, and it assigns no formal charge:
      https://github.com/openmm/pdbfixer/blob/master/Manual.html
    - ParmEd reads names and coordinates and adds bonds inside standard
      residues from a name-keyed template table. It generates no
      protonation variants. It derives no formal charge from chemistry;
      it copies the PDB charge column when the file sets it:
      https://github.com/ParmEd/ParmEd/blob/master/parmed/formats/pdb.py
    - mdtraj reads names and coordinates and guesses bonds from a
      standard residue table and distances. It derives no formal charge
      from chemistry; it copies the PDB charge column when the file sets
      it:
      https://github.com/mdtraj/mdtraj/blob/master/mdtraj/formats/pdb/pdbfile.py
    """

    variant: object
    record_atoms: dict
    missing: set
    expects_prior: bool
    expects_posterior: bool
    expects_crosslink: bool


def _bond_separation(variant, start, end, max_bonds):
    """Return the number of bonds between two atoms of one template.

    The search is a breadth-first walk over the cached adjacency map of
    ``ResidueTemplate.bonded_names``. It stops after ``max_bonds``
    steps, so its cost does not grow with the size of the template.

    Parameters
    ----------
    variant : ResidueTemplate
        Template whose bonds define the graph.
    start : str
        Canonical name of the first atom.
    end : str
        Canonical name of the second atom.
    max_bonds : int
        Largest separation that the search reports.

    Returns
    -------
    int or None
        The separation in bonds, or None when the two atoms are more
        than ``max_bonds`` apart or are not connected.
    """
    visited = {start}
    frontier = {start}
    for separation in range(1, max_bonds + 1):
        frontier = {
            name
            for atom_name in frontier
            for name in variant.bonded_names(atom_name)
            if name not in visited
        }
        if end in frontier:
            return separation
        visited |= frontier
    return None


def _record_pos(group, atom_name):
    """Return the position of a named PDB record of one residue, or None.

    Parameters
    ----------
    group : _PdbResidue
        The parsed residue whose records are searched.
    atom_name : str
        The record name to find, as the file writes it.

    Returns
    -------
    numpy.ndarray or None
        The position in nm, or None when no record carries the name.
    """
    return next(
        (record.pos for record in group.records if record.name == atom_name), None
    )


def _ter_break_reason(earlier, later, earlier_variant, later_variant):
    """Report why a TER between two residues ends the chain.

    The wwPDB Format Guide v3.30, section 9 (Coordinate Section, TER),
    states that the TER record ends the chain of ATOM and HETATM
    records that comes before it:
    https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
    A strict reader therefore ends the polymer at every TER.

    A preparation tool that writes the file from a topology puts a TER
    at the end of each topology chain. It does not put one at the end
    of each PDB chain. OpenMM's ``PDBFile.writeModel`` prints a TER
    after the last residue of every ``Topology`` chain:
    https://github.com/openmm/openmm/blob/05472c9a812927c863be67abbb3376e944b2c7ef/wrappers/python/openmm/app/pdbfile.py#L404
    With ``keepIds=True`` it takes the chain identifier from the chain
    object, so two topology chains can carry one identifier. A cap
    (ACE, NME) or a ligand that the topology holds in its own chain
    then follows a TER inside one PDB chain. Every writer that goes
    through OpenMM inherits this behavior.

    The callers have already checked that the two residues share a
    chain identifier. This function applies the two remaining tests.

    The residue numbering must increase across the TER. The pair
    ``(resnum, icode)`` of the later residue must be greater than the
    pair of the earlier residue. Strict consecutiveness is not
    required. OpenMM writes a numbering gap where a loop is missing,
    and it writes insertion codes, and the loader accepts both shapes
    when no TER is present.

    The candidate peptide bond must also be short enough. The distance
    from the C atom of the earlier residue to the N atom of the later
    residue must stay below ``_ADVISORY_TER_MAX_C_N``. The numbering
    alone cannot tell one polymer from two separate molecules that
    share a chain identifier.

    A TER that passes both tests is advisory. The atom records then
    still describe one polymer. The residue before the TER is missing
    its OXT and HXT atoms, and the residue after it is missing its H2
    atom. Only a peptide bond explains that.

    Parameters
    ----------
    earlier : _PdbResidue
        The residue that carries the TER record.
    later : _PdbResidue
        The residue that follows it in the file.
    earlier_variant : mbuild.biopolymers.ccd.ResidueTemplate
        The base template of the earlier residue. It names the atom
        that carries the posterior peptide bond.
    later_variant : mbuild.biopolymers.ccd.ResidueTemplate
        The base template of the later residue. It names the atom that
        carries the prior peptide bond.

    Returns
    -------
    str or None
        The reason the TER ends the chain, or None when the TER is
        advisory.
    """
    if (later.resnum, later.icode) <= (earlier.resnum, earlier.icode):
        return (
            "the residue numbering does not increase across it "
            f"({earlier.resnum}{earlier.icode} then "
            f"{later.resnum}{later.icode})"
        )
    # The link atom names C and N are the same in PDB format version 2
    # and version 3, so a record can be found by the template name here,
    # before the records are assigned to template atoms.
    carbon = _record_pos(earlier, earlier_variant.posterior_link_atom)
    nitrogen = _record_pos(later, later_variant.prior_link_atom)
    if carbon is None or nitrogen is None:
        return (
            f"{earlier.label} or {later.label} carries no backbone C or "
            "N record, so the peptide bond cannot be measured"
        )
    distance = float(np.linalg.norm(carbon - nitrogen))
    if distance > _ADVISORY_TER_MAX_C_N:
        return (
            f"the C atom of {earlier.label} and the N atom of "
            f"{later.label} are {distance * 10:.2f} A apart, which is "
            f"too far for a peptide bond (limit "
            f"{_ADVISORY_TER_MAX_C_N * 10:.1f} A)"
        )
    return None


def _assign_records(group, variant):
    """Assign the records of one PDB residue to template atoms.

    The first pass gives every record the single atom that
    ``name_to_atom`` reports for its name. That pass resolves files
    written with wwPDB version 3 atom names, and it is the only pass
    such files need.

    Two of its rejection reasons come from the atom names alone: a name
    the template does not carry, and two records that claim the same
    template atom. Both happen on files whose hydrogen names are valid
    but not canonical. Digit-first names such as ``2HB`` are PDB format
    version 2 names, which Amber-style tools still write, and they fail
    the first reason. The wwPDB publishes the atom naming rule on the
    Nomenclature page of the version 3.0 guide:
    https://www.wwpdb.org/documentation/file-format-content/format30/sect12.html
    Glycine written with the version 2 alpha-hydrogen names
    ``HA1``/``HA2`` fails the second reason. The CCD gives the version 3
    atom ``HA2`` the alternative name ``HA1``, and it gives the atom
    ``HA3`` the alternative name ``HA2``. ``parse_ccd_cif`` reads these
    alternative names from the ``_chem_comp_atom.alt_atom_id`` column of
    the CCD CIF file, and the table ``_ATOM_NAME_SYNONYMS`` adds more
    names per residue.
    ``mbuild.biopolymers.ccd.ResidueTemplate.atoms_named`` reports every
    template atom that one name can denote. After either reason, a
    second pass runs ``_assign_records_bipartite`` over the full
    candidate list of every record.

    Parameters
    ----------
    group : _ResidueGroup
        The records of one PDB residue, in file order.
    variant : mbuild.biopolymers.ccd.ResidueTemplate
        The template variant to assign the records to.

    Returns
    -------
    record_atoms : dict
        Maps ``id(record)`` to the assigned AtomTemplate. Complete only
        when ``reason`` is None.
    reason : str or None
        Why the variant does not fit, or None on success.
    fallback : bool
        True when the second pass produced the assignment.
    """
    name_to_atom = variant.name_to_atom
    record_atoms = {}
    used = set()
    reason = None
    names_disagree = False
    for record in group.records:
        atom = name_to_atom.get(record.name)
        if atom is None:
            reason = f"atom name {record.name!r} is not in the template"
            names_disagree = True
            break
        if atom.name in used:
            reason = f"two records match template atom {atom.name!r}"
            names_disagree = True
            break
        if record.element and record.element.upper() != atom.element.upper():
            reason = (
                f"element {record.element!r} of atom {record.name!r} "
                f"conflicts with template element {atom.element!r}"
            )
            break
        used.add(atom.name)
        record_atoms[id(record)] = atom
    if not names_disagree:
        return record_atoms, reason, False
    record_atoms, reason = _assign_records_bipartite(group, variant, reason)
    return record_atoms, reason, reason is None


def _assign_records_bipartite(group, variant, reason):
    """Pair the records of one residue with template atoms when names alone
    do not decide.

    The loader gives a PDB residue its chemistry by pairing each ATOM
    record with one atom of the CCD template. A record has a name, an
    element and coordinates. The template atom has the formal charge and
    the bonds. ``_assign_records`` does this pairing by name: record
    ``CA`` is template atom ``CA``. That pass is enough for a file with
    wwPDB version 3 atom names. This function then runs only for
    variants the first pass rejected by name, and for such a file it
    rejects them again.

    Older files use PDB format version 2 names, and the CCD stores those
    names as synonyms. For glycine the version 3 atoms are ``HA2`` and
    ``HA3``. Version 2 files call them ``HA1`` and ``HA2``, so the CCD
    lists ``HA1`` as a synonym of ``HA2`` and ``HA2`` as a synonym of
    ``HA3``. In a version 2 file, record ``HA2`` then fits two template
    atoms, and record ``HA1`` fits one of them. The first pass takes the
    first fit for each record, so both records land on ``HA2``. Two
    records on one atom is a collision, and the first pass rejects the
    residue.

    This function solves the collision by a bipartite matching. Each
    record lists every template atom it can be: the atoms whose name or
    synonym matches, with the same element. A Kuhn augmenting-path
    search then pairs records with atoms so that no atom is used twice.
    When a record needs an atom that another record holds, the other
    record moves to its next candidate and the search continues from
    there. For the glycine above the result
    is record ``HA1`` to ``HA2`` and record ``HA2`` to ``HA3``. Records
    are visited in file order and candidates stay in template order, so
    one file always gives one assignment.

    Either every record gets an atom or the residue is rejected. A record
    with no template atom has no element, charge or bonds, so a partial
    pairing cannot build the residue.

    Parameters
    ----------
    group : _ResidueGroup
        The records of one PDB residue, in file order.
    variant : mbuild.biopolymers.ccd.ResidueTemplate
        The template variant to assign the records to.
    reason : str
        The rejection reason of the first pass. It is reported again
        when the assignment fails, so that the error message keeps
        naming the record that the user must inspect.

    Returns
    -------
    record_atoms : dict
        Maps ``id(record)`` to the assigned AtomTemplate, or empty on
        failure.
    reason : str or None
        None on success, else the reason passed in.
    """
    candidates = []
    for record in group.records:
        atoms = [
            atom
            for atom in variant.atoms_named(record.name)
            if not record.element or record.element.upper() == atom.element.upper()
        ]
        if not atoms:
            return {}, reason
        candidates.append(atoms)

    holder = {}

    def augment(index, visited):
        """Give record ``index`` an atom, moving earlier records on."""
        for atom in candidates[index]:
            if atom.name in visited:
                continue
            visited.add(atom.name)
            held_by = holder.get(atom.name)
            if held_by is None or augment(held_by, visited):
                holder[atom.name] = index
                return True
        return False

    for index in range(len(candidates)):
        if not augment(index, set()):
            return {}, reason

    atom_of = {index: name for name, index in holder.items()}
    name_to_atom = variant.name_to_atom
    return {
        id(record): name_to_atom[atom_of[index]]
        for index, record in enumerate(group.records)
    }, None


def _leaving_expectations(variant, missing):
    """Report which leaving fragments the absent atoms of a residue match.

    ``missing`` holds the template atom names that the records of the
    residue do not fill. A residue loses a leaving fragment when it
    forms a peptide bond or a crosslink, so an absent fragment is
    expected. Returns the three expectations and the union of the
    fragments that they explain.

    Parameters
    ----------
    variant : _Variant
        The template variant under test.
    missing : set of str
        The template atom names that no record fills.

    Returns
    -------
    tuple of (bool, bool, bool, set of str)
        Whether the residue expects a prior bond, a posterior bond and
        a crosslink, and the atom names that these bonds explain.
    """
    prior = variant.prior_fragment
    posterior = variant.posterior_fragment
    crosslink_fragment = set()
    if variant.crosslink and variant.crosslink[0] in variant.atom_names:
        crosslink_fragment = variant.leaving_fragment_of(variant.crosslink[0])
    expects_prior = bool(prior) and prior <= missing
    expects_posterior = bool(posterior) and posterior <= missing
    expects_crosslink = bool(crosslink_fragment) and crosslink_fragment <= missing
    explained = (
        (prior if expects_prior else set())
        | (posterior if expects_posterior else set())
        | (crosslink_fragment if expects_crosslink else set())
    )
    return expects_prior, expects_posterior, expects_crosslink, explained


def _no_match_message(group, variants, reasons):
    """Return the error text for a residue that fits no template variant."""
    details = "\n  ".join(reasons)
    # The per-variant reasons name the records that failed, but not
    # the names the templates take. List them once, so the user can
    # compare the file against them. The list is the union over
    # every variant: a variant can hold an atom that the base
    # variant does not, such as the H3 of an N-terminal residue.
    accepted = sorted({name for variant in variants for name in variant.name_to_atom})
    return (
        f"Could not match residue {group.label} against any "
        f"template variant:\n  {details}\n"
        f"The {group.resname} template accepts these {len(accepted)} "
        f"atom names: {', '.join(accepted)}.\n"
        "Check that the file is fully protonated (e.g. run pdbfixer "
        "or reduce) and uses standard PDB atom names."
    )


def _rescued_names(group, record_atoms):
    """Return the record names that the second assignment pass renamed."""
    return sorted(
        record.name
        for record in group.records
        if record.name != record_atoms[id(record)].name
    )


def _match_residue(group, variants, prior_possible, posterior_possible):
    """Match one PDB residue against its template variants.

    Returns the valid matches. Raises MBuildError with the per-variant
    rejection reasons when nothing matches. Logs at info level when the
    second assignment pass rescued the residue, so that the tolerance
    is visible in the log.
    """
    matches = []
    reasons = []
    rescued = []
    for variant in variants:
        record_atoms, reason, fallback = _assign_records(group, variant)
        if reason is not None:
            reasons.append(f"{variant.description}: {reason}")
            continue
        used = {atom.name for atom in record_atoms.values()}
        missing = variant.atom_names - used
        expects_prior, expects_posterior, expects_crosslink, explained = (
            _leaving_expectations(variant, missing)
        )
        if missing != explained:
            reasons.append(
                f"{variant.description}: atoms {sorted(missing - explained)} "
                "are missing but not part of a leaving fragment"
            )
            continue
        if expects_prior and not prior_possible:
            reasons.append(
                f"{variant.description}: expects a bond to a preceding "
                "residue, but none is adjacent"
            )
            continue
        if expects_posterior and not posterior_possible:
            reasons.append(
                f"{variant.description}: expects a bond to a following "
                "residue, but none is adjacent"
            )
            continue
        matches.append(
            _Match(
                variant=variant,
                record_atoms=record_atoms,
                missing=missing,
                expects_prior=expects_prior,
                expects_posterior=expects_posterior,
                expects_crosslink=expects_crosslink,
            )
        )
        if fallback and not rescued:
            rescued = _rescued_names(group, record_atoms)
    if not matches:
        raise MBuildError(_no_match_message(group, variants, reasons))
    if rescued:
        logger.info(
            f"Residue {group.label}: the first-hit atom names did not fit, "
            f"and the second pass read {rescued} as alternative names."
        )
    return matches


def _bridge_scope_message(label1, atom1_name, label2, atom2_name, remedy):
    """Return the error text for a bridge that mBuild does not build.

    ``_add_disulfide`` in ``mbuild.biopolymers.ccd`` gives the SG-SG
    crosslink to CYS only. A CONECT record between the sulfur or
    selenium atoms of two other residues therefore describes a bond
    that no template predicts. The text names both atoms, states the
    limit, gives the caller's remedy, and points at the issue tracker.

    The caller gives the remedy, because the two error sites need two
    different actions. One site reports a bond that the templates do
    not predict, and the file loads after the CONECT record is removed.
    The other site reports a residue that also lost the hydrogen of its
    bridging atom, and that file needs the hydrogen back as well.

    Parameters
    ----------
    label1, label2 : str
        Labels of the two residues, as ``_PdbResidue.label`` writes them.
    atom1_name, atom2_name : str
        Names of the two bridging atoms.
    remedy : str
        One sentence that gives the action which loads the file.

    Returns
    -------
    str
        The message.
    """
    return (
        f"CONECT record between {label1} {atom1_name} and {label2} "
        f"{atom2_name} joins two sulfur or selenium atoms, but mBuild "
        "forms disulfide bridges between CYS residues only. Other "
        "bridging residues such as SEC, DCY and HCS are not supported "
        f"yet. {remedy} Open an issue at "
        "https://github.com/mosdef-hub/mbuild/issues if you need this "
        "residue."
    )


def _conect_partners(conects):
    """Map each atom serial to the serials that CONECT records bond it to.

    ``conects`` holds one frozenset per bonded serial pair.
    ``_parse_pdb`` expands a multi-partner CONECT record into one pair
    per partner. A frozenset with one serial comes from a record that
    names an atom as its own partner. It describes no pair, so this
    function drops it.
    """
    partners_of = {}
    for pair in conects:
        serials = tuple(pair)
        if len(serials) != 2:
            continue
        partners_of.setdefault(serials[0], set()).add(serials[1])
        partners_of.setdefault(serials[1], set()).add(serials[0])
    return partners_of


def _bridge_scope_conflict(group, groups, conects, library):
    """Return the bridge message for a residue that failed to match, or None.

    A bridged residue is missing the hydrogen of its sulfur or selenium
    atom, and a CONECT record joins that atom to a second bridging
    residue. Only CYS carries a crosslink, so every other bridged
    residue keeps that hydrogen in every template variant and matches
    none of them. The match error then asks the user to protonate the
    file, and the new hydrogen breaks the bridge. This function finds
    the case, so that the caller reports the true limit instead.

    Parameters
    ----------
    group : mbuild.biopolymers.protein_pdb_io._PdbResidue
        The residue that matched no template variant.
    groups : list of _PdbResidue
        Every parsed residue of the file, to resolve a CONECT partner.
    conects : list of frozenset
        The atom serials of each CONECT record.
    library : mbuild.biopolymers.ccd.CCDLibrary
        The template library. Every residue name of the file resolves
        in it, because ``_load_pdb`` looks all of them up first.

    Returns
    -------
    str or None
        The message, or None when the residue failed for another reason.
    """
    partners_of = _conect_partners(conects)
    record_by_serial = {
        record.serial: (other, record) for other in groups for record in other.records
    }

    def bridging_atom(other, record):
        """Return the template atom of a record when it can bridge."""
        atom = library[other.resname][0].name_to_atom.get(record.name)
        if atom is None or atom.element.upper() not in _BRIDGING_ELEMENTS:
            return None
        return atom

    template = library[group.resname][0]
    present = {}
    for record in group.records:
        atom = template.name_to_atom.get(record.name)
        if atom is not None:
            present[atom.name] = record
    for name, record in present.items():
        atom = bridging_atom(group, record)
        if atom is None:
            continue
        hydrogens = [
            other
            for other in template.bonded_names(name)
            if template.name_to_atom[other].element == "H"
        ]
        if not hydrogens or all(other in present for other in hydrogens):
            continue
        for serial in partners_of.get(record.serial, ()):
            partner = record_by_serial.get(serial)
            if partner is None:
                continue
            partner_group, partner_record = partner
            partner_atom = bridging_atom(partner_group, partner_record)
            if partner_atom is None:
                continue
            if group.resname == "CYS" and partner_group.resname == "CYS":
                continue
            # The residue lost the hydrogen of its bridging atom. The
            # file therefore loads only when both the CONECT record
            # goes and that hydrogen comes back. Name the hydrogen and
            # the residue, so that the user makes both changes.
            missing = [other for other in hydrogens if other not in present]
            noun = "hydrogen" if len(missing) == 1 else "hydrogens"
            names = " and ".join(missing)
            return _bridge_scope_message(
                group.label,
                name,
                partner_group.label,
                partner_atom.name,
                "To load the entry without the bridge, remove that CONECT "
                f"record and add the {noun} {names} to {group.label}.",
            )
    return None


def _crosslink_message(group, rejected):
    """Return the error text for a residue that the crosslink rule emptied.

    ``rejected`` holds the candidate matches that the rule refused, each
    with the CONECT state that refused it. The reported candidate is the
    first one that carries a CONECT record and expects no crosslink,
    because that candidate names the atoms the file must lose to form
    the bridge. The first rejected candidate is reported when no
    candidate is of that kind.
    """
    match, linked = rejected[0]
    for candidate in rejected:
        if candidate[1] and not candidate[0].expects_crosslink:
            match, linked = candidate
            break
    name = match.variant.crosslink[0]
    leaving = sorted(match.variant.leaving_fragment_of(name))
    if linked:
        return (
            f"Residue {group.label}: a CONECT record joins its "
            f"{name} atom to the {name} atom of another residue, "
            f"which signals a disulfide, but its {leaving} atoms "
            "are present. Remove the CONECT record, or remove the "
            f"{leaving} atoms to form the disulfide."
        )
    return (
        f"Residue {group.label} is missing its {leaving} atoms, "
        "which signals a crosslink, but no CONECT record connects "
        f"it to a crosslink partner. Add a CONECT record between "
        f"the two {name} atoms, or restore the {leaving} atoms."
    )


def _filter_crosslink_candidates(groups, all_candidates, conects):
    """Reject crosslink candidate matches that CONECT records contradict.

    A bridged cysteine (HG absent) matches two kinds of variants: the
    neutral crosslink variants (``expects_crosslink=True``) and the
    deprotonated thiolate variants (``expects_crosslink=False``). The
    two disagree on the SG formal charge, so ``_matches_agree`` would
    reject every disulfide-containing file. The CONECT records decide
    between them. A candidate whose variant carries a crosslink atom
    must agree with the file. Its crosslink expectation must equal the
    presence of an SS CONECT to a crosslink-capable partner residue. A
    partner is crosslink-capable when any of its own pre-filter
    candidates expects the crosslink. Capability is computed before any
    rejection, so two bridged residues validate each other. Candidates
    whose variant has
    no crosslink atom are kept unchanged. This mirrors openff-pablo's
    ``filter_on_crosslinks`` rule.

    Returns the filtered candidate lists. Raises MBuildError when the
    rule leaves a residue with no candidate.
    """

    def crosslink_serial(group, match):
        """Return the serial of the match's crosslink atom, or None."""
        name = match.variant.crosslink[0]
        for record in group.records:
            atom = match.record_atoms.get(id(record))
            if atom is not None and atom.name == name:
                return record.serial
        return None

    serial_owner = {}
    for index, (group, candidates) in enumerate(zip(groups, all_candidates)):
        for match in candidates:
            if match.expects_crosslink:
                serial = crosslink_serial(group, match)
                if serial is not None:
                    serial_owner[serial] = index

    partners_of = _conect_partners(conects)

    filtered = []
    for index, (group, candidates) in enumerate(zip(groups, all_candidates)):
        kept = []
        rejected = []
        for match in candidates:
            if not match.variant.crosslink:
                kept.append(match)
                continue
            serial = crosslink_serial(group, match)
            if serial is None:
                kept.append(match)
                continue
            linked = any(
                serial_owner.get(other, index) != index
                for other in partners_of.get(serial, ())
            )
            if match.expects_crosslink == linked:
                kept.append(match)
            else:
                rejected.append((match, linked))
        if not kept:
            raise MBuildError(_crosslink_message(group, rejected))
        filtered.append(kept)
    return filtered


def _matches_agree(matches, group):
    """Verify that all valid matches assign the same chemistry.

    Matches may differ in absent atoms. A neutral and a deprotonated
    C-terminal template differ that way when OXT itself is absent. The
    matches must agree on the charges of the atoms present, on the bonds
    among them, and on the expected links. This function returns the
    first match, and it raises on disagreement.

    The reference is the first match, and the match order is the
    variant order of the CCDLibrary, which is the order in which
    ``_protonation_variants`` generates the variants. That order does
    not depend on the file, so the same residue always returns the same
    match. A disagreement that reaches this point raises instead of
    picking a variant. The two variants give the atoms different
    chemistry, and the loader must never guess which one the file means.
    """
    reference = matches[0]

    def fingerprint(match):
        present = {atom.name for atom in match.record_atoms.values()}
        charges = tuple(
            sorted(
                (atom.name, atom.formal_charge) for atom in match.record_atoms.values()
            )
        )
        bonds = tuple(
            sorted(
                (*sorted((bond.atom1, bond.atom2)), bond.order)
                for bond in match.variant.bonds
                if bond.atom1 in present and bond.atom2 in present
            )
        )
        return (
            charges,
            bonds,
            match.expects_prior,
            match.expects_posterior,
            match.expects_crosslink,
        )

    reference_fingerprint = fingerprint(reference)
    for match in matches[1:]:
        if fingerprint(match) != reference_fingerprint:
            raise MBuildError(
                f"Residue {group.label} matches multiple template variants "
                "that disagree on chemistry. This usually means the "
                "protonation state is incomplete or inconsistent."
            )
    return reference
