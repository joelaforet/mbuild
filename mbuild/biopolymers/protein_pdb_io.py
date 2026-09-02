"""PDB record parsing and writing for protein structures.

The reader is strict about the records the template-matching loader
needs (serials, TER, CONECT, insertion codes); the writer produces
RCSB-conformant files that residue-template loaders ingest.
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from mbuild.box import Box
from mbuild.exceptions import MBuildError

logger = logging.getLogger(__name__)

__all__ = ["write_pdb"]


@dataclass
class _PdbRecord:
    serial: int
    name: str
    resname: str
    chain_id: str
    resnum: int
    icode: str
    pos: np.ndarray
    element: str
    hetatm: bool


@dataclass
class _PdbResidue:
    resname: str
    chain_id: str
    resnum: int
    icode: str
    records: list = field(default_factory=list)
    ter_after: bool = False

    @property
    def label(self):
        """Return the residue identity for error messages and logs."""
        return f"{self.resname} {self.chain_id}:{self.resnum}{self.icode}"


def _decode_index(field, width):
    """Decode an atom serial or a residue number from its PDB columns.

    The wwPDB Format Guide v3.30, section 9 (Coordinate Section,
    ATOM/HETATM), gives the atom serial number five columns (7-11) and
    the residue sequence number four columns (23-26). A system with
    more than 99999 atoms, or a chain with more than 9999 residues,
    does not fit in those columns. OpenMM writes a value that does not
    fit in hexadecimal, in ``openmm/app/pdbfile.py::_formatIndex``: a
    value below ``10 ** width`` fills the field in decimal, and a
    larger value fills it with
    ``value - 10 ** width + 10 * 16 ** (width - 1)`` in hexadecimal.
    Atom serials therefore run 99999, A0000, A0001 up to AFFFF, then
    B0000. Residue numbers run 9999, A000, A001, and so on. Any system
    above 99999 atoms or 9999 residues carries such fields.

    This function inverts that rule. A field that reads as a decimal
    number keeps its decimal value, so a file inside the column widths
    reads as before. Every other field is read as hexadecimal and
    shifted back. The OpenMM rule takes the shifted value modulo
    ``16 ** width``, so the encoding repeats. This function decodes the
    first cycle, which holds 493215 atoms and 34575 residues.

    Parameters
    ----------
    field : str
        The text of the field, with or without its column padding.
    width : int
        The number of columns of the field: 5 for an atom serial
        number, 4 for a residue sequence number.

    Returns
    -------
    int
        The decoded number.

    Raises
    ------
    MBuildError
        If the field is neither a decimal nor a hexadecimal number.
    """
    text = field.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        shifted = int(text, 16)
    except ValueError:
        raise MBuildError(
            f"PDB field {field!r} is not a decimal or hexadecimal number."
        )
    return shifted + 10**width - 10 * 16 ** (width - 1)


def _parse_pdb(text):
    """Parse ATOM/HETATM/TER/CONECT/CRYST1 records of the first model.

    A disordered atom carries an alternate location indicator in
    column 17, which the wwPDB Format Guide v3.30, section 9
    (Coordinate Section, ATOM), names altLoc. One conformer only is
    read: the records whose altLoc is blank, plus the records that
    carry the first non-blank altLoc of the file. The other conformers
    are skipped, and one warning names the residues that lose them.

    Returns a tuple ``(residues, conects, box)``: the ``_PdbResidue``
    groups in file order, the CONECT pairs as a set of frozensets of
    two serials, and the ``Box`` or None.
    """
    residues = []
    conects = set()
    box = None
    kept_alt_loc = None
    skipped_alt_loc_keys = {}
    in_extra_model = False
    last_key = None
    for line in text.splitlines():
        record_type = line[:6]
        if record_type == "ENDMDL":
            in_extra_model = True
        elif record_type.startswith("MODEL") and in_extra_model:
            logger.warning("PDB file has multiple models; only model 1 is read.")
        elif record_type in ("ATOM  ", "HETATM") and not in_extra_model:
            record = _PdbRecord(
                serial=_decode_index(line[6:11], 5),
                name=line[12:16].strip(),
                resname=line[17:20].strip(),
                chain_id=line[21].strip(),
                resnum=_decode_index(line[22:26], 4),
                icode=line[26].strip(),
                pos=np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
                / 10.0,
                element=line[76:78].strip(),
                hetatm=record_type == "HETATM",
            )
            key = (record.resname, record.chain_id, record.resnum, record.icode)
            # A crystal structure gives a disordered atom one record
            # per conformer, each with its own altLoc letter. The
            # loader builds one structure, so it keeps the blank
            # records and the first non-blank letter of the file and
            # it skips the rest. openff-pablo keeps the same pair
            # (_pdb_data.py, _allowed_alt_locs).
            alt_loc = line[16].strip()
            if alt_loc:
                if kept_alt_loc is None:
                    kept_alt_loc = alt_loc
                if alt_loc != kept_alt_loc:
                    skipped_alt_loc_keys[key] = None
                    continue
            if key != last_key:
                residues.append(_PdbResidue(*key))
                last_key = key
            residues[-1].records.append(record)
        elif record_type.startswith("TER") and residues:
            residues[-1].ter_after = True
        elif record_type == "CONECT":
            fields = [line[start : start + 5].strip() for start in (6, 11, 16, 21, 26)]
            serials = [int(value) for value in fields if value]
            for partner in serials[1:]:
                conects.add(frozenset((serials[0], partner)))
        elif record_type == "CRYST1":
            lengths = (
                float(line[6:15]) / 10.0,
                float(line[15:24]) / 10.0,
                float(line[24:33]) / 10.0,
            )
            angles = (float(line[33:40]), float(line[40:47]), float(line[47:54]))
            if any(length > 0.2 for length in lengths):
                box = Box(lengths=lengths, angles=angles)
    if skipped_alt_loc_keys:
        labels = ", ".join(_PdbResidue(*key).label for key in skipped_alt_loc_keys)
        logger.warning(
            f"Alternate location {kept_alt_loc!r} is read for {labels}. "
            "The other conformers of these residues are skipped."
        )
    if not residues:
        raise MBuildError("No ATOM or HETATM records found in the PDB file.")
    return residues, conects, box


def write_pdb(protein, filename, overwrite=False):
    """Write a prepared PDB file for downstream residue-template loaders.

    Bonds made through ``add_port_at`` + ``force_overlap`` get
    CONECT records but no ``cross_bonds`` record, so downstream
    loaders need residue definitions for them from you.

    mBuild's generic PDB writer (ParmEd via ``save``) cannot express
    residue numbers, chain identifiers, HETATM records, or a
    selective CONECT policy, so this recipe has its own writer. The
    conventions follow the RCSB standard and residue-template
    readers: ``ATOM`` for residues loaded
    from ATOM records, ``HETATM`` for attached fragments and
    heteroatoms, ``TER`` after every chain, ``CRYST1`` when a box is
    set, and ``CONECT`` records **only** for bonds between
    non-adjacent residues (disulfides, attached fragments, branch
    links) — peptide bonds are implied by residue adjacency, and a
    CONECT that the residue templates cannot explain makes strict
    loaders fail.

    Parameters
    ----------
    protein : mbuild.biopolymers.Protein
        The protein to write.
    filename : str
        Path of the PDB file to write.
    overwrite : bool, optional, default=False
        Overwrite the file if it exists.
    """
    if os.path.exists(filename) and not overwrite:
        raise IOError(f"{filename} exists; not overwriting")

    lines = []
    if protein.box is not None:
        lines.append(_cryst1_line(protein.box))
    atom_lines, particle_serial, particle_residue, residue_order = _atom_and_ter_lines(
        protein
    )
    lines.extend(atom_lines)

    # The layout above only visits particles inside a Chain -> Residue
    # path. A particle outside that path would be absent from the file
    # and its bonds would be absent from the CONECT records.
    _check_residue_membership(protein.particles(), particle_serial)

    lines.extend(
        _conect_lines(protein, particle_serial, particle_residue, residue_order)
    )
    lines.append("END")

    with open(filename, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _cryst1_line(box):
    """Format the CRYST1 record of a unit cell.

    Parameters
    ----------
    box : mbuild.Box
        The unit cell. Its lengths are in nanometres; the record holds
        angstroms.

    Returns
    -------
    str
        The CRYST1 line, with space group P 1 and Z value 1.
    """
    a, b, c = (length * 10.0 for length in box.lengths)
    alpha, beta, gamma = box.angles
    return (
        f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
        f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1"
    )


def _atom_and_ter_lines(protein):
    """Lay out the coordinate section of the file.

    Residues are written sorted by number: template readers form
    polymer links only between record-adjacent residues, so backbone
    order in the file must follow residue numbers, not attachment
    order. Adjacency is a property of the record order alone. The
    wwPDB Format Guide v3.30, section 9 (Coordinate Section,
    ATOM/HETATM/TER), states that the records of one chain follow each
    other in sequence order and that a TER record closes the chain, so
    a reader takes the polymer sequence from the record order and the
    TER records.

    Parameters
    ----------
    protein : mbuild.biopolymers.Protein
        The protein to lay out.

    Returns
    -------
    tuple
        ``(lines, particle_serial, particle_residue, residue_order)``.
        ``lines`` holds the ATOM, HETATM and TER records. The three
        maps give the serial and the residue of every written
        particle, and the ``(chain id, position)`` key of every
        residue, which ``_conect_lines`` reads.
    """
    lines = []
    serial = 0
    particle_serial = {}
    particle_residue = {}
    residue_order = {}
    for chain in protein.chains:
        residue = None
        for index, residue in enumerate(
            sorted(
                protein.residues(chain.chain_id),
                key=lambda res: (res.resnum, res.icode),
            )
        ):
            # The order key holds the chain, so the peptide-bond test
            # in _conect_lines cannot pair residues across a TER.
            residue_order[id(residue)] = (chain.chain_id, index)
            if residue.resnum > 9999:
                raise MBuildError(
                    "PDB residue numbers larger than 9999 are not supported."
                )
            for particle in residue.particles():
                serial += 1
                if serial > 99999:
                    raise MBuildError(
                        "PDB atom serials larger than 99999 are not supported."
                    )
                particle_serial[particle] = serial
                particle_residue[particle] = residue
                lines.append(_pdb_atom_line(serial, particle, residue, chain.chain_id))
        if residue is not None:
            serial += 1
            lines.append(
                f"TER   {serial:5d}      {residue.name:<3s} "
                f"{chain.chain_id or ' ':1s}{residue.resnum:4d}"
                f"{residue.icode or ' ':1s}"
            )
    return lines, particle_serial, particle_residue, residue_order


def _check_residue_membership(particles, assigned):
    """Raise when a particle of a protein sits outside every Residue.

    ``Protein.to_rdkit`` and ``write_pdb`` both resolve each particle
    through its residue, so a particle outside a ``Chain -> Residue``
    path would leave the export, together with its bonds. Both exports
    stop here instead of writing an incomplete structure.

    Parameters
    ----------
    particles : iterable of mbuild.Compound
        The particles of the protein.
    assigned : container
        The particles that a Residue claims. The check reports every
        particle that is not in it.
    """
    orphans = [particle for particle in particles if particle not in assigned]
    if orphans:
        raise MBuildError(
            "Every atom of a Protein must belong to a Residue, but "
            f"{[p.name for p in orphans[:5]]} "
            f"{'(and more) ' if len(orphans) > 5 else ''}do not. Add "
            "atoms through attach() or into a Residue, not directly "
            "onto the Protein."
        )


def _pdb_name_field(name):
    """Return the atom name in the four columns a PDB record gives it.

    The wwPDB Format Guide v3.30, section 9 (Coordinate Section, ATOM),
    puts the atom name in columns 13-16 and the element symbol,
    right-justified, in columns 13-14. A name of three characters or
    less therefore starts in column 14, and a four-character name fills
    the field. ``Protein.to_rdkit`` writes the same field into the
    RDKit PDB residue info, so both exports name atoms alike.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    str
        The padded name field.
    """
    return name.center(4) if len(name) >= 4 else f" {name:<3s}"


def _pdb_atom_line(serial, particle, residue, chain_id):
    """Format one ATOM or HETATM record for the particle."""
    record = "HETATM" if residue.hetatm else "ATOM  "
    name_field = _pdb_name_field(particle.name)
    x, y, z = particle.pos * 10.0
    element = particle.element.symbol.upper() if particle.element else ""
    return (
        f"{record}{serial:5d} {name_field[:4]} {residue.name:<3.3s} "
        f"{chain_id or ' ':1.1s}{residue.resnum:4d}{residue.icode or ' ':1.1s}"
        f"   {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
        f"          {element:>2.2s}"
    )


def _conect_lines(protein, particle_serial, particle_residue, residue_order):
    """Yield CONECT lines, following the RCSB convention.

    The wwPDB Format Guide v3.30, section 10 (Connectivity Section,
    CONECT), states that CONECT records give the connectivity of
    HETATM residues and of bonds that the standard residue chemistry
    does not describe, such as disulfide bridges. It also states that
    a CONECT record holds one atom serial plus up to four bonded
    serials, so a wider set of partners needs more than one record.

    Every bond that touches a HETATM residue is listed, because PDB
    viewers (e.g. PyMOL) treat CONECT records as the complete bond
    list for HETATM atoms and skip distance-based perception for
    them. Bonds between different ATOM residues are listed too
    (disulfides), except the peptide bond, which residue adjacency
    implies. A bond is a peptide bond only when its C atom belongs to
    a residue and its N atom belongs to the next residue of the same
    chain; any other C-N bond gets a CONECT record. Strict template
    readers accept these records because their residue definitions
    predict all of them.
    """
    partners = {}
    for particle1, particle2 in protein.bonds():
        residue1 = particle_residue.get(particle1)
        residue2 = particle_residue.get(particle2)
        if residue1 is None or residue2 is None:
            continue
        if residue1 is residue2:
            if not residue1.hetatm:
                continue
        elif not (residue1.hetatm or residue2.hetatm):
            chain1, index1 = residue_order[id(residue1)]
            chain2, index2 = residue_order[id(residue2)]
            if chain1 == chain2 and index2 - index1 == 1:
                earlier, later = particle1, particle2
            elif chain1 == chain2 and index1 - index2 == 1:
                earlier, later = particle2, particle1
            else:
                earlier = later = None
            if earlier is not None and (earlier.name, later.name) == ("C", "N"):
                continue  # implied peptide bond
        serial1 = particle_serial[particle1]
        serial2 = particle_serial[particle2]
        partners.setdefault(serial1, []).append(serial2)
        partners.setdefault(serial2, []).append(serial1)
    for serial in sorted(partners):
        bonded = sorted(partners[serial])
        for start in range(0, len(bonded), 4):
            chunk = bonded[start : start + 4]
            yield (
                "CONECT" + f"{serial:5d}" + "".join(f"{other:5d}" for other in chunk)
            )
