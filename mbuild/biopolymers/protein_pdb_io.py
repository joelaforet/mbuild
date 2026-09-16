"""PDB record parsing for protein structures.

The reader is strict about the records the template-matching loader
needs: atom serials (including hybrid-36), TER records, CONECT records,
insertion codes, altLoc selection, and the CRYST1 unit cell.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from mbuild.box import Box
from mbuild.exceptions import MBuildError

logger = logging.getLogger(__name__)


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


def _format_residue_label(resname, chain_id, resnum, icode):
    """Return one residue identity as text, such as ``CYS A:22``.

    Every error and every warning of the biopolymers package names a
    residue in this one format.

    Parameters
    ----------
    resname : str
        Residue name.
    chain_id : str
        Chain identifier.
    resnum : int
        Residue sequence number.
    icode : str
        Insertion code, or an empty string.

    Returns
    -------
    str
        The residue name, the chain identifier, and the residue number
        with its insertion code.
    """
    return f"{resname} {chain_id}:{resnum}{icode}"


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
        return _format_residue_label(
            self.resname, self.chain_id, self.resnum, self.icode
        )


def _decode_index(field, width, line_number):
    """Decode an atom serial or a residue number from its PDB columns.

    The wwPDB Format Guide v3.30, section 9 (Coordinate Section,
    ATOM/HETATM), gives the atom serial number five columns (7-11) and
    the residue sequence number four columns (23-26):
    https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
    A system with more than 99999 atoms, or a chain with more than
    9999 residues, does not fit in those columns. OpenMM writes a
    value that does not fit in hexadecimal, in
    ``openmm/app/pdbfile.py::_formatIndex``:
    https://github.com/openmm/openmm/blob/05472c9a812927c863be67abbb3376e944b2c7ef/wrappers/python/openmm/app/pdbfile.py#L482-L491
    A value below ``10 ** width`` fills the field in decimal, and a
    larger value fills it with
    ``value - 10 ** width + 10 * 16 ** (width - 1)`` in hexadecimal.
    Atom serials therefore run 99999, A0000, A0001 up to AFFFF, then
    B0000. Residue numbers run 9999, A000, A001, and so on. Any system
    above 99999 atoms or 9999 residues carries such fields. Section 10
    (Connectivity Section, CONECT) gives the CONECT atom serials the
    same five columns. OpenMM encodes them by the same rule, so those
    fields need this function too.

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
    line_number : int
        The number of the line the field comes from, counted from 1.
        The error message names it, so that the user can find the
        record in a file of millions of lines.

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
            f"PDB field {field!r} on line {line_number} is not a decimal "
            "or hexadecimal number."
        )
    return shifted + 10**width - 10 * 16 ** (width - 1)


def _encode_index(value, width, name):
    """Format an atom serial or a residue number into its PDB columns.

    This function is the inverse of ``_decode_index``. It follows
    OpenMM's ``openmm/app/pdbfile.py::_formatIndex``. A value below
    ``10 ** width`` fills the field in decimal. A larger value fills it
    with ``value - 10 ** width + 10 * 16 ** (width - 1)`` in
    hexadecimal. A reader that knows the rule, such as this module or
    OpenMM, reads the file back.

    The OpenMM rule takes the shifted value modulo ``16 ** width``, so
    the encoding repeats above ``10 ** width + 6 * 16 ** (width - 1)``.
    A repeated value cannot be decoded, so this function raises instead
    of writing a file that no reader can read back.

    Parameters
    ----------
    value : int
        The atom serial number or residue sequence number.
    width : int
        The number of columns of the field: 5 for an atom serial
        number, 4 for a residue sequence number.
    name : str
        The name of the field, for the error message.

    Returns
    -------
    str
        The field, of exactly ``width`` characters.

    Raises
    ------
    MBuildError
        If the value is above the range the encoding covers.
    """
    if value < 10**width:
        return f"{value:{width}d}"
    limit = 10**width + 6 * 16 ** (width - 1)
    if value >= limit:
        raise MBuildError(
            f"PDB {name} {value} is too large to write. The field has "
            f"{width} columns, and the hexadecimal encoding of OpenMM "
            f"covers values below {limit} only. Above that value the "
            "encoding repeats, so no reader could read the file back."
        )
    return f"{value - 10**width + 10 * 16 ** (width - 1):{width}X}"


def _parse_pdb(text):
    """Parse ATOM/HETATM/TER/CONECT/CRYST1 records of the first model.

    A disordered atom carries an alternate location indicator in
    column 17, which the wwPDB Format Guide v3.30, section 9
    (Coordinate Section, ATOM), names altLoc:
    https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
    One conformer only is read per residue. A residue keeps the
    records whose altLoc is blank. It also keeps the records that
    carry its own first non-blank altLoc. The other conformers are
    skipped, and one warning names the residues that lose them.

    Returns a tuple ``(residues, conects, box)``: the ``_PdbResidue``
    groups in file order, the CONECT pairs as a set of frozensets of
    two serials, and the ``Box`` or None.
    """
    residues = []
    conects = set()
    box = None
    kept_alt_loc_by_key = {}
    skipped_alt_loc_keys = {}
    in_extra_model = False
    last_key = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        record_type = line[:6]
        if record_type == "ENDMDL":
            in_extra_model = True
        elif record_type.startswith("MODEL") and in_extra_model:
            logger.warning("PDB file has multiple models; only model 1 is read.")
        elif record_type in ("ATOM  ", "HETATM") and not in_extra_model:
            record = _PdbRecord(
                serial=_decode_index(line[6:11], 5, line_number),
                name=line[12:16].strip(),
                # The wwPDB Format Guide v3.30, section 9
                # (Coordinate Section, ATOM), declares the residue
                # name in columns 18-20 and column 21 blank:
                # https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
                # A four-character name, such as a lipid name that a
                # membrane builder writes, fills column 21 too, so
                # the field is read over all four columns and
                # stripped. A three-character name is unchanged.
                # openff-pablo reads the same four columns when it
                # is not strict (_pdb_data.py):
                # https://github.com/openforcefield/openff-pablo/blob/main/openff/pablo/_pdb_data.py
                resname=line[17:21].strip(),
                chain_id=line[21].strip(),
                resnum=_decode_index(line[22:26], 4, line_number),
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
            # loader builds one structure, so every residue keeps its
            # blank records and its own first non-blank letter, and it
            # skips the rest. The letters belong to the disorder group
            # of one residue. A letter chosen over the whole file would
            # drop every record of a residue that uses other letters.
            # openff-pablo follows another rule
            # (_pdb_data.py, _allowed_alt_locs). It collects the
            # altLoc values of the whole file. If the file holds one
            # value, pablo keeps that value. If the file holds more
            # than one, pablo keeps the blank value and the letter 'A'
            # only, and it warns. Take a file whose records carry
            # blank, 'B' and 'C'. Pablo keeps the blank records alone.
            # This reader keeps the blank records and the first letter
            # of each residue.
            alt_loc = line[16].strip()
            if alt_loc:
                kept = kept_alt_loc_by_key.setdefault(key, alt_loc)
                if alt_loc != kept:
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
            serials = [
                _decode_index(value, 5, line_number) for value in fields if value
            ]
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
        labels = ", ".join(
            f"{_PdbResidue(*key).label} keeps {kept_alt_loc_by_key[key]!r}"
            for key in skipped_alt_loc_keys
        )
        logger.warning(
            f"One alternate location is read per residue: {labels}. "
            "The other conformers of these residues are skipped."
        )
    if not residues:
        raise MBuildError("No ATOM or HETATM records found in the PDB file.")
    return residues, conects, box
