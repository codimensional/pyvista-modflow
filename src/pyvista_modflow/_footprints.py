"""Read cell footprints (plan-view x, y outlines) from a GSF or a polygon shapefile."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyvista_modflow._arrays import FloatArray, IntArray, offsets, ragged_positions, unique_rows

if TYPE_CHECKING:
    from typing import TextIO

# A GSF node record is "inode x y z lay m ivert(1..m)": six fields before the vertex ids.
_GSF_RECORD_HEAD = 6
_MIN_CORNERS = 3
_SHP_POLYGONS = (5, 15, 25)  # Polygon, PolygonZ, PolygonM
_SHP_HEADER = 100
_SHP_RECORD_HEADER = 8
_DBF_FIELD_DESCRIPTOR = 32
_DBF_HEADER_END = 0x0D
_DBF_LIVE_RECORD = ord(' ')


@dataclass(frozen=True, eq=False)
class Footprints:
    """Plan-view cell footprints as rings of shared corners.

    Attributes
    ----------
    corner_xy : numpy.ndarray
        ``(ncorner, 2)`` distinct corner coordinates.
    ring : numpy.ndarray
        Corner ids of every footprint, concatenated.
    nside : numpy.ndarray
        Number of corners in each footprint.
    layer : numpy.ndarray, optional
        Layer of each footprint when the source assigns one (a GSF lists every node).
    crs_wkt : str, optional
        Coordinate reference system, when the source carries one.

    """

    corner_xy: FloatArray
    ring: IntArray
    nside: IntArray
    layer: IntArray | None = None
    crs_wkt: str | None = None


def _corner_ids(xy: FloatArray) -> tuple[FloatArray, IntArray]:
    """Return the distinct x/y corners (exact match) and each row's corner id."""
    first, inverse = unique_rows(xy[:, 0], xy[:, 1])
    return xy[first], inverse


def read_gsf(path: str | Path) -> Footprints:
    """Read node footprints from a MODFLOW-USG grid specification file (GSF).

    Vertex elevations are ignored: a GSF vertex carries one z shared by every cell that
    touches it, so it cannot hold the cells' own tops and bottoms.

    Parameters
    ----------
    path : str or pathlib.Path
        The GSF file.

    Returns
    -------
    Footprints
        One footprint per node, in node order.

    Raises
    ------
    ValueError
        If the file is not a GSF or its node records are malformed.

    """
    path = Path(path)
    with path.open(encoding='utf-8-sig', errors='replace') as f:
        try:
            return _parse_gsf(f)
        except IndexError:
            msg = f'{path}: the grid specification file ends early'
            raise ValueError(msg) from None
        except ValueError as err:
            msg = f'{path}: {err}'
            raise ValueError(msg) from None


def _parse_gsf(f: TextIO) -> Footprints:
    """Parse an open GSF; errors leave out the path, which the caller adds."""
    if f.readline().upper().split()[:1] != ['UNSTRUCTURED']:
        msg = 'not a grid specification file (first line must be UNSTRUCTURED)'
        raise ValueError(msg)
    nnode = int(f.readline().split()[0])
    nvert = int(f.readline().split()[0])
    verts = np.array(' '.join(islice(f, nvert)).split(), dtype=float)
    tokens = np.array(f.read().split(), dtype=float)
    if verts.size != 3 * nvert:
        msg = f'expected {nvert} vertices of x y z'
        raise ValueError(msg)
    xyz = verts.reshape(nvert, 3)

    # Each record's start depends on the previous record's vertex count.
    start = np.empty(nnode, np.int64)
    pos = 0
    for i in range(nnode):
        start[i] = pos
        pos += _GSF_RECORD_HEAD + int(tokens[pos + 5])
    if pos != tokens.size or not np.array_equal(tokens[start], np.arange(1, nnode + 1)):
        msg = f'node records are not numbered 1..{nnode} in order'
        raise ValueError(msg)
    m = tokens[start + 5].astype(np.int64)
    if np.any(m % 2) or np.any(m < 2 * _MIN_CORNERS):
        msg = 'a node must list a top and a bottom ring of at least three vertices each'
        raise ValueError(msg)
    nside = m // 2
    ring_pos = ragged_positions(start + _GSF_RECORD_HEAD, nside)
    top_ring = tokens[ring_pos].astype(np.int64) - 1
    bot_ring = tokens[ring_pos + np.repeat(nside, nside)].astype(np.int64) - 1
    if not np.array_equal(xyz[top_ring, :2], xyz[bot_ring, :2]):
        msg = "a node's bottom ring does not lie under its top ring"
        raise ValueError(msg)
    corner_xy, vert_corner = _corner_ids(xyz[:, :2])
    layer = tokens[start + 4].astype(np.int64)
    return Footprints(corner_xy, vert_corner[top_ring], nside, layer=layer)


def _read_dbf_field(path: Path, name: str) -> IntArray:
    """Return one field of a dBASE table whose values are all whole numbers."""
    buf = np.fromfile(path, np.uint8)
    nrec = int(buf[4:8].view('<u4')[0])
    header_len, rec_len = (int(v) for v in buf[8:12].view('<u2'))
    fields: dict[str, tuple[int, int]] = {}  # name -> (column, width)
    pos, col = _DBF_FIELD_DESCRIPTOR, 1  # each record opens with a one-byte deletion flag
    while buf[pos] != _DBF_HEADER_END:
        # Latin-1 decodes any byte, so a name in a Windows code page cannot fail.
        field = bytes(buf[pos : pos + 11]).split(b'\0')[0].decode('latin-1')
        width = int(buf[pos + 16])
        fields[field] = (col, width)
        col += width
        pos += _DBF_FIELD_DESCRIPTOR
    match = next((f for f in fields if f.upper() == name.upper()), None)
    if match is None:
        msg = f'{path}: no field named {name!r}; the fields are {", ".join(fields)}'
        raise ValueError(msg)
    col, width = fields[match]
    recs = buf[header_len : header_len + nrec * rec_len].reshape(nrec, rec_len)
    if np.any(recs[:, 0] != _DBF_LIVE_RECORD):
        msg = f'{path}: the table has deleted records'
        raise ValueError(msg)
    msg = f'{path}: field {match!r} does not hold whole numbers'
    # Numeric (N, F) and text (C) fields both store numbers as text.
    try:
        values = recs[:, col : col + width].copy().view(f'S{width}').ravel().astype(float)
    except ValueError:
        raise ValueError(msg) from None
    if np.any(values != np.round(values)):
        raise ValueError(msg)
    return values.astype(np.int64)


def _sidecar(path: Path, suffix: str) -> Path:
    """Return the shapefile part beside ``path``, preferring the case of its suffix."""
    upper, lower = path.with_suffix(suffix.upper()), path.with_suffix(suffix)
    first, second = (upper, lower) if path.suffix.isupper() else (lower, upper)
    return second if not first.exists() and second.exists() else first


def _int32_at(buf: npt.NDArray[np.uint8], at: IntArray) -> IntArray:
    """Return the little-endian int32 at each byte offset."""
    return buf[at[:, None] + np.arange(4)].view('<i4').ravel().astype(np.int64)


def read_shapefile(path: str | Path, *, id_field: str = 'ID') -> Footprints:
    """Read one layer's node footprints from a polygon shapefile.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.shp`` file; its ``.shx`` and ``.dbf`` must sit beside it, and a ``.prj``
        there is read as the coordinate reference system.
    id_field : str, default: 'ID'
        The attribute holding each polygon's node number, ``1 .. n``.

    Returns
    -------
    Footprints
        One footprint per polygon, ordered by ``id_field``.

    Raises
    ------
    ValueError
        If a record is not a single-ring polygon or ``id_field`` is not ``1 .. n``.

    """
    path = Path(path)
    buf = np.fromfile(path, np.uint8)
    index = np.fromfile(_sidecar(path, '.shx'), np.uint8)
    # The .shx gives each record's offset in 16-bit words, big-endian.
    content = index[_SHP_HEADER:].view('>i4').reshape(-1, 2)[:, 0].astype(np.int64) * 2
    content += _SHP_RECORD_HEADER

    shape_type = _int32_at(buf, content)
    if not np.all(np.isin(shape_type, _SHP_POLYGONS)):
        msg = f'{path}: every record must be a Polygon; found shape types {np.unique(shape_type)}'
        raise ValueError(msg)
    # Polygon content: shape type, bounding box (4 doubles), NumParts, NumPoints, parts, points
    # (PolygonZ and PolygonM append their z and m after the points).
    nparts, npoints = _int32_at(buf, content + 36), _int32_at(buf, content + 40)
    if np.any(nparts != 1):
        msg = f'{path}: {np.count_nonzero(nparts != 1)} polygons have holes or several parts'
        raise ValueError(msg)
    xy = buf[ragged_positions(content + 44 + 4 * nparts, 16 * npoints)].view('<f8').reshape(-1, 2)

    # Rings are closed (the last point repeats the first); drop the repeat.
    first = offsets(npoints)[:-1]
    if not np.array_equal(xy[first], xy[first + npoints - 1]):
        msg = f'{path}: a polygon ring is not closed'
        raise ValueError(msg)
    nside = npoints - 1
    corner_xy, ring = _corner_ids(xy[ragged_positions(first, nside)])

    ids = _read_dbf_field(_sidecar(path, '.dbf'), id_field)
    if ids.size != nside.size or not np.array_equal(np.sort(ids), np.arange(1, ids.size + 1)):
        msg = f'{path}: {id_field} is not a permutation of 1..{nside.size}'
        raise ValueError(msg)
    order = np.argsort(ids)
    ring = ring[ragged_positions(offsets(nside)[:-1][order], nside[order])]
    prj = _sidecar(path, '.prj')
    crs = prj.read_text(encoding='utf-8-sig', errors='replace').strip() if prj.exists() else ''
    return Footprints(corner_xy, ring, nside[order], crs_wkt=crs or None)
