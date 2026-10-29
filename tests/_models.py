"""Writers for small synthetic MODFLOW-USG grids, and one stepped, refined model."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

DATA = Path(__file__).parent / 'data'


def write_disu(path, nodelay, top, bot, area, ivsd=-1):
    """Write a minimal DISU: item 1, LAYCBD, NODELAY, TOP, BOT, AREA (connections are not read)."""
    nlay = len(nodelay)
    lines = [f'{sum(nodelay)} {nlay} 0 {ivsd} 1 4 2 0', ' '.join(['0'] * nlay)]
    lines.append('INTERNAL 1 (FREE) -1 NODELAY')
    lines.append(' '.join(str(n) for n in nodelay))
    start = np.concatenate([[0], np.cumsum(nodelay)])
    for arr in (top, bot):
        for k in range(nlay):
            lines.append(f'INTERNAL 1.0 (FREE) -1 layer {k + 1}')
            lines.append(' '.join(repr(float(v)) for v in arr[start[k] : start[k + 1]]))
    for k in range(1 if ivsd == -1 else nlay):
        lines.append('INTERNAL 1.0 (FREE) -1 AREA')
        lines.append(' '.join(repr(float(v)) for v in area[start[k] : start[k + 1]]))
    Path(path).write_text('\n'.join(lines) + '\n')


def write_gsf(path, rings, layer):
    """Write a GSF with private vertices: each node lists its top ring then its bottom ring."""
    verts, records = [], []
    for i, (ring, lay) in enumerate(zip(rings, layer), start=1):
        ids = []
        for z in (1.0, 0.0):  # vertex z is ignored by the reader, so any value works
            for x, y in ring:
                verts.append((x, y, z))
                ids.append(len(verts))
        cx, cy = np.mean(ring, axis=0)
        records.append(
            f'{i} {float(cx)!r} {float(cy)!r} 0.5 {lay} {len(ids)} ' + ' '.join(map(str, ids))
        )
    head = ['UNSTRUCTURED GWF', f'{len(rings)} {max(layer)} 1 1', str(len(verts))]
    body = [f'{x!r} {y!r} {z!r}' for x, y, z in verts]
    Path(path).write_text('\n'.join(head + body + records) + '\n')


def write_shapefile(path, rings, ids, prj=None, id_decimals=0, field=b'ID', kind=b'N'):
    """Write single-part polygons (``.shp``, ``.shx``, ``.dbf``) with one ID field.

    Outer rings are clockwise, as the shapefile specification requires.
    """
    path = Path(path)
    records, offsets = [], []
    pos = 100
    for ring in rings:
        pts = [ring[0], *ring[:0:-1], ring[0]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        body = struct.pack('<i4dii', 5, min(xs), min(ys), max(xs), max(ys), 1, len(pts))
        body += struct.pack('<i', 0) + b''.join(struct.pack('<2d', *p) for p in pts)
        offsets.append((pos // 2, len(body) // 2))
        records.append(struct.pack('>2i', len(records) + 1, len(body) // 2) + body)
        pos += 8 + len(body)
    allx = [p[0] for r in rings for p in r]
    ally = [p[1] for r in rings for p in r]
    bbox = struct.pack('<4d', min(allx), min(ally), max(allx), max(ally)) + bytes(32)

    def header(length_bytes):
        return (
            struct.pack('>7i', 9994, 0, 0, 0, 0, 0, length_bytes // 2)
            + struct.pack('<2i', 1000, 5)
            + bbox
        )

    shp = b''.join(records)
    path.write_bytes(header(100 + len(shp)) + shp)
    shx = b''.join(struct.pack('>2i', o, n) for o, n in offsets)
    path.with_suffix('.shx').write_bytes(header(100 + len(shx)) + shx)
    width = 12
    field = field.ljust(11, b'\0') + kind + bytes(4) + bytes([width, id_decimals]) + bytes(14)
    dbf_head = struct.pack('<4BIHH', 3, 126, 1, 1, len(ids), 32 + 32 + 1, 1 + width) + bytes(20)
    rows = b''.join(b' ' + f'{i:{width}.{id_decimals}f}'.encode() for i in ids)
    path.with_suffix('.dbf').write_bytes(dbf_head + field + b'\r' + rows + b'\x1a')
    if prj is not None:
        path.with_suffix('.prj').write_text(prj)


# A coarse cell beside two fine ones, so the coarse cell's east edge has a corner part way
# along it (a quadtree T-junction), at UTM-sized coordinates.
X0, Y0 = 512_000.0, 6_420_000.0
STEPPED_RINGS = [
    [(X0, Y0), (X0 + 40, Y0), (X0 + 40, Y0 + 40), (X0, Y0 + 40)],
    [(X0 + 40, Y0), (X0 + 60, Y0), (X0 + 60, Y0 + 20), (X0 + 40, Y0 + 20)],
    [(X0 + 40, Y0 + 20), (X0 + 60, Y0 + 20), (X0 + 60, Y0 + 40), (X0 + 40, Y0 + 40)],
]
STEPPED_AREA = np.array([1600.0, 400.0, 400.0] * 2)
# Two layers per column, contiguous within a column and breaking at different elevations
# from column to column.
STEPPED_TOP = np.array([310.0, 307.5, 304.0, 296.0, 299.5, 293.0])
STEPPED_BOT = np.array([296.0, 299.5, 293.0, 281.0, 288.25, 284.5])
