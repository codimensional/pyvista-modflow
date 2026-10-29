"""Geometry oracles computed from a grid's raw arrays, independent of the reader's own code."""

from __future__ import annotations

import numpy as np
import pyvista as pv


def _unpack(stream):
    """Split a ``[n, v0 .. vn-1, n, ...]`` stream into its runs' lengths and values."""
    lengths, values, i = [], [], 0
    while i < len(stream):
        n = int(stream[i])
        lengths.append(n)
        values.extend(stream[i + 1 : i + 1 + n])
        i += n + 1
    return np.array(lengths, np.int64), np.array(values, np.int64)


def _faces(grid):
    """Return each face's cell, the point ids of every face (concatenated) and their counts."""
    face_len, point_ids = _unpack(grid.polyhedron_faces)
    faces_per_cell, face_ids = _unpack(grid.polyhedron_face_locations)
    cell_of_face = np.empty(face_ids.size, np.int64)
    cell_of_face[face_ids] = np.repeat(np.arange(grid.n_cells), faces_per_cell)
    return cell_of_face, point_ids, face_len


def _fans(face_len):
    """Index (face, k) of every fan triangle (v0, vk, vk+1) of every face."""
    ntri = face_len - 2
    face = np.repeat(np.arange(face_len.size), ntri)
    k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
    return face, k


def cell_volumes(grid):
    """Volume of every polyhedron by the divergence theorem over its faces."""
    cell_of_face, ids, face_len = _faces(grid)
    pts = np.asarray(grid.points)
    start = np.concatenate([[0], np.cumsum(face_len)[:-1]])
    face, k = _fans(face_len)
    origin = pts[ids[start[face]]]
    p1, p2 = pts[ids[start[face] + k]], pts[ids[start[face] + k + 1]]
    # Measure from one point of each cell so the sum stays well conditioned.
    ref = np.zeros((grid.n_cells, 3))
    ref[cell_of_face[face]] = origin
    r = ref[cell_of_face[face]]
    six_v = np.einsum('ij,ij->i', origin - r, np.cross(p1 - r, p2 - r))
    return np.bincount(cell_of_face[face], weights=six_v, minlength=grid.n_cells) / 6.0


def unshared_face_area(grid, decimals=6):
    """Total area of the faces no other cell has, matching faces by their points' coordinates."""
    _, ids, face_len = _faces(grid)
    pts = np.round(np.asarray(grid.points), decimals)
    start = np.concatenate([[0], np.cumsum(face_len)[:-1]])
    keys = {}
    areas = []
    for f, (s, n) in enumerate(zip(start, face_len)):
        poly = pts[ids[s : s + n]]
        key = tuple(sorted(map(tuple, poly)))
        keys.setdefault(key, []).append(f)
        # Newell's method for the area of a planar polygon in 3D.
        areas.append(0.5 * np.linalg.norm(np.cross(poly, np.roll(poly, -1, axis=0)).sum(axis=0)))
    areas = np.array(areas)
    return sum(areas[fs[0]] for fs in keys.values() if len(fs) == 1)


def box_exterior_area(lo, hi, atol=1e-6):
    """Exterior area of a union of axis-aligned boxes that touch but never overlap.

    Every box face counts, less twice the area two boxes share across a common plane
    (planes within ``atol``, absolute, since the coordinates are UTM-sized).
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ext = hi - lo
    total = 2 * (ext[:, 0] * ext[:, 1] + ext[:, 1] * ext[:, 2] + ext[:, 0] * ext[:, 2]).sum()
    contact = 0.0
    for i in range(len(lo) - 1):
        olo, ohi = np.maximum(lo[i], lo[i + 1 :]), np.minimum(hi[i], hi[i + 1 :])
        overlap = np.clip(ohi - olo, 0, None)
        for axis in range(3):
            touch = (np.abs(hi[i, axis] - lo[i + 1 :, axis]) <= atol) | (
                np.abs(lo[i, axis] - hi[i + 1 :, axis]) <= atol
            )
            others = [a for a in range(3) if a != axis]
            contact += (touch * overlap[:, others[0]] * overlap[:, others[1]]).sum()
    return total - 2 * contact


def independent_prisms(rings, top, bot):
    """One prism per node with private points: the unsplit construction a conforming grid fixes."""
    cells, points = [], []
    for ring, t, b in zip(rings, top, bot):
        n = len(ring)
        base = len(points)
        points += [(x, y, b) for x, y in ring] + [(x, y, t) for x, y in ring]
        faces = [list(range(base, base + n))[::-1], list(range(base + n, base + 2 * n))]
        faces += [
            [base + i, base + (i + 1) % n, base + n + (i + 1) % n, base + n + i] for i in range(n)
        ]
        cells.append([len(faces)] + [v for f in faces for v in (len(f), *f)])
    flat = []
    for c in cells:
        flat += [len(c), *c]
    return pv.UnstructuredGrid(
        flat, [pv.CellType.POLYHEDRON] * len(cells), np.array(points, float)
    )
