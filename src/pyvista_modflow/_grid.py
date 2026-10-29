"""Build a conforming polyhedral grid from footprints and per-node elevations."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from pyvista_modflow._arrays import (
    FloatArray,
    IntArray,
    offsets,
    ragged_positions,
    unique,
    unique_rows,
)

_MIN_CORNERS = 3
# A corner this close to an edge lies on it: a fraction of the edge's length, but at least
# 0.01 units, since grid files round coordinates to two or three decimals and a rotated
# grid's corners then miss the line through its neighbors' edges.
_COLLINEAR_RTOL = 1e-6
_COLLINEAR_ATOL = 1e-2
# SetPolyhedralCells and the face-stream storage it fills arrived in VTK 9.4.
_POLYHEDRAL_VTK = (9, 4)


def _ring_links(nside: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Return each ring's start, each entry's successor and each entry's mirror in its ring."""
    start = offsets(nside)[:-1]
    succ = np.arange(nside.sum()) + 1
    succ[start + nside - 1] = start
    j = np.arange(succ.size) - np.repeat(start, nside)
    return start, succ, np.repeat(start + nside - 1, nside) - j


def _insert_hanging_corners(
    corner_xy: FloatArray, ring: IntArray, nside: IntArray, layer: IntArray
) -> tuple[IntArray, IntArray]:
    """Add to each counterclockwise footprint the corners that lie inside its edges.

    A refined grid (quadtree, nested) lists only a coarse cell's own corners, while its
    finer neighbors put a corner part way along the shared edge. Only an edge no footprint
    in the same layer runs in reverse can hold one, and only a corner of such an edge can
    be one; layers can be refined differently.
    """
    _, succ, _ = _ring_links(nside)
    a, b = ring, ring[succ]
    ncorner = corner_xy.shape[0]
    fwd = a * ncorner + b
    in_layer = np.repeat(layer, nside) * ncorner * ncorner
    sorted_fwd = np.sort(in_layer + fwd)
    rev = in_layer + b * ncorner + a
    at = np.minimum(np.searchsorted(sorted_fwd, rev), sorted_fwd.size - 1)
    he = np.flatnonzero(sorted_fwd[at] != rev)
    if he.size == 0:
        return ring, nside
    # Each distinct lone edge is searched once; layers repeat them.
    first, he_edge = unique_rows(fwd[he])
    ea, eb = a[he[first]], b[he[first]]
    cand = unique(np.concatenate([ea, eb]))
    cand = cand[np.argsort(corner_xy[cand, 0], kind='stable')]
    cx = corner_xy[cand, 0]

    # Candidates whose x lies within each lone edge's x extent, as (edge, candidate) pairs.
    pa, pb = corner_xy[ea], corner_xy[eb]
    seg = pb - pa
    length = np.hypot(seg[:, 0], seg[:, 1])
    tol = np.maximum(_COLLINEAR_RTOL * length, _COLLINEAR_ATOL)
    x0, x1 = np.minimum(pa[:, 0], pb[:, 0]) - tol, np.maximum(pa[:, 0], pb[:, 0]) + tol
    i0, i1 = np.searchsorted(cx, x0, 'left'), np.searchsorted(cx, x1, 'right')
    pair_e = np.repeat(np.arange(ea.size), i1 - i0)
    pair_c = cand[ragged_positions(i0, i1 - i0)]
    rel = corner_xy[pair_c] - pa[pair_e]
    t = (rel[:, 0] * seg[pair_e, 0] + rel[:, 1] * seg[pair_e, 1]) / length[pair_e] ** 2
    dist = np.abs(rel[:, 0] * seg[pair_e, 1] - rel[:, 1] * seg[pair_e, 0]) / length[pair_e]
    margin = tol[pair_e] / length[pair_e]
    hit = (dist <= tol[pair_e]) & (t > margin) & (t < 1 - margin)
    if not hit.any():
        return ring, nside
    by_edge = np.argsort(pair_e[hit], kind='stable')
    hit_e, hit_c, hit_t = pair_e[hit][by_edge], pair_c[hit][by_edge], t[hit][by_edge]
    nhit = np.bincount(hit_e, minlength=ea.size)

    # Every lone half-edge takes its edge's hits: its start corner (t = 0), then the hanging
    # corners by t.
    count = nhit[he_edge]
    src = ragged_positions(offsets(nhit)[:-1][he_edge], count)
    slot = np.concatenate([np.arange(ring.size), np.repeat(he, count)])
    along = np.concatenate([np.zeros(ring.size), hit_t[src]])
    corner = np.concatenate([ring, hit_c[src]])
    order = np.lexsort((along, slot))
    cell = np.repeat(np.arange(nside.size), nside)
    return corner[order], np.bincount(cell[slot], minlength=nside.size).astype(np.int64)


def _prepare_rings(
    corner_xy: FloatArray, ring: IntArray, nside: IntArray, layer: IntArray
) -> tuple[IntArray, IntArray, FloatArray]:
    """Return counterclockwise footprints with hanging corners added, and their areas."""
    _, succ, _ = _ring_links(nside)
    keep = ring != ring[succ]  # a digitized outline can repeat a corner
    if not keep.all():
        cell = np.repeat(np.arange(nside.size), nside)
        ring, nside = ring[keep], np.bincount(cell[keep], minlength=nside.size).astype(np.int64)
    if np.any(nside < _MIN_CORNERS):
        msg = f'node {int(np.argmax(nside < _MIN_CORNERS)) + 1} has fewer than three corners'
        raise ValueError(msg)
    start, succ, flipped = _ring_links(nside)
    # Shoelace area relative to each ring's first corner, so map coordinates keep precision.
    xy = corner_xy[ring] - np.repeat(corner_xy[ring[start]], nside, axis=0)
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * np.add.reduceat(x * y[succ] - x[succ] * y, start)
    ring = np.where(np.repeat(area < 0, nside), ring[flipped], ring)
    ring, nside = _insert_hanging_corners(corner_xy, ring, nside, layer)
    return ring, nside, np.abs(area)


def build_grid(  # noqa: PLR0913, PLR0915 -- one vectorized pass that frees large arrays as it goes
    corner_xy: FloatArray,
    ring: IntArray,
    nside: IntArray,
    top: FloatArray,
    bot: FloatArray,
    layer: IntArray,
) -> tuple[pv.UnstructuredGrid, FloatArray]:
    """Build one flat-topped polyhedral column segment per node.

    Neighboring columns break at different elevations, so each side wall is split wherever
    a neighbor's top or bottom meets it, and every vertical edge carries every elevation
    used at its corner. A wall shared by two cells is then the same face in both, so the
    extracted surface has no interior walls.

    Parameters
    ----------
    corner_xy : numpy.ndarray
        ``(ncorner, 2)`` distinct corner coordinates.
    ring : numpy.ndarray
        Each node's footprint as corner ids, concatenated in node order.
    nside : numpy.ndarray
        Number of corners in each node's footprint.
    top : numpy.ndarray
        Top elevation of each node.
    bot : numpy.ndarray
        Bottom elevation of each node, below its top.
    layer : numpy.ndarray
        Layer of each node; hanging corners are found within each layer.

    Returns
    -------
    pyvista.UnstructuredGrid
        One polyhedron per node, in node order.
    numpy.ndarray
        The footprint area of each node.

    Raises
    ------
    ValueError
        If a footprint has fewer than three corners.
    pyvista.VTKVersionError
        If VTK is older than 9.4.

    """
    if pv.vtk_version_info < _POLYHEDRAL_VTK:  # pragma: no cover
        msg = 'pyvista-modflow needs VTK 9.4 or later for its polyhedral cell storage'
        raise pv.VTKVersionError(msg)
    nnode = nside.size
    ring, nside, area = _prepare_rings(corner_xy, ring, nside, layer)
    ring_start, succ, flipped = _ring_links(nside)

    zs = unique(np.concatenate([top, bot]))
    nz = zs.size
    z_top, z_bot = np.searchsorted(zs, top), np.searchsorted(zs, bot)

    # Every half-edge (cell, a -> b) of every footprint.
    he_cell = np.repeat(np.arange(nnode), nside)
    he_a, he_b = ring, ring[succ]
    he_lo, he_hi = z_bot[he_cell], z_top[he_cell]

    # Points: every elevation used by any cell at a corner. Sorted keys, so a corner's points
    # are contiguous and ascending in z, and a point's id is its key's rank.
    pkeys = unique(np.concatenate([he_a * nz + he_lo, he_a * nz + he_hi]))
    points = np.column_stack([corner_xy[pkeys // nz], zs[pkeys % nz]])

    # Breakpoints along each footprint edge: the TOP and BOT of every cell on either side.
    ncorner = corner_xy.shape[0]
    _, edge = unique_rows(np.minimum(he_a, he_b) * ncorner + np.maximum(he_a, he_b))
    ekeys = unique(np.concatenate([edge * nz + he_lo, edge * nz + he_hi]))
    p0 = np.searchsorted(ekeys, edge * nz + he_lo)
    npieces = np.searchsorted(ekeys, edge * nz + he_hi) - p0

    # One wall face per piece between consecutive breakpoints, walking (a,z0) -> (b,z0) -> up
    # b -> (b,z1) -> (a,z1) -> down a, so each face carries every point on its vertical edges.
    piece = np.repeat(np.arange(edge.size), npieces)
    p = ragged_positions(p0, npieces)
    z0, z1 = ekeys[p] % nz, ekeys[p + 1] % nz
    a, b = he_a[piece], he_b[piece]
    sa, ea = np.searchsorted(pkeys, a * nz + z0), np.searchsorted(pkeys, a * nz + z1)
    sb, eb = np.searchsorted(pkeys, b * nz + z0), np.searchsorted(pkeys, b * nz + z1)
    nup = eb - sb + 1
    wall_len = 1 + nup + (ea - sa)
    del a, b, z0, z1, p

    # Each corner's points at the cell's BOT and TOP; caps are the bottom ring reversed
    # (facing down) and the top ring (facing up).
    corner_lo = np.searchsorted(pkeys, ring * nz + np.repeat(z_bot, nside))
    corner_hi = np.searchsorted(pkeys, ring * nz + np.repeat(z_top, nside))
    caps = ((corner_lo[flipped], 0), (corner_hi, 1))

    # Faces per cell: its walls (grouped by cell already, since half-edges are), then the caps.
    nwall = np.add.reduceat(npieces, ring_start)
    nfaces = nwall + 2
    wall_dest = np.arange(piece.size) + 2 * he_cell[piece]
    bot_dest = offsets(nfaces)[:-1] + nwall
    face_len = np.empty(nfaces.sum(), np.int64)
    face_len[wall_dest], face_len[bot_dest], face_len[bot_dest + 1] = wall_len, nside, nside
    face_start = offsets(face_len)[:-1]
    del piece

    # Face ids are the running sum of their differences, so only O(faces) positions are
    # scattered: a wall is sa, sb..eb (step +1), ea..sa+1 (step -1); a cap's ids are given.
    ws = face_start[wall_dest]
    dv = np.zeros(int(face_len.sum()) + 1, np.int64)
    dv[ws + 2] += 1
    dv[ws + 1 + nup] -= 1
    dv[ws + 2 + nup] -= 1
    dv[ws + wall_len] += 1
    np.cumsum(dv, out=dv)
    cap_pos = np.concatenate([ragged_positions(face_start[bot_dest + k], nside) for _, k in caps])
    dv[cap_pos[1:]] = np.diff(np.concatenate([ids for ids, _ in caps]))
    dv[ws + 1], dv[ws + 1 + nup] = sb - sa, ea - eb
    # Jump into each face from the last id of the face before it (a wall ends on sa + 1).
    first, last = np.empty(face_start.size, np.int64), np.empty(face_start.size, np.int64)
    first[wall_dest], last[wall_dest] = sa, sa + 1
    for ids, k in caps:
        first[bot_dest + k], last[bot_dest + k] = ids[ring_start], ids[ring_start + nside - 1]
    dv[face_start] = first - np.concatenate([[0], last[:-1]])
    face_ids = np.cumsum(dv[:-1])
    del dv, ws, cap_pos, wall_dest, sa, ea, sb, eb

    # A cell's distinct points are those on its corners' vertical edges, BOT to TOP; ordering
    # corners by id within each cell sorts them, since point ids ascend by corner then z.
    by_corner = np.lexsort((ring, he_cell))
    count = (corner_hi - corner_lo + 1)[by_corner]
    cell_points = ragged_positions(corner_lo[by_corner], count)
    npts = np.add.reduceat(count, ring_start)

    # PyVista has no setter for VTK's polyhedral arrays (its legacy cell format exhausts
    # memory at millions of cells), so this is the one raw VTK call. CellArray.from_arrays
    # is annotated NDArray[int] and rejects the int64 arrays it accepts: arg-type ignores.
    grid = pv.UnstructuredGrid()
    grid.points = points
    grid.SetPolyhedralCells(
        pv.convert_array(np.full(nnode, pv.CellType.POLYHEDRON, np.uint8)),
        pv.CellArray.from_arrays(offsets(npts), cell_points),  # type: ignore[arg-type]
        pv.CellArray.from_arrays(offsets(nfaces), np.arange(face_len.size)),  # type: ignore[arg-type]
        pv.CellArray.from_arrays(offsets(face_len), face_ids),  # type: ignore[arg-type]
    )
    return grid, area
