from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

import pyvista_modflow as pm

from ._models import (
    STEPPED_AREA,
    STEPPED_BOT,
    STEPPED_RINGS,
    STEPPED_TOP,
    X0,
    Y0,
    write_disu,
    write_gsf,
    write_shapefile,
)
from ._oracle import box_exterior_area, cell_volumes, independent_prisms, unshared_face_area


def _footprint_areas(fp: pm.Footprints) -> np.ndarray:
    start = np.concatenate([[0], np.cumsum(fp.nside)[:-1]])
    out = []
    for s, n in zip(start, fp.nside):
        x, y = fp.corner_xy[fp.ring[s : s + n]].T
        out.append(abs(0.5 * (x * np.roll(y, -1) - np.roll(x, -1) * y).sum()))
    return np.array(out)


def test_registered_as_the_pyvista_reader():
    sources = {r.extension: r.source for r in pv.registered_readers()}
    assert sources['.dis'] == sources['.disu'] == 'pyvista_modflow:read'


def test_freyberg_cells_are_exact_flat_prisms(freyberg):
    disu_path, gsf_path = freyberg
    with pytest.warns(UserWarning, match='skipped 116 nodes whose TOP <= BOT'):
        grid = pv.read(disu_path)
    disu, fp = pm.read_disu(disu_path), pm.read_gsf(gsf_path)
    keep = disu.top > disu.bot
    node = np.asarray(grid.cell_data['node'])
    np.testing.assert_array_equal(node, np.flatnonzero(keep) + 1)
    np.testing.assert_array_equal(grid.cell_data['layer'], np.repeat([1, 2, 3], 1499)[keep])
    assert grid.active_scalars_name == 'layer'

    expected = (_footprint_areas(fp) * (disu.top - disu.bot))[node - 1]
    np.testing.assert_allclose(cell_volumes(grid), expected, rtol=1e-12)
    # Every cell's points sit at its own TOP or BOT, never at the GSF's shared vertex z.
    conn, off = np.asarray(grid.cell_connectivity), np.asarray(grid.cell_offsets)
    z = np.asarray(grid.points)[conn, 2]
    cell = np.repeat(np.arange(grid.n_cells), np.diff(off))
    assert np.all(z <= disu.top[node - 1][cell] + 1e-9)
    assert np.all(z >= disu.bot[node - 1][cell] - 1e-9)


def test_freyberg_surface_has_no_interior_walls(freyberg):
    disu_path, gsf_path = freyberg
    with pytest.warns(UserWarning, match='TOP <= BOT'):
        grid = pv.read(disu_path)
    disu, fp = pm.read_disu(disu_path), pm.read_gsf(gsf_path)
    node = np.asarray(grid.cell_data['node']) - 1
    xy = fp.corner_xy[fp.ring].reshape(-1, 4, 2)[node]
    lo = np.column_stack([xy.min(axis=1), disu.bot[node]])
    hi = np.column_stack([xy.max(axis=1), disu.top[node]])
    exact = box_exterior_area(lo, hi)
    assert unshared_face_area(grid) == pytest.approx(exact, rel=1e-9)

    rings = [fp.corner_xy[fp.ring[4 * i : 4 * i + 4]] for i in node]
    naive = independent_prisms(rings, disu.top[node], disu.bot[node])
    assert unshared_face_area(naive) > exact * 1.01  # the oracle fails the unsplit walls


def test_hanging_corner_splits_the_coarse_wall(stepped):
    grid = pv.read(stepped / 'stepped.disu')
    lo = [(*np.min(r, axis=0), b) for r, b in zip(STEPPED_RINGS * 2, STEPPED_BOT)]
    hi = [(*np.max(r, axis=0), t) for r, t in zip(STEPPED_RINGS * 2, STEPPED_TOP)]
    exact = box_exterior_area(lo, hi)
    assert unshared_face_area(grid) == pytest.approx(exact, rel=1e-9)
    naive = independent_prisms(STEPPED_RINGS * 2, STEPPED_TOP, STEPPED_BOT)
    assert unshared_face_area(naive) > exact
    expected = STEPPED_AREA * (STEPPED_TOP - STEPPED_BOT)
    np.testing.assert_allclose(cell_volumes(grid), expected, rtol=1e-12)


def test_layers_refined_differently_each_get_their_hanging_corners(tmp_path):
    # Layer 1 has two fine cells east of the coarse cell; layer 2 has one coarse cell there,
    # whose edge runs the reverse of the coarse cell's east edge in the other layer.
    east = [(X0 + 40, Y0), (X0 + 80, Y0), (X0 + 80, Y0 + 40), (X0 + 40, Y0 + 40)]
    rings = [*STEPPED_RINGS, STEPPED_RINGS[0], east]
    top = np.array([310.0, 307.5, 304.0, 200.0, 200.0])
    bot = np.array([296.0, 299.5, 293.0, 100.0, 100.0])
    area = np.array([1600.0, 400.0, 400.0, 1600.0, 1600.0])
    write_disu(tmp_path / 'm.disu', [3, 2], top, bot, area, ivsd=0)
    write_gsf(tmp_path / 'm.gsf', rings, [1, 1, 1, 2, 2])
    grid = pm.read(tmp_path / 'm.disu')
    lo = [(*np.min(r, axis=0), b) for r, b in zip(rings, bot)]
    hi = [(*np.max(r, axis=0), t) for r, t in zip(rings, top)]
    assert unshared_face_area(grid) == pytest.approx(box_exterior_area(lo, hi), rel=1e-9)


def test_shapefile_and_gsf_build_the_same_grid(stepped):
    from_gsf = pm.read(stepped / 'stepped.disu')
    from_shp = pv.read(stepped / 'stepped.disu', footprints=stepped / 'cells.shp')
    # The shapefile's rings are clockwise and start elsewhere, so faces list their points
    # from a different corner; the points, each cell's points and volumes are the same.
    np.testing.assert_array_equal(from_shp.points, from_gsf.points)
    for name in ('cell_connectivity', 'cell_offsets'):
        np.testing.assert_array_equal(getattr(from_shp, name), getattr(from_gsf, name))
    np.testing.assert_allclose(cell_volumes(from_shp), cell_volumes(from_gsf), rtol=1e-12)
    assert from_shp.field_data['crs_wkt'][0] == 'PROJCS["test"]'
    assert 'crs_wkt' not in from_gsf.field_data


def test_shapefile_ids_order_the_polygons(stepped):
    # Polygons stored out of node order, labeled by ID, read the same as in order.
    write_shapefile(stepped / 'shuffled.shp', STEPPED_RINGS[::-1], [3, 2, 1])
    shuffled = pm.read(stepped / 'stepped.disu', stepped / 'shuffled.shp')
    np.testing.assert_allclose(
        cell_volumes(shuffled), STEPPED_AREA * (STEPPED_TOP - STEPPED_BOT), rtol=1e-12
    )


def test_mislabeled_polygons_are_refused(stepped):
    write_shapefile(stepped / 'swapped.shp', STEPPED_RINGS, [2, 1, 3])
    with pytest.raises(ValueError, match='disagree with the DISU AREA'):
        pm.read(stepped / 'stepped.disu', stepped / 'swapped.shp')
    grid = pm.read(stepped / 'stepped.disu', stepped / 'swapped.shp', area_rtol=None)
    assert grid.n_cells == 6


def test_shapefile_needs_a_layered_disu(stepped):
    write_disu(stepped / 'flat.disu', [3, 3], STEPPED_TOP, STEPPED_BOT, STEPPED_AREA, ivsd=0)
    with pytest.raises(ValueError, match='IVSD = -1'):
        pm.read(stepped / 'flat.disu', stepped / 'cells.shp')


def test_nodes_without_thickness_are_skipped(stepped):
    top = STEPPED_TOP.copy()
    top[4] = STEPPED_BOT[4]  # node 5 has no thickness
    write_disu(stepped / 'stepped.disu', [3, 3], top, STEPPED_BOT, STEPPED_AREA)
    with pytest.warns(UserWarning, match='skipped 1 nodes'):
        grid = pv.read(stepped / 'stepped.disu')
    np.testing.assert_array_equal(grid.cell_data['node'], [1, 2, 3, 4, 6])
    np.testing.assert_array_equal(grid.cell_data['layer'], [1, 1, 1, 2, 2])


def test_a_repeated_corner_is_dropped(stepped, tmp_path):
    # A digitized outline that lists a corner twice in a row reads as the clean outline.
    first = STEPPED_RINGS[0]
    rings = [[first[0], first[1], first[1], *first[2:]], *STEPPED_RINGS[1:]] * 2
    (tmp_path / 'dup').mkdir()
    write_gsf(tmp_path / 'dup' / 'stepped.gsf', rings, [1, 1, 1, 2, 2, 2])
    (tmp_path / 'dup' / 'stepped.disu').write_bytes((stepped / 'stepped.disu').read_bytes())
    clean, grid = pm.read(stepped / 'stepped.disu'), pm.read(tmp_path / 'dup' / 'stepped.disu')
    for name in ('cell_connectivity', 'polyhedron_faces'):
        np.testing.assert_array_equal(getattr(grid, name), getattr(clean, name))


def test_a_model_without_thickness_is_refused(stepped):
    write_disu(stepped / 'stepped.disu', [3, 3], STEPPED_BOT, STEPPED_BOT, STEPPED_AREA)
    with pytest.raises(ValueError, match='every node has TOP <= BOT'):
        pm.read(stepped / 'stepped.disu')


def test_missing_footprints_name_the_files_tried(tmp_path):
    write_disu(tmp_path / 'lonely.dis', [3], np.full(3, 2.0), np.zeros(3), np.full(3, 400.0))
    with pytest.raises(FileNotFoundError, match=r'lonely\.gsf.*lonely\.shp.*footprints='):
        pv.read(tmp_path / 'lonely.dis')


def test_unknown_footprint_format_is_refused(stepped):
    with pytest.raises(ValueError, match=r'must be a \.gsf or a \.shp'):
        pm.read(stepped / 'stepped.disu', stepped / 'cells.dbf')


def test_gsf_disagreeing_with_disu_is_refused(stepped):
    write_gsf(stepped / 'stepped.gsf', STEPPED_RINGS * 2, [1, 1, 2, 2, 2, 2])
    with pytest.raises(ValueError, match='disagree on the node count'):
        pm.read(stepped / 'stepped.disu')
