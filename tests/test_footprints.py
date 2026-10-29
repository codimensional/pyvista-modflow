from __future__ import annotations

import struct

import numpy as np
import pytest

import pyvista_modflow as pm

from ._models import STEPPED_RINGS, write_gsf, write_shapefile


def test_gsf_corners_are_shared_by_exact_xy(tmp_path):
    write_gsf(tmp_path / 'm.gsf', STEPPED_RINGS, [1, 1, 1])
    fp = pm.read_gsf(tmp_path / 'm.gsf')
    # 3 outlines, 12 vertex rings, but only the 8 distinct corners of the layout.
    assert fp.corner_xy.shape == (8, 2)
    np.testing.assert_array_equal(fp.nside, [4, 4, 4])
    np.testing.assert_array_equal(fp.corner_xy[fp.ring[:4]], STEPPED_RINGS[0])


@pytest.mark.parametrize(
    ('text', 'match'),
    [
        ('STRUCTURED\n', 'not a grid specification file'),
        ('UNSTRUCTURED\n1 1 1 1\n2\n0 0 0\n', 'expected 2 vertices'),
        ('UNSTRUCTURED\n1 1 1 1\n1\n0 0 0\n2 0 0 0 1 2 1 1\n', 'numbered 1..1'),
        ('UNSTRUCTURED\n1 1 1 1\n1\n0 0 0\n1 0 0 0 1 3 1 1 1\n', 'top and a bottom ring'),
        ('UNSTRUCTURED\n1 1 1 1\n1\n0 0 x\n1 0 0 0 1 6 1 1 1 1 1 1\n', "convert .*'x'"),
        ('UNSTRUCTURED\n\n', 'ends early'),
        ('UNSTRUCTURED\n2 1 1 1\n1\n0 0 0\n1 0 0 0 1 6 1 1 1 1 1 1\n', 'ends early'),
    ],
)
def test_malformed_gsf_is_refused(tmp_path, text, match):
    (tmp_path / 'bad.gsf').write_text(text)
    with pytest.raises(ValueError, match=rf'bad\.gsf: .*{match}'):
        pm.read_gsf(tmp_path / 'bad.gsf')


def test_gsf_bottom_ring_must_lie_under_the_top(tmp_path):
    text = 'UNSTRUCTURED\n1 1 1 1\n4\n0 0 1\n1 0 1\n0 1 1\n5 5 0\n1 0 0 0 1 6 1 2 3 1 2 4\n'
    (tmp_path / 'bad.gsf').write_text(text)
    with pytest.raises(ValueError, match='does not lie under'):
        pm.read_gsf(tmp_path / 'bad.gsf')


def _patch(path, offset, data):
    raw = bytearray(path.read_bytes())
    raw[offset : offset + len(data)] = data
    path.write_bytes(bytes(raw))


def test_shapefile_without_prj_has_no_crs(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3])
    assert pm.read_shapefile(tmp_path / 'c.shp').crs_wkt is None


def test_shapefile_must_hold_polygons(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3])
    _patch(tmp_path / 'c.shp', 108, struct.pack('<i', 3))  # first record becomes a PolyLine
    with pytest.raises(ValueError, match='must be a Polygon'):
        pm.read_shapefile(tmp_path / 'c.shp')


def test_shapefile_polygons_must_have_one_part(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3])
    _patch(tmp_path / 'c.shp', 108 + 36, struct.pack('<i', 2))
    with pytest.raises(ValueError, match='holes or several parts'):
        pm.read_shapefile(tmp_path / 'c.shp')


def test_shapefile_rings_must_close(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3])
    _patch(tmp_path / 'c.shp', 108 + 48, struct.pack('<2d', 1.0, 2.0))  # first point moves
    with pytest.raises(ValueError, match='not closed'):
        pm.read_shapefile(tmp_path / 'c.shp')


def test_shapefile_ids_must_number_every_polygon(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 2])
    with pytest.raises(ValueError, match=r'not a permutation of 1\.\.3'):
        pm.read_shapefile(tmp_path / 'c.shp')


def test_missing_id_field_lists_the_fields(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3])
    with pytest.raises(ValueError, match="no field named 'NODE'; the fields are ID"):
        pm.read_shapefile(tmp_path / 'c.shp', id_field='NODE')


def test_whole_numbers_in_a_decimal_field_are_ids(tmp_path):
    # GIS tools often store an integer ID as a decimal field, written "3.00".
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS[::-1], [3, 2, 1], id_decimals=2)
    fp = pm.read_shapefile(tmp_path / 'c.shp', id_field='id')
    first = STEPPED_RINGS[0]
    np.testing.assert_array_equal(fp.corner_xy[fp.ring[:4]], [first[0], *first[:0:-1]])


def test_field_name_in_a_windows_code_page(tmp_path):
    # "N°" in Windows-1252, which is not valid UTF-8.
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3], field=b'N\xb0')
    np.testing.assert_array_equal(
        pm.read_shapefile(tmp_path / 'c.shp', id_field='n°').nside, [4, 4, 4]
    )
    with pytest.raises(ValueError, match="no field named 'ID'; the fields are N°"):
        pm.read_shapefile(tmp_path / 'c.shp')


def test_whole_numbers_in_a_text_field_are_ids(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS[::-1], [3, 2, 1], kind=b'C')
    fp = pm.read_shapefile(tmp_path / 'c.shp')
    first = STEPPED_RINGS[0]
    np.testing.assert_array_equal(fp.corner_xy[fp.ring[:4]], [first[0], *first[:0:-1]])


def test_text_ids_that_are_not_numbers_are_refused(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3], kind=b'C')
    _patch(tmp_path / 'c.dbf', 66, b'A')
    with pytest.raises(ValueError, match="'ID' does not hold whole numbers"):
        pm.read_shapefile(tmp_path / 'c.shp')


def test_upper_case_shapefile_reads_its_upper_case_sidecars(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3], prj='PROJCS["x"]')
    for ext in ('shp', 'shx', 'dbf', 'prj'):
        (tmp_path / f'c.{ext}').rename(tmp_path / f'C.{ext.upper()}')
    fp = pm.read_shapefile(tmp_path / 'C.SHP')
    np.testing.assert_array_equal(fp.nside, [4, 4, 4])
    assert fp.crs_wkt == 'PROJCS["x"]'


def test_upper_case_shapefile_reads_lower_case_sidecars(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3], prj='PROJCS["x"]')
    (tmp_path / 'c.shp').rename(tmp_path / 'c.SHP')
    fp = pm.read_shapefile(tmp_path / 'c.SHP')
    np.testing.assert_array_equal(fp.nside, [4, 4, 4])
    assert fp.crs_wkt == 'PROJCS["x"]'


@pytest.mark.parametrize('shape_type', [15, 25])
def test_polygon_z_and_m_records_read_as_polygons(tmp_path, shape_type):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3])
    _patch(tmp_path / 'c.shp', 108, struct.pack('<i', shape_type))
    np.testing.assert_array_equal(pm.read_shapefile(tmp_path / 'c.shp').nside, [4, 4, 4])


def test_fractional_ids_are_refused(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2.5, 3], id_decimals=1)
    with pytest.raises(ValueError, match="'ID' does not hold whole numbers"):
        pm.read_shapefile(tmp_path / 'c.shp')


def test_shapefile_with_deleted_records_is_refused(tmp_path):
    write_shapefile(tmp_path / 'c.shp', STEPPED_RINGS, [1, 2, 3])
    _patch(tmp_path / 'c.dbf', 65, b'*')
    with pytest.raises(ValueError, match='deleted records'):
        pm.read_shapefile(tmp_path / 'c.shp')
