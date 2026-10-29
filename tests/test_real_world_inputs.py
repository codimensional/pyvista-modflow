"""Inputs as Windows GUIs, GIS tools and older MODFLOW utilities write them."""

from __future__ import annotations

import os

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
)
from ._oracle import unshared_face_area

# Each layer's arrays in their own file, as the GUIs that write OPEN/CLOSE models save them.
ARRAYS = {
    'Top1': STEPPED_TOP[:3],
    'Top2': STEPPED_TOP[3:],
    'Bot1': STEPPED_BOT[:3],
    'Bot2': STEPPED_BOT[3:],
    'Area': STEPPED_AREA[:3],
}


def _write(path, text, *, newline='\n', prefix=b''):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prefix + text.replace('\n', newline).encode())


def _write_model(root, folder, ref, *, newline='\n', prefix=b''):
    """Write the stepped model as ``root/model/MODEL.DISU`` with its arrays under ``folder``."""
    for name, values in ARRAYS.items():
        text = ' '.join(repr(float(v)) for v in values) + '\n'
        _write(root / folder / 'Arrays' / f'{name}.txt', text, newline=newline)
    controls = [f'OPEN/CLOSE {ref.format(name)} 1 (FREE) -1  ** {name}' for name in ARRAYS]
    disu = '\n'.join(
        [
            '# Unstructured discretization, exported from a GUI',
            '6 2 0 -1 1 4 2 0 ** Part 1 NODES NLAY NJAG IVSD NPER ITMUNI LENUNI IDSYMRD',
            '0 0 ** Part 2 LAYCBD',
            'internal 1 (free) -1 ** Part 3 NODELAY',
            '3 3',
            *controls,
        ]
    )
    _write(root / 'model' / 'MODEL.DISU', disu + '\n', newline=newline, prefix=prefix)
    write_gsf(root / 'model' / 'MODEL.GSF', STEPPED_RINGS * 2, [1, 1, 1, 2, 2, 2])
    gsf = (root / 'model' / 'MODEL.GSF').read_text()
    _write(root / 'model' / 'MODEL.GSF', gsf, newline=newline, prefix=prefix)
    return root / 'model' / 'MODEL.DISU'


def _assert_same_grid(grid, expected):
    np.testing.assert_array_equal(grid.points, expected.points)
    for name in ('cell_connectivity', 'polyhedron_faces'):
        np.testing.assert_array_equal(getattr(grid, name), getattr(expected, name))
    for name in ('node', 'layer'):
        np.testing.assert_array_equal(grid.cell_data[name], expected.cell_data[name])


@pytest.mark.parametrize(
    ('folder', 'ref'),
    [
        ('externalfiles', r'..\externalfiles\Arrays\{}.txt'),
        ('externalfiles', '../externalfiles/Arrays/{}.txt'),
        ('external files', r"'..\external files\Arrays\{}.txt'"),
        ('external files', '"../external files/Arrays/{}.txt"'),
    ],
)
@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_open_close_paths_as_windows_writes_them(stepped, tmp_path, folder, ref, newline):
    disu = _write_model(tmp_path / 'win', folder, ref, newline=newline)
    _assert_same_grid(pv.read(disu), pm.read(stepped / 'stepped.disu'))


def test_absolute_open_close_path(stepped, tmp_path):
    root = tmp_path / 'abs'
    disu = _write_model(root, 'externalfiles', f'{root}/externalfiles/Arrays/{{}}.txt')
    _assert_same_grid(pv.read(disu), pm.read(stepped / 'stepped.disu'))


def test_byte_order_mark_and_windows_code_page(stepped, tmp_path):
    # A UTF-8 BOM, then a comment in Windows-1252 ("café"), which is not valid UTF-8.
    prefix = b'\xef\xbb\xbf# caf\xe9\r\n'
    disu = _write_model(tmp_path / 'bom', 'externalfiles', r'..\externalfiles\Arrays\{}.txt')
    _write(disu, disu.read_text(), prefix=prefix)
    gsf = disu.with_suffix('.GSF')
    gsf.write_bytes(b'\xef\xbb\xbf' + gsf.read_bytes())
    _assert_same_grid(pv.read(disu), pm.read(stepped / 'stepped.disu'))


def test_relative_path_from_another_directory(stepped, tmp_path, monkeypatch):
    disu = _write_model(tmp_path / 'rel', 'externalfiles', r'..\externalfiles\Arrays\{}.txt')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    grid = pv.read(os.path.relpath(disu, elsewhere))
    _assert_same_grid(grid, pm.read(stepped / 'stepped.disu'))


def test_last_line_without_newline(tmp_path):
    text = '4 1 0 0 1 4 2 0\n0\nCONSTANT 4\nCONSTANT 2\nCONSTANT 1\nINTERNAL 1 (FREE) -1\n5 6 7 8'
    (tmp_path / 'm.disu').write_text(text)
    np.testing.assert_array_equal(pm.read_disu(tmp_path / 'm.disu').area, [5, 6, 7, 8])


def test_lower_case_gsf_header(tmp_path):
    write_gsf(tmp_path / 'm.gsf', STEPPED_RINGS, [1, 1, 1])
    text = (tmp_path / 'm.gsf').read_text().replace('UNSTRUCTURED GWF', 'unstructured gwf')
    (tmp_path / 'm.gsf').write_text(text)
    np.testing.assert_array_equal(pm.read_gsf(tmp_path / 'm.gsf').nside, [4, 4, 4])


def _rotated(rings, degrees, decimals):
    """Rotate rings about the model origin and round them as a grid file would print them."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return [
        [
            (
                round(float(X0 + c * (x - X0) - s * (y - Y0)), decimals),
                round(float(Y0 + s * (x - X0) + c * (y - Y0)), decimals),
            )
            for x, y in ring
        ]
        for ring in rings
    ]


def test_rotated_grid_with_rounded_coordinates(stepped, tmp_path):
    # Rounding moves a fine cell's corner off the line through the coarse cell's edge.
    rot = tmp_path / 'rotated'
    rot.mkdir()
    write_gsf(rot / 'stepped.gsf', _rotated(STEPPED_RINGS * 2, 30, 3), [1, 1, 1, 2, 2, 2])
    (rot / 'stepped.disu').write_bytes((stepped / 'stepped.disu').read_bytes())
    grid = pm.read(rot / 'stepped.disu')
    straight = pm.read(stepped / 'stepped.disu')
    np.testing.assert_array_equal(grid.cell_offsets, straight.cell_offsets)
    assert unshared_face_area(grid) == pytest.approx(unshared_face_area(straight), rel=1e-4)


def test_small_cells_at_map_coordinates(tmp_path):
    x, y = 512345.678, 6420987.654
    ring = [(x, y), (x + 0.5, y), (x + 0.5, y + 0.5), (x, y + 0.5)]
    write_gsf(tmp_path / 'm.gsf', [ring], [1])
    write_disu(tmp_path / 'm.disu', [1], np.array([1.0]), np.array([0.0]), np.array([0.25]))
    assert pm.read(tmp_path / 'm.disu', area_rtol=1e-9).n_cells == 1


def test_area_printed_to_four_digits(tmp_path):
    ring = [(0.0, 0.0), (35.13, 0.0), (35.13, 35.14), (0.0, 35.14)]
    write_gsf(tmp_path / 'm.gsf', [ring], [1])
    text = '1 1 0 0 1 4 2 0\n0\nCONSTANT 1\nCONSTANT 1\nCONSTANT 0\n'
    (tmp_path / 'm.disu').write_text(text + 'INTERNAL 1.0 (1E12.4) -1\n  0.1234E+04\n')
    assert pm.read(tmp_path / 'm.disu').n_cells == 1


def test_structured_dis_is_named_without_footprints(tmp_path):
    (tmp_path / 'model.dis').write_text('3 10 10 1 4 2\n0 0 0\nCONSTANT 100.0\n')
    with pytest.raises(ValueError, match='structured DIS files are not supported'):
        pv.read(tmp_path / 'model.dis')
