from __future__ import annotations

import numpy as np
import pytest

import pyvista_modflow as pm

from ._models import DATA

HEADER = '4 1 0 0 1 4 2 0\n0\nCONSTANT 4 NODELAY\n'


def _disu(tmp_path, top, bot='CONSTANT -3.5\n', area='CONSTANT 25.0\n', name='m.disu'):
    path = tmp_path / name
    path.write_text(HEADER + top + bot + area)
    return path


@pytest.mark.parametrize(
    ('block', 'expected'),
    [
        ('INTERNAL 1.0 (FREE) -1\n1.5 2.5\n3.5 4.5\n', [1.5, 2.5, 3.5, 4.5]),
        ('INTERNAL 2.0 (FREE) -1\n1.5,2.5, 3.5 4.5\n', [3.0, 5.0, 7.0, 9.0]),
        ('INTERNAL 1.0 (FREE) -1\n3*1.25 7\n', [1.25, 1.25, 1.25, 7.0]),
        # Fixed-width fields with no separator between them.
        ('INTERNAL 1.0 (2F6.2) -1\n 12.50-13.25\n  9.75 -0.50\n', [12.5, -13.25, 9.75, -0.5]),
        (
            'INTERNAL 1.0 (4E10.3) -1\n 1.250D+01-1.325E+01 9.750E+00-5.000E-01\n',
            [12.5, -13.25, 9.75, -0.5],
        ),
        # MODFLOW ignores a zero multiplier.
        ('INTERNAL 0.0 (FREE) -1\n1.5 2.5 3.5 4.5\n', [1.5, 2.5, 3.5, 4.5]),
        ('INTERNAL 1.0 (FREE) -1\n1.25D+01 2.5 3.5d0 4.5\n', [12.5, 2.5, 3.5, 4.5]),
        # Free format stops after the values it needs; the rest of that line is ignored.
        ('INTERNAL 1.0 (FREE) -1\n1.5 2.5\n3.5 4.5 9.9\n', [1.5, 2.5, 3.5, 4.5]),
        ('INTERNAL 1.0 (FREE) -1\n1.5 2.5 3.5 4.5  # top\n', [1.5, 2.5, 3.5, 4.5]),
        # A real field written without a point has the format's decimals implied.
        ('INTERNAL 1.0 (4F5.2) -1\n  125  250 1.25-1.25\n', [1.25, 2.5, 1.25, -1.25]),
        # A scale factor or a skip before the fields.
        (
            'INTERNAL 1.0 (1P,4E11.4) -1\n 1.2500E+01-1.3250E+01 9.7500E+00-5.0000E-01\n',
            [12.5, -13.25, 9.75, -0.5],
        ),
        (
            'INTERNAL 1.0 (1X,4F7.2) -1\n    1250  -1325    975    -50\n',
            [12.5, -13.25, 9.75, -0.5],
        ),
        # A blank fixed-width field is zero.
        ('INTERNAL 1.0 (4F5.1) -1\n  1.5       3.5  4.5\n', [1.5, 0.0, 3.5, 4.5]),
        ('CONSTANT 8.125\n', [8.125] * 4),
        ('         0     8.125(FREE)               -1\n', [8.125] * 4),
    ],
)
def test_array_controls(tmp_path, block, expected):
    disu = pm.read_disu(_disu(tmp_path, block))
    np.testing.assert_array_equal(disu.top, expected)
    np.testing.assert_array_equal(disu.bot, [-3.5] * 4)


@pytest.mark.parametrize('ref', ["'arrays/top.ref'", r'arrays\top.ref'])
def test_open_close_reads_beside_the_disu(tmp_path, ref):
    (tmp_path / 'arrays').mkdir()
    (tmp_path / 'arrays' / 'top.ref').write_text('11.5 12.5\n13.5 14.5\n')
    disu = pm.read_disu(_disu(tmp_path, f'OPEN/CLOSE {ref} 2.0 (FREE) -1\n'))
    np.testing.assert_array_equal(disu.top, [23.0, 25.0, 27.0, 29.0])


def test_short_array_is_refused(tmp_path):
    with pytest.raises(ValueError, match='file ended'):
        pm.read_disu(_disu(tmp_path, 'INTERNAL 1.0 (FREE) -1\n1 2 3\n', bot='', area=''))


def test_bad_value_names_the_file_and_array(tmp_path):
    path = _disu(tmp_path, 'INTERNAL 1.0 (FREE) -1\n1.5 2.5 oops 4.5\n')
    with pytest.raises(ValueError, match=r"m\.disu: TOP layer 1: .*'oops'") as err:
        pm.read_disu(path)
    assert str(err.value).count('m.disu') == 1


@pytest.mark.parametrize(
    ('text', 'error', 'match'),
    [
        (HEADER + 'INTERNAL 1.0 (4F5.1) -1\n', ValueError, 'file ended'),
        (HEADER + '\n', ValueError, 'empty array-control line'),
        (HEADER + "OPEN/CLOSE 'top.bin' 1.0 (BINARY) -1\n", NotImplementedError, 'binary'),
        (HEADER + 'EXTERNAL 11 1.0 (FREE) -1\n', NotImplementedError, "'EXTERNAL'"),
        ('4 1 0 0 1 4 2 0\n1\n', NotImplementedError, 'confining beds'),
        ('4 1 0 0 1 4 2 0\nx\n', ValueError, r"LAYCBD must be 1 integers, found \['x'\]"),
        ('4 1 0 0 1 4 2 0\n', ValueError, 'file ended early'),
        ('5 1 0 0 1 4 2 0\n0\nCONSTANT 4\n', ValueError, 'NODELAY sums to 4'),
        (
            '7 2 0 -1 1 4 2 0\n0 0\nINTERNAL 1 (FREE) -1\n4 3\n' + 'CONSTANT 1\n' * 4,
            ValueError,
            'different node counts',
        ),
    ],
)
def test_malformed_disu_is_refused(tmp_path, text, error, match):
    (tmp_path / 'm.disu').write_text(text)
    with pytest.raises(error, match=match):
        pm.read_disu(tmp_path / 'm.disu')


def test_unit_number_control_needs_the_name_file(tmp_path):
    with pytest.raises(NotImplementedError, match='LOCAT 11'):
        pm.read_disu(_disu(tmp_path, '        11       1.0(FREE)              -1\n1 2 3 4\n'))


def test_modflow6_input_is_named(tmp_path):
    (tmp_path / 'm.disv').write_text('BEGIN OPTIONS\nEND OPTIONS\n')
    with pytest.raises(NotImplementedError, match='MODFLOW 6'):
        pm.read_disu(tmp_path / 'm.disv')


def test_structured_dis_is_named(tmp_path):
    (tmp_path / 'm.dis').write_text('3 10 12 1 4 2\n')
    with pytest.raises(ValueError, match='structured DIS'):
        pm.read_disu(tmp_path / 'm.dis')


def test_layered_area_is_shared_by_every_layer():
    disu = pm.read_disu(DATA / 'mfusg_test' / '45usg' / '45usg.disu')
    assert disu.layered
    np.testing.assert_array_equal(disu.nodelay, [15, 15, 15])
    np.testing.assert_array_equal(disu.layer, np.repeat([1, 2, 3], 15))
    np.testing.assert_array_equal(disu.area.reshape(3, 15), np.tile(disu.area[:15], (3, 1)))
    assert np.all(disu.top > disu.bot)


@pytest.mark.parametrize(
    ('name', 'nodelay'),
    [
        ('01A_nestedgrid_nognc/flow.disu', [121]),
        ('rch_evt_tests/usg_rch_evt.disu', [9, 9, 9]),
    ],
)
def test_open_models_parse(name, nodelay):
    disu = pm.read_disu(DATA / 'mfusg_test' / name)
    np.testing.assert_array_equal(disu.nodelay, nodelay)
    assert disu.top.size == disu.bot.size == disu.area.size == sum(nodelay)
    assert np.all(disu.top > disu.bot)
    assert np.all(disu.area > 0)
