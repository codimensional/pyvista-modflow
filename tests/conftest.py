"""Fixtures: the vendored open models, and a small stepped, refined synthetic model."""

from __future__ import annotations

import gzip
import shutil

import pytest
import pyvista as pv

from ._models import (
    DATA,
    STEPPED_AREA,
    STEPPED_BOT,
    STEPPED_RINGS,
    STEPPED_TOP,
    write_disu,
    write_gsf,
    write_shapefile,
)

pv.OFF_SCREEN = True


@pytest.fixture
def freyberg(tmp_path):
    """FloPy's freyberg_usg model: a DISU and its GSF, decompressed into ``tmp_path``."""
    out = []
    for name in ('freyberg.usg.disu', 'freyberg.usg.gsf'):
        with (
            gzip.open(DATA / 'freyberg_usg' / f'{name}.gz') as src,
            (tmp_path / name).open('wb') as dst,
        ):
            shutil.copyfileobj(src, dst)
        out.append(tmp_path / name)
    return tuple(out)


@pytest.fixture
def stepped(tmp_path):
    """A layered (IVSD = -1) two-layer model of ``STEPPED_RINGS``, as DISU + GSF + shapefile."""
    write_disu(tmp_path / 'stepped.disu', [3, 3], STEPPED_TOP, STEPPED_BOT, STEPPED_AREA)
    write_gsf(tmp_path / 'stepped.gsf', STEPPED_RINGS * 2, [1, 1, 1, 2, 2, 2])
    write_shapefile(tmp_path / 'cells.shp', STEPPED_RINGS, [1, 2, 3], prj='PROJCS["test"]')
    return tmp_path
