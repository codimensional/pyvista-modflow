"""Render the README figure: FloPy's Freyberg MODFLOW-USG model with its layers pulled apart."""

import gzip
from pathlib import Path
import shutil
import tempfile

import pyvista as pv

DATA = Path(__file__).parents[1] / 'tests' / 'data' / 'freyberg_usg'

pv.OFF_SCREEN = True

with tempfile.TemporaryDirectory() as tmp:
    for name in ('freyberg.usg.disu', 'freyberg.usg.gsf'):
        with gzip.open(DATA / f'{name}.gz') as src, (Path(tmp) / name).open('wb') as dst:
            shutil.copyfileobj(src, dst)
    grid = pv.read(Path(tmp) / 'freyberg.usg.disu')

grid.cell_data['elevation'] = grid.cell_centers().points[:, 2]
grid = grid.scale([1, 1, 60])
layers = pv.MultiBlock(
    [grid.threshold([k, k], scalars='layer').translate([0, 0, -2500 * k]) for k in (1, 2, 3)]
)

pl = pv.Plotter(window_size=(2000, 1200))
pl.add_mesh(layers, scalars='elevation', cmap='gist_earth', show_edges=True, show_scalar_bar=False)
pl.camera_position = [(-1, -0.2, 0.6), (0, 0, 0), (0, 0, 1)]
pl.reset_camera()
pl.camera.zoom(1.2)
pl.screenshot('docs/images/freyberg.png')
