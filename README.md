<p align="center">
  <a href="https://codimensional.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://codimensional.com/partners/pyvista-codim-banner-dark.png">
      <img src="https://codimensional.com/partners/pyvista-codim-banner-light.png" width="516" alt="PyVista and CoDimensional">
    </picture>
  </a>
</p>

<h1 align="center">pyvista-modflow</h1>

<p align="center">
  Read MODFLOW-USG unstructured grids into <a href="https://pyvista.org">PyVista</a> as exact, flat-layered volumes.
</p>

<p align="center">
  <a href="https://pypi.org/project/pyvista-modflow/"><img src="https://img.shields.io/pypi/v/pyvista-modflow" alt="PyPI"></a>
  <a href="https://github.com/codimensional/pyvista-modflow/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/codimensional/pyvista-modflow/ci.yml?branch=main" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

Built and maintained by [CoDimensional PBC](https://codimensional.com).

<p align="center">
  <img src="https://raw.githubusercontent.com/codimensional/pyvista-modflow/main/docs/images/freyberg.png" width="760" alt="The three layers of the Freyberg MODFLOW-USG model pulled apart, colored by elevation, showing quadtree refinement along the river and the stepped tops of the cells">
</p>
<p align="center"><sub>FloPy's Freyberg MODFLOW-USG model, layers pulled apart, 60× vertical exaggeration (<a href="https://github.com/codimensional/pyvista-modflow/blob/main/scripts/render_readme.py"><code>scripts/render_readme.py</code></a>).</sub></p>

## Install

```sh
pip install pyvista-modflow
```

It depends only on NumPy and PyVista (0.49 or later, with VTK 9.4 or later), and it registers itself as the PyVista reader for `.dis` and `.disu` files.

## Quickstart

```python
import pyvista as pv

grid = pv.read('model.disu')  # finds model.gsf or model.shp beside it
grid = pv.read('model.disu', footprints='cells.shp')  # or name the outlines
grid.plot(show_edges=True)  # colored by layer
```

`pyvista_modflow.read(...)` takes the same arguments. The grid carries two cell arrays: `layer`, and `node`, the 1-based MODFLOW node number. Use `node` to attach heads or any other per-node result:

```python
import numpy as np

heads = np.loadtxt('heads.txt')  # one value per MODFLOW node, from any post-processor
grid.cell_data['head'] = heads[grid.cell_data['node'] - 1]
```

## Command line

Installing the package also lets the `pyvista` command read DISU files. The footprint file must sit beside the DISU with the same name (`model.gsf` or `model.shp`).

```sh
pyvista convert model.disu model.vtu  # one model
pyvista convert "*.disu" .vtu         # every model in the folder
pyvista plot model.disu
```

## Recipes

Aquifers are thin, so exaggerate z before looking at them:

```python
grid.scale([1, 1, 20]).plot(scalars='head', show_edges=True)
```

One layer, a cross-section, or a cutaway of whole cells:

```python
layer2 = grid.threshold([2, 2], scalars='layer')
section = grid.slice(normal='y')
cutaway = grid.clip(normal='x', crinkle=True)
```

Save it for ParaView or any other VTK reader:

```python
grid.save('model.vtu')
```

## Background

A MODFLOW-USG `DISU` file holds each cell's top, bottom and area, but no x or y coordinates. The cell outlines come from a sidecar file, and the usual conversions build the cells from that sidecar's vertices. Those vertices are shared between neighboring cells and carry a single z, so every cell is bent to follow an interpolated surface instead of its own flat top and bottom.

This reader takes each cell's outline from the sidecar and its elevations from the `DISU`, so every cell is a flat-topped prism. Neighboring columns break at different elevations, so the reader splits every side wall wherever a neighbor's top or bottom meets it, and inserts the extra corner a refined (quadtree or nested) neighbor puts part way along a coarse cell's edge. A wall shared by two cells is the same face in both, so the surface PyVista extracts has no interior walls. Tops and bottoms are not split the same way: where a layer is refined differently from the one above it, the coarse cell's top and the fine cells' bottoms touch without sharing faces.

## Inputs

| File                                                            | Holds                                                                                                                                                                                    |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DISU` (`.dis`, `.disu`)                                        | Nodes per layer and every node's `TOP`, `BOT` and `AREA`. `INTERNAL`, `CONSTANT`, `OPEN/CLOSE`, free and fixed-width Fortran formats are read.                                           |
| GSF (`.gsf`)                                                    | One outline per node. Vertex z is ignored.                                                                                                                                               |
| Polygon shapefile (`.shp` with `.shx`, `.dbf`, optional `.prj`) | One layer's outlines, numbered `1 .. n` by a whole-number `ID` field (`id_field=`), reused by every layer of a layered (`IVSD = -1`) `DISU`. The `.prj` is kept as `crs_wkt` field data. |

Each outline's area is checked against the `DISU` `AREA`, which catches outlines matched to the wrong nodes; `area_rtol=None` skips it. A node whose `TOP` is not above its `BOT` has no volume and is skipped with a warning, and the `node` cell array maps every cell back to its MODFLOW node.

MODFLOW 6 (`DISV`, `DISU`, binary grid files) and structured `DIS` files are not read yet.

## Development

```sh
just sync   # uv sync with the dev group
just test   # pytest
just lint   # pre-commit on every file
just figure # re-render the figure above
```

The test models in `tests/data/` come from [FloPy](https://github.com/modflowpy/flopy) (public domain, CC0); see `tests/data/README.md`.
