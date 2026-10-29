"""Read MODFLOW unstructured grids into PyVista as exact, flat-layered volumes."""

from __future__ import annotations

from importlib.metadata import version as _dist_version
from pathlib import Path
import warnings

import numpy as np
import pyvista as pv

from pyvista_modflow._arrays import offsets, ragged_positions
from pyvista_modflow._disu import Disu, read_disu
from pyvista_modflow._footprints import Footprints, read_gsf, read_shapefile
from pyvista_modflow._grid import build_grid

__version__ = _dist_version('pyvista-modflow')

__all__ = ['Disu', 'Footprints', '__version__', 'read', 'read_disu', 'read_gsf', 'read_shapefile']

_FOOTPRINT_SUFFIXES = ('.gsf', '.shp')


def _find_footprints(disu: Path) -> Path:
    """Return the GSF or shapefile beside a DISU file with the same name."""
    for suffix in _FOOTPRINT_SUFFIXES:
        for candidate in (disu.with_suffix(suffix), disu.with_suffix(suffix.upper())):
            if candidate.exists():
                return candidate
    tried = ', '.join(str(disu.with_suffix(s)) for s in _FOOTPRINT_SUFFIXES)
    msg = (
        f'{disu}: a DISU file holds no x, y coordinates, so the cell footprints come from a '
        f'GSF or a polygon shapefile. None found at {tried}; pass footprints=<path>.'
    )
    raise FileNotFoundError(msg)


def _every_layer(fp: Footprints, disu: Disu, path: Path) -> Footprints:
    """Repeat one layer's footprints for every layer of a layered DISU."""
    nfoot = fp.nside.size
    if not disu.layered or np.any(disu.nodelay != nfoot):
        msg = (
            f'{path}: {nfoot} footprints for one layer need a layered DISU (IVSD = -1) with '
            f'that many nodes in every layer; this one has IVSD = {disu.ivsd} and '
            f'{disu.nodelay.tolist()} nodes per layer'
        )
        raise ValueError(msg)
    foot = np.tile(np.arange(nfoot), disu.nodelay.size)
    nside = fp.nside[foot]
    ring = fp.ring[ragged_positions(offsets(fp.nside)[:-1][foot], nside)]
    return Footprints(fp.corner_xy, ring, nside, crs_wkt=fp.crs_wkt)


def read(
    path: str | Path,
    /,
    footprints: str | Path | None = None,
    *,
    id_field: str = 'ID',
    area_rtol: float | None = 2e-3,
) -> pv.UnstructuredGrid:
    """Read a MODFLOW-USG DISU grid as a volume of flat-topped polyhedral cells.

    A DISU file has no x, y coordinates, so the footprints come from a GSF (one per node)
    or a polygon shapefile (one layer's, reused by every layer of an ``IVSD = -1`` DISU).
    This is also the :func:`pyvista.read` handler for ``.dis`` and ``.disu`` files.

    Parameters
    ----------
    path : str or pathlib.Path
        The DISU file.
    footprints : str or pathlib.Path, optional
        A ``.gsf`` or ``.shp`` file. By default, the file beside ``path`` with the same
        name and either suffix.
    id_field : str, default: 'ID'
        For a shapefile, the attribute holding each polygon's layer-1 node number.
    area_rtol : float, optional, default: 2e-3
        Maximum relative difference between each footprint's area and the DISU ``AREA``,
        which catches footprints matched to the wrong nodes. The default allows an
        ``AREA`` printed to four significant digits. ``None`` skips the check.

    Returns
    -------
    pyvista.UnstructuredGrid
        One polyhedron per node whose ``TOP`` is above its ``BOT``, in node order. Cell
        arrays ``node`` (the 1-based MODFLOW node) and ``layer`` (active); ``crs_wkt``
        field data when the shapefile has a ``.prj``.

    Raises
    ------
    FileNotFoundError
        If no footprint file is given or found beside ``path``.
    ValueError
        If the footprints and the DISU disagree, or no node has any thickness.
    NotImplementedError
        If the DISU uses an input this reader does not support.

    See Also
    --------
    read_disu : The DISU's ``TOP``, ``BOT`` and ``AREA`` arrays alone.

    Examples
    --------
    Read a model whose GSF sits beside its DISU, then plot it with a vertical
    exaggeration.

    >>> import pyvista as pv
    >>> grid = pv.read('model.disu')  # doctest: +SKIP
    >>> grid.scale([1, 1, 10]).plot(show_edges=True)  # doctest: +SKIP

    Name the footprints explicitly, here a shapefile.

    >>> grid = pv.read('model.disu', footprints='cells.shp')  # doctest: +SKIP

    """
    path = Path(path)
    disu = read_disu(path)  # first, so a structured .dis is named as one
    fp_path = Path(footprints) if footprints is not None else _find_footprints(path)
    layer = disu.layer
    suffix = fp_path.suffix.lower()
    if suffix == '.gsf':
        fp = read_gsf(fp_path)
        if fp.layer is None or not np.array_equal(fp.layer, layer):
            msg = f'{fp_path} and {path} disagree on the node count or the nodes in each layer'
            raise ValueError(msg)
    elif suffix == '.shp':
        fp = _every_layer(read_shapefile(fp_path, id_field=id_field), disu, fp_path)
    else:
        msg = f'{fp_path}: footprints must be a .gsf or a .shp file'
        raise ValueError(msg)

    # A node with TOP <= BOT has no volume (models carry them, often inactive).
    keep = disu.top > disu.bot
    if not keep.any():
        msg = f'{path}: every node has TOP <= BOT, so there is no volume to build'
        raise ValueError(msg)
    if not keep.all():
        msg = (
            f'{path}: skipped {np.count_nonzero(~keep)} nodes whose TOP <= BOT; the `node` '
            'cell array maps each cell to its MODFLOW node'
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
    ring, nside = fp.ring[np.repeat(keep, fp.nside)], fp.nside[keep]
    grid, area = build_grid(
        fp.corner_xy, ring, nside, top=disu.top[keep], bot=disu.bot[keep], layer=layer[keep]
    )
    node = np.flatnonzero(keep) + 1
    if area_rtol is not None:
        expected = disu.area[keep]
        rel = np.abs(area - expected) / np.abs(expected)
        if np.max(rel) > area_rtol:
            worst = int(np.argmax(rel))
            msg = (
                f'{np.count_nonzero(rel > area_rtol)} footprints in {fp_path} disagree with '
                f'the DISU AREA of {path}; worst is node {node[worst]} ({area[worst]:.6g} vs '
                f'{expected[worst]:.6g}). Check the footprint order or id_field, or pass '
                'area_rtol=None to skip this check.'
            )
            raise ValueError(msg)
    grid.cell_data['node'] = node.astype(np.int32)
    grid.cell_data['layer'] = layer[keep].astype(np.int32)
    grid.set_active_scalars('layer', preference='cell')  # so grid.plot() colors by layer
    if fp.crs_wkt:
        grid.field_data['crs_wkt'] = [fp.crs_wkt]
    return grid
