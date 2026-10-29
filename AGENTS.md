# pyvista-modflow

A PyVista reader for MODFLOW-USG DISU grids, with cell outlines from a GSF or a polygon
shapefile. Start with `README.md` and `just --list`.

## Anti-patterns specific to this repo

- Taking cell elevations from GSF vertex z. A GSF vertex is shared by every cell touching
  it and carries one z; `TOP` and `BOT` come from the DISU only.
- Matching outline edges by corner id alone. Refined grids put a neighbor's corner part
  way along a coarse cell's edge; `_insert_hanging_corners` adds it before walls are split.
- `np.unique` on the large arrays. It is several times slower than the sort-based helpers
  in `_arrays.py` on NumPy 2.x.
- A test that asserts on VTK's own cell volume or `extract_surface().volume`. VTK's
  polyhedron volume is unreliable and the stepped surface has legitimate non-manifold
  edges; `tests/_oracle.py` computes both independently.
- Testing a parser only on files the test writers produce. Real models come from Windows
  GUIs and GIS tools: backslash and quoted OPEN/CLOSE paths, CRLF, a byte-order mark,
  code-page comments, upper-case names, clockwise shapefile rings, PolygonZ records. A new
  input bug gets a case in `tests/test_real_world_inputs.py` that fails without the fix.
- Test data that is not openly licensed. Every file in `tests/data/` names its source and
  license in `tests/data/README.md`.
