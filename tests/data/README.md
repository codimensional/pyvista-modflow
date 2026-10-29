# Test models

Copied from [FloPy](https://github.com/modflowpy/flopy) `examples/data` at commit
`fb62d047c2e492d405eba30e3b6383671fdc9f7c`. FloPy's data is in the public domain in the
United States (USGS) and dedicated worldwide under CC0 1.0.

| Path                                                                    | Source                             | Notes                                                                           |
| ----------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------- |
| `freyberg_usg/freyberg.usg.disu.gz`, `freyberg_usg/freyberg.usg.gsf.gz` | `freyberg_usg/`                    | gzipped; 4497 nodes, 3 layers, quadtree refinement, 116 nodes with `TOP <= BOT` |
| `mfusg_test/01A_nestedgrid_nognc/flow.disu`                             | `mfusg_test/01A_nestedgrid_nognc/` | fixed-width `(5G13.0)` arrays                                                   |
| `mfusg_test/45usg/45usg.disu`                                           | `mfusg_test/45usg/`                | layered, `IVSD = -1`                                                            |
| `mfusg_test/rch_evt_tests/usg_rch_evt.disu`                             | `mfusg_test/rch_evt_tests/`        | layered, `IVSD = -1`                                                            |
