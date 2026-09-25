# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/) and follows [Semantic Versioning](https://semver.org/).

---

## v0.4.0 - 25 Sep 2026

A structures-and-export release. It adds a printed inverted-F antenna, building
blocks for your own structures, an experimental wave port for the FEM backend,
and dark and light plot themes. The Gerber and STEP exports now handle vias,
their antipads, inner layers and transformed objects well enough to send a
design to a board house. A round of FDTD mesher fixes makes simulations smaller
and more accurate, and scikit-rf is no longer a dependency.

### Added
- `InvertedFAntenna` / `InvertedFAntennaParams`: a printed inverted-F antenna
  designed from a target frequency, with an example, a tutorial and API docs
  (fd9f964, 961cdb5, 7f8e3ac)
- `GenericParams` / `GenericStructure`: build your own structure from
  ready-made pieces -- traces, bends, tapers, stubs, vias, coplanar waveguides
  and ports -- with an example and a "building a custom structure" tutorial
  (c1e2be2, 9bd6fe7)
- A `SimParams` subclass can supply `substrate_width_mm` and
  `substrate_length_mm` as plain fields instead of properties (32fcdd5)
- Dark and light plot themes, `use_dark_theme()` and `use_light_theme()`, with
  an API docs page. Plots now default to dark (1035fa8, 5adff82)
- FEM: an experimental wave port, `FEM_port_type="waveport"`. It solves the
  mode of the line at the port -- microstrip, coplanar waveguide and so on --
  instead of driving one fixed field direction. The port sits on the edge of
  the simulation and sizes itself from the trace and substrate;
  `FEM_waveport_width_mm` and `FEM_waveport_height_mm` set that size directly,
  and `FEM_port_mode_*` pick the mode and override its impedance or
  permittivity. Lumped ports stay the default (9f000ab, ffe29f1, 41b731f)
- FEM: `FEM_boundary="pec"` closes the model in a shielded box, for
  transmission lines, filters and other structures that do not radiate. The
  backend also reports how much signal a line loses (41b731f)
- FEM: `FEM_mesh_freq` sets the frequency the mesh is sized at; see Changed
  (81d6483)
- FEM: `FEM_air_pad_mm` also takes `[x, y, z]` or three `[low, high]` pairs, so
  each face of the air box gets only the air it needs. Zero puts the boundary
  right on a ground plane. A single number works as before (76a9b63)
- FEM: `FEM_max_solve_points` caps the extra solves the sweep spends correcting
  a physically impossible curve (see Fixed) (43f1335)
- FEM: the progress output shows the number of frequencies to solve, the time
  each solve took, the elapsed time, the degrees of freedom and the peak memory
  (3423bba)
- `simulate_model()` records an NF2FF box, so a standalone model can plot its
  radiation pattern and directivity (a20d7cc, 7839433, b669af6)
- `simulation_box` on `SimParams` sets the FDTD simulation box as sizes or as
  bounds. The mesher and the field dump both use it; left as `None`, both
  derive it from the geometry as before (39cb572)
- `export_touchstone`, simpleEMS's own Touchstone writer and reader. Its data
  lines match the old scikit-rf output byte for byte.
  `SimTools.export_touchstone()` takes a new `s_matrix` argument to export
  every S-parameter of a multi-port network, not only S11 and S21, and returns
  the path of the file it wrote (6a48784)
- Via support in the STEP export and, through it, in the FEM backend (d9cbaf0)
- The simpleEMS logo in the README and the docs header, and figures redrawn on
  a dark background (4139ef5, 2784643, 3593620, b317cbd, e99e912)

### Changed
- **Breaking:** removed `SimTools.run_all_post_processing()`. Call the plot and
  export methods you need directly (b7de19a)
- **Breaking:** the Gerber export writes one file per copper layer, chosen by
  the z position of each metal, plus an Excellon drill file and a board outline,
  instead of writing only the top layer (6504ee6)
- **Breaking:** `substrate_cells` defaults to 7 instead of 4, and values below 2
  raise an error. The mesher places that many evenly spaced z-lines through the
  substrate, counting both faces (07a4531)
- FEM: the mesh and the air box follow `main_freq` instead of the ends of
  `freq_range`. Widening the range to see more of a plot no longer makes the
  mesh finer and the air box bigger (81d6483)
- The microstrip line's board is now as long as the trace, with a port on each
  edge, instead of a dielectric wavelength of padding around it: a 3 mm trace
  used to sit on a 120 x 133 mm board (6d906be)
- The band-pass filter and the IFA short their stubs to ground with real vias
  instead of boxes, so the shorts now reach the Gerber export (b76030a,
  4e5053e)
- The FDTD simulation box pads by a full wavelength instead of half of one, so
  the NF2FF box encloses the whole structure instead of cutting into the
  substrate (a2e93c5)
- Dropped the `scikit-rf` dependency (59a9b0c)

### Fixed
- FEM: a sweep could return a curve with more power leaving the structure than
  entered it, flagged only by a warning. The sweep now spends extra solves to
  correct itself, stops once they stop helping, and names the setting to change
  when it cannot. The curves it returns also track the real response more
  closely (43f1335)
- FEM: boards with vias failed to simulate, because the mesher left a broken
  element on the via barrels (d6027e1)
- FEM: on multilayer boards a port could stop just short of an inner ground
  plane, so the simulation saw a broken connection and reported a bad match
  (713ad93)
- FEM: board loss came out about a fifth too high (41b731f)
- FDTD: the auto mesher missed the narrow slot beside an inset patch's feed, so
  the antenna came out badly mismatched. The mesher now places lines on the
  slot and spends fewer cells elsewhere, so the same simulation is smaller and
  faster (86895dd)
- FDTD: above about 25 GHz the mesher packed five z-lines into the 35 um copper
  instead of one. The 60 GHz patch example now runs 2.3x lighter (3a3f933)
- FDTD: polygons and curved bends produced so many fine mesh lines that the
  simulation could not run, and the IFA came out overmeshed (92191dd, 3169c24,
  fdbbe46)
- FDTD: the mesher ignored a user-defined simulation box (39cb572)
- FDTD: the thin-layer collapse could reduce a low-frequency substrate to a
  single z-line (07a4531)
- `simulate_model()` on a model with no ports crashed while creating the NF2FF
  box instead of raising "No ports found" (42c8d79)
- The Gerber and STEP exports placed rotated or translated objects in their
  original position, and the STEP export failed on a structure that reused a
  name (086fa36, 5c83cd6)
- The antipad around a via never reached the Gerber layers, so a board made
  from them would have plated the via onto the plane it should pass through,
  and the STEP export left the planes whole. Both now cut the clearance, and
  the ground stitching around it stays connected (2843d16, b76df9f)

---

## v0.3.0 - 09 Aug 2026

Mostly a correctness release for the FEM backend: several results it reported
were wrong, and the antenna radiation plots were the worst affected. If you
have used the FEM backend for radiation patterns or gain, re-run those
simulations. Sweeps are also faster, and STEP files can now be simulated with
the FDTD backend too. This release also brings the project its first automated
test suite and CI, so these results stay checked from now on.

### Added
- STEP-file import for the FDTD backend (`simulate_step_FDTD`), matching what
  the FEM backend already offered, with an inset-fed patch example
- `add_fdtd_setup()` adds simulation settings (frequency range, boundary
  conditions, timestep, end criteria) to a geometry-only CSXCAD file, so a
  model saved by `CSX.Write2XML()` can be simulated. It also changes the
  frequency band of an existing model without rebuilding its geometry
- `simulate_model(freqs=...)` to report results over a frequency band you
  choose, rather than the one built into the model
- `add_field_dump(dump_freq=...)` lets you pick the frequency a frequency-domain
  dump records; it defaults to the model's main frequency
- FEM: `FEM_air_pad_mm` to set the air padding around a structure directly,
  instead of letting it scale with wavelength. Useful for filters and other
  non-radiating structures, whose air box need not grow with a wide sweep
- FEM: `simulate_step_FEM()` now shows the meshed geometry with PyVista after
  meshing, so you can check the mesh before the sweep runs. `show_mesh` turns
  it on or off, and `mesh_style` and `theme` match
  `SimTools.write_and_show_structure()`
- Documentation: solver-backend and STEP-export sections in the README, and a
  geometry-only model section in the standalone model tutorial
- A first automated test suite and CI workflow. Tests cover the design formulas,
  both solver backends, the exporters, the CLI and the plots, and the examples
  now fail the build if the API moves under them. Tests that need openEMS or
  GetDP installed skip themselves, so the rest still run anywhere. CI runs the
  fast tests on every push and the full suite on master

### Changed
- **Breaking:** `SimTools.export_stl()` now takes `sim` as its first argument.
  STL files are built directly with CadQuery rather than by launching
  AppCSXCAD, so the export no longer needs that program installed
- **Breaking:** FEM and FDTD settings now live on `SimParams` as `FEM_*` and
  `FDTD_*` fields, instead of being passed to `setup_simulation()`
  (`num_FEM_solve_points` is now `FEM_num_solve_points`)
- **Breaking:** removed the `port_type` / `FEM_port_type` `"wave"` option. It
  was never a real wave port, and a genuine one, prototyped and measured,
  performed worse than the lumped port it would have replaced. Ports are
  lumped-only for now
- FEM: every port at a frequency is now solved together rather than one at a
  time. This roughly halves a two-port sweep, and saves more the more ports
  there are. Results are unchanged
- FEM: each material is now meshed to the detail its own wavelength needs, so
  the substrate is resolved properly without spending elements on empty air
- FEM: a sweep whose results are physically impossible is now reported as such,
  rather than returned as a curve with a false resonance in it
- FEM: the formulation now uses the `e^{+jωt}` engineering time convention that
  openEMS uses, instead of the physics `e^{-jωt}` one. A phase, and an
  inductance or capacitance read off the Smith chart, now mean the same thing
  whichever backend produced it
- `simulate_model()` and `add_fdtd_setup()` now write to `Sim_Path` when given
  no output path, as the other modules do. `simulate_model()`'s `output_path`
  is optional as a result
- Modules renamed to make clear which backend they belong to: `mesh.py` to
  `fdtd_mesh.py`, `standalone_model.py` to `fdtd_standalone_model.py`, and
  `export_step.py` to `export_cad.py`
- The FEM modules' documentation was rewritten to a consistent style

### Fixed
- FEM antenna gain and radiation patterns were wrong in four separate ways,
  each of which alone could shift a pattern or its level:
  - the far field was computed in the wrong time convention, which affected
    every radiation plot and the directivity read off it
  - radiated power was measured in a way that reads near zero behind a PML
    boundary, leaving those gain plots around 14 dB low
  - with a symmetry plane, only half the antenna was accounted for, and part
    of what was measured sat outside the simulated region
  - efficiency was taken from the strongest point of the sweep rather than the
    frequency the pattern was computed at

  A half model of the 24 GHz patch example now matches the full model to
  0.25 dB, and a PML run matches a Silver-Muller one to 0.16 dB. Loss in
  lossy conductors is counted for the first time as part of this
- FEM: when the usual far-field pattern file cannot be read and the fallback
  reads it point by point, each sample's coordinates were being mixed up, so
  every sample pointed the wrong way. The fallback now reads them correctly
- FEM S-parameters between ports that differ from each other -- in substrate
  thickness or reference impedance -- were wrong, which could also make a
  reciprocal structure look non-reciprocal. Ports that match each other, as in
  every shipped example, were unaffected
- FEM S21 was reported with the wrong sign of phase, giving negative group
  delay
- FEM: group delay had the wrong scale as well, dielectric loss was measured
  over the wrong region, and a solid whose name sounded like a port (say
  `port_feed_1`) could be classified as metal, silently dropping the
  excitation. All three are corrected
- FEM: a sweep over a single frequency used to crash, and when it did not it
  solved that frequency five times and threw four of the results away, because
  the seeding step spread its points across a band with no width. A single-point
  sweep now solves once
- FEM simulations no longer reuse a stale mesh or solver setup after the
  geometry or the settings change, so sweep and optimise loops are correct
- Box-shaped STEP solids are rebuilt as real boxes rather than approximated by
  flat faces, restoring the mesh detail at metal edges and with it the
  resonance the model predicts
- The default output path when none is given, for both backends, and a relative
  `output_path` is now made absolute before running, so openEMS no longer fails
  with a confusing error about its working directory
- `simulate_model()` on a model with no simulation settings now says what it
  found and points at `add_fdtd_setup()`, and an unsupported excitation now
  gives a clear error instead of a `TypeError`. It also accepts a `pathlib.Path`
  for the model file, as documented, not only a plain string
- `SimTools.run_all_post_processing()` stopped partway through: it passed the
  wrong value to the STL export and errored out before ever writing the
  Touchstone and Gerber files. It now finishes everything it promises
- Errors no longer dump local variables into the traceback, where they were
  mostly noise
- `simpleems install openems` no longer reports a failed install after a
  successful build. The Python bindings were being installed into a virtual
  environment that nothing activates, and the health check ran before the
  freshly installed binaries were on `PATH`. AppCSXCAD, the optional Qt
  viewer, is now reported but no longer fails an install on a machine without
  Qt. On Windows, `--force` now really replaces the old files instead of
  leaving them behind
- Documentation CI build, the demo screenshots that PyPI showed as broken, the
  project's GitHub link in the docs header, and the description of what
  `run=False` does when importing a STEP file (it reads back results and fails
  when there are none, rather than stopping before the solve)

---

## 0.2.0 - 25 Jul 2026

### Added
- New GetDP-based FEM (finite-element) solver backend as an alternative to the
  FDTD (openEMS) backend, with shared FDTD/FEM modules unified for dual-backend
  simulation
- `simpleems install getdp` CLI command to install the GetDP solver binary
- Documentation: GetDP installation guide, `fem_backend` API reference, and a
  "Standalone FEM STEP Model" tutorial

### Fixed
- Updated examples for the `add_field_dump`/`compute_sim_data` signature changes
- Fixed README installation, PyPI, and demo links
- Fixed Sphinx docs CI build (mocked heavy/native dependencies for a
  docs-only build) and added GitHub Pages deployment

---

## 0.1.0 - 11 Jul 2026

- Initial public release
---

