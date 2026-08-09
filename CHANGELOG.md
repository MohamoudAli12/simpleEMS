# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/) and follows [Semantic Versioning](https://semver.org/).

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

