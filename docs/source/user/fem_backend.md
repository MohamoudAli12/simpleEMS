# FEM (GetDP) Backend

Every structure in simpleEMS (patch antennas, microstrip lines, quarter-wave
filters, ...) is solved by openEMS's FDTD engine by default. simpleEMS also
ships a second, independent solver: a Gmsh + GetDP finite-element
frequency-domain (FEM) backend. It solves the exact same geometry and returns
the same {func}`~simpleEMS.sim_tools.SimTools` results, so every plotting and
export helper used elsewhere in this guide works unchanged.

```{note}
The FEM backend requires the `getdp` binary to be on `PATH`, in addition to a
working simpleEMS/openEMS installation. Run `simpleems install getdp` to
install it.
```

There are two ways to use it:

1. **Switch the backend on an existing design** -- keep building the
   structure with `CSXCAD` as usual and set `backend_engine="FEM"` on the
   parameters. This is the easiest path and works with any structure class.
2. **Solve a STEP file directly** -- skip `CSXCAD` entirely and mesh an
   existing `.step` CAD file with {func}`~simpleEMS.fem_backend.simulate_step_FEM`.

## Switching an existing design to FEM

First create a file named `microstrip_line_fem.py` in your favourite editor.

### Import Modules

The imports are identical to the FDTD tutorial -- {class}`~simpleEMS.microstrip_line.MicrostripLineParams`,
{class}`~simpleEMS.microstrip_line.MicrostripLine` and {func}`~simpleEMS.sim_tools.setup_simulation`.

```{literalinclude} ../../../examples/MicrostripLine_FEM.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

### Parameter definition

The only difference from the FDTD example is `backend_engine="FEM"` plus two
FEM-only knobs:

```{literalinclude} ../../../examples/MicrostripLine_FEM.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

- `backend_engine="FEM"` routes `write_and_show_structure` / `run_simulation`
  / `compute_sim_data` to the Gmsh + GetDP pipeline instead of openEMS FDTD.
- `num_FEM_solve_points` sets how many full FEM solves the adaptive
  rational-interpolation sweep performs (must be `>= 4`); `num_points` is
  still the number of *interpolated* output points, same as for FDTD.

Finer FEM mesh/solver tuning -- boundary condition, symmetry, element order,
port type, mesh density -- is *not* set on the params object. It is passed as
keyword arguments straight to `setup_simulation`, e.g.
`setup_simulation(params, FEM_port_type="wave")` -- see
`InsetFedPatch_24GHz_FEM.py` for a worked example. `setup_simulation`
assembles them into a {class}`~simpleEMS.fem_materials.FEMOptions` internally.

```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
- {class}`simpleEMS.fem_materials.FEMOptions` for what each `fem_*` keyword
  does -- its field names and defaults match `setup_simulation`'s.
```

### Setup simulation

```{literalinclude} ../../../examples/MicrostripLine_FEM.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

`setup_simulation` still builds a `CSX` geometry and `FDTD` object -- the FEM
backend reads the CSXCAD primitives and properties from `CSX`, it just never
launches the openEMS FDTD engine.

### Microstrip Line Creation

```{literalinclude} ../../../examples/MicrostripLine_FEM.py
:language: python
:start-after: BUILD
:end-before: BUILD
```

`create_mesh()` still builds the FDTD mesh (used for visualisation only in
this mode), and `write_and_show_structure` now meshes the geometry with Gmsh
and shows the resulting tetrahedral/triangular mesh instead of the FDTD grid.
The mesh is always colored by Gmsh physical-group id (dielectric, PEC, port,
absorbing boundary, ...); pass `mesh_style="surface"`/`"points"` for a
different PyVista representation (default is `"wireframe"`), and
`theme="default"`/`"document"`/`"paraview"` to change the PyVista plot theme
(default is `"dark"`) -- applied globally via `pyvista.set_plot_theme`.
If `output_path` already has a mesh from an earlier call, it is reused instead
of being rebuilt -- delete `fem_mesh.json` there (or use a fresh
`output_path`) to force a remesh after changing the geometry or FEM options.

### GetDP simulation

```{literalinclude} ../../../examples/MicrostripLine_FEM.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This drives the adaptive GetDP sweep: a handful of full finite-element solves
(`num_FEM_solve_points` of them) are fitted with a barycentric rational
model and evaluated over the dense `num_points` frequency grid. Wait for the
sweep to finish.

```{seealso}
- {func}`simpleEMS.fem_backend.build_mesh`
- {func}`simpleEMS.fem_backend.run_sweep`
- {func}`simpleEMS.fem_sweep.rational_sweep`
```

### Post-processing

`compute_sim_data` reads the sweep results back into the same `SimData`
named tuple the FDTD backend returns, so every plot below is unchanged from
the FDTD tutorial:

```{literalinclude} ../../../examples/MicrostripLine_FEM.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

### Complete script

```{literalinclude} ../../../examples/MicrostripLine_FEM.py
:language: python
```

## Solving a STEP file directly

If you already have geometry as a `.step` file -- exported from simpleEMS
itself (see {func}`~simpleEMS.sim_tools.SimTools.export_step`) or from other
CAD software -- you can skip `CSXCAD` and mesh it directly with
{func}`~simpleEMS.fem_backend.simulate_step_FEM`. This is a CSX-free entry
point that mirrors {func}`~simpleEMS.standalone_model.simulate_model`
described in {doc}`standalone_model`.

The example below reuses the STEP file exported by `InsetFedPatch_24GHz.py`
(run that script first to generate
`InsetFedPatch_24GHz/step/structure.step`), and reconstructs its dielectrics,
PEC solids, and port from their solid names:

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
```

`simulate_step_FEM` takes the STEP file, the dense output frequency grid, and
plain dictionaries describing which solids are dielectrics (`{name: (eps_r,
tan_d)}`), which are perfect conductors (`pec=[...]`), and which are ports
(`{name: {"z0": ..., "direction": ..., "number": ...}}`). It returns a
`SimData` named tuple, exactly like every other backend, ready for
`SimTools` post-processing.

Radiation post-processing (`plot_2d_directivity`, `plot_3d_gain`, ...) uses
{class}`~simpleEMS.fem_radiation.FEMNF2FF`, which wraps the FEM near-field
solve and near-to-far-field transform behind the same interface as
`openEMS.nf2ff.nf2ff`, so it works with the existing `SimTools` radiation
plots unchanged.

```{seealso}
- {func}`simpleEMS.fem_backend.simulate_step_FEM`
- {class}`simpleEMS.fem_materials.FEMOptions`
- {class}`simpleEMS.fem_radiation.FEMNF2FF`
```
