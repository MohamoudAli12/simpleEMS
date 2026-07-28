# Standalone FEM STEP Model

Do you have an existing `.step` CAD file and want to simulate it directly with
the FEM (GetDP) backend, without building a CSXCAD geometry first? Then you
are in the right place.

This tutorial shows how to feed a raw STEP file straight into simpleEMS's FEM
solver and postprocess the results with the same `SimTools` used by the FDTD
workflows.

```{note}
This backend needs the `getdp` binary on your `PATH`. See
[Installation](installation.md) for how to install it.
```

First create a file named `inset_fed_patch_step_fem.py` in your favourite editor.

## Import Modules

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`simulate_step_FEM` comes from the FEM backend, and `SimTools` is the same
postprocessing/plotting toolkit used across simpleEMS.

```{seealso}
- {func}`simpleEMS.fem_backend.simulate_step_FEM`
```

## Model Structure

Unlike the CSXCAD-driven tutorials, a standalone STEP model has no
`ConductingSheet`/`Metal`/`Material`/`LumpedElement` properties to read roles
from -- each solid is just a named shape in the CAD file. Roles are assigned
by matching each solid's name against the arguments you pass to
{func}`simpleEMS.fem_backend.simulate_step_FEM`:

- `dielectrics={name: (eps_r, tan_d)}` -- dielectric volumes (e.g. the substrate)
- `pec=[name, ...]` -- perfect-conductor solids
- `lossy_conductor={name: sigma}` -- lossy (surface-impedance) conductors
- `ports={name: {"z0": ..., "direction": "x|y|z", "number": ...}}` -- excitation ports

Any solid not covered by these overrides falls back to a name-based guess
(e.g. a name containing `"substrate"` is guessed as a dielectric, `"ground"`
or `"patch"` as PEC, `"port"` as a port) -- but explicit overrides always win,
so naming your solids sensibly in the CAD tool is enough to skip most of the
bookkeeping.

The example below uses a STEP file with five named solids:

| Solid          | Role assigned via     | Meaning                     |
|----------------|------------------------|------------------------------|
| `substrate`    | `dielectrics`          | dielectric substrate         |
| `patch_inset`  | `pec`                  | radiating patch              |
| `feed`         | `pec`                  | inset feed line               |
| `ground`       | `pec`                  | ground plane                 |
| `port_resist_1`| `ports`                | lumped excitation port       |

## Parameter definition

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`STEP_FILE` points at the CAD file to mesh and solve, and `freqs` is the dense
output frequency grid centred on the resonant frequency.

## Simulating the model

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

{func}`simpleEMS.fem_backend.simulate_step_FEM` meshes the STEP geometry,
generates the GetDP problem, runs the adaptive frequency sweep, and reads the
results back -- all in one call. Solver/mesh tuning (outer boundary type,
symmetry, element order, port type, mesh density, number of adaptive solve
points, ...) is set directly via the flat `FEM_*` keyword arguments, the same
way `SimParams` exposes them -- no separate options object to build. It
returns a single `SimData` named tuple with `freqs`, `s11`, `s21` (`None` for
a single-port problem), `z11`, `vswr`, `input_power`, `port_voltage`,
`port_current`, and `ref_impedance`.

## Postprocessing

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

This plots S11, the Smith chart, VSWR, and input impedance with the same
`SimTools` methods used across simpleEMS. For radiation patterns, get the
NF2FF adapter from `SimTools.create_nf2ff()` and feed it into the usual
`plot_2d_directivity` / `plot_2d_rad_pattern` / `plot_3d_directivity` /
`plot_3d_gain` / `plot_3d_power` helpers.

```{seealso}
- {class}`simpleEMS.sim_tools.SimTools`
```

## External Export

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: EXPORT
:end-before: EXPORT
```

Exports the S11 results to a Touchstone file for use in other RF tools.

## Complete script

Below is the complete script: it simulates an inset-fed patch antenna at
24.125 GHz directly from `structure.step`, then plots S11, the Smith chart,
VSWR, input impedance, and the 2D/3D radiation pattern before exporting a
Touchstone file.

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
```
