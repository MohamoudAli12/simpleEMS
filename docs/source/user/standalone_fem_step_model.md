# Standalone FEM STEP Model

This tutorial tells you how to simulate a STEP CAD file with the FEM (GetDP)
backend. You do not have to build a CSXCAD geometry first.

The FDTD backend has a related function. See
[Standalone FDTD STEP Model](standalone_fdtd_step_model.md).

```{note}
This backend needs the `getdp` program on your `PATH`. See
[Installation](installation.md).
```

Make a file with the name `inset_fed_patch_step_fem.py`.

## Import the modules

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`simulate_step_FEM` reads the STEP file and runs the FEM solver. `SimTools`
shows and exports the results. All the simpleEMS workflows use `SimTools`.

```{seealso}
- {func}`simpleEMS.fem_backend.simulate_step_FEM`
```

## Model structure

A STEP file contains only named solids. It contains no material data and no
port data. You must tell the function what each solid is.

Four arguments give the role of each solid:

- `dielectrics={name: (eps_r, tan_d)}` — the dielectric solids, for example the
  substrate
- `pec=[name, ...]` — the solids that are perfect conductors
- `lossy_conductor={name: sigma}` — the conductors that have losses
- `ports={name: {"z0": ..., "direction": "x|y|z", "number": ...}}` — the ports

The FEM backend also reads the name of each solid that these arguments do not
name. A name that contains `substrate` becomes a dielectric. A name that
contains `ground` or `patch` becomes a perfect conductor. A name that contains
`port` becomes a port. The four arguments above always win.

The example uses a STEP file with five named solids:

| Solid           | Role from     | Function              |
|-----------------|---------------|-----------------------|
| `substrate`     | `dielectrics` | dielectric substrate  |
| `patch_inset`   | `pec`         | radiant patch         |
| `feed`          | `pec`         | inset feed line       |
| `ground`        | `pec`         | ground plane          |
| `port_resist_1` | `ports`       | lumped port           |

## Set the parameters

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`STEP_FILE` gives the path to the CAD file. `freqs` gives the frequency points
of the results.

## Simulate the model

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

One call does all of these steps:

1. It reads the named solids from the STEP file.
2. It makes the mesh.
3. It runs the GetDP solver at a small number of frequencies.
4. It calculates the results at all the frequencies you asked for.

The `FEM_*` arguments control the solver and the mesh. They set the outer
boundary, the symmetry plane, the element order, the air padding, the density
of the mesh, and the number of frequencies to solve at. `SimParams` gives the
same settings the same names.

```{seealso}
- {class}`simpleEMS.fem_backend.FEMOptions`
```

The function returns the S-parameters, the impedance, the VSWR, and the port
power in one `SimData`.

## Show the results

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

These commands show the S11 curve, the Smith chart, the VSWR, and the input
impedance.

To show a radiation pattern, first get the far-field object from
`SimTools.create_nf2ff()`. Then give it to the radiation plots, for example
`plot_2d_directivity` or `plot_3d_gain`.

```{seealso}
- {class}`simpleEMS.sim_tools.SimTools`
```

## Export the results

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
:start-after: EXPORT
:end-before: EXPORT
```

This command writes the S11 results to a Touchstone file. Other RF tools can
read this file.

## Complete script

The complete script is below. It simulates an inset-fed patch antenna at
24.125 GHz directly from `structure.step`. Then it shows the S11 curve, the
Smith chart, the VSWR, the input impedance, and the radiation patterns. At the
end it writes a Touchstone file.

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFEM.py
:language: python
```
