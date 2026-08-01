# Standalone FDTD STEP Model

This tutorial tells you how to simulate a STEP CAD file with the FDTD (openEMS)
backend. You do not have to build a CSXCAD geometry first.

The FEM backend has a related function. See
[Standalone FEM STEP Model](standalone_fem_step_model.md).

```{note}
This function is a prototype. It can change in a later release.
```

Make a file with the name `inset_fed_patch_step_fdtd.py`.

## Import the modules

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFDTD.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`simulate_step_FDTD` reads the STEP file and runs the FDTD solver. `SimTools`
shows and exports the results. All the simpleEMS workflows use `SimTools`.

```{seealso}
- {func}`simpleEMS.fdtd_import_step.simulate_step_FDTD`
```

## Model structure

A STEP file contains only named solids. It contains no material data and no
port data. You must tell the function what each solid is.

Three arguments give the role of each solid:

- `dielectrics={name: (eps_r, tan_d)}` — the dielectric solids, for example the
  substrate
- `pec=[name, ...]` — the solids that are perfect conductors
- `ports={name: {"z0": ..., "direction": "x|y|z", "number": ...}}` — the ports

```{important}
Give the name of each solid that you want in the simulation. The FDTD backend
does not set a role automatically from the name of a solid. It ignores each
solid that these three arguments do not name.
```

The function stops and shows an error if a name is not in the STEP file. It also
stops if you give no ports.

The example uses a STEP file with five named solids:

| Solid           | Role from     | Function              |
|-----------------|---------------|-----------------------|
| `substrate`     | `dielectrics` | dielectric substrate  |
| `patch_inset`   | `pec`         | radiant patch         |
| `feed`          | `pec`         | inset feed line       |
| `ground`        | `pec`         | ground plane          |
| `port_resist_1` | `ports`       | lumped port           |

## Set the parameters

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFDTD.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`STEP_FILE` gives the path to the CAD file. `freqs` gives the frequency points
of the results. The lowest and the highest of these frequencies also set the
band of the excitation signal.

## Simulate the model

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFDTD.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

One call does all of these steps:

1. It reads the named solids from the STEP file.
2. It builds the CSXCAD geometry.
3. It makes the mesh.
4. It runs the openEMS solver.
5. It reads the results back.

These arguments control the solver:

- `FDTD_end_criteria` sets when the solver stops. A smaller value gives more
  accuracy, but the solver runs for a longer time.
- `FDTD_timestep` sets the maximum number of time steps.
- `FDTD_mesh_resolution_factor` and `FDTD_metal_mesh_resolution_factor` set how
  fine the mesh is. The second one applies near the metal.

These arguments control the sequence:

- `show_structure` shows the geometry in AppCSXCAD before the simulation. The
  default value is `True`. Close the AppCSXCAD window to continue.
- `run` starts the solver. The default value is `True`. Set it to `False` to
  examine the geometry only.

The function returns two items:

- `sim_data` — the S-parameters, the impedance, the VSWR, and the port power
- `nf2ff` — the object that records the far-field data

```{note}
The function makes the `nf2ff` object before the solver runs. openEMS records
the far-field data only while the solver runs. You cannot get a radiation
pattern after the simulation if this object does not exist.
```

## Show the results

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFDTD.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

These commands show the S11 curve, the Smith chart, the VSWR, and the input
impedance. They also show the 2D and the 3D radiation patterns. `save_plots`
writes each plot to a file. `show_plots` shows the plots on the screen.

```{seealso}
- {class}`simpleEMS.sim_tools.SimTools`
```

## Export the results

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFDTD.py
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

```{literalinclude} ../../../examples/InsetFedPatch_24GHz_StepFDTD.py
:language: python
```
