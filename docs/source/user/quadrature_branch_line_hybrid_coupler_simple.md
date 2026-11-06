# Quadrature Branch-Line Hybrid Coupler Design - simple

This tutorial tells you how to design a simple 3 dB quadrature branch-line
hybrid coupler with simpleEMS. This coupler divides the input power equally
between two output ports, with a 90° phase difference between them.

```{note}
You must have simpleEMS and openEMS on your computer. See
[Installation](installation.md).
```

Make a file with the name `quadrature_branch_line_hybrid_coupler.py`.

## Import the modules

Import `QuadratureBranchLineHybridCouplerParams`,
`QuadratureBranchLineHybridCoupler`, and `setup_simulation` from `simpleEMS`.

```{literalinclude} ../../../examples/QuadratureBranchLineHybridCoupler.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`QuadratureBranchLineHybridCouplerParams` holds the parameters of the coupler
and of the simulation.

```{seealso}
- {class}`simpleEMS.coupler.QuadratureBranchLineHybridCouplerParams`
```

`QuadratureBranchLineHybridCoupler` makes the coupler.

```{seealso}
- {class}`simpleEMS.coupler.QuadratureBranchLineHybridCoupler`
```

`setup_simulation` makes the CSXCAD geometry and the solver. It also sets the
frequency range of the simulation.

```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

## Set the parameters

```{literalinclude} ../../../examples/QuadratureBranchLineHybridCoupler.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`QuadratureBranchLineHybridCouplerParams` holds the centre frequency, the
frequency range, the characteristic impedance, and the properties of the
substrate. It calculates the width and the length of each arm and of each feed
line from these values.

The coupler has four quarter-wave arms. The two series arms have an impedance
of Z0/√2. The two shunt arms have an impedance of Z0. Use
`shunt_arm_elec_length_deg` to make the shunt arms longer if the junctions
move the centre frequency.

`sim_method` selects the solver. `"FDTD"` is the default. `"FEM"` uses the
experimental frequency-domain solver.

## Set up the simulation

```{literalinclude} ../../../examples/QuadratureBranchLineHybridCoupler.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

`setup_simulation` prepares the solver and the CSXCAD geometry. It returns a
`SimSetup` with the `CSX`, `FDTD`, and `freqs` attributes.

## Build the coupler

```{literalinclude} ../../../examples/QuadratureBranchLineHybridCoupler.py
:language: python
:start-after: BUILD
:end-before: BUILD
```

These commands build the substrate, the ground plane, the branch arms, the
feed lines, the ports, and the mesh. Then they show the structure in
AppCSXCAD. Close the AppCSXCAD window to continue. The coupler is now ready for
the simulation.

`build_quadrature_branch_line_hybrid_coupler()` returns the four ports:

| Port | Position | Function |
|---|---|---|
| 1 | Top left | Input |
| 2 | Top right | Through |
| 3 | Bottom right | Coupled |
| 4 | Bottom left | Isolated |

## Run the simulation

```{literalinclude} ../../../examples/QuadratureBranchLineHybridCoupler.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This command runs the solver. Wait until the solver stops.

## Show and export the results

```{literalinclude} ../../../examples/QuadratureBranchLineHybridCoupler.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

These commands show the S11, S21, S31, and S41 curves, the input impedance,
the Smith chart, and the phase of S21 and S31.

At the centre frequency, a good coupler has:

- S21 and S31 near -3 dB. The power divides equally between the two outputs.
- S11 and S41 well below -20 dB. The input is matched and port 4 is isolated.
- A phase difference of 90° between S21 and S31.

`generate_report()` writes a report with these key results.
`export_gerber()` writes the layout for a PCB manufacturer, and
`export_step()` writes the 3D model for CAD tools.

## Complete script

The complete script is below. It designs the coupler, runs the simulation, and
shows the results.

```{literalinclude} ../../../examples/QuadratureBranchLineHybridCoupler.py
:language: python
```
