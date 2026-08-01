# Microstrip Line Design - simple

This tutorial tells you how to design a simple microstrip transmission line
with simpleEMS.

```{note}
You must have simpleEMS and openEMS on your computer. See
[Installation](installation.md).
```

Make a file with the name `microstrip_line.py`.

## Import the modules

Import `MicrostripLineParams`, `MicrostripLine`, and `setup_simulation` from
`simpleEMS`.

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`MicrostripLineParams` holds the parameters of the line and of the simulation.

```{seealso}
- {class}`simpleEMS.microstrip_line.MicrostripLineParams`
```

`MicrostripLine` makes the line.

```{seealso}
- {class}`simpleEMS.microstrip_line.MicrostripLine`
```

`setup_simulation` makes the CSXCAD geometry and the FDTD solver. It also sets
the frequency range of the simulation.

```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

## Set the parameters

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`MicrostripLineParams` holds the target frequency, the properties of the
substrate, and the characteristic impedance of the line.

## Set up the simulation

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

`setup_simulation` prepares the FDTD solver and the CSXCAD geometry. It returns
a `SimSetup` with the `CSX`, `FDTD`, and `freqs` attributes.

## Build the line

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: BUILD
:end-before: BUILD
```

These commands build the substrate, the ground plane, the trace, the ports, and
the mesh. Then they show the structure in AppCSXCAD. Close the AppCSXCAD window
to continue. The line is now ready for the simulation.

## Run the simulation

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This command runs the openEMS solver. Wait until the solver stops.

## Show the results

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

These commands show the S11 and the S21 curves, the complex impedance, the
phase, and the group delay. A good line has a low S11 and a group delay that
stays constant with frequency.

## Complete script

The complete script is below. It designs the line, runs the simulation, and
shows the results.

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
```
