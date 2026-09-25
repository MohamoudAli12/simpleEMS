# Generic Structure from Components - simple

This tutorial tells you how to build your own structure from the primitives in
`simpleEMS.components`. The structure uses every primitive: it starts as a
microstrip line, turns two corners, passes an open radial stub and a shorted
stub, and ends as a coplanar waveguide.

```{note}
You must have simpleEMS and openEMS on your computer. See
[Installation](installation.md).
```

Make a file with the name `simple_generic_structure.py`.

## Import the modules

Import `GenericParams`, `GenericStructure`, and `setup_simulation` from
`simpleEMS`.

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`GenericParams` holds the parameters of the board and of the simulation.

```{seealso}
- {class}`simpleEMS.components.GenericParams`
```

`GenericStructure` gives you one `create_*` method for each primitive.

```{seealso}
- {class}`simpleEMS.components.GenericStructure`
```

`setup_simulation` makes the CSXCAD geometry and the FDTD solver. It also sets
the frequency range of the simulation.

```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

## Set the parameters

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`GenericParams` does not design anything for you. You give it the frequency
range, the design frequency, the size of the board, and the properties of the
substrate. `target_freq` sets the mesh size.

## Set up the simulation

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

`setup_simulation` prepares the FDTD solver and the CSXCAD geometry. It returns
a `SimSetup` with the `CSX`, `FDTD`, and `freqs` attributes.

## Make the board

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: BOARD
:end-before: BOARD
```

`create_substrate` fills the board size from `params`. `create_ground` puts a
copper plane under the board. Here the plane stops at `y = 14.5`, so the
coplanar waveguide at the end has no copper under it.

## Draw the traces

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: TRACES
:end-before: TRACES
```

Each primitive starts at `position`, the middle of its input edge, and runs
along `+y`. `rotation` turns it counter-clockwise about that point, so the next
primitive starts where the last one stops. Every trace sits on top of the
substrate unless you give it a `z_elevation_mm`.

| Method | What it draws |
|---|---|
| `create_microstrip` | A straight trace |
| `create_miter` | A 90° corner with the outer corner cut off |
| `create_curved_bend` | A corner that follows a circular arc |
| `create_radial_stub` | A fan-shaped open stub |
| `create_via` | A plated hole that joins two copper layers |
| `create_taper` | A trace that changes width along its length |
| `create_gcpw` | A coplanar waveguide with a ground plane and stitching vias |
| `create_cpw` | A coplanar waveguide with no ground plane |

The shorted stub is a microstrip trace with a via at its end. The via connects
the trace to the ground plane.

## Add the ports and the mesh

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: PORTS
:end-before: PORTS
```

`create_lumped_port` drives the microstrip end. Add it before `create_mesh`, so
the mesher puts lines on its edges.

`create_cpw_port` terminates the coplanar waveguide end. It measures the field
across both slots at once, so it sees the CPW mode correctly. `start` is the
reference plane at the board edge, and `stop` runs 4 mm back into the line. Add
it **after** `create_mesh`, because it places its probes on the mesh lines.

Then the commands show the structure in AppCSXCAD. Close the AppCSXCAD window
to continue.

```{tip}
`create_cpw_lumped_port` is the other CPW port. It puts one lumped port across
each slot and fills the full copper thickness, so it suits thick copper. Add it
**before** `create_mesh`.
```

## Run the simulation

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This command runs the openEMS solver. Wait until the solver stops.

## Show the results

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

These commands show S11 and S21, the Smith chart, the impedance, and the phase
of S21.

## Export the results

You can write the model to different formats. Use these files in other tools,
or send them to a PCB manufacturer.

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
:start-after: EXPORT
:end-before: EXPORT
```

## Complete script

The complete script is below. It builds the structure, runs the simulation, and
shows the results.

```{literalinclude} ../../../examples/simple_generic_structure.py
:language: python
```
