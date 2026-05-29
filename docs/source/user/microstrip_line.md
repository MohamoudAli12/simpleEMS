# Microstrip Line Design - simple
This tutorial will walk you through the process of designing a simple Microstrip Transmission Line using simpleEMS.

```{note}
You will need to have a working installation of simpleEMS and openEMS.
```

First create a file named `microstrip_line.py` in your favourite editor.

## Import Modules

Import `MicrostripLineParams`, `MicrostripLine` and `setup_simulation` from `simpleEMS`

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

This import will give us access to `MicrostripLineParams` class which holds all parameters that will be used to define the microstrip line and simulation domain.
```{seealso}
- {class}`simpleEMS.microstrip_line.MicrostripLineParams`

```

We also get access to the `MicrostripLine` class which will create the microstrip line object.
```{seealso}
- {class}`simpleEMS.microstrip_line.MicrostripLine`
```

The third imported function is `setup_simulation` function which will create the `CSX` geometry and `FDTD` engine and we can use to setup various aspects of the simulation.
```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

Now we are ready to define all the parameters needed to create the `Microstrip Line` and simulation domain.
## Parameter definition

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

The `MicrostripLineParams` class defines parameters such as target frequency, substrate properties, and characteristic impedance of the microstrip line.

## Setup simulation

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

The `setup_simulation` function is used to setup the `FDTD` simulation engine and `CSXCAD` geometry which will be used to visualise the design. The function returns `CSX`, `FDTD`, and `freqs` objects.

## Microstrip Line Creation

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: BUILD
:end-before: BUILD
```

The above code creates the `MicrostripLine` and launches `AppCSXCAD` to visualise the microstrip structure.
The line is constructed by creating the substrate, ground plane, microstrip trace, ports, FDTD mesh, and near-field to far-field transformation.
Now we have finished the microstrip structure and it is ready for simulation with `openEMS`.

## openEMS simulation

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This calls the openEMS FDTD engine to run the simulation of the microstrip line. Wait for the simulation to finish.

## Post-processing

Once the simulation is finished, it is time to postprocess the simulation result to see some amazing plots and 3D visualisation.
The visualisation includes plotting S11, S21, VSWR, Complex Impedance, Radiation Pattern, Directivity and 3D view of radiation pattern.


```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

## External Export
You can export the model to various format for further processing/visualisation and exporting to other CAD software.

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
:start-after: EXPORT
:end-before: EXPORT
```

## Complete script

Below is the complete script to design, simulate and post-process the microstrip transmission line.

```{literalinclude} ../../../examples/MicrostripLine.py
:language: python
```

Wow! how simple is designing a microstrip line.
