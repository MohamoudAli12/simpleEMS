# Inset Fed Patch Antenna Design - simple

This tutorial tells you how to design a simple inset-fed patch antenna with
simpleEMS.

```{note}
You must have simpleEMS and openEMS on your computer. See
[Installation](installation.md).
```

Make a file with the name `inset_fed_patch_antenna.py`.

## Import the modules

Import `InsetFedPatchParams`, `InsetFedPatchAntenna`, and `setup_simulation`
from `simpleEMS`.

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`InsetFedPatchParams` holds the parameters of the antenna and of the simulation.

```{seealso}
- {class}`simpleEMS.patch_antenna.InsetFedPatchParams`
```

`InsetFedPatchAntenna` makes the antenna.

```{seealso}
- {class}`simpleEMS.patch_antenna.InsetFedPatchAntenna`
```

`setup_simulation` makes the CSXCAD geometry and the FDTD solver. It also sets
the frequency range of the simulation.

```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

## Set the parameters

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`InsetFedPatchParams` holds the resonant frequency, the properties of the
substrate, and the characteristic impedance of the antenna.

## Set up the simulation

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

`setup_simulation` prepares the FDTD solver and the CSXCAD geometry. It returns
a `SimSetup` with the `CSX`, `FDTD`, and `freqs` attributes.

## Build the antenna

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: BUILD
:end-before: BUILD
```

These commands build the antenna. Then they show the structure in AppCSXCAD.
Close the AppCSXCAD window to continue. The antenna is now ready for the
simulation.

## Run the simulation

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This command runs the openEMS solver. Wait until the solver stops.

## Show the results

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

These commands show the S11 curve, the VSWR, and the complex impedance. They
also show the radiation pattern and the directivity in 2D and in 3D.

## Export the results

You can write the model to different formats. Use these files in other tools,
or send them to a PCB manufacturer.

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: EXPORT
:end-before: EXPORT
```

## Complete script

The complete script is below. It designs the antenna, runs the simulation, and
shows the results.

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
```
