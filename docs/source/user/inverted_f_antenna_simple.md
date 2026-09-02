# Inverted-F Antenna Design - simple

This tutorial tells you how to design a simple printed inverted-F antenna
(IFA) with simpleEMS. An IFA is a compact antenna made of a shorting leg, a
radiating tip, and a feed line printed on a substrate.

```{note}
You must have simpleEMS and openEMS on your computer. See
[Installation](installation.md).
```

The example script below is `InvertedFAntenna_2.45GHz.py`, which designs an
IFA that resonates at 2.45 GHz.

## Import the modules

Import `InvertedFAntennaParams`, `InvertedFAntenna`, and `setup_simulation`
from `simpleEMS`.

```{literalinclude} ../../../examples/InvertedFAntenna_2.45GHz.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`InvertedFAntennaParams` holds the parameters of the antenna and of the
simulation.

```{seealso}
- {class}`simpleEMS.ifa_antenna.InvertedFAntennaParams`
```

`InvertedFAntenna` makes the antenna.

```{seealso}
- {class}`simpleEMS.ifa_antenna.InvertedFAntenna`
```

`setup_simulation` makes the CSXCAD geometry and the FDTD solver. It also sets
the frequency range of the simulation.

```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

## Set the parameters

```{literalinclude} ../../../examples/InvertedFAntenna_2.45GHz.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`InvertedFAntennaParams` derives the IFA geometry — the shorting leg, the
radiating tip, and the feed spacing — from the resonant frequency and the
properties of the substrate.

## Set up the simulation

```{literalinclude} ../../../examples/InvertedFAntenna_2.45GHz.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

`setup_simulation` prepares the FDTD solver and the CSXCAD geometry. It returns
a `SimSetup` with the `CSX`, `FDTD`, and `freqs` attributes.

## Build the antenna

```{literalinclude} ../../../examples/InvertedFAntenna_2.45GHz.py
:language: python
:start-after: BUILD
:end-before: BUILD
```

`build_inverted_f_antenna` builds the substrate, the ground plane, the
shorting leg and via, the feed line, the radiating tip, the port, and the
mesh in one call. Then these commands show the structure in AppCSXCAD. Close
the AppCSXCAD window to continue. The antenna is now ready for the
simulation.

## Run the simulation

```{literalinclude} ../../../examples/InvertedFAntenna_2.45GHz.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This command runs the openEMS solver. Wait until the solver stops.

## Show the results

```{literalinclude} ../../../examples/InvertedFAntenna_2.45GHz.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

These commands show the S11 curve, the VSWR, and the complex impedance. They
also show the radiation pattern and the directivity in 2D and in 3D. Finally,
they export the model to a STEP file and to Gerber files, which you can send
to a PCB manufacturer.

## Complete script

The complete script is below. It designs the antenna, runs the simulation,
shows the results, and exports the model.

```{literalinclude} ../../../examples/InvertedFAntenna_2.45GHz.py
:language: python
```
