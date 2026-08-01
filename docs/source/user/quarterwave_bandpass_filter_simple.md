# Quarter-Wave Bandpass Filter Design - simple

This tutorial tells you how to design a simple quarter-wave bandpass stub
filter with simpleEMS. This filter passes the frequencies in one band and stops
the others.

```{note}
You must have simpleEMS and openEMS on your computer. See
[Installation](installation.md).
```

Make a file with the name `quarterwave_bandpass_filter.py`.

## Import the modules

Import `QuarterWaveFilterParams`, `BandPassQuarterWaveFilter`, and
`setup_simulation` from `simpleEMS`.

```{literalinclude} ../../../examples/BandPassQuarterWaveFilter.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

`QuarterWaveFilterParams` holds the parameters of the filter and of the
simulation.

```{seealso}
- {class}`simpleEMS.quarterwave_stub_filter.QuarterWaveFilterParams`
```

`BandPassQuarterWaveFilter` makes the filter.

```{seealso}
- {class}`simpleEMS.quarterwave_stub_filter.BandPassQuarterWaveFilter`
```

`setup_simulation` makes the CSXCAD geometry and the FDTD solver. It also sets
the frequency range of the simulation.

```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

## Set the parameters

```{literalinclude} ../../../examples/BandPassQuarterWaveFilter.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

`QuarterWaveFilterParams` holds the centre frequency, the bandwidth, the type
of response, the order of the filter, and the properties of the substrate.

## Set up the simulation

```{literalinclude} ../../../examples/BandPassQuarterWaveFilter.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

`setup_simulation` prepares the FDTD solver and the CSXCAD geometry. It returns
a `SimSetup` with the `CSX`, `FDTD`, and `freqs` attributes.

## Build the filter

```{literalinclude} ../../../examples/BandPassQuarterWaveFilter.py
:language: python
:start-after: BUILD
:end-before: BUILD
```

These commands build the substrate, the ground plane, the series lines, the
stubs, the ports, and the mesh. Then they show the structure in AppCSXCAD.
Close the AppCSXCAD window to continue. The filter is now ready for the
simulation.

The bandpass filter has a short circuit at the end of each stub. The bandstop
filter does not. `create_shunt_line_short()` makes this short circuit.

## Run the simulation

```{literalinclude} ../../../examples/BandPassQuarterWaveFilter.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This command runs the openEMS solver. Wait until the solver stops.

## Show the results

```{literalinclude} ../../../examples/BandPassQuarterWaveFilter.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

These commands show the S11 and the S21 curves. Use them to examine the
passband of the filter.

## Export the results

You can write the model to the Gerber format for a PCB manufacturer. You can
also write the results to the Touchstone format for other EDA tools.

```{literalinclude} ../../../examples/BandPassQuarterWaveFilter.py
:language: python
:start-after: EXPORT
:end-before: EXPORT
```

## Complete script

The complete script is below. It designs the filter, runs the simulation, and
shows the results.

```{literalinclude} ../../../examples/BandPassQuarterWaveFilter.py
:language: python
```
