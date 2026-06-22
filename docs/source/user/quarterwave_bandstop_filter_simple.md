# Quarter-Wave Bandstop Filter Design - simple
This tutorial will walk you through the process of designing a simple Quarter-Wave Bandstop Stub Filter using simpleEMS.

```{note}
You will need to have a working installation of simpleEMS and openEMS.
```

First create a file named `quarterwave_bandstop_filter.py` in your favourite editor.

## Import Modules

Import `QuarterWaveFilterParams`, `BandStopQuarterWaveFilter` and `setup_simulation` from `simpleEMS`

```{literalinclude} ../../../examples/QuarterWaveFilter.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

This import will give us access to `QuarterWaveFilterParams` class which holds all parameters that will be used to define the filter and simulation domain.
```{seealso}
- {class}`simpleEMS.quarterwave_stub_filter.QuarterWaveFilterParams`

```

We also get access to the `BandStopQuarterWaveFilter` class which will create the filter object.
```{seealso}
- {class}`simpleEMS.quarterwave_stub_filter.BandStopQuarterWaveFilter`
```

The third imported function is `setup_simulation` function which will create the `CSX` geometry and `FDTD` engine and we can use to setup various aspects of the simulation.
```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

Now we are ready to define all the parameters needed to create the `Quarter-Wave Bandstop Filter` and simulation domain.
## Parameter definition

```{literalinclude} ../../../examples/QuarterWaveFilter.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

The `QuarterWaveFilterParams` class defines parameters such as centre frequency, bandwidth, filter response type, filter order, and substrate properties of the filter.

## Setup simulation

```{literalinclude} ../../../examples/QuarterWaveFilter.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

The `setup_simulation` function is used to setup the `FDTD` simulation engine and `CSXCAD` geometry which will be used to visualise the design. The function returns a `SimSetupFDTD` named tuple with `.CSX`, `.FDTD`, and `.freqs` attributes.

## Filter Creation

```{literalinclude} ../../../examples/QuarterWaveFilter.py
:language: python
:start-after: BUILD
:end-before: BUILD
```

The above code creates the `BandStopQuarterWaveFilter` and launches `AppCSXCAD` to visualise the filter structure.
The filter is constructed by creating the substrate, ground plane, series transmission line sections, shunt stubs, ports, and FDTD mesh.
Now we have finished the filter structure and it is ready for simulation with `openEMS`.

## openEMS simulation

```{literalinclude} ../../../examples/QuarterWaveFilter.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```

This calls the openEMS FDTD engine to run the simulation of the filter. Wait for the simulation to finish.

## Post-processing

Once the simulation is finished, it is time to postprocess the simulation result to see the filter response.
The visualisation includes plotting S11 and S21 to observe the stopband characteristic.


```{literalinclude} ../../../examples/QuarterWaveFilter.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

## External Export
You can export the model to Gerber format for PCB fabrication.

```{literalinclude} ../../../examples/QuarterWaveFilter.py
:language: python
:start-after: EXPORT
:end-before: EXPORT
```

## Complete script

Below is the complete script to design, simulate and post-process the quarter-wave bandstop stub filter.

```{literalinclude} ../../../examples/QuarterWaveFilter.py
:language: python
```

Wow! how simple is designing a filter.
