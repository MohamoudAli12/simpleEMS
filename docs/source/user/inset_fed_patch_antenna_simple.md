# Inset Fed Patch Antenna Design - simple
This tutorial will walk you through the process of designing a simple Inset-Fed Patch Antenna using simpleEMS.

```{note}
You will need to have a working installation of simpleEMS and openEMS.
```

First create a file named `inset_fed_patch_antenna.py` in your favourite editor.

## Import Modules

import `InsetFedPatchParams`, `InsetFedPatchAntenna` and `setup_simulation` from `simpleEMS`

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: IMPORTS
:end-before: IMPORTS
```

This import will give us access to `InsetFedPatchParams` class which holds all parameters that will be used to define the antenna and simulation domain.
```{seealso}
- {class}`simpleEMS.patch_antenna.InsetFedPatchParams`

```

We also get access to the `InsetFedPatchAntenna` class which will create the antenna object.
```{seealso}
- {class}`simpleEMS.patch_antenna.InsetFedPatchAntenna`
```

The third imported function is `setup_simulation` function which will create the `CSX` geometry and `FDTD` engine and we can use to setup various aspects of the simulation and `freqs` which define the frequency range of the simulation.
```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

Now we are ready to define all the parameters needed to create the `Inset Fed Patch antenna` and simulation domain.
## Parameter definition

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: PARAMS
:end-before: PARAMS
```

The `InsetFedPatchParams` class defines parameters such as resonant frequency, substrate properties, and characteristic impedance of the antenna.
## Setup simulation

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: SETUP
:end-before: SETUP
```

The `setup_simulation` function is used to setup the `FDTD` simulation engine and `CSXCAD` geometry which will be used to visualise the design. The function returns `CSX`, `FDTD`, and `freqs` objects.

## Antenna Creation

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: BUILD
:end-before: BUILD
```
The above code creates the `Inset Fed Patch Antenna` and launches `AppCSXCAD` to visualise the antenna structure.
Now we have finished the antenna structure and it is ready for simulation with `openEMS`.

## openEMS simulation

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: SIMULATE
:end-before: SIMULATE
```
This calls the openEMS FDTD engine to run the simulation of the antenna. wait for the simulation to finish.

## Post-processing

Once the simulation is finished, it is time to postprocess the simulation result to see some amazing plots and 3D visualisation.
The visualisation includes plotting S11, VSWR, Complex Impedance, Radiation Pattern, Directivity and 3D view of radiation pattern.


```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: PPROCESS
:end-before: PPROCESS
```

## External Export
You can export the model to various format for further processsing/visualisation and exporting to other CAD software.

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
:start-after: EXPORT
:end-before: EXPORT
```

## Complete script

Below is the compelete script to design, simulate and post-process the inset fed patch antenna.

```{literalinclude} ../../../examples/simple_inset_fed_patch_antenna.py
:language: python
```

Wow! how simple is designing an antenna. 
