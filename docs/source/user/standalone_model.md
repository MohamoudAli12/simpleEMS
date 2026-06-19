# Standalone openEMS FDTD Model

Do you have an existing openEMS `.xml` model? if the answer is yes, then you are in the right place.

This tutorial will show you how to load and simulate your existing model with simpleEMS and take advantage of all
the tools that simpleEMS offers.

## Model Structure

Your model will need to have `FDTD` simulation data along with `CSXCAD` properties and primitives.

The supported model should be similar to the below
```{literalinclude} ../../../examples/structure.xml
:language: xml
```
The model should be generated from openEMS `Write2XML` method to have `FDTD` properties.

## Loading and simulating the model

To load the model and simulate it we will use {func}`simpleEMS.model.simulate_model` function from simpleEMS.
This function returns the `network_parameters` such as `S11`, `S21` etc., `CSX` structure, and `characteristic impedance` extracted
from the model. all of the returned values are used for postprocessing.

## Postprocessing
once the simulation is finished, you can use all available tools from simpleEMS's `SimTools` class to postprocess and plot any data
for visualisation.

## Example code

The example code below shows how load, simulate and postprocess an openEMS `structure.xml` model.

```{literalinclude} ../../../examples/run_model.py
:language: python
```

