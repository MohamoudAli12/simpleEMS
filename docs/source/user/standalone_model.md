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

## Geometry-only models

A model written by CSXCAD's `Write2XML` instead (root element `<ContinuousStructure>`) carries the
mesh, materials, metals, lumped ports and probes, but none of the solver settings — the excitation
waveform, the boundary conditions and the run limits all live in the `<FDTD>` section that only
openEMS' `Write2XML` writes. Simulating such a file directly raises a `ValueError`.

Use {func}`simpleEMS.fdtd_standalone_model.add_fdtd_setup` to supply those settings and re-write the
model as a full `<openEMS>` document:

```python
from simpleEMS import add_fdtd_setup, simulate_model

xml = add_fdtd_setup(
    "structure.xml",                       # root <ContinuousStructure>
    freq_range=(1e9, 2e9),                 # Hz
    FDTD_boundary=["MUR"] * 4 + ["PML_8"] * 2,
)
sim_data, sim, charac_imp = simulate_model(xml, output_path)
```

The result is written to `structure_fdtd.xml` beside the source file unless `output_xml_path` says
otherwise. An existing `<openEMS>` file is accepted too — its `<FDTD>` section is replaced, so you
can re-band or re-terminate a model without rebuilding its geometry. The geometry itself is never
touched: the model must already carry its own mesh and at least one lumped port.

## Loading and simulating the model

To load the model and simulate it we will use {func}`simpleEMS.model.simulate_model` function from simpleEMS.
This function returns a tuple containing `sim_data` (a `SimData` named tuple with `freqs`, `s11`, `s21`, `z11`, `vswr`, and `input_power`), `sim` (a `SimSetup` named tuple with the `CSX` geometry and `FDTD` solver), and `charac_imp` (the characteristic impedance). All of the returned values are used for postprocessing.

## Postprocessing
once the simulation is finished, you can use all available tools from simpleEMS's `SimTools` class to postprocess and plot any data
for visualisation.

## Example code

The example code below shows how load, simulate and postprocess an openEMS `structure.xml` model.

```{literalinclude} ../../../examples/run_model.py
:language: python
```

