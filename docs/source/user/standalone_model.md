# Standalone openEMS FDTD Model

This tutorial tells you how to simulate an openEMS model that you have
already. You do not have to build the geometry again. All the simpleEMS tools
apply to the results.

## Model structure

The model must contain the FDTD simulation settings and the CSXCAD properties
and primitives. The openEMS `Write2XML` method writes a model of this type.

A supported model looks like this:

```{literalinclude} ../../../examples/structure.xml
:language: xml
```

## Models with geometry only

The CSXCAD `Write2XML` method writes a different model. The root element of
that model is `<ContinuousStructure>`. It contains the mesh, the materials, the
metals, the ports, and the probes, but no simulation settings. The excitation,
the boundary conditions, and the run limits are all absent.

You cannot simulate such a model. The function stops and shows a `ValueError`.

Use {func}`simpleEMS.fdtd_standalone_model.add_fdtd_setup` to add the
simulation settings and write a complete model:

```python
from simpleEMS import add_fdtd_setup, simulate_model

xml = add_fdtd_setup(
    "structure.xml",                       # root <ContinuousStructure>
    freq_range=(1e9, 2e9),                 # Hz
    FDTD_boundary=["MUR"] * 4 + ["PML_8"] * 2,
)
sim_data, sim, charac_imp = simulate_model(xml, output_path)
```

`add_fdtd_setup` writes the result to `structure_fdtd.xml`, next to the source
file. Give `output_xml_path` to write it somewhere else.

`add_fdtd_setup` also accepts a complete model and replaces its simulation
settings. Use this to change the frequency band or the boundary conditions of a
model. It does not change the geometry, so the model must already contain a
mesh and a minimum of one port.

## Simulate the model

Use {func}`simpleEMS.fdtd_standalone_model.simulate_model` to load the model
and simulate it. The function returns three items:

- `sim_data` — the S-parameters, the impedance, the VSWR, and the port power
- `sim` — the CSXCAD geometry and the FDTD solver
- `charac_imp` — the characteristic impedance

Use all three to examine and export the results.

## Show the results

Use the `SimTools` class to plot and to export the results. The same tools
apply to a model that you build with simpleEMS.

```{seealso}
- {class}`simpleEMS.sim_tools.SimTools`
```

## Example code

The example below loads `structure.xml`, simulates it, shows the results, and
writes them to different formats.

```{literalinclude} ../../../examples/run_model.py
:language: python
```
