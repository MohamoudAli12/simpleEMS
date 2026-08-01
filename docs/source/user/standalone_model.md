# Standalone openEMS FDTD Model

This tutorial tells you how to simulate an openEMS model that you have
already. You do not have to build the geometry again. All the simpleEMS tools
apply to the results.

## Model structure

The model must contain the FDTD simulation settings and the CSXCAD properties
and primitives. The openEMS `Write2XML` method writes a model of this type. Its
root element is `<openEMS>`.

The model must also contain a mesh and a minimum of one port.

```{important}
simpleEMS finds the ports by their names. The names must agree with the names
that `openEMS.ports.LumpedPort` gives them:

- `port_resist_<N>` — the lumped element
- `port_excite_<N>` — the excitation
- `port_ut_<N>` — the voltage probe
- `port_it_<N>` — the current probe

`<N>` is the port number. simpleEMS ignores a port with different names, and
then stops with a `RuntimeError` because it found no ports.
```

A supported model looks like this:

```{literalinclude} ../../../examples/structure.xml
:language: xml
:lines: 1-20
```

The full file is `examples/structure.xml`.

if the model is missing these information or you want to modify it refer to the section below
on how to do it.

## Models with geometry only

The CSXCAD `Write2XML` method writes a different model. The root element of
that model is `<ContinuousStructure>`. It contains the mesh, the materials, the
metals, the ports, and the probes, but no simulation settings. The excitation,
the boundary conditions, and the run limits are all absent.

You cannot simulate such a model. `simulate_model` stops and shows a
`ValueError` that tells you to add the settings first.

Use {func}`simpleEMS.fdtd_standalone_model.add_fdtd_setup` to add the
simulation settings and write a complete model:

```python
from simpleEMS import add_fdtd_setup, simulate_model

xml = add_fdtd_setup(
    "structure.xml",                       # root <ContinuousStructure>
    freq_range=(1e9, 2e9),                 # Hz
    FDTD_boundary=["MUR"] * 4 + ["PML_8"] * 2,
    overwrite=True,                        # replace an earlier result
)
sim_data, sim, charac_imp = simulate_model(xml)
```

`add_fdtd_setup` writes the result to `structure_fdtd.xml`, in the `Sim_Path`
directory of the current directory. Give `output_xml_path` to write it
somewhere else.

```{caution}
`add_fdtd_setup` does not replace a file that exists already. It stops with a
`FileExistsError`. Give `overwrite=True` to replace the file, as above.
```

`add_fdtd_setup` also accepts a complete model and replaces its simulation
settings. Use this to change the frequency band or the boundary conditions of a
model. It does not change the geometry.

The default excitation is a Gaussian pulse, which sets a frequency band in the
model. Give `excitation="sinus"`, `"dirac"`, or `"step"` for a different one.
These three set no band, so you must then give `freqs` to `simulate_model`.

```{seealso}
- {func}`simpleEMS.fdtd_standalone_model.add_fdtd_setup`
```

## Simulate the model

Use {func}`simpleEMS.fdtd_standalone_model.simulate_model` to load the model
and simulate it.

```{note}
`simulate_model` shows the geometry in AppCSXCAD before the solver runs. Close
the AppCSXCAD window to continue.
```

These arguments control the function:

- `output_path` — the directory for the results. The default is the `Sim_Path`
  directory of the current directory.
- `num_points` — the number of frequency points in the results. The default
  value is `1000`.
- `freqs` — the frequency points to report the results at. The default is the
  band of the model's own excitation. Give this argument to use a smaller
  band, or when the excitation sets no band.
- `run` — whether to run the solver. The default value is `True`. Set it to
  `False` to examine the results of an earlier run again.

The function returns three items:

- `sim_data` — the S-parameters, the impedance, the VSWR, and the port power
- `sim` — the CSXCAD geometry and the FDTD solver
- `charac_imp` — the characteristic impedance, read from the model's own port

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
