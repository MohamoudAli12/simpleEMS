# Probe Fed Patch Antenna Design - simple
This tutorial will walk you through the process of designing a simple Probe-Fed/Coaxial-Fed Patch Antenna using simpleEMS.

```{note}
You will need to have a working installation of simpleEMS and openEMS.
```

First create a file named `probe_fed_patch_antenna.py` in your favourite editor.

## Import Modules

Now import `ProbeFedPatchParams`, `ProbeFedPatchAntenna` and `setup_simulation` from `simpleEMS`

```python
from simpleEMS import (
    ProbeFedPatchParams,
    ProbeFedPatchAntenna,
    setup_simulation,
)
```

This import will give us access to `ProbeFedPatchParams` class which holds all parameters that will be used to define the antenna and simulation domain.
```{seealso}
- {class}`simpleEMS.patch_antenna.ProbeFedPatchParams`

```

We also get access to the `ProbeFedPatchAntenna` class which will create the antenna object.
```{seealso}
- {class}`simpleEMS.patch_antenna.ProbeFedPatchAntenna`
```

The third imported function is `setup_simulation` function which will create the `CSX` geometry and `FDTD` engine and we can use to setup various aspects of the simulation.
```{seealso}
- {func}`simpleEMS.sim_tools.setup_simulation`
```

Now we are ready to define all the parameters needed to create the `Probe Fed Patch antenna` and simulation domain.
## Parameter definition

```python
params = ProbeFedPatchParams(
    resonant_freq=2.45e9,
    corner_freq=0.5e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)
``` 
The `ProbeFedPatchParams` class defines parameters such as resonant frequency, substrate properties, and 
and characteristic impedance of the antenna.

- *`resonant_freq`*: defines the resonant frequency of the antenna
- *`corner_freq`*: is used to generate a span centered around the resonant frequency. for example, `0.5e9` will
generate a frequency span of `0.5e9` higher than the `resonant_freq` and `0.5e9` lower than the `resonant_freq`.
- *`substrate_thickness_mm`*: defines the thickness of the substrate material in millimeters.
- *`substrate_eps_r`*: defines the relative permittivity of the substrate material.
- *`charac_imp`*: defines the characteristic impedance of the input port.

## Setup simulation

```python
CSX, FDTD, freqs = setup_simulation(params)

```
The `setup_simulation` function is used to setup the `FDTD` simulation engine and `CSXCAD` geometry which will be used to visualise the design. The function returns `CSX`, `FDTD`, and `freqs` objects.

## Antenna Creation

```python
patch = ProbeFedPatchAntenna(params, CSX, FDTD)
patch.print_and_save_params(params)
patch.create_probe_fed_patch()
patch.create_substrate()
patch.create_ground()
port = patch.create_port()
patch.create_mesh()
nf2ff = patch.create_nf2ff(FDTD)
patch.add_field_dump(CSX, params)
patch.write_and_show_structure(FDTD)
```
The above code creates the `Probe Fed Patch Antenna` and launches `CSXCAD` to visualise the antenna structure.
The code builds the individual antenna parts and then calls `AppCSXCAD` to view the model.

- *`patch = ProbeFedPatchAntenna()`*: This creates the Probe Fed Patch Antenna object and takes the `params`, `CSX`, and `FDTD` object.
- *`patch.print_and_save_params()`*: Once the antenna is created, all parameters and properties of the antenna is printed and saved to `.txt` file for review.
- *`patch.create_probe_fed_patch()`*: Creates probe fed patch antenna element.
- *`patch.create_substrate()`*: Creates the substrate material.
- *`patch.create_ground()`*: Creates the ground below beneath the substrate.
- *`patch.create_port()`*: Creates a lumped port with the specified characteristic impedance.
- *`patch.create_mesh()`*: Creates the mesh over the simulation domain.
- *`patch.create_nf2ff()`*: Creates a near to far field recording box which will be used for radiation pattern calculation and plotting.
- *`patch.add_field_dump()`*: This is used to add a field dump such as `electric field`, `magnetic field`to be visualised in `Paraview`.
- *`patch.write_and_show_structure()`*: This writes the antenna structure to `.xml` file and launches the `AppCSXCAD` to view the structure.

Now we have finished the antenna structure and it is ready for simulation with `openEMS`.

## openEMS simulation

```python
patch.run_simulation(FDTD)
```
This calls the openEMS FDTD engine to run the simulation of the antenna. wait for the simulation to finish.

## Post-processing

Once the simulation is finished, it is time to postprocess the simulation result to see some amazing plots and 3D visualisation.

```python
network_params = patch.compute_network_params(
    port,
    freqs,
    params.charac_imp,
)
nf2ff_3d_result = patch.compute_nf2ff_3d(nf2ff, params.resonant_freq)
patch.plot_s_param(network_params.freqs, network_params.s11)
patch.plot_smith_chart(network_params.freqs, network_params.s11)
patch.plot_vswr(network_params.freqs, network_params.vswr)
patch.plot_impedance(network_params.freqs, network_params.z11)
patch.plot_2d_directivity(nf2ff, params.resonant_freq)
patch.plot_2d_rad_pattern(nf2ff, params.resonant_freq)
patch.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq)
patch.plot_3d_gain(nf2ff_3d_result, params.resonant_freq, network_params.input_power)
patch.plot_3d_power(nf2ff_3d_result, params.resonant_freq)
patch.save_plots()
patch.show_plots()

```

- *`network_params = patch.compute_network_params()`*: This function calculates various network parameters such as `S11`, `VSWR`, from the simulation result and returns the computed result.
- *`nf2ff_3d_result = patch.compute_nf2ff_3d()`*: Computes the 3D radiation pattern from nf2ff recording box.
- *`patch.plot_s_param()`*: plots the reflection coefficient `S11` which indicates how well matched the antenna is, and optionally `S21`.
- *`patch.plot_smith_chart()`*:plots the reflection coefficient on a smith chart.
- *`patch.plot_vswr()`*: plots the voltage standing wave ratio `VSWR` of the antenna.
- *`patch.plot_impedance()`*: plots the complex input impedance `Z11` of the antenna.
- *`patch.plot_2d_directivity()`*: plots the directivity of the antenna on a polar plot.
- *`patch.plot_2d_rad_pattern()`*: plots the `theta` and `phi` components of the radiation pattern of the antenna on a polar plot.
- *`patch.plot_3d_directivity()`*: plots the directivity of the antenna in 3D.
- *`patch.plot_3d_gain()`*: plots the gain of the antenna in 3D.
- *`patch.plot_3d_power()`*: plots the 3D power of the antenna.
- *`patch.save_plots()`*: Saves the 2D plots as an `.png` images.
- *`patch.show_plots()`*: Displays all 2D and 3D plots to the user.

## External Export
simpleEMS can also export the created model to other formats for further processing.

```python
patch.export_stl()
patch.export_touchstone(
    network_params.freqs,
    network_params.s11,
    output_path=output_path,
    charac_imp=params.charac_imp,
)
patch.export_gerber(CSX)
```

- *`patch.export_stl()`*: Exports the geometry of the antenna to 3D `STL` file.
- *`patch.export_touchstone()`*: Exports the `S11` parameters to touchstone format.
- *`patch.export_gerber()`*: Exports the patch antenna model to gerber file.


## Complete script

Below is the compelete script to design, simulate and post-process the probe fed patch antenna.

```{code-block} python 

#!/usr/bin/env python3
from simpleEMS import (
    ProbeFedPatchParams,
    ProbeFedPatchAntenna,
    setup_simulation,
)

params = ProbeFedPatchParams(
    resonant_freq=2.45e9,
    corner_freq=0.5e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)

CSX, FDTD, freqs = setup_simulation(params)

patch = ProbeFedPatchAntenna(params, CSX, FDTD)
patch.print_and_save_params(params)
patch.create_probe_fed_patch()
patch.create_substrate()
patch.create_ground()
port = patch.create_port()
patch.create_mesh()
nf2ff = patch.create_nf2ff(FDTD)
patch.add_field_dump(CSX, params)
patch.write_and_show_structure(FDTD)

patch.run_simulation(FDTD)
network_params = patch.compute_network_params(
    port,
    freqs,
    params.charac_imp,
)
nf2ff_3d_result = patch.compute_nf2ff_3d(nf2ff, params.resonant_freq)
patch.plot_s_param(freqs, network_params.s11, network_params.s21)
patch.plot_smith_chart(freqs, network_params.s11)
patch.plot_vswr(freqs, network_params.vswr)
patch.plot_impedance(freqs, network_params.z11)
patch.plot_2d_directivity(nf2ff, params.resonant_freq)
patch.plot_2d_rad_pattern(nf2ff, params.resonant_freq)
patch.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq)
patch.plot_3d_gain(nf2ff_3d_result, params.resonant_freq, network_params.input_power)
patch.plot_3d_power(nf2ff_3d_result, params.resonant_freq)
patch.save_plots()
patch.show_plots()
patch.export_stl()
patch.export_touchstone(
    network_params.freqs,
    network_params.s11,
    output_path=output_path,
    charac_imp=params.charac_imp,
)
patch.export_gerber(CSX)
``` 
## Simplifying The Code 

This is a lot of code. who has the time to type all of this code.
The above code is very verbose for a reason; to make the user understand what is going on under the hood.

Below is much simpler version of the same code.

```{code-block} python
:linenos
#!/usr/bin/env python3
from simpleEMS import (
    ProbeFedPatchParams,
    ProbeFedPatchAntenna,
    setup_simulation,
)

params = ProbeFedPatchParams(
    resonant_freq=2.45e9,
    corner_freq=0.5e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)

CSX, FDTD, freqs = setup_simulation(params)

patch = ProbeFedPatchAntenna(params, CSX, FDTD)
patch.print_and_save_params(params)
port, nf2ff = patch.build_probe_fed_patch_antenna()
patch.add_field_dump(CSX, params)
patch.write_and_show_structure(FDTD)

patch.run_simulation(FDTD)

network_params = patch.compute_network_params(
    port,
    freqs,
    params.charac_imp,
)
nf2ff_3d_result = patch.compute_nf2ff_3d(nf2ff, params.resonant_freq)
patch.run_all_post_processing(
    CSX,
    network_params.freqs,
    network_params.s11,
    network_params.vswr,
    network_params.z11,
    network_params.input_power,
    nf2ff,
    nf2ff_3d_result,
    params,
    s21=network_params.s21,
)
```

Wow! how simple is designing an antenna. 
