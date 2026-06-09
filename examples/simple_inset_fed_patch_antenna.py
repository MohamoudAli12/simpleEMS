#!/usr/bin/env python3
"""Simple inset-fed patch antenna example at 2.45 GHz."""

# IMPORTS
from simpleEMS import (
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    setup_simulation,
)

# IMPORTS

# PARAMS
params = InsetFedPatchParams(
    resonant_freq=2.45e9,
    span_freq=0.5e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)
# PARAMS

# SETUP
CSX, FDTD, freqs = setup_simulation(params)
# SETUP

# BUILD
patch = InsetFedPatchAntenna(params, CSX, FDTD)
patch.print_and_save_params(params)
port, nf2ff = patch.build_inset_fed_patch_antenna()
patch.create_mesh()
patch.add_field_dump(CSX, params)
patch.write_and_show_structure(FDTD)
# BUILD

# SIMULATE
patch.run_simulation(FDTD)
# SIMULATE

# PPROCESS
network_params = patch.compute_network_params(
    port,
    freqs,
    params.charac_imp,
)
nf2ff_3d_result = patch.compute_nf2ff_3d(nf2ff, params.resonant_freq)
patch.plot_s_param(freqs, network_params.s11)
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
# PPROCESS

# EXPORT
patch.export_stl()
patch.export_touchstone(
    network_params.freqs,
    network_params.s11,
    charac_imp=params.charac_imp,
)
patch.export_gerber(CSX)
# EXPORT
