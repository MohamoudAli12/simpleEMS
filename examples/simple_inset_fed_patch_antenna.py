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
    backend_engine="FEM",
)
# PARAMS

# SETUP
sim = setup_simulation(params)
# SETUP

# BUILD
patch = InsetFedPatchAntenna(params, sim)
patch.print_and_save_params(params)
port = patch.build_inset_fed_patch_antenna()
patch.create_mesh()
nf2ff = patch.create_nf2ff(sim)
patch.add_field_dump(sim, params)
patch.write_and_show_structure(sim)
# BUILD

# SIMULATE
patch.run_simulation(sim)
# SIMULATE

# PPROCESS
sim_data = patch.compute_sim_data(
    port, sim.freqs, params.charac_imp, backend_engine=params.backend_engine
)
nf2ff_3d_result = patch.compute_nf2ff_3d(nf2ff, params.resonant_freq)
patch.plot_s_param(sim_data.freqs, sim_data.s11)
patch.plot_smith_chart(sim_data.freqs, sim_data.s11)
patch.plot_vswr(sim_data.freqs, sim_data.vswr)
patch.plot_impedance(sim_data.freqs, sim_data.z11)
patch.plot_2d_directivity(nf2ff, params.resonant_freq)
patch.plot_2d_rad_pattern(nf2ff, params.resonant_freq)
patch.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq)
patch.plot_3d_gain(nf2ff_3d_result, params.resonant_freq, sim_data.input_power)
patch.plot_3d_power(nf2ff_3d_result, params.resonant_freq)
patch.save_plots()
patch.show_plots()
# PPROCESS

# EXPORT
patch.export_stl()
patch.export_touchstone(
    sim_data.freqs,
    sim_data.s11,
    charac_imp=params.charac_imp,
)
patch.export_gerber(sim)
# EXPORT
