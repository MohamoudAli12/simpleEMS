#!/usr/bin/env python3
"""Microstrip-fed printed quarter-wave monopole design example at 2.45 GHz."""

# IMPORTS
from simpleEMS import (
    PrintedMonopoleAntenna,
    PrintedMonopoleAntennaParams,
    setup_simulation,
)

# IMPORTS

# PARAMS
params = PrintedMonopoleAntennaParams(
    min_freq=3.5e9,
    max_freq=7.5e9,
    resonant_freq=5.21e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
    # backend_engine="FEM",
)
# PARAMS

# SETUP
sim = setup_simulation(params)
# SETUP

# BUILD
monopole = PrintedMonopoleAntenna(params, sim)
monopole.print_and_save_params(params)
port = monopole.build_printed_monopole_antenna()
monopole.create_mesh()
nf2ff = monopole.create_nf2ff(sim)
monopole.add_field_dump(sim, params)
monopole.write_and_show_structure(sim)
# BUILD

# SIMULATE
monopole.run_simulation(sim)
# SIMULATE

# PPROCESS
sim_data = monopole.compute_sim_data(sim, port)
nf2ff_3d_result = monopole.compute_nf2ff_3d(nf2ff, params.resonant_freq)
monopole.plot_s_param(sim_data.freqs, sim_data.s11)
monopole.plot_smith_chart(sim_data.freqs, sim_data.s11)
monopole.plot_vswr(sim_data.freqs, sim_data.vswr)
monopole.plot_impedance(sim_data.freqs, sim_data.z11)
monopole.plot_2d_directivity(nf2ff, params.resonant_freq)
monopole.plot_2d_rad_pattern(nf2ff, params.resonant_freq)
monopole.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq)
monopole.plot_3d_gain(nf2ff_3d_result, params.resonant_freq, sim_data.input_power)
monopole.save_plots()
monopole.show_plots()
monopole.export_step(sim)
monopole.export_gerber(sim)
# PPROCESS
