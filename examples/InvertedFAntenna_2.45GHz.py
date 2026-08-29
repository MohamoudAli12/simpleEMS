#!/usr/bin/env python3
# IMPORTS
from simpleEMS import (
    InvertedFAntenna,
    InvertedFAntennaParams,
    setup_simulation,
)

# IMPORTS

# PARAMS
params = InvertedFAntennaParams(
    resonant_freq=2.45e9,
    span_freq=1e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)
# PARAMS

# SETUP
sim = setup_simulation(params)
# SETUP


# BUILD
ifa = InvertedFAntenna(params, sim)
ifa.print_and_save_params(params)
port = ifa.build_inverted_f_antenna()
nf2ff = ifa.create_nf2ff(sim)
ifa.add_field_dump(sim, params)
ifa.write_and_show_structure(sim)
# BUILD

# SIMULATE
ifa.run_simulation(sim)
# SIMULATE

# PPROCESS
sim_data = ifa.compute_sim_data(sim, port)
nf2ff_3d_result = ifa.compute_nf2ff_3d(nf2ff, params.resonant_freq)
ifa.plot_s_param(sim_data.freqs, sim_data.s11)
ifa.plot_smith_chart(sim_data.freqs, sim_data.s11)
ifa.plot_vswr(sim_data.freqs, sim_data.vswr)
ifa.plot_impedance(sim_data.freqs, sim_data.z11)
ifa.plot_2d_directivity(nf2ff, params.resonant_freq)
ifa.plot_2d_rad_pattern(nf2ff, params.resonant_freq)
ifa.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq)
ifa.plot_3d_gain(nf2ff_3d_result, params.resonant_freq, sim_data.input_power)
ifa.plot_3d_power(nf2ff_3d_result, params.resonant_freq)
ifa.save_plots()
ifa.show_plots()
ifa.export_step(sim)
ifa.export_gerber(sim)
# PPROCESS
