#!/usr/bin/env python3
"""Microstrip transmission line design example at 1.7 GHz."""

# IMPORTS
from simpleEMS import (
    MicrostripLine,
    MicrostripLineParams,
    setup_simulation,
)

# IMPORTS

# PARAMS
params = MicrostripLineParams(
    min_freq=1e9,
    max_freq=2e9,
    target_freq=1.7e9,
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
microstrip = MicrostripLine(params, sim)
microstrip.print_and_save_params(params)
ports = microstrip.build_microstrip_line()
microstrip.create_mesh()
microstrip.add_field_dump(sim, params)
microstrip.write_and_show_structure(sim)
# BUILD

# SIMULATE
microstrip.run_simulation(sim)
# SIMULATE

# PPROCESS
sim_data = microstrip.compute_sim_data(
    ports,
    sim.freqs,
    params.charac_imp,
)

microstrip.plot_s_param(sim_data.freqs, sim_data.s11, sim_data.s21)
microstrip.plot_impedance(sim_data.freqs, sim_data.z11)
microstrip.plot_phase(sim_data.freqs, sim_data.s21)
microstrip.show_plots()
# PPROCESS
