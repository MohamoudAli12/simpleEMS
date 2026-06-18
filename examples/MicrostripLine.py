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
CSX, FDTD, freqs = setup_simulation(params)
# SETUP

# BUILD
microstrip = MicrostripLine(params, CSX, FDTD)
microstrip.print_and_save_params(params)
ports = microstrip.build_microstrip_line()
microstrip.create_mesh()
microstrip.add_field_dump(CSX, params)
microstrip.write_and_show_structure(FDTD)
# BUILD

# SIMULATE
microstrip.run_simulation(FDTD)
# SIMULATE

# PPROCESS
network_params = microstrip.compute_network_params(
    ports,
    freqs,
    params.charac_imp,
)

microstrip.plot_s_param(freqs, network_params.s11, network_params.s21)
microstrip.plot_impedance(freqs, network_params.z11)
microstrip.plot_phase(freqs, network_params.phase)
microstrip.show_plots()
# PPROCESS
