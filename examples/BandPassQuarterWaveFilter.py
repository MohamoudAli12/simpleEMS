#!/usr/bin/env python3

"""Quarter-wave bandpass stub filter design example at 2.45 GHz."""

# IMPORTS
from simpleEMS import QuarterWaveFilterParams, BandPassQuarterWaveFilter
from simpleEMS import setup_simulation

# IMPORTS

# PARAMS
params = QuarterWaveFilterParams(
    substrate_eps_r=3.3,
    substrate_tand=0.001,
    substrate_thickness_mm=1.6,
    min_freq=0.5e9,
    max_freq=3e9,
    centre_freq=1.5e9,
    bandwidth_freq=1e9,
    filter_type="bandpass",
    filter_response="butterworth",
    filter_order=3,
)
# PARAMS

# SETUP
sim = setup_simulation(params)
# SETUP

# BUILD
filter = BandPassQuarterWaveFilter(params, sim.CSX, sim.FDTD)
filter.print_and_save_params(params)
ports = filter.build_band_pass_quarter_wave_filter()
filter.create_mesh()
filter.write_and_show_structure(sim.FDTD)
# BUILD

# SIMULATE
filter.run_simulation(sim.FDTD)
# SIMULATE

# PPROCESS
network_params = filter.compute_network_params(ports, sim.freqs, params.charac_imp)
filter.plot_s_param(sim.freqs, network_params.s11, network_params.s21)
filter.show_plots()
# PPROCESS

# EXPORT
filter.export_gerber(sim.CSX)
filter.export_touchstone(
    freqs=sim.freqs,
    s11=network_params.s11,
    s21=network_params.s21,
)
# EXPORT
