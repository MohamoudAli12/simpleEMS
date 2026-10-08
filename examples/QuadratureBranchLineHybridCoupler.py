#!/usr/bin/env python3
"""3 dB quadrature branch-line hybrid coupler design example at 2.45 GHz."""

# IMPORTS
from simpleEMS import (
    QuadratureBranchLineHybridCoupler,
    QuadratureBranchLineHybridCouplerParams,
    setup_simulation,
)

# IMPORTS

# PARAMS
params = QuadratureBranchLineHybridCouplerParams(
    min_freq=1e9,
    max_freq=3.5e9,
    centre_freq=2.45e9,
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
coupler = QuadratureBranchLineHybridCoupler(params, sim)
coupler.print_and_save_params(params)
ports = coupler.build_quadrature_branch_line_hybrid_coupler()
coupler.create_mesh()
coupler.add_field_dump(sim, params)
coupler.write_and_show_structure(sim)
# BUILD

# SIMULATE
coupler.run_simulation(sim)
# SIMULATE

# PPROCESS
sim_data = coupler.compute_sim_data(sim, ports)

coupler.plot_s_param(
    sim_data.freqs,
    sim_data.s11,
    sim_data.s21,
    sim_data.s31,
    sim_data.s41,
)
coupler.plot_impedance(sim_data.freqs, sim_data.z11)
coupler.plot_smith_chart(sim_data.freqs, sim_data.s11)
coupler.plot_phase(sim_data.freqs, sim_data.s21, sim_data.s31)
coupler.show_plots()
coupler.export_gerber(sim)
coupler.export_step(sim)
# PPROCESS
