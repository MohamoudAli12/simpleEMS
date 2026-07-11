#!/usr/bin/env python3
"""Microstrip transmission line (2-port) solved with the FEM (GetDP) backend.

Same structure as ``MicrostripLine.py``; only ``backend_engine="FEM"`` changes.
Because it is a 2-port line the FEM sweep produces S21, so the existing
``plot_phase`` and ``plot_group_delay`` SimTools methods are reused unchanged.

Requires the ``getdp`` binary on ``PATH`` or in ``SIMPLEEMS_GETDP_BIN``.
"""

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
    num_points=201,  # interpolated output points
    backend_engine="FEM",  # <-- the only change vs. the FDTD example
    num_fem_solve_points=8,  # number of full FEM solves
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
microstrip.write_and_show_structure(sim)  # FEM: shows the mesh
# BUILD

# SIMULATE
microstrip.run_simulation(sim)  # FEM: adaptive GetDP sweep
# SIMULATE

# PPROCESS
sim_data = microstrip.compute_sim_data(
    ports,
    sim.freqs,
    params.charac_imp,
    backend_engine=params.backend_engine,
)

microstrip.plot_s_param(sim_data.freqs, sim_data.s11, sim_data.s21)
microstrip.plot_impedance(sim_data.freqs, sim_data.z11)
microstrip.plot_phase(sim_data.freqs, sim_data.s21)
microstrip.plot_group_delay(sim_data.freqs, sim_data.s21)
microstrip.show_plots()
# PPROCESS
