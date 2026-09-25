#!/usr/bin/env python3
"""Simple structure built from the primitives in simpleEMS.components."""

# IMPORTS
from simpleEMS import GenericParams, GenericStructure, setup_simulation

# IMPORTS

# PARAMS
params = GenericParams(
    min_freq=1e9,
    max_freq=6e9,
    target_freq=2.45e9,
    substrate_width_mm=40,
    substrate_length_mm=60,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)
# PARAMS

# SETUP
sim = setup_simulation(params)
# SETUP

# BOARD
structure = GenericStructure(params, sim)
structure.create_substrate()
structure.create_ground(start=[-20, -30, -0.035], stop=[20, 14.5, 0])
# BOARD

# TRACES
structure.create_microstrip("feed", position=(-10, -30), width_mm=3, length_mm=10)
structure.create_miter(
    position=(-10, -20), width_mm=3, miter_distance_mm=1.8, turn="right"
)
structure.create_microstrip(
    "run", position=(-8.5, -18.5), width_mm=3, length_mm=7, rotation=-90
)
structure.create_curved_bend(
    position=(-1.5, -18.5), width_mm=3, bend_radius_mm=4, rotation=-90
)
structure.create_microstrip("line", position=(2.5, -14.5), width_mm=3, length_mm=12)
structure.create_radial_stub(
    position=(4, -8.5),
    inner_radius_mm=0.5,
    outer_radius_mm=8,
    angle_start=-30,
    angle_end=30,
)
structure.create_microstrip(
    "short_stub", position=(1, -5), width_mm=1.5, length_mm=16, rotation=90
)
structure.create_via(
    "short_via",
    position=(-14.4, -5),
    via_diameter_mm=0.8,
    z_bottom_mm=-0.035,
    z_top_mm=1.635,
)
structure.create_taper(position=(2.5, -2.5), width1_mm=3, width2_mm=2, length_mm=3)
structure.create_gcpw(
    position=(2.5, 0.5),
    trace_width_mm=2,
    gap_mm=0.3,
    ground_width_mm=4,
    length_mm=14,
    via_diameter_mm=0.6,
    via_pitch_mm=2,
    via_z_bottom_mm=-0.035,
    via_z_top_mm=1.635,
)
structure.create_cpw(
    position=(2.5, 14.5),
    trace_width_mm=2,
    gap_mm=0.3,
    ground_width_mm=4,
    length_mm=15.5,
)
# TRACES

# PORTS
port_1 = structure.create_lumped_port(
    port_nr=1, start=[-11.5, -30, 0], stop=[-8.5, -30, 1.635], excite=1
)
structure.create_mesh()
port_2 = structure.create_cpw_port(
    port_nr=2, start=[1.5, 30, 1.6], stop=[3.5, 26, 1.6], gap_mm=0.3
)
structure.write_and_show_structure(sim)
# PORTS

# SIMULATE
structure.run_simulation(sim)
# SIMULATE

# PPROCESS
sim_data = structure.compute_sim_data(sim, [port_1, port_2])
structure.plot_s_param(sim_data.freqs, sim_data.s11, sim_data.s21)
structure.plot_smith_chart(sim_data.freqs, sim_data.s11)
structure.plot_impedance(sim_data.freqs, sim_data.z11)
structure.plot_phase(sim_data.freqs, sim_data.s21)
structure.save_plots()
structure.show_plots()
# PPROCESS

# EXPORT
structure.export_touchstone(
    sim_data.freqs,
    sim_data.s11,
    s21=sim_data.s21,
    charac_imp=params.charac_imp,
)
structure.export_gerber(sim)
# EXPORT
