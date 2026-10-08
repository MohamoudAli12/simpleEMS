#!/usr/bin/env python3
"""Centre-fed printed half-wave dipole design example at 2.45 GHz."""

# IMPORTS
from simpleEMS import (
    PrintedDipoleAntenna,
    PrintedDipoleAntennaParams,
    setup_simulation,
)

# IMPORTS

# PARAMS
params = PrintedDipoleAntennaParams(
    min_freq=4e9,
    max_freq=7e9,
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
dipole = PrintedDipoleAntenna(params, sim)
dipole.print_and_save_params(params)
port = dipole.build_printed_dipole_antenna()
dipole.create_mesh()
nf2ff = dipole.create_nf2ff(sim)
dipole.add_field_dump(sim, params)
dipole.write_and_show_structure(sim)
# BUILD

# SIMULATE
dipole.run_simulation(sim)
# SIMULATE

# PPROCESS
sim_data = dipole.compute_sim_data(sim, port)
nf2ff_3d_result = dipole.compute_nf2ff_3d(nf2ff, params.resonant_freq)
dipole.plot_s_param(sim_data.freqs, sim_data.s11)
dipole.plot_smith_chart(sim_data.freqs, sim_data.s11)
dipole.plot_vswr(sim_data.freqs, sim_data.vswr)
dipole.plot_impedance(sim_data.freqs, sim_data.z11)
dipole.plot_2d_directivity(nf2ff, params.resonant_freq)
dipole.plot_2d_rad_pattern(nf2ff, params.resonant_freq)
dipole.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq)
dipole.plot_3d_gain(nf2ff_3d_result, params.resonant_freq, sim_data.input_power)
dipole.save_plots()
dipole.show_plots()
dipole.export_step(sim)
dipole.export_gerber(sim)
# PPROCESS
