#!/usr/bin/env python3
"""Inset-fed patch antenna at 24.125 GHz, FEM simulation directly from a standalone STEP file."""

# IMPORTS
from pathlib import Path
import numpy as np
from simpleEMS import FEMNF2FF, FEMOptions, SimTools, simulate_step_FEM
# IMPORTS

# PARAMS
STEP_FILE = Path(__file__).parent / "structure.step"

resonant_freq = 24.125e9  # Hz
span_freq = 2e9  # Hz
freqs = np.linspace(resonant_freq - span_freq, resonant_freq + span_freq, 501)
# PARAMS

# SIMULATE
sim_data = simulate_step_FEM(
    STEP_FILE,
    freqs,
    dielectrics={"substrate": (3.48, 0.0037)},  # (eps_r, tan_d)
    pec=["patch_inset", "feed", "ground"],
    ports={"port_resist_1": {"z0": 50.0, "direction": "z", "number": 1}},
    num_solve_points=12,
    FEM_options=FEMOptions(boundary="silver_muller", port_type="lumped"),
    charac_imp=50.0,
)
# SIMULATE

# PPROCESS
SimTools.plot_s_param(sim_data.freqs, sim_data.s11)
SimTools.plot_smith_chart(
    sim_data.freqs, sim_data.s11, charac_imp=sim_data.ref_impedance
)
SimTools.plot_vswr(sim_data.freqs, sim_data.vswr)
SimTools.plot_impedance(sim_data.freqs, sim_data.z11)

nf2ff = FEMNF2FF()
SimTools.plot_2d_directivity(nf2ff, resonant_freq)
SimTools.plot_2d_rad_pattern(nf2ff, resonant_freq)
nf2ff_3d = SimTools.compute_nf2ff_3d(nf2ff, resonant_freq)
SimTools.plot_3d_directivity(nf2ff_3d, resonant_freq)
SimTools.plot_3d_gain(nf2ff_3d, resonant_freq, sim_data.input_power)
SimTools.plot_3d_power(nf2ff_3d, resonant_freq)

SimTools.save_plots()
SimTools.show_plots()
# PPROCESS

# EXPORT
SimTools.export_touchstone(freqs=sim_data.freqs, s11=sim_data.s11, charac_imp=50.0)
# EXPORT
