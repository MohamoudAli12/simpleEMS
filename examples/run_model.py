#!/usr/bin/env python3
from pathlib import Path
from simpleEMS import simulate_model, SimTools

# PATH
output_path = Path(__file__).with_suffix("")
output_path.mkdir(parents=True, exist_ok=True)

field_dump = output_path / "field_dump"
field_dump.mkdir(parents=True, exist_ok=True)
# PATH

# SIMULATE
sim_data, sim, charac_imp = simulate_model(
    "structure.xml",
    output_path,
)
# SIMULATE

# PPROCESS
SimTools.plot_s_param(
    sim_data.freqs,
    sim_data.s11,
    sim_data.s21,
)
SimTools.plot_impedance(sim_data.freqs, sim_data.z11)
SimTools.plot_smith_chart(sim_data.freqs, sim_data.s11)
SimTools.plot_vswr(sim_data.freqs, sim_data.vswr)
SimTools.show_plots()
SimTools.save_plots(output_path)
# PPROCESS

# EXPORT
SimTools.export_stl(output_path)
SimTools.export_touchstone(
    sim_data.freqs,
    sim_data.s11,
    output_path=output_path,
    charac_imp=charac_imp,
)
SimTools.export_gerber(sim, output_path)
# EXPORT
