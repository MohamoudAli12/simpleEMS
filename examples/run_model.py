#!/usr/bin/env python3
from pathlib import Path
from simpleEMS import simulate_model, SimTools

output_path = Path(__file__).with_suffix("")
output_path.mkdir(parents=True, exist_ok=True)


network_params, CSX, charac_imp = simulate_model(
    "structure.xml",
    output_path,
)
SimTools.plot_s_param(
    network_params.freqs,
    network_params.s11,
    network_params.s21,
)
SimTools.plot_impedance(network_params.freqs, network_params.z11)
SimTools.plot_smith_chart(network_params.freqs, network_params.s11)
SimTools.plot_vswr(network_params.freqs, network_params.vswr)
SimTools.show_plots()
SimTools.save_plots(output_path)
SimTools.export_stl(output_path)
SimTools.export_touchstone(
    network_params.freqs,
    network_params.s11,
    output_path=output_path,
    charac_imp=charac_imp,
)
SimTools.export_gerber(CSX, output_path)
