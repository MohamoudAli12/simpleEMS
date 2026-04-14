#!/usr/bin/env python3
from simpleEMS import (
    InsetFedPatchParams,
    InsetFedPatchAntenna,
    setup_simulation,
)

params = InsetFedPatchParams(
    resonant_freq=2.45e9,
    corner_freq=0.5e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)

CSX, FDTD = setup_simulation(params)

patch = InsetFedPatchAntenna(params, CSX, FDTD)
patch.print_and_save_params(params)
patch.create_patch_with_inset()
patch.create_feed()
patch.create_substrate()
patch.create_ground()
port = patch.create_port()
patch.create_mesh()
nf2ff = patch.create_nf2ff(FDTD)
patch.add_field_dump(CSX, params)
patch.write_and_show_structure(FDTD)

patch.run_simulation(FDTD)

network_params = patch.compute_network_params(port, params)
nf2ff_3d_result = patch.compute_nf2ff_3d(nf2ff, params.resonant_freq)
patch.plot_s11(network_params.freqs, network_params.s11)
patch.plot_smith_chart(network_params.freqs, network_params.s11)
patch.plot_vswr(network_params.freqs, network_params.vswr)
patch.plot_impedance(network_params.freqs, network_params.z11)
patch.plot_2d_directivity(nf2ff, params.resonant_freq)
patch.plot_2d_rad_pattern(nf2ff, params.resonant_freq)
patch.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq)
patch.plot_3d_gain(nf2ff_3d_result, params.resonant_freq, network_params.input_power)
patch.plot_3d_power(nf2ff_3d_result, params.resonant_freq)
patch.save_plots()
patch.show_plots()
patch.export_stl()
patch.export_touchstone(network_params)
patch.export_gerber(
    CSX,
    options={"ignore": ["ground"]},
)
