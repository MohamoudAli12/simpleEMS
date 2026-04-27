#!/usr/bin/env python3
from simpleEMS import (
    InsetFedPatchParams,
    InsetFedPatchAntenna,
    setup_simulation,
)

params = InsetFedPatchParams(
    operating_freq=2.45e9,
    corner_freq=0.5e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)

CSX, FDTD = setup_simulation(params, boundary_cond=["MUR"] * 6)

patch = InsetFedPatchAntenna(params, CSX, FDTD)
patch.print_and_save_params(params)
port, nf2ff = patch.build_inset_fed_patch_antenna()
patch.add_field_dump(CSX, params)
patch.write_and_show_structure(FDTD)

patch.run_simulation(FDTD)

network_params = patch.compute_network_params(
    port,
    params.operating_freq,
    params.corner_freq,
    params.num_points,
    params.charac_imp,
)

nf2ff_3d_result = patch.compute_nf2ff_3d(nf2ff, params.operating_freq)
patch.run_all_post_processing(
    CSX,
    network_params.freqs,
    network_params.s11,
    network_params.vswr,
    network_params.z11,
    network_params.input_power,
    nf2ff,
    nf2ff_3d_result,
    params,
)
