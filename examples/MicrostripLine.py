#!/usr/bin/env python3
from simpleEMS import (
    MicrostripLineParams,
    MicrostripLine,
    setup_simulation,
)

params = MicrostripLineParams(
    min_freq=1e9,
    max_freq=2e9,
    target_freq=1.7e9,
    substrate_thickness_mm=1.6,
    substrate_eps_r=4.4,
    substrate_tand=0.001,
    charac_imp=50,
)

CSX, FDTD, freqs = setup_simulation(params, boundary_cond=["MUR"] * 6)

microstrip = MicrostripLine(params, CSX, FDTD)
microstrip.print_and_save_params(params)
microstrip.create_substrate()
microstrip.create_ground()
microstrip.create_microstrip_line()
port = microstrip.create_ports()
microstrip.create_mesh()
nf2ff = microstrip.create_nf2ff(FDTD)
microstrip.add_field_dump(CSX, params)
microstrip.write_and_show_structure(FDTD)

microstrip.run_simulation(FDTD)

network_params = microstrip.compute_network_params(
    port,
    freqs,
    params.charac_imp,
)

nf2ff_3d_result = microstrip.compute_nf2ff_3d(nf2ff, params.target_freq)
microstrip.run_all_post_processing(
    CSX,
    freqs,
    network_params.s11,
    network_params.vswr,
    network_params.z11,
    network_params.input_power,
    nf2ff,
    nf2ff_3d_result,
    params,
    network_params.s21,
)
