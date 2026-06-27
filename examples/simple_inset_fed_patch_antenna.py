#!/usr/bin/env python3
"""Simple inset-fed patch antenna example at 2.45 GHz."""

from pathlib import Path

from simpleEMS import (
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    setup_simulation,
)

# PARAMS
params = InsetFedPatchParams(
    resonant_freq=24.125e9,  # Hz
    span_freq=2e9,  # Hz
    substrate_thickness_mm=0.254,  # mm
    substrate_eps_r=3.48,
    substrate_tand=0.0037,
    charac_imp=50.0,
    end_criteria=1e-5,
    backend="FEM",
)

params.patch_length_mm = 3.15
params.patch_width_mm = 3.26
params.inset_length_mm = 1.075
params.inset_width_mm = 0.475
params.feed_width_mm = 0.275

# PARAMS

sim = setup_simulation(params)

patch = InsetFedPatchAntenna(params, sim)
patch.print_and_save_params(params)
port = patch.build_inset_fed_patch_antenna()
patch.create_mesh()

if params.backend == "FEM":
    from simpleEMS.emerge_fem import run_fem_simulation

    patch.write_and_show_structure(sim)
    patch.export_step(sim)

    result = run_fem_simulation(Path.cwd() / "step" / "structure.step", params)
    sim_data = result.sim_data
    nf2ff = result.nf2ff
    nf2ff_3d = result.nf2ff_3d_result
else:
    nf2ff = patch.create_nf2ff(sim)
    patch.add_field_dump(sim, params)
    patch.write_and_show_structure(sim)
    patch.run_simulation(sim)
    sim_data = patch.compute_sim_data(port, sim.freqs, params.charac_imp)
    nf2ff_3d = patch.compute_nf2ff_3d(nf2ff, params.resonant_freq)

patch.plot_s_param(sim_data.freqs, sim_data.s11)
patch.plot_smith_chart(sim_data.freqs, sim_data.s11)
patch.plot_vswr(sim_data.freqs, sim_data.vswr)
patch.plot_impedance(sim_data.freqs, sim_data.z11)
patch.plot_2d_directivity(nf2ff, params.resonant_freq)
patch.plot_2d_rad_pattern(nf2ff, params.resonant_freq)
# patch.save_plots()
patch.show_plots()

# patch.export_touchstone(
#     sim_data.freqs,
#     sim_data.s11,
#     charac_imp=params.charac_imp,
# )
# patch.export_step(sim)

if params.backend == "FDTD":
    patch.export_stl()
    patch.export_gerber(sim)
