#!/usr/bin/env python3
"""Inset-fed patch antenna at 24.125 GHz solved with the FEM (GetDP) backend.

This is the same structure as ``InsetFedPatch_24GHz.py``; the only change is
``backend_engine="FEM"`` on the parameters. The pipeline then meshes the
geometry with Gmsh, solves with GetDP over an adaptive rational-interpolation
frequency sweep (``num_fem_solve_points`` full solves interpolated onto
``num_points``), and returns the usual ``SimData`` so all plotting works.

Requires the ``getdp`` binary on ``PATH`` or in ``SIMPLEEMS_GETDP_BIN``.
"""

from pathlib import Path


from simpleEMS import (
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    setup_simulation,
)


def simulate(output_path: Path) -> None:
    params = InsetFedPatchParams(
        resonant_freq=24.125e9,  # Hz
        span_freq=2e9,  # Hz
        substrate_thickness_mm=0.254,  # mm
        substrate_eps_r=3.48,
        substrate_tand=0.0037,
        charac_imp=50.0,
        num_points=201,  # interpolated output points
        backend_engine="FEM",  # <-- the only change vs. the FDTD example
        num_fem_solve_points=12,  # number of full FEM solves
    )

    params.patch_length_mm = 3.15
    params.patch_width_mm = 3.26
    params.inset_length_mm = 1.075
    params.inset_width_mm = 0.475
    params.feed_width_mm = 0.275

    sim = setup_simulation(params)

    patch = InsetFedPatchAntenna(params, sim)
    patch.print_and_save_params(params, output_path)
    port = patch.build_inset_fed_patch_antenna()
    patch.create_mesh()
    nf2ff = patch.create_nf2ff(sim)
    patch.write_and_show_structure(sim, output_path)  # FEM: shows the mesh

    patch.run_simulation(sim, output_path)  # FEM: adaptive GetDP sweep
    sim_data = patch.compute_sim_data(
        sim,
        port,
        output_path,
    )

    patch.plot_s_param(sim_data.freqs, sim_data.s11)
    patch.plot_smith_chart(sim_data.freqs, sim_data.s11)
    patch.plot_vswr(sim_data.freqs, sim_data.vswr)
    patch.plot_impedance(sim_data.freqs, sim_data.z11)

    # Radiation post-processing at resonance, reusing the existing SimTools plots.
    patch.plot_2d_directivity(nf2ff, params.resonant_freq, output_path)
    patch.plot_2d_rad_pattern(nf2ff, params.resonant_freq, output_path)
    nf2ff_3d = patch.compute_nf2ff_3d(nf2ff, params.resonant_freq, output_path)
    patch.plot_3d_directivity(nf2ff_3d, params.resonant_freq, output_path)
    patch.plot_3d_gain(
        nf2ff_3d, params.resonant_freq, sim_data.input_power, output_path
    )
    patch.plot_3d_power(nf2ff_3d, params.resonant_freq, output_path)

    patch.save_plots(output_path)
    patch.show_plots()
    patch.export_touchstone(
        freqs=sim_data.freqs,
        s11=sim_data.s11,
        charac_imp=params.charac_imp,
        output_path=output_path,
    )


def main() -> None:
    output_path = Path(__file__).with_suffix("")
    output_path.mkdir(parents=True, exist_ok=True)
    simulate(output_path=output_path)


if __name__ == "__main__":
    main()
