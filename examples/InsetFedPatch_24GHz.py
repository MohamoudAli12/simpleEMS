#!/usr/bin/env python3
"""Inset-fed patch antenna design at 24.125 GHz with sweep and optimisation."""

from pathlib import Path

from simpleEMS import (
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    optimize_s_params,
    param_sweep,
    setup_simulation,
    optimize_s11,
)


def simulate(output_path, sweep=False, sweep_val=[], optimize=False, optimize_val=[]):
    params = InsetFedPatchParams(
        resonant_freq=24.125e9,  # Hz
        span_freq=2e9,  # Hz
        substrate_thickness_mm=0.254,  # mm
        substrate_eps_r=3.48,
        substrate_tand=0.0037,
        charac_imp=50.0,
        end_criteria=1e-5,
    )

    params.patch_length_mm = 3.15
    params.patch_width_mm = 3.26
    params.inset_length_mm = 1.075
    params.inset_width_mm = 0.475
    params.feed_width_mm = 0.275

    if sweep:
        params.inset_length_mm = sweep_val[0]
        params.inset_width_mm = sweep_val[1]

    if optimize:
        params.inset_length_mm = optimize_val[0]
        params.inset_width_mm = optimize_val[1]
        params.patch_length_mm = optimize_val[2]
        params.patch_width_mm = optimize_val[3]

    sim = setup_simulation(params)

    patch = InsetFedPatchAntenna(params, sim)
    patch.print_and_save_params(params, output_path)
    port = patch.build_inset_fed_patch_antenna()
    patch.create_mesh()
    nf2ff = patch.create_nf2ff(sim)
    patch.add_field_dump(sim, params, output_path)
    patch.write_and_show_structure(sim, output_path)
    sim_data = None

    if sweep:
        patch.run_simulation(sim, output_path)
        sim_data = patch.compute_sim_data(
            port,
            sim.freqs,
            params.charac_imp,
            output_path,
        )
        return sim_data

    if optimize:
        patch.run_simulation(sim, output_path)
        sim_data = patch.compute_sim_data(
            port,
            sim.freqs,
            params.charac_imp,
            output_path,
        )
        return optimize_s11(
            sim_data.freqs,
            sim_data.s11,
            params.resonant_freq,
        )

    if not (sweep or optimize):
        patch.run_simulation(sim, output_path)
        sim_data = patch.compute_sim_data(
            port,
            sim.freqs,
            params.charac_imp,
            output_path,
        )

        nf2ff_3d_result = patch.compute_nf2ff_3d(
            nf2ff, params.resonant_freq, output_path
        )
        patch.plot_s_param(sim_data.freqs, sim_data.s11)
        patch.plot_smith_chart(sim_data.freqs, sim_data.s11)
        patch.plot_vswr(sim_data.freqs, sim_data.vswr)
        patch.plot_impedance(sim_data.freqs, sim_data.z11)
        patch.plot_2d_directivity(nf2ff, params.resonant_freq, output_path)
        patch.plot_2d_rad_pattern(nf2ff, params.resonant_freq, output_path)
        patch.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq, output_path)
        patch.plot_3d_gain(
            nf2ff_3d_result,
            params.resonant_freq,
            sim_data.input_power,
            output_path,
        )
        patch.plot_3d_power(nf2ff_3d_result, params.resonant_freq, output_path)
        patch.save_plots(output_path)
        patch.show_plots()
        patch.export_touchstone(
            freqs=sim_data.freqs,
            s11=sim_data.s11,
            charac_imp=params.charac_imp,
            output_path=output_path,
        )
        patch.export_stl(output_path)
        patch.export_gerber(sim, output_path)
        patch.export_step(sim, output_path)


def main():
    output_path = Path(__file__).with_suffix("")
    output_path.mkdir(parents=True, exist_ok=True)

    optimize = False
    if optimize:
        x0 = {
            "inset_length_mm": 0.533,
            "inset_width_mm": 0.148,
            "patch_length_mm": 1.389,
            "patch_width_mm": 1.767,
        }
        optimize_s_params(simulate, x0, output_path)

    sweep = False
    if sweep:
        sweep_vals = {
            "inset_length_mm": (0.533, 0.750, 3),
            # "inset_width_mm": (0.148, 0.158, 3),
        }
        param_sweep(
            simulate_fn=simulate,
            sweep_vals=sweep_vals,
            output_path=output_path,
        )

    if not (sweep or optimize):
        simulate(output_path=output_path)


if __name__ == "__main__":
    main()
