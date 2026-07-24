#!/usr/bin/env python3
"""Probe-fed patch antenna design at 2.45 GHz with parameter sweep and optimisation."""

from pathlib import Path

from simpleEMS import (
    ProbeFedPatchAntenna,
    ProbeFedPatchParams,
    optimize_s11,
    optimize_s_params,
    param_sweep,
    setup_simulation,
)


def simulate(output_path, sweep=False, sweep_val=[], optimize=False, optimize_val=[]):
    params = ProbeFedPatchParams(
        resonant_freq=2.45e9,  # Hz
        span_freq=0.5e9,  # Hz
        substrate_thickness_mm=1.6,  # mm
        substrate_eps_r=4.4,
        substrate_tand=0.001,
        charac_imp=50,
    )

    if sweep:
        params.patch_length_mm = sweep_val[0]
        params.patch_width_mm = sweep_val[1]

    if optimize:
        params.probe_pos_mm = optimize_val[0]
        params.patch_width_mm = optimize_val[1]
        params.patch_length_mm = optimize_val[2]

    params.probe_pos_mm = round(params.probe_pos_mm - 7, params.fp_precision)
    params.patch_length_mm = round(params.patch_length_mm - 1.14, params.fp_precision)

    sim = setup_simulation(params)
    patch = ProbeFedPatchAntenna(params, sim)
    patch.print_and_save_params(params, output_path)
    port = patch.build_probe_fed_patch_antenna()
    patch.create_mesh()
    nf2ff = patch.create_nf2ff(sim)
    patch.add_field_dump(sim, params, output_path)
    patch.write_and_show_structure(sim, output_path)
    sim_data = None

    if sweep:
        patch.run_simulation(sim, output_path)
        sim_data = patch.compute_sim_data(sim, port, output_path)

        return sim_data

    if optimize:
        patch.run_simulation(sim, output_path)
        sim_data = patch.compute_sim_data(sim, port, output_path)
        return optimize_s11(
            sim_data.freqs,
            sim_data.s11,
            target_freq=params.resonant_freq,
        )

    elif not (sweep and optimize):
        patch.run_simulation(sim, output_path)
        sim_data = patch.compute_sim_data(sim, port, output_path)

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
        patch.export_stl(output_path)
        patch.export_touchstone(
            freqs=sim_data.freqs,
            s11=sim_data.s11,
            charac_imp=params.charac_imp,
            output_path=output_path,
        )

        patch.export_gerber(sim, output_path)


def main():
    output_path = Path(__file__).with_suffix("")
    output_path.mkdir(parents=True, exist_ok=True)

    optimize = False
    if optimize:
        x0 = {
            "probe_pos_mm": 11.501,
            "patch_width_mm": 37.234,
            "patch_length_mm": 28.809,
        }
        optimize_s_params(simulate, x0, output_path)

    sweep = False
    if sweep:
        sweep_vals = {
            "patch_length_mm": (0.533, 0.750, 3),
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
