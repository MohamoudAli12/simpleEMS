#!/usr/bin/env python3
"""Inset-fed patch antenna design at 2.45 GHz with parameter sweep and optimisation."""

from pathlib import Path

from simpleEMS import (
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    optimize_s11,
    optimize_s_params,
    param_sweep,
    setup_simulation,
)


def simulate(output_path, sweep=False, sweep_val=[], optimize=False, optimize_val=[]):
    params = InsetFedPatchParams(
        resonant_freq=2.45e9,
        span_freq=0.5e9,
        substrate_thickness_mm=1.6,
        substrate_eps_r=4.4,
        substrate_tand=0.001,
        charac_imp=50,
    )

    params.inset_length_mm = 7.63006
    params.inset_width_mm = 1.62696
    params.patch_length_mm = 28.14265
    params.patch_width_mm = 51.49757

    if sweep:
        params.inset_length_mm = sweep_val[0]
        params.inset_width_mm = sweep_val[1]

    if optimize:
        params.inset_length_mm = optimize_val[0]
        params.inset_width_mm = optimize_val[1]
        params.patch_length_mm = optimize_val[2]
        params.patch_width_mm = optimize_val[3]

    sim = setup_simulation(params)

    patch = InsetFedPatchAntenna(params, sim.CSX, sim.FDTD)
    patch.print_and_save_params(params, output_path)
    port = patch.build_inset_fed_patch_antenna()
    patch.create_mesh()
    nf2ff = patch.create_nf2ff(sim.FDTD)
    patch.add_field_dump(sim.CSX, params, output_path)
    patch.write_and_show_structure(sim.FDTD, output_path)
    network_params = None

    if sweep:
        patch.run_simulation(sim.FDTD, output_path)
        network_params = patch.compute_network_params(
            port,
            sim.freqs,
            params.charac_imp,
            output_path,
        )

        return network_params
    if optimize:
        patch.run_simulation(sim.FDTD, output_path)
        network_params = patch.compute_network_params(
            port,
            sim.freqs,
            params.charac_imp,
            output_path,
        )

        return optimize_s11(
            network_params.freqs,
            network_params.s11,
            target_freq=params.resonant_freq,
        )

    if not (sweep or optimize):
        patch.run_simulation(sim.FDTD, output_path)
        network_params = patch.compute_network_params(
            port,
            sim.freqs,
            params.charac_imp,
            output_path,
        )

        nf2ff_3d_result = patch.compute_nf2ff_3d(
            nf2ff, params.resonant_freq, output_path
        )
        patch.plot_s_param(network_params.freqs, network_params.s11)
        patch.plot_smith_chart(network_params.freqs, network_params.s11)
        patch.plot_vswr(network_params.freqs, network_params.vswr)
        patch.plot_impedance(network_params.freqs, network_params.z11)
        patch.plot_2d_directivity(nf2ff, params.resonant_freq, output_path)
        patch.plot_2d_rad_pattern(nf2ff, params.resonant_freq, output_path)
        patch.plot_3d_directivity(nf2ff_3d_result, params.resonant_freq, output_path)
        patch.plot_3d_gain(
            nf2ff_3d_result,
            params.resonant_freq,
            network_params.input_power,
            output_path,
        )
        patch.plot_3d_power(nf2ff_3d_result, params.resonant_freq, output_path)
        patch.save_plots(output_path)
        patch.show_plots()
        patch.export_touchstone(
            freqs=network_params.freqs,
            s11=network_params.s11,
            charac_imp=params.charac_imp,
            output_path=output_path,
        )
        patch.export_stl(output_path)
        patch.export_gerber(sim.CSX, output_path)
        patch.export_step(sim.CSX, output_path)


def main():
    output_path = Path(__file__).with_suffix("")
    output_path.mkdir(parents=True, exist_ok=True)

    optimize = False
    if optimize:
        x0 = {
            "inset_length_mm": 11.501,
            "inset_width_mm": 1.531,
            "patch_length_mm": 28.809,
            "patch_width_mm": 37.234,
        }
        optimize_s_params(simulate, x0, output_path)

    sweep = False
    if sweep:
        sweep_vals = {
            "inset_length_mm": (0.533, 0.750, 3),
            "inset_width_mm": (0.148, 0.158, 3),
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
