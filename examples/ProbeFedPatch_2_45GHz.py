#!/usr/bin/env python3
from pathlib import Path
from simpleEMS import (
    ProbeFedPatchParams,
    ProbeFedPatchAntenna,
    setup_simulation,
    optimize_s_params,
    optimize_s11,
    param_sweep,
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

    # params.probe_pos_mm = np.round(params.probe_pos_mm - 7, params.fp_precision)
    # params.patch_length_mm = np.round(
    #     params.patch_length_mm - 1.14, params.fp_precision
    # )

    CSX, FDTD, freqs = setup_simulation(params)
    patch = ProbeFedPatchAntenna(params, CSX, FDTD)
    patch.print_and_save_params(params, output_path)
    patch.create_probe_fed_patch()
    port = patch.create_port()
    patch.create_substrate()
    patch.create_ground()
    patch.create_mesh()
    patch.add_field_dump(CSX, params, output_path)
    nf2ff = patch.create_nf2ff(FDTD)
    # patch.write_and_show_structure(FDTD, output_path)
    network_params = None

    if sweep:
        patch.run_simulation(FDTD, output_path)
        network_params = patch.compute_network_params(
            port,
            freqs,
            params.charac_imp,
            output_path,
        )

        return network_params

    if optimize:
        patch.run_simulation(FDTD, output_path)
        network_params = patch.compute_network_params(
            port,
            freqs,
            params.charac_imp,
            output_path,
        )
        return optimize_s11(
            network_params.freqs,
            network_params.s11,
            target_freq=params.resonant_freq,
        )

    elif not (sweep and optimize):
        patch.run_simulation(FDTD, output_path)
        network_params = patch.compute_network_params(
            port,
            freqs,
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
        patch.export_stl(output_path)
        patch.export_touchstone(
            freqs = network_params.freqs,
            s11=network_params.s11,
            charac_imp=params.charac_imp,
            output_path=output_path,
        )

        patch.export_gerber(CSX, output_path)


def main():
    output_path = Path(__file__).with_suffix("")
    output_path.mkdir(parents=True, exist_ok=True)

    optimize = True
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
