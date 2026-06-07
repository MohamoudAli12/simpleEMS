#!/usr/bin/env python3
"""Simpler inset-fed patch antenna example at 60 GHz using high-level helper methods."""

from pathlib import Path


from simpleEMS import (
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    optimize_s_params,
    optimize_s11,
    param_sweep,
    setup_simulation,
)


def simulate(output_path, sweep=False, sweep_val=[], optimize=False, optimize_val=[]):
    params = InsetFedPatchParams(
        resonant_freq=60e9,  # Hz
        span_freq=3e9,  # Hz
        substrate_thickness_mm=0.125,  # mm
        substrate_eps_r=3.00,
        substrate_tand=0.001,
        charac_imp=50.0,
    )

    # parameters values after optimization
    params.inset_length_mm = 0.489096
    params.inset_width_mm = 0.156097
    params.patch_length_mm = 1.378419
    params.patch_width_mm = 1.74283

    if sweep:
        params.inset_length_mm = sweep_val[0]
        params.inset_width_mm = sweep_val[1]

    if optimize:
        params.inset_length_mm = optimize_val[0]
        params.inset_width_mm = optimize_val[1]
        params.patch_length_mm = optimize_val[2]
        params.patch_width_mm = optimize_val[3]

    CSX, FDTD, freqs = setup_simulation(params, boundary_cond=["MUR"] * 6)

    patch = InsetFedPatchAntenna(params, CSX, FDTD)
    patch.print_and_save_params(params, output_path)
    port, nf2ff = patch.build_inset_fed_patch_antenna()
    patch.add_field_dump(CSX, params, output_path)
    patch.write_and_show_structure(FDTD, output_path)
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
            freqs,
            network_params.s11,
            params.resonant_freq,
        )

    if not (sweep or optimize):
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
            output_path,
        )


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
