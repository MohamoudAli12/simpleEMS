#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from simpleEMS import (
    InsetFedPatchParams,
    InsetFedPatchAntenna,
    setup_simulation,
    optimize_s11,
    param_sweep,
)


def simulate(output_path, sweep=False, sweep_val=[], optimize=False, optimize_val=[]):
    params = InsetFedPatchParams(
        resonant_freq=2.45e9,
        corner_freq=0.5e9,
        substrate_thickness_mm=1.6,
        substrate_eps_r=4.4,
        substrate_tand=0.001,
        charac_imp=50,
    )

    if sweep:
        params.inset_length_mm = sweep_val[0]
        params.inset_width_mm = sweep_val[1]

    if optimize:
        params.inset_length_mm = optimize_val[0]
        params.inset_width_mm = optimize_val[1]
        params.patch_length_mm = optimize_val[2]
        params.patch_width_mm = optimize_val[3]

    CSX, FDTD = setup_simulation(params)

    patch = InsetFedPatchAntenna(params, CSX, FDTD)
    patch.print_and_save_params(params, output_path)
    patch.create_patch_with_inset()
    patch.create_feed()
    port = patch.create_port()
    patch.create_substrate()
    patch.create_ground()
    patch.create_mesh()
    nf2ff = patch.create_nf2ff(FDTD)
    patch.add_field_dump(CSX, params, output_path)
    patch.write_and_show_structure(FDTD, output_path)
    network_params = None

    if sweep:
        patch.run_simulation(FDTD, output_path)
        network_params = patch.compute_network_params(port, params, output_path)
        return network_params
    if optimize:
        patch.run_simulation(FDTD, output_path)
        network_params = patch.compute_network_params(port, params, output_path)

        freqs = network_params.freqs
        s11 = network_params.s11
        s11 = 20.0 * np.log10(np.abs(s11))
        idx = (np.abs(freqs - params.resonant_freq)).argmin()
        return s11[idx]

    if not (sweep or optimize):
        patch.run_simulation(FDTD, output_path)
        network_params = patch.compute_network_params(port, params, output_path)
        nf2ff_3d_result = patch.compute_nf2ff_3d(
            nf2ff, params.resonant_freq, output_path
        )
        patch.plot_s11(network_params.freqs, network_params.s11)
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
        patch.export_touchstone(network_params, output_path)
        patch.export_gerber(
            CSX,
            output_path,
            options={"ignore": ["ground"]},
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
        optimize_s11(simulate, x0, output_path)

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
