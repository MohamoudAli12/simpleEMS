#!/usr/bin/env python3
"""Inset-fed patch antenna design at 60 GHz with parameter sweep and optimisation."""

from pathlib import Path

from simpleEMS import (
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    optimize_s11,
    optimize_s_params,
    param_sweep,
    setup_simulation,
    DumpType,
)


def simulate(output_path, sweep=False, sweep_val=[], optimize=False, optimize_val=[]):
    params = InsetFedPatchParams(
        resonant_freq=60e9,  # Hz
        span_freq=3e9,  # Hz
        substrate_thickness_mm=0.125,  # mm
        substrate_eps_r=3.00,
        substrate_tand=0.001,
        charac_imp=50.0,
        end_criteria=1e-4,
        min_trace_spacing_mm=0.05,
        backend_engine="FEM",
    )

    # parameters values after optimization
    params.inset_length_mm = 0.3733943584019176
    params.inset_width_mm = 0.16049029225835224
    params.patch_length_mm = 1.3351978511911158
    params.patch_width_mm = 1.9588453758383704
    params.feed_width_mm = 0.3143571692930055

    if sweep:
        params.inset_length_mm = sweep_val[0]
        params.inset_width_mm = sweep_val[1]

    if optimize:
        params.inset_length_mm = optimize_val[0]
        params.inset_width_mm = optimize_val[1]
        params.patch_length_mm = optimize_val[2]
        params.patch_width_mm = optimize_val[3]
        params.feed_width_mm = optimize_val[4]

    sim = setup_simulation(params)

    patch = InsetFedPatchAntenna(params, sim)
    patch.print_and_save_params(params, output_path)
    port = patch.build_inset_fed_patch_antenna()
    patch.create_mesh()
    nf2ff = patch.create_nf2ff(sim)
    patch.add_field_dump(
        sim,
        params,
        output_path,
        dump_type=DumpType.current_density_time,
    )
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
            target_freq=params.resonant_freq,
        )

    if not (sweep or optimize):
        patch.write_and_show_structure(sim, output_path)
        patch.run_simulation(sim, output_path)
        sim_data = patch.compute_sim_data(
            port,
            sim.freqs,
            params.charac_imp,
            output_path,
            backend_engine=params.backend_engine,
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
        patch.show_plots()
        patch.save_plots(output_path)
        patch.export_stl(output_path)
        patch.export_touchstone(
            sim_data.freqs,
            sim_data.s11,
            output_path=output_path,
            charac_imp=params.charac_imp,
        )
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
            "feed_width_mm": 0.296,
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
