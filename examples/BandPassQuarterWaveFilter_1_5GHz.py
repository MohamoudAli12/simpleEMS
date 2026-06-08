#!/usr/bin/env python3
"""Quarter-wave bandpass stub filter design at 1.5 GHz with parameter sweep and optimisation."""

from pathlib import Path


from simpleEMS import (
    BandPassQuarterWaveFilter,
    QuarterWaveFilterParams,
    optimize_s21,
    optimize_s_params,
    param_sweep,
    setup_simulation,
)


def simulate(output_path, sweep=False, sweep_val=[], optimize=False, optimize_val=[]):
    params = QuarterWaveFilterParams(
        substrate_eps_r=3.48,
        substrate_tand=0.0037,
        substrate_thickness_mm=1.6,
        min_freq=0.5e9,
        max_freq=2.5e9,
        centre_freq=1.5e9,
        bandwidth_freq=1e9,
        filter_type="bandpass",
        filter_response="butterworth",
        filter_order=3,
    )

    if sweep:
        params.line_length_mm = sweep_val[0]
        params.series_line_width_mm = sweep_val[1]

    if optimize:
        params.line_length_mm = optimize_val[0]
        params.series_line_width_mm = optimize_val[1]
        params.shunt_line_width_mm = optimize_val[2:5]
        params.shunt_line_length_mm = optimize_val[5:]

    CSX, FDTD, freqs = setup_simulation(params)

    filter = BandPassQuarterWaveFilter(params, CSX, FDTD)
    filter.print_and_save_params(params, output_path)
    filter.create_substrate()
    filter.create_ground()
    filter.create_series_line()
    filter.create_shunt_line()
    filter.create_shunt_line_short()
    ports = filter.create_ports()
    filter.create_mesh()
    filter.add_field_dump(CSX, params, output_path)
    filter.write_and_show_structure(FDTD, output_path)
    network_params = None

    if sweep:
        filter.run_simulation(FDTD, output_path)
        network_params = filter.compute_network_params(
            ports,
            freqs,
            params.charac_imp,
            output_path,
        )

        return network_params

    if optimize:
        filter.run_simulation(FDTD, output_path)
        network_params = filter.compute_network_params(
            ports,
            freqs,
            params.charac_imp,
            output_path,
        )

        s21_cost = optimize_s21(
            network_params.freqs,
            network_params.s21,
            params.centre_freq,
        )

        return s21_cost

    if not (sweep or optimize):
        filter.run_simulation(FDTD, output_path)
        network_params = filter.compute_network_params(
            ports,
            freqs,
            params.charac_imp,
            output_path,
        )

        filter.plot_s_param(
            network_params.freqs, network_params.s11, network_params.s21
        )
        filter.save_plots(output_path)
        filter.show_plots()
        filter.export_gerber(CSX, output_path)
        filter.export_touchstone(
            freqs=freqs,
            s11=network_params.s11,
            s21=network_params.s21,
            output_path=output_path,
        )


def main():
    output_path = Path(__file__).with_suffix("")
    output_path.mkdir(parents=True, exist_ok=True)

    optimize = False
    if optimize:
        x0 = {
            "line_length_mm": 30.314,
            "series_line_width_mm": 3.599,
            "shunt_line_width_mm_0": 0.988,
            "shunt_line_width_mm_1": 3.874,
            "shunt_line_width_mm_2": 0.988,
            "shunt_line_length_mm_0": 31.412,
            "shunt_line_length_mm_1": 30.045,
            "shunt_line_length_mm_2": 31.412,
        }
        optimize_s_params(simulate, x0, output_path)

    sweep = False
    if sweep:
        sweep_vals = {
            "line_length_mm": (18.0, 22.0, 3),
            "series_line_width_mm": (2.0, 4.0, 3),
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
