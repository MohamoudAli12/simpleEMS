from simpleEMS import QuarterWaveFilterParams, BandStopQuarterWaveFilter
from simpleEMS import setup_simulation

params = QuarterWaveFilterParams(
    substrate_eps_r=3.3,
    substrate_tand=0.001,
    substrate_thickness_mm=1.6,
    min_freq=1e9,
    max_freq=2e9,
    centre_freq=1.5e9,
    bandwidth_freq=1e9,
    filter_type="bandstop",
    filter_response="butterworth",
    filter_order=1,
)
CSX, FDTD, freqs = setup_simulation(params)
filter = BandStopQuarterWaveFilter(params, CSX, FDTD)
filter.print_and_save_params(params)
filter.create_substrate()
filter.create_ground()
filter.create_series_line()
filter.create_shunt_line()
ports = filter.create_ports()
filter.create_mesh()
filter.write_and_show_structure(FDTD)
filter.run_simulation(FDTD)
network_params = filter.compute_network_params(ports, freqs, params.charac_imp)
filter.plot_s_param(freqs, network_params.s11, network_params.s21)
filter.show_plots()
