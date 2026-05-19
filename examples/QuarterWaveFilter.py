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
    filter_order=3,
)
CSX, FDTD, freqs = setup_simulation(params)
filter = BandStopQuarterWaveFilter(params, CSX, FDTD)
filter.print_and_save_params(params)
filter.create_substrate()
filter.create_ground()
filter.create_series_line()
filter.create_shunt_line()
filter.write_and_show_structure(FDTD)
