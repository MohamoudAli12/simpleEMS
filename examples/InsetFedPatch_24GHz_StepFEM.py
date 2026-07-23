#!/usr/bin/env python3
"""Inset-fed patch antenna at 24.125 GHz, solved directly from a standalone STEP file.

Unlike ``InsetFedPatch_24GHz_FEM.py``, no CSXCAD geometry is built here — the
mesh is generated straight from an existing ``.step`` CAD file via
``simulate_step_fem()``. It reuses the STEP file exported by
``InsetFedPatch_24GHz.py`` (run that script first to generate
``InsetFedPatch_24GHz/step/structure.step`` if it doesn't exist yet).

Requires the ``getdp`` binary on ``PATH`` or in ``SIMPLEEMS_GETDP_BIN``.
"""

from pathlib import Path

import numpy as np

from simpleEMS import FemOptions, SimTools, simulate_step_fem
from simpleEMS.fem_radiation import FemNF2FF

STEP_FILE = Path(__file__).parent / "InsetFedPatch_24GHz" / "step" / "structure.step"


def simulate(output_path: Path) -> None:
    if not STEP_FILE.exists():
        raise FileNotFoundError(
            f"{STEP_FILE} not found - run examples/InsetFedPatch_24GHz.py first "
            "to generate it."
        )

    resonant_freq = 24.125e9  # Hz
    span_freq = 2e9  # Hz
    freqs = np.linspace(resonant_freq - span_freq, resonant_freq + span_freq, 201)

    sim_data = simulate_step_fem(
        str(STEP_FILE),
        freqs,
        dielectrics={"substrate": (3.48, 0.0037)},  # (eps_r, tan_d)
        pec=["patch_inset", "feed", "ground"],
        ports={"port_resist_1": {"z0": 50.0, "direction": "z", "number": 1}},
        num_solve_points=12,
        fem_options=FemOptions(boundary="silver_muller", port_type="lumped"),
        charac_imp=50.0,
        output_path=output_path,
    )

    SimTools.plot_s_param(sim_data.freqs, sim_data.s11)
    SimTools.plot_smith_chart(sim_data.freqs, sim_data.s11)
    SimTools.plot_vswr(sim_data.freqs, sim_data.vswr)
    SimTools.plot_impedance(sim_data.freqs, sim_data.z11)

    # Radiation post-processing at resonance, reusing the existing SimTools plots.
    nf2ff = FemNF2FF()
    SimTools.plot_2d_directivity(nf2ff, resonant_freq, output_path)
    SimTools.plot_2d_rad_pattern(nf2ff, resonant_freq, output_path)
    nf2ff_3d = SimTools.compute_nf2ff_3d(nf2ff, resonant_freq, output_path)
    SimTools.plot_3d_directivity(nf2ff_3d, resonant_freq, output_path)
    SimTools.plot_3d_gain(nf2ff_3d, resonant_freq, sim_data.input_power, output_path)
    SimTools.plot_3d_power(nf2ff_3d, resonant_freq, output_path)

    SimTools.save_plots(output_path)
    SimTools.show_plots()
    SimTools.export_touchstone(
        freqs=sim_data.freqs,
        s11=sim_data.s11,
        charac_imp=50.0,
        output_path=output_path,
    )


def main() -> None:
    output_path = Path(__file__).with_suffix("")
    output_path.mkdir(parents=True, exist_ok=True)
    simulate(output_path=output_path)


if __name__ == "__main__":
    main()
