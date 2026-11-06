# simpleEMS
# Copyright (C) 2026 Mohamoud Ali
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
PDF design report of a simulated structure.

:func:`generate_report` gathers the design target, the model and the simulated
results of one structure into a printable PDF. The report speaks to the reader
of the design rather than the person running the simulation: it shows the
layout, the dimensions and the performance, and leaves the solver settings out.
"""

from __future__ import annotations

import datetime
import inspect
import re
import textwrap
from dataclasses import fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import EngFormatter, MaxNLocator
from pysmithchart import S_PARAMETER

from . import __version__
from .console import console
from .export_layout import infer_layers

if TYPE_CHECKING:
    from .sim_params import SimParams
    from .sim_tools import NF2FFData, SimData, SimSetup

__all__ = ["generate_report"]

PAGE_SIZE = (8.27, 11.69)  # A4 portrait, in inches
SUBSTRATE_COLOR = "#3f7d4e"
COPPER_COLOR = "#c8893a"
PORT_COLOR = "#d62728"

# Solver settings and derived quantities the reader of a design has no use for
HIDDEN_FIELDS = {
    "unit",
    "num_points",
    "sim_method",
    "simulation_box",
    "substrate_cells",
    "substrate_kappa",
    "lambda0",
    "fp_precision",
    "min_trace_width_mm",
    "min_trace_spacing_mm",
}
READABLE_LABELS = {
    "substrate_eps_r": "Substrate relative permittivity",
    "substrate_tand": "Substrate loss tangent",
    "charac_imp": "Port impedance",
}
SOLVER_NAMES = {
    "FDTD": "Full-wave, time domain (openEMS FDTD)",
    "FEM": "Full-wave, frequency domain (finite elements, GetDP)",
}

freq_formatter = EngFormatter(unit="Hz", places=3)
# Tick labels drop the trailing zeros the tables keep, so they fit side by side
freq_axis_formatter = EngFormatter(unit="Hz")


def generate_report(
    params: SimParams,
    sim: SimSetup,
    sim_data: SimData,
    nf2ff_data: NF2FFData | None = None,
    output_path: Path | None = None,
    title: str | None = None,
    author: str | None = None,
    filename: str = "design_report",
) -> Path:
    """
    Write a PDF design report of a simulated structure.

    The report opens with the design target and the key results, then shows
    the model -- the copper layout with its ports and the design dimensions --
    and the simulated performance: matching, transmission, input impedance
    and, for an antenna, the radiation pattern and efficiency. Each page ends
    with a short note on how to read it.

    The key results follow the kind of structure. A one-port structure, or
    one passed with ``nf2ff_data``, reads as an antenna: match and -10 dB
    bandwidth. A params object with ``filter_type`` reads as a band-pass or
    band-stop filter: insertion loss, rejection and the 3 dB passband or
    stopband. A four-port result reads as a coupler: through, coupled and
    isolated ports, amplitude and phase balance. Any other two-port reads as
    a transmission line: insertion and return loss.

    Parameters
    ----------
    params : SimParams
        Parameter object of the structure; supplies the dimensions, the
        substrate and the design frequency ``main_freq``.
    sim : SimSetup
        Simulation setup the structure was built in; supplies the geometry
        drawn on the model page.
    sim_data : SimData
        Network results returned by ``compute_sim_data``.
    nf2ff_data : NF2FFData, optional
        Far-field results returned by ``compute_nf2ff_data``. Adds the
        radiation page and the antenna figures of the summary. Default is
        ``None``.
    output_path : Path, optional
        Directory to create the ``report`` folder under; the PDF is written
        into that folder. Defaults to ``cwd / "Sim_Path"``.
    title : str, optional
        Report title. Defaults to ``"<Structure> design report"``, with the
        structure named after the parameter class.
    author : str, optional
        Name printed on the summary page and stored in the PDF metadata.
    filename : str, optional
        Base name of the PDF file. Default is ``"design_report"``.

    Returns
    -------
    Path
        Path of the written PDF.
    """
    if output_path is None:
        output_path = Path.cwd() / "Sim_Path"

    report_folder = output_path / "report"
    report_folder.mkdir(parents=True, exist_ok=True)
    report_path = report_folder / f"{filename}.pdf"

    class_name = type(params).__name__.removesuffix("Params")
    structure_name = re.sub(r"(?<!^)(?=[A-Z])", " ", class_name)
    report_title = title or f"{structure_name} design report"

    # ---- Key figures --------------------------------------------------------
    freqs = sim_data.freqs
    main_freq = params.main_freq
    target = freq_formatter(main_freq)
    main_index = int(np.argmin(np.abs(freqs - main_freq)))
    with np.errstate(divide="ignore"):
        s11_db = 20.0 * np.log10(np.abs(sim_data.s11))

    transmissions = []
    for name, values in (
        ("S21", sim_data.s21),
        ("S31", sim_data.s31),
        ("S41", sim_data.s41),
    ):
        if values is None:
            continue
        with np.errstate(divide="ignore"):
            values_db = 20.0 * np.log10(np.abs(values))
        transmissions.append((name, values, values_db))

    # Judge each kind of structure by the figures its designer works to: an
    # antenna by its match, a filter by its passband or stopband, a coupler by
    # its split and isolation, and a line by its loss
    filter_type = getattr(params, "filter_type", None)
    if nf2ff_data is not None or not transmissions:
        structure_kind = "antenna"
    elif filter_type in ("bandpass", "bandstop"):
        structure_kind = filter_type
    elif sim_data.s41 is not None:
        structure_kind = "coupler"
    else:
        structure_kind = "transmission"

    if structure_kind == "bandpass":
        s21_db = transmissions[0][2]
        anchor_index = int(np.argmax(s21_db))
        band_threshold_db = s21_db[anchor_index] - 3.0
        in_band = s21_db >= band_threshold_db
        band_name = "Passband (S21 within 3 dB of its peak)"
        band_legend = "Passband"
        threshold_label = "3 dB below the S21 peak"
    elif structure_kind == "bandstop":
        s21_db = transmissions[0][2]
        anchor_index = int(np.argmin(s21_db))
        band_threshold_db = -3.0
        in_band = s21_db <= band_threshold_db
        band_name = "Stopband (S21 below -3 dB)"
        band_legend = "Stopband"
        threshold_label = "-3 dB"
    elif structure_kind == "coupler":
        s41_db = transmissions[2][2]
        anchor_index = main_index
        band_threshold_db = -15.0
        in_band = (s11_db <= band_threshold_db) & (s41_db <= band_threshold_db)
        band_name = "Bandwidth (S11 and S41 below -15 dB)"
        band_legend = "Operating band"
        threshold_label = "-15 dB"
    else:
        anchor_index = int(np.argmin(s11_db))
        band_threshold_db = -10.0
        in_band = s11_db <= band_threshold_db
        band_name = "Bandwidth (S11 below -10 dB)"
        band_legend = "Matched band"
        threshold_label = "-10 dB"

    # Walk out from the anchor point for as long as the band criterion holds
    lower_index = upper_index = anchor_index
    while lower_index > 0 and in_band[lower_index - 1]:
        lower_index -= 1
    while upper_index < len(freqs) - 1 and in_band[upper_index + 1]:
        upper_index += 1
    has_band = bool(in_band[anchor_index])
    if has_band:
        bandwidth = freqs[upper_index] - freqs[lower_index]
        band_centre = (freqs[upper_index] + freqs[lower_index]) / 2
        bandwidth_text = (
            f"{freq_formatter(freqs[lower_index])} to "
            f"{freq_formatter(freqs[upper_index])} ({freq_formatter(bandwidth)}, "
            f"{100.0 * bandwidth / band_centre:.1f} %)"
        )
        if lower_index == 0 or upper_index == len(freqs) - 1:
            bandwidth_text += ", runs past the simulated range"
    else:
        bandwidth_text = "not reached in the simulated range"

    impedance = sim_data.z11[main_index]
    sign = "+" if impedance.imag >= 0 else "-"
    impedance_text = f"{impedance.real:.1f} {sign} j{abs(impedance.imag):.1f} Ω"
    designed_bandwidth = getattr(params, "bandwidth_freq", None)

    if structure_kind == "antenna":
        result_rows = [
            [
                "Best match",
                f"{s11_db[anchor_index]:.1f} dB at "
                f"{freq_formatter(freqs[anchor_index])}",
            ],
            [band_name, bandwidth_text],
            [f"Reflection S11 at {target}", f"{s11_db[main_index]:.1f} dB"],
            [f"VSWR at {target}", f"{sim_data.vswr[main_index]:.2f}"],
            [f"Input impedance at {target}", impedance_text],
        ]
    elif structure_kind == "bandpass":
        result_rows = [
            [f"Insertion loss at {target}", f"{-s21_db[main_index]:.2f} dB"],
            [
                "Lowest insertion loss",
                f"{-s21_db[anchor_index]:.2f} dB at "
                f"{freq_formatter(freqs[anchor_index])}",
            ],
            [band_name, bandwidth_text],
        ]
        if designed_bandwidth is not None:
            result_rows.append(
                ["Designed bandwidth", freq_formatter(designed_bandwidth)]
            )
        result_rows += [
            [f"Return loss at {target}", f"{-s11_db[main_index]:.1f} dB"],
            [
                f"Rejection at {freq_formatter(freqs[0])} / "
                f"{freq_formatter(freqs[-1])}",
                f"{-s21_db[0]:.1f} dB / {-s21_db[-1]:.1f} dB",
            ],
        ]
    elif structure_kind == "bandstop":
        result_rows = [
            [f"Rejection at {target}", f"{-s21_db[main_index]:.1f} dB"],
            [
                "Deepest rejection",
                f"{-s21_db[anchor_index]:.1f} dB at "
                f"{freq_formatter(freqs[anchor_index])}",
            ],
            [band_name, bandwidth_text],
        ]
        if designed_bandwidth is not None:
            result_rows.append(
                ["Designed bandwidth", freq_formatter(designed_bandwidth)]
            )
        result_rows.append(
            [
                f"Insertion loss at {freq_formatter(freqs[0])} / "
                f"{freq_formatter(freqs[-1])}",
                f"{-s21_db[0]:.2f} dB / {-s21_db[-1]:.2f} dB",
            ]
        )
    elif structure_kind == "coupler":
        through, coupled, _isolated = (values for _name, values, _db in transmissions)
        through_db, coupled_db = transmissions[0][2], transmissions[1][2]
        result_rows = [
            [f"Through (S21) at {target}", f"{through_db[main_index]:.2f} dB"],
            [f"Coupled (S31) at {target}", f"{coupled_db[main_index]:.2f} dB"],
            [f"Isolation (S41) at {target}", f"{-s41_db[main_index]:.1f} dB"],
            [f"Return loss at {target}", f"{-s11_db[main_index]:.1f} dB"],
            [
                f"Amplitude imbalance at {target}",
                f"{through_db[main_index] - coupled_db[main_index]:.2f} dB",
            ],
            [
                f"Phase difference (S31 - S21) at {target}",
                f"{np.angle(coupled[main_index] / through[main_index], deg=True):.1f}°",
            ],
            [band_name, bandwidth_text],
        ]
    else:
        s21_db = transmissions[0][2]
        worst_transmission_index = int(np.argmin(s21_db))
        worst_match_index = int(np.argmax(s11_db))
        result_rows = [
            [f"Insertion loss at {target}", f"{-s21_db[main_index]:.2f} dB"],
            [f"Return loss at {target}", f"{-s11_db[main_index]:.1f} dB"],
            [
                "Highest insertion loss",
                f"{-s21_db[worst_transmission_index]:.2f} dB at "
                f"{freq_formatter(freqs[worst_transmission_index])}",
            ],
            [
                "Lowest return loss",
                f"{-s11_db[worst_match_index]:.1f} dB at "
                f"{freq_formatter(freqs[worst_match_index])}",
            ],
            [band_name, bandwidth_text],
        ]
        result_rows.append(
            [
                f"Transmission phase (S21) at {target}",
                f"{np.angle(sim_data.s21[main_index], deg=True):.1f}°",
            ]
        )
        for name, values, values_db in transmissions[1:]:
            result_rows.append(
                [
                    f"Transmission to port {name[1]} ({name}) at {target}",
                    f"{values_db[main_index]:.2f} dB, "
                    f"phase {np.angle(values[main_index], deg=True):.1f}°",
                ]
            )
        result_rows += [
            [f"VSWR at {target}", f"{sim_data.vswr[main_index]:.2f}"],
            [f"Input impedance at {target}", impedance_text],
        ]

    if nf2ff_data is not None:
        sphere = nf2ff_data.sphere
        far_field_freq = freq_formatter(nf2ff_data.freq)
        total_efficiency = nf2ff_data.efficiency * nf2ff_data.mismatch_efficiency
        sphere_field = np.squeeze(sphere.E_norm)
        theta_index, phi_index = np.unravel_index(
            np.argmax(sphere_field), sphere_field.shape
        )
        result_rows += [
            [
                f"Peak directivity at {far_field_freq}",
                f"{10.0 * np.log10(sphere.Dmax[0]):.2f} dBi",
            ],
            [
                f"Peak realized gain at {far_field_freq}",
                f"{10.0 * np.log10(total_efficiency * sphere.Dmax[0]):.2f} dBi",
            ],
            [
                "Main beam direction",
                f"θ = {np.rad2deg(sphere.theta[theta_index]):.0f}°, "
                f"φ = {np.rad2deg(sphere.phi[phi_index]):.0f}°",
            ],
            ["Radiation efficiency", f"{100.0 * nf2ff_data.efficiency:.1f} %"],
            ["Total efficiency (incl. mismatch)", f"{100.0 * total_efficiency:.1f} %"],
        ]

    layout = infer_layers(sim.CSX)
    board_size_text = "n/a"
    if layout.outline is not None:
        x_min, y_min, x_max, y_max = layout.outline
        board_size_text = f"{x_max - x_min:.2f} mm × {y_max - y_min:.2f} mm"

    target_rows = [
        ["Structure", structure_name],
        ["Design frequency", target],
        [
            "Simulated range",
            f"{freq_formatter(freqs[0])} to {freq_formatter(freqs[-1])}",
        ],
        ["Port impedance", f"{sim_data.ref_impedance:.4g} Ω"],
        [
            "Substrate",
            f"εr = {params.substrate_eps_r:g}, tan δ = {params.substrate_tand:g}, "
            f"{params.substrate_thickness_mm:g} mm thick",
        ],
        ["Copper thickness", f"{params.copper_thickness_mm * 1000.0:g} µm"],
        ["Board size", board_size_text],
        ["Simulation method", SOLVER_NAMES.get(sim.sim_method, sim.sim_method)],
    ]

    # Design dimensions: the dataclass fields plus any *_mm property a structure
    # derives, e.g. a substrate size computed from the patch
    parameter_names = [field.name for field in fields(params)]
    parameter_names += [
        name
        for name, _member in inspect.getmembers(
            type(params),
            lambda member: isinstance(member, property | cached_property),
        )
        if name.endswith("_mm") and name not in parameter_names
    ]
    parameter_rows = []
    for name in parameter_names:
        if name in HIDDEN_FIELDS or name.startswith(("FDTD_", "FEM_")):
            continue
        value = getattr(params, name)
        label = READABLE_LABELS.get(
            name, name.removesuffix("_mm").replace("_", " ").capitalize()
        )
        if isinstance(value, bool) or not isinstance(value, int | float):
            value_text = str(value)
        elif name.endswith("_mm"):
            value_text = f"{value:.{params.fp_precision}f} mm"
        elif "freq" in name:
            value_text = freq_formatter(value)
        elif name == "charac_imp":
            value_text = f"{value:g} Ω"
        else:
            value_text = f"{value:.4g}"
        parameter_rows.append([label, value_text])

    pages: list[tuple[Figure, str, str]] = []
    with plt.style.context("default"):
        # ---- Summary --------------------------------------------------------
        summary_figure = Figure(figsize=PAGE_SIZE)
        # A long title wraps upwards, clear of the line beneath it
        summary_figure.text(
            0.06,
            0.87,
            textwrap.fill(report_title, width=38),
            fontsize=22,
            weight="bold",
            va="bottom",
        )
        summary_figure.text(
            0.06,
            0.85,
            " · ".join(
                part
                for part in (
                    datetime.date.today().isoformat(),
                    author,
                    f"simpleEMS {__version__}",
                )
                if part
            ),
            fontsize=10,
            color="0.4",
        )
        target_axes = summary_figure.add_axes((0.06, 0.6, 0.88, 0.22))
        results_axes = summary_figure.add_axes((0.06, 0.1, 0.88, 0.46))
        tables = []
        for axes, heading, rows in (
            (target_axes, "Design target", target_rows),
            (results_axes, "Key results", result_rows),
        ):
            axes.axis("off")
            tables.append(
                axes.table(
                    cellText=rows,
                    colLabels=[heading, ""],
                    colWidths=[0.45, 0.55],
                    cellLoc="left",
                    loc="upper left",
                )
            )
        pages.append(
            (
                summary_figure,
                "Summary",
                "The results come from a full-wave electromagnetic simulation of "
                "the model shown on the next page. The design frequency is the one "
                "the structure was dimensioned for; the best match can sit slightly "
                "off it.",
            )
        )

        # ---- Model ----------------------------------------------------------
        model_figure = Figure(figsize=PAGE_SIZE, layout="constrained")
        layer_count = max(len(layout.layers), 1)
        model_grid = model_figure.add_gridspec(2, layer_count, height_ratios=[1, 1])
        port_centres = [
            (
                (primitive.GetStart()[0] + primitive.GetStop()[0]) / 2,
                (primitive.GetStart()[1] + primitive.GetStop()[1]) / 2,
            )
            for prop in sim.CSX.GetAllProperties()
            if prop.__class__.__name__ == "CSPropLumpedElement"
            for primitive in prop.GetAllPrimitives()
        ]
        for column, layer in enumerate(layout.layers):
            layer_axes = model_figure.add_subplot(model_grid[0, column])
            if layout.outline is not None:
                x_min, y_min, x_max, y_max = layout.outline
                layer_axes.add_patch(
                    Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        facecolor=SUBSTRATE_COLOR,
                        edgecolor="black",
                        linewidth=0.8,
                    )
                )
            for _name, primitive in layer.regions:
                kind = primitive.__class__.__name__
                if kind == "CSPrimBox":
                    start, stop = primitive.GetStart(), primitive.GetStop()
                    corners = [
                        [start[0], start[1], start[2]],
                        [stop[0], start[1], start[2]],
                        [stop[0], stop[1], start[2]],
                        [start[0], stop[1], start[2]],
                    ]
                elif kind in ("CSPrimPolygon", "CSPrimLinPoly"):
                    x_coords, y_coords = primitive.GetCoords()
                    elevation = primitive.GetElevation()
                    corners = [
                        [x, y, elevation]
                        for x, y in zip(x_coords, y_coords, strict=True)
                    ]
                else:
                    continue
                if primitive.HasTransform():
                    transform = primitive.GetTransform()
                    corners = [list(transform.Transform(corner)) for corner in corners]
                layer_axes.add_patch(
                    Polygon(
                        [corner[:2] for corner in corners],
                        closed=True,
                        facecolor=COPPER_COLOR,
                        edgecolor="#7a4f1c",
                        linewidth=0.5,
                    )
                )
            for drill in layout.drills:
                layer_axes.add_patch(
                    Circle(
                        (drill.x, drill.y),
                        drill.diameter / 2,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.5,
                    )
                )
            if column == 0:
                for port_number, (x_centre, y_centre) in enumerate(port_centres, 1):
                    layer_axes.plot(x_centre, y_centre, "o", color=PORT_COLOR)
                    layer_axes.annotate(
                        f"Port {port_number}",
                        (x_centre, y_centre),
                        xytext=(6, 6),
                        textcoords="offset points",
                        color=PORT_COLOR,
                        weight="bold",
                    )
            if layer.index == 1:
                layer_title = "Top copper"
            elif layer.index == layer.count:
                layer_title = "Bottom copper"
            else:
                layer_title = f"Inner copper {layer.index - 1}"
            layer_axes.set_title(layer_title)
            layer_axes.set_aspect("equal")
            layer_axes.autoscale_view()
            layer_axes.set_xlabel("x (mm)")
            layer_axes.set_ylabel("y (mm)")
        parameter_axes = model_figure.add_subplot(model_grid[1, :])
        parameter_axes.axis("off")
        tables.append(
            parameter_axes.table(
                cellText=parameter_rows,
                colLabels=["Design parameter", "Value"],
                colWidths=[0.6, 0.4],
                cellLoc="left",
                loc="upper center",
            )
        )
        pages.append(
            (
                model_figure,
                "Model",
                "Top view of each copper layer on the board (green). The red "
                "markers show where the ports connect. All dimensions are in "
                "millimetres.",
            )
        )

        # ---- Matching and transmission --------------------------------------
        response_figure = Figure(figsize=PAGE_SIZE, layout="constrained")
        response_axes = response_figure.subplots(
            3 if transmissions else 2, 1, sharex=True, squeeze=False
        )[:, 0]
        if structure_kind == "coupler":
            port_roles = {"S21": "through", "S31": "coupled", "S41": "isolated"}
        else:
            port_roles = {}
        response_axes[0].plot(freqs, s11_db, label="S11 (reflection)")
        for name, _values, values_db in transmissions:
            response_axes[0].plot(
                freqs,
                values_db,
                label=f"{name} ({port_roles.get(name, 'transmission')})",
            )
        response_axes[0].axhline(
            band_threshold_db, linestyle=":", color="0.4", label=threshold_label
        )
        if has_band:
            response_axes[0].axvspan(
                freqs[lower_index],
                freqs[upper_index],
                color="C2",
                alpha=0.12,
                label=band_legend,
            )
        response_axes[0].set_ylabel("Magnitude (dB)")
        response_axes[0].set_title("S-parameters")
        response_axes[1].plot(freqs, sim_data.vswr, label="VSWR")
        response_axes[1].axhline(2.0, linestyle=":", color="0.4", label="VSWR = 2")
        response_axes[1].set_ylim(
            1.0, max(2.5, min(10.0, float(np.max(sim_data.vswr))))
        )
        response_axes[1].set_ylabel("VSWR")
        response_axes[1].set_title("Voltage standing wave ratio")
        if transmissions:
            for name, values, _values_db in transmissions:
                response_axes[2].plot(freqs, np.angle(values, deg=True), label=name)
            if structure_kind == "coupler":
                response_axes[2].plot(
                    freqs,
                    np.angle(sim_data.s31 / sim_data.s21, deg=True),
                    color="black",
                    label="S31 - S21 (quadrature)",
                )
            response_axes[2].set_ylabel("Phase (°)")
            response_axes[2].set_title("Transmission phase")
        for axes in response_axes:
            axes.axvline(
                main_freq, linestyle="--", color="0.3", label="Design frequency"
            )
            axes.xaxis.set_major_formatter(freq_axis_formatter)
            axes.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axes.grid(True)
            axes.legend(fontsize=8)
        response_axes[-1].set_xlabel("Frequency")
        if structure_kind == "bandpass":
            response_caption = (
                "S21 is the share of the signal that passes from port 1 to port 2. "
                "The passband is where S21 stays within 3 dB of its peak; outside "
                "it, the lower S21, the stronger the rejection. S11 shows how much "
                "of the signal reflects back at port 1: inside the passband it "
                "should sit low."
            )
        elif structure_kind == "bandstop":
            response_caption = (
                "S21 is the share of the signal that passes from port 1 to port 2. "
                "The stopband is where S21 falls below -3 dB; the deeper the dip, "
                "the stronger the rejection. Outside the stopband S21 should sit "
                "close to 0 dB and S11 low."
            )
        elif structure_kind == "coupler":
            response_caption = (
                "Port 1 feeds the coupler. An ideal 3 dB quadrature hybrid splits "
                "the power equally between the through (S21) and coupled (S31) "
                "ports, about -3 dB each, 90° apart, while the isolated port (S41) "
                "and the reflection (S11) stay low. The shaded band is where both "
                "S11 and S41 stay below -15 dB."
            )
        elif structure_kind == "transmission":
            response_caption = (
                "S21 shows how much of the signal passes from port 1 to the other "
                "port: the closer to 0 dB, the lower the loss. S11 is the share "
                "reflected back at port 1: below -10 dB (VSWR below 2) less than "
                "10 % of the power reflects. The transmission phase gives the "
                "electrical length of the structure."
            )
        else:
            response_caption = (
                "S11 is the share of the signal reflected back at port 1: the "
                "lower, the better the match. Below -10 dB (VSWR below 2) less "
                "than 10 % of the power is reflected, and the antenna accepts the "
                "rest."
            )
        pages.append(
            (
                response_figure,
                "Matching and transmission",
                response_caption + " The dashed line marks the design frequency.",
            )
        )

        # ---- Input impedance ------------------------------------------------
        impedance_figure = Figure(figsize=PAGE_SIZE, layout="constrained")
        impedance_grid = impedance_figure.add_gridspec(2, 1, height_ratios=[1, 1.4])
        impedance_axes = impedance_figure.add_subplot(impedance_grid[0])
        impedance_axes.plot(freqs, sim_data.z11.real, label="Resistance (real part)")
        impedance_axes.plot(
            freqs, sim_data.z11.imag, label="Reactance (imaginary part)"
        )
        impedance_axes.axhline(
            sim_data.ref_impedance,
            linestyle=":",
            color="0.4",
            label=f"Port impedance {sim_data.ref_impedance:.4g} Ω",
        )
        impedance_axes.axvline(main_freq, linestyle="--", color="0.3")
        impedance_axes.xaxis.set_major_formatter(freq_axis_formatter)
        impedance_axes.xaxis.set_major_locator(MaxNLocator(nbins=5))
        impedance_axes.set_xlabel("Frequency")
        impedance_axes.set_ylabel("Impedance (Ω)")
        impedance_axes.set_title("Input impedance at port 1")
        impedance_axes.grid(True)
        impedance_axes.legend(fontsize=8)
        smith_axes = impedance_figure.add_subplot(
            impedance_grid[1],
            projection="smith",
            axes_impedance=sim_data.ref_impedance,
            axes_normalize=True,
        )
        smith_axes.plot(sim_data.s11, datatype=S_PARAMETER, marker="", label="S11")
        smith_axes.plot(
            sim_data.s11[main_index],
            datatype=S_PARAMETER,
            marker="o",
            linestyle="",
            label=f"At {target}",
        )
        smith_axes.set_title("Smith chart")
        smith_axes.legend(loc="lower right", fontsize=8)
        pages.append(
            (
                impedance_figure,
                "Input impedance",
                "A well-matched port has a resistance close to the port impedance "
                "(dotted line) and a reactance close to zero. On the Smith chart, "
                "the closer the curve passes to the centre, the better the match.",
            )
        )

        # ---- Radiation ------------------------------------------------------
        if nf2ff_data is not None:
            radiation_figure = Figure(figsize=PAGE_SIZE, layout="constrained")
            radiation_grid = radiation_figure.add_gridspec(2, 2)
            peak_dbi = 10.0 * np.log10(sphere.Dmax[0])
            floor_dbi = peak_dbi - 30.0
            for position, (cut, angles, plane) in enumerate(
                (
                    (nf2ff_data.xz_cut, nf2ff_data.xz_cut.theta, "xz-plane (φ = 0°)"),
                    (nf2ff_data.yz_cut, nf2ff_data.yz_cut.theta, "yz-plane (φ = 90°)"),
                    (nf2ff_data.xy_cut, nf2ff_data.xy_cut.phi, "xy-plane (θ = 90°)"),
                )
            ):
                # Reference every cut to the sphere's peak, as plot_2d_directivity does
                with np.errstate(divide="ignore"):
                    pattern_dbi = peak_dbi + 20.0 * np.log10(
                        np.squeeze(cut.E_norm) / np.max(sphere.E_norm)
                    )
                polar_axes = radiation_figure.add_subplot(
                    radiation_grid[divmod(position, 2)], projection="polar"
                )
                polar_axes.plot(
                    np.ravel(angles), np.maximum(pattern_dbi, floor_dbi), color="C0"
                )
                polar_axes.set_theta_zero_location("N")
                polar_axes.set_theta_direction(-1)
                polar_axes.set_ylim(floor_dbi, peak_dbi + 1.0)
                polar_axes.set_title(f"Directivity (dBi), {plane}", fontsize=10)
            efficiency_axes = radiation_figure.add_subplot(radiation_grid[1, 1])
            curve = nf2ff_data.efficiency_curve
            if len(curve.freqs):
                efficiency_axes.plot(
                    curve.freqs, 100.0 * curve.radiation, label="Radiation efficiency"
                )
                efficiency_axes.plot(
                    curve.freqs, 100.0 * curve.total, label="Total efficiency"
                )
                efficiency_axes.xaxis.set_major_formatter(freq_axis_formatter)
                # The panel is half the page wide, so it fits fewer labels
                efficiency_axes.xaxis.set_major_locator(MaxNLocator(nbins=3))
                efficiency_axes.legend(fontsize=8)
            else:
                efficiency_axes.text(
                    0.5,
                    0.5,
                    "No -10 dB band to report",
                    ha="center",
                    transform=efficiency_axes.transAxes,
                )
            efficiency_axes.set_ylim(0.0, 100.0)
            efficiency_axes.set_xlabel("Frequency")
            efficiency_axes.set_ylabel("Efficiency (%)")
            efficiency_axes.set_title("Efficiency", fontsize=10)
            efficiency_axes.grid(True)
            pages.append(
                (
                    radiation_figure,
                    f"Radiation at {far_field_freq}",
                    "Directivity is how strongly the antenna concentrates its "
                    "radiation in each direction, in dBi (relative to an ideal "
                    "radiator that sends equally in all directions). Radiation "
                    "efficiency is the share of the accepted power that is "
                    "radiated; total efficiency also counts the power reflected "
                    "at the port.",
                )
            )

        table_font_size = 9
        for table in tables:
            table.auto_set_font_size(False)
            table.set_fontsize(table_font_size)
            table.scale(1.0, 1.5)
            # A table does not wrap its text, so wrap each cell to its column
            # and make its row tall enough for the lines; a long value would
            # otherwise run off the page. The layout has to settle first, for
            # the width of the table's axes to be known.
            table_figure = table.get_figure()
            table_figure.draw_without_rendering()
            axes_width_points = (
                table.axes.get_position().width * table_figure.get_figwidth() * 72
            )
            row_line_counts: dict[int, int] = {}
            for (row, _column), cell in table.get_celld().items():
                row_line_counts.setdefault(row, 1)
                text = cell.get_text().get_text()
                if not text.strip():
                    continue
                # The text starts a tenth of the cell width in from its left edge.
                room_points = 0.85 * cell.get_width() * axes_width_points
                text_points = (
                    TextPath((0, 0), text, size=table_font_size).get_extents().width
                )
                characters_per_line = int(len(text) * room_points / text_points)
                wrapped_text = textwrap.fill(text, width=max(characters_per_line, 1))
                cell.get_text().set_text(wrapped_text)
                row_line_counts[row] = max(
                    row_line_counts[row], wrapped_text.count("\n") + 1
                )
            for (row, _column), cell in table.get_celld().items():
                cell.set_height(cell.get_height() * row_line_counts[row])
                cell.set_edgecolor("0.85")
                if row == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("0.93")

        metadata = {"Title": report_title, "Creator": f"simpleEMS {__version__}"}
        if author:
            metadata["Author"] = author
        with PdfPages(report_path, metadata=metadata) as pdf:
            for page_number, (figure, heading, caption) in enumerate(pages, 1):
                if figure.get_layout_engine() is not None:
                    # leave room for the header and the caption
                    figure.get_layout_engine().set(rect=(0.04, 0.1, 0.92, 0.83))
                figure.text(0.06, 0.955, heading, fontsize=16, weight="bold")
                figure.text(
                    0.94, 0.96, report_title, ha="right", fontsize=8, color="0.4"
                )
                figure.text(
                    0.06,
                    0.035,
                    textwrap.fill(caption, width=110),
                    fontsize=9,
                    color="0.25",
                )
                figure.text(
                    0.94,
                    0.015,
                    f"Page {page_number} of {len(pages)}",
                    ha="right",
                    fontsize=8,
                    color="0.4",
                )
                pdf.savefig(figure)

    console.rule(f"[info]{len(pages)}-page report written to {report_path}[/info]")
    return report_path
