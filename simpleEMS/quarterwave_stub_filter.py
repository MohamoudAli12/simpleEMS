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
Quarter-wave stub filter design and simulation.

Provides parameter and geometry classes for band-pass and band-stop
quarter-wave stub filters using Bessel, Butterworth, or Chebyshev
responses. Includes `QuarterWaveFilterParams` for parameter
definition and `BandStopQuarterWaveFilter` and
`BandPassQuarterWaveFilter` for structure construction.
"""

import math
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from openEMS.ports import LumpedPort

from simpleEMS.sim_tools import SimTools, SimSetup
from .sim_params import SimParams
from .mesh import Mesh
from .calc import microstrip_width_from_impedance, calculate_electrical_length_mm
from .filter_coefficient import get_filter_coefficient


# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "QuarterWaveFilterParams",
    "BandStopQuarterWaveFilter",
    "BandPassQuarterWaveFilter",
]


def _get_total_length(
    filter_order: int, line_length_mm: float, shunt_line_width_mm: list[float]
) -> float:
    """Compute total physical length of the filter structure along the x-axis.

    Parameters
    ----------
    filter_order : int
        Order of the filter (number of shunt stubs).
    line_length_mm : float
        Length of each series line section in mm.
    shunt_line_width_mm : list of float
        Widths of each shunt stub in mm.

    Returns
    -------
    float
        Total physical length in mm.
    """
    total_length = (filter_order + 1) * line_length_mm
    for i in range(filter_order):
        total_length += shunt_line_width_mm[i]
    return total_length


@dataclass
class QuarterWaveFilterParams(SimParams):
    """
    Parameters for a quarter-wave stub band-pass or band-stop filter.

    Extends `SimParams` and computes the derived geometric parameters
    (series line, shunt stub widths and lengths) required to build and
    simulate a quarter-wave stub filter, using Bessel, Butterworth, or
    Chebyshev prototype coefficients to size each shunt stub.

    Parameters
    ----------
    min_freq : float
        Minimum simulation frequency in Hz.
    max_freq : float
        Maximum simulation frequency in Hz.
    centre_freq : float
        Centre frequency of the filter in Hz.
    bandwidth_freq : float
        Bandwidth in Hz.
    filter_type : str
        Filter type: ``"bandpass"`` or ``"bandstop"``.
    filter_response : str
        Filter response prototype: ``"bessel"``, ``"butterworth"``, or ``"chebyshev"``.
    filter_order : int
        Order of the filter (number of shunt stubs).
    ripple_db : float, optional
        Passband ripple in dB, used only for the Chebyshev response.
        Default is None, which is treated as 0.1 dB.
    elec_length_deg : int, optional
        Electrical length, in degrees, of the series line section and of
        each shunt stub. Default is 90 (quarter-wave).

    Attributes
    ----------
    frac_bandwidth : float
        Fractional bandwidth (bandwidth_freq / centre_freq).
    series_line_width_mm : float
        Width of the series transmission line in mm.
    shunt_line_width_mm : list of float
        Widths of each shunt stub in mm.
    shunt_line_length_mm : list of float
        Lengths of each shunt stub in mm.
    line_length_mm : float
        Physical length of each series line section in mm.

    Raises
    ------
    ValueError
        If `filter_type` is not ``"bandpass"`` or ``"bandstop"``; if the
        resulting fractional bandwidth is not in (0, 1]; if the series line
        or any shunt stub width falls below `min_trace_width_mm`; if
        adjacent shunt stubs would overlap; or if a required characteristic
        impedance cannot be realized on this substrate.

    Notes
    -----
    As a subclass of `SimParams`, this class also accepts all of
    `SimParams`'s keyword-only constructor arguments (e.g.
    `substrate_eps_r`, `substrate_tand`, `substrate_thickness_mm`,
    `charac_imp`); see that class's docstring for details.
    """

    min_freq: float
    max_freq: float
    centre_freq: float
    bandwidth_freq: float
    filter_type: str  # "bandpass" or "bandstop"
    filter_response: str  # "bessel", "butterworth", or "chebyshev"
    filter_order: int
    ripple_db: float | None = None
    elec_length_deg: int = 90
    frac_bandwidth: float = field(init=False)
    series_line_width_mm: float = field(init=False)
    shunt_line_width_mm: list[float] = field(init=False)
    shunt_line_length_mm: list[float] = field(init=False)
    line_length_mm: float = field(init=False)

    @property
    def freq_range(self) -> tuple[float, float]:
        """
        Return the simulation frequency range.

        Returns
        -------
        tuple of (float, float)
            A tuple containing (min_freq, max_freq).
        """
        return self.min_freq, self.max_freq

    @property
    def main_freq(self) -> float:
        """
        Return the primary frequency of interest for analysis.

        Returns
        -------
        float
            The centre frequency of the filter.
        """
        return self.centre_freq

    @property
    def substrate_length_mm(self) -> float:
        """
        Return the total substrate length based on filter geometry.

        Returns
        -------
        float
            Substrate length in mm.
        """
        return _get_total_length(
            self.filter_order,
            self.line_length_mm,
            self.shunt_line_width_mm,
        )

    @property
    def substrate_width_mm(self) -> float:
        """
        Return the substrate width based on filter geometry.

        Computed as the series line width plus one line-section length,
        which approximates the space needed for the series line and a
        shunt stub extending alongside it.

        Returns
        -------
        float
            Substrate width in mm.
        """
        return self.series_line_width_mm + self.line_length_mm

    @property
    def simulation_box(self) -> NDArray:
        """
        Return the 3D simulation bounding box dimensions.

        Returns
        -------
        NDArray
            Array of shape (3,) with [x, y, z] dimensions in mm.
        """
        return self._create_simulation_box(
            self.substrate_length_mm,
            self.substrate_width_mm,
            self.lambda0,
        )

    def __post_init__(self) -> None:
        """
        Validate the filter type and bandwidth, then compute filter geometry.

        Raises
        ------
        ValueError
            If `filter_type` is not ``"bandpass"`` or ``"bandstop"``, or if
            the fractional bandwidth (`bandwidth_freq` / `centre_freq`) is
            not in (0, 1].
        """
        super().__post_init__()
        if self.filter_type not in ("bandpass", "bandstop"):
            raise ValueError(f"Quarter-wave filter does not support {self.filter_type}")

        self.frac_bandwidth = self.bandwidth_freq / self.centre_freq

        if not 0 < self.frac_bandwidth <= 1:
            raise ValueError(
                f"Fractional bandwidth must be in (0, 1], got {self.frac_bandwidth}"
            )

        self._compute_geometry()

    def _get_shunt_width_and_length(self, idx: int) -> tuple[float, float]:
        """
        Compute the microstrip width and quarter-wave length of one shunt stub.

        Looks up the filter prototype coefficient for element `idx`, derives
        the required shunt stub impedance from it (using a different formula
        for band-pass vs. band-stop), then converts that impedance to a
        microstrip trace width and computes the trace length needed for the
        electrical length given by `elec_length_deg`.

        Parameters
        ----------
        idx : int
            Index of the shunt stub (0-based).

        Returns
        -------
        tuple[float, float]
            ``(width_shunt_mm, length_shunt_mm)`` -- the shunt stub's
            microstrip width and length, both in mm.

        Raises
        ------
        ValueError
            If the computed stub width is below `min_trace_width_mm`, or if
            the required shunt impedance cannot be realized on this
            substrate.
        """
        g = get_filter_coefficient(
            idx, self.filter_response, self.filter_order, self.ripple_db
        )
        shunt_imped = 0.0
        if self.filter_type == "bandpass":
            shunt_imped = (math.pi * self.charac_imp * self.frac_bandwidth) / (4 * g)
        if self.filter_type == "bandstop":
            shunt_imped = (4 * self.charac_imp) / (math.pi * self.frac_bandwidth * g)
        width_shunt_mm, er_eff_shunt = microstrip_width_from_impedance(
            shunt_imped,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.centre_freq,
        )

        if width_shunt_mm < self.min_trace_width_mm:
            raise ValueError(
                f"Shunt stub {idx} width {width_shunt_mm:.4f} mm "
                f"< minimum trace width {self.min_trace_width_mm} mm"
            )

        length_shunt_mm = calculate_electrical_length_mm(
            self.elec_length_deg,
            er_eff_shunt,
            self.centre_freq,
        )
        return width_shunt_mm, length_shunt_mm

    def _compute_geometry(self) -> None:
        """
        Compute all derived geometric parameters for the quarter-wave stub filter.

        Calculates the series line width and length, and the width and
        length of each shunt stub, based on the characteristic impedance,
        substrate properties, filter coefficients, and target frequency.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the series line width or any shunt stub width falls below
            `min_trace_width_mm`, or if adjacent shunt stubs would overlap.
        """
        self.series_line_width_mm, er_eff = microstrip_width_from_impedance(
            self.charac_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )

        if self.series_line_width_mm < self.min_trace_width_mm:
            raise ValueError(
                f"Series line width {self.series_line_width_mm:.4f} mm "
                f"< minimum trace width {self.min_trace_width_mm} mm"
            )

        self.line_length_mm = calculate_electrical_length_mm(
            self.elec_length_deg,
            er_eff,
            self.main_freq,
        )
        self.shunt_line_width_mm = []
        self.shunt_line_length_mm = []
        for i in range(self.filter_order):
            width_shunt, length_shunt = self._get_shunt_width_and_length(i)
            self.shunt_line_width_mm.append(width_shunt)
            self.shunt_line_length_mm.append(length_shunt)

        for i in range(self.filter_order - 1):
            gap = (
                self.line_length_mm
                - (self.shunt_line_width_mm[i] + self.shunt_line_width_mm[i + 1]) / 2
            )
            if gap < 0:
                raise ValueError(
                    f"Adjacent stubs {i} and {i + 1} overlap by {-gap:.4f} mm"
                )

        self._round_outputs()

    def _round_outputs(self) -> None:
        """
        Round all geometric parameters to the configured floating-point precision.
        This method iterates over key geometric and mesh attributes and rounds
        them to `self.fp_precision` decimal places.

        Returns
        -------
        None
        """
        for attr in [
            "line_length_mm",
            "series_line_width_mm",
            "shunt_line_width_mm",
            "shunt_line_length_mm",
            "frac_bandwidth",
            "lambda0",
            "FDTD_mesh_resolution",
            "FDTD_metal_mesh_resolution",
            "FDTD_thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class QuarterWaveFilter(SimTools):
    """
    Base class for quarter-wave stub filter models using openEMS.

    Provides methods for creating the substrate and ground plane shared
    by all quarter-wave stub filter variants. Specific stub terminations
    (open for band-stop, shorted for band-pass) should be implemented by
    subclasses.

    Parameters
    ----------
    params : QuarterWaveFilterParams
        Data container holding geometric and material properties (e.g.,
        substrate thickness, permittivity, and line/stub dimensions).
    sim : SimSetup
        Named tuple containing the CSXCAD geometry and openEMS FDTD engine.

    Attributes
    ----------
    params : QuarterWaveFilterParams
        Stored reference to the simulation parameters.
    CSX : ContinuousStructure
        Stored reference to the geometry engine (from sim).
    FDTD : openEMS
        Stored reference to the simulation engine (from sim).
    """

    def __init__(
        self,
        params: QuarterWaveFilterParams,
        sim: SimSetup,
    ) -> None:
        """Initialise the QuarterWaveFilter with parameters and simulation objects."""
        self.params = params
        self.CSX = sim.CSX
        self.FDTD = sim.FDTD

    def create_substrate(self) -> None:
        """
        Define and add the dielectric substrate to the simulation.

        This method creates a material using the permittivity and
        loss tangent.

        Returns
        -------
        None
        """
        substrate = self.CSX.AddMaterial(
            "substrate",
            epsilon=self.params.substrate_eps_r,
            kappa=self.params.substrate_kappa,
        )
        substrate.SetColor("#0F8A00", 100)
        substrate_start = [
            -self.params.substrate_length_mm / 2,
            -self.params.lambda0 / 4,
            0,
        ]
        substrate_stop = [
            self.params.substrate_length_mm + self.params.substrate_length_mm / 2,
            self.params.substrate_width_mm + self.params.lambda0 / 4,
            self.params.substrate_thickness_mm,
        ]
        substrate.AddBox(priority=0, start=substrate_start, stop=substrate_stop)

    def create_ground(self) -> None:
        """
        Define and add the copper ground plane to the geometry.

        Adds a metallic box (PEC) below the substrate. The ground plane
        thickness is defined by `copper_thickness_mm` and
        extends to the edges of the substrate.

        Returns
        -------
        None
        """
        ground = self.CSX.AddMetal("ground")
        ground.SetColor("#B87333", 255)
        ground_start = [
            -self.params.substrate_length_mm / 2,
            -self.params.lambda0 / 4,
            0,
        ]
        ground_stop = [
            self.params.substrate_length_mm + self.params.substrate_length_mm / 2,
            self.params.substrate_width_mm + self.params.lambda0 / 4,
            -self.params.copper_thickness_mm,
        ]
        ground.AddBox(priority=2, start=ground_start, stop=ground_stop)


class BandStopQuarterWaveFilter(QuarterWaveFilter):
    """
    Band-stop quarter-wave stub filter model.

    Extends `QuarterWaveFilter` with the series transmission line sections,
    open-circuited shunt stubs, excitation ports, and FDTD mesh needed for
    a band-stop (notch) response. Each shunt stub is left open at its far
    end, which reflects energy near `centre_freq` back into the through
    line. Use `build_band_stop_quarter_wave_filter` to construct the full
    geometry, or call the individual ``create_*`` methods directly for
    finer control.
    """

    def create_series_line(self) -> None:
        """
        Create the series transmission line sections.

        Adds multiple rectangular metal boxes for the series line
        segments of the filter, separated by shunt stub locations.

        Returns
        -------
        None
        """
        first_series_line = self.CSX.AddMetal("series_line_1")
        first_series_line.SetColor("#B87333", 255)
        f_line_start = [0, 0, self.params.substrate_thickness_mm]
        f_line_stop = [
            self.params.line_length_mm,
            self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        first_series_line.AddBox(
            priority=1,
            start=f_line_start,
            stop=f_line_stop,
        )
        for i in range(self.params.filter_order):
            remaining_series_line = self.CSX.AddMetal(f"series_line_{i + 2}")
            remaining_series_line.SetColor("#B87333", 255)
            line_start = [
                (i + 1) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1]),
                0,
                self.params.substrate_thickness_mm,
            ]
            line_stop = [
                (i + 2) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1]),
                self.params.series_line_width_mm,
                self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
            ]
            remaining_series_line.AddBox(
                priority=1,
                start=line_start,
                stop=line_stop,
            )

    def create_shunt_line(self) -> None:
        """
        Create the shunt stub line sections.

        Adds multiple rectangular metal boxes for the shunt stub
        elements of the band-stop filter, connecting from the series
        line to the open-circuit end.

        Returns
        -------
        None
        """
        first_shunt_line = self.CSX.AddMetal("shunt_line_1")
        first_shunt_line.SetColor("#B87333", 255)
        f_line_start = [
            self.params.line_length_mm,
            0,
            self.params.substrate_thickness_mm,
        ]
        f_line_stop = [
            self.params.line_length_mm + self.params.shunt_line_width_mm[0],
            self.params.shunt_line_length_mm[0] + self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        first_shunt_line.AddBox(
            priority=1,
            start=f_line_start,
            stop=f_line_stop,
        )

        for i in range(self.params.filter_order - 1):
            shunt_line = self.CSX.AddMetal(f"shunt_line_{i + 2}")
            shunt_line.SetColor("#B87333", 255)
            line_start = [
                (i + 2) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1]),
                0,
                self.params.substrate_thickness_mm,
            ]
            line_stop = [
                (i + 2) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1])
                + self.params.shunt_line_width_mm[i + 1],
                self.params.series_line_width_mm
                + self.params.shunt_line_length_mm[i + 1],
                self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
            ]
            shunt_line.AddBox(
                priority=1,
                start=line_start,
                stop=line_stop,
            )

    def create_ports(self) -> list[LumpedPort]:
        """
        Define the excitation lumped ports at both ends of the filter.

        Creates Port 1 (excited) at the input and Port 2 (unexcited)
        at the output for S-parameter measurement.

        Returns
        -------
        list of LumpedPort
            A list of two LumpedPort objects [port1, port2].
        """
        total_length = _get_total_length(
            self.params.filter_order,
            self.params.line_length_mm,
            self.params.shunt_line_width_mm,
        )
        port_1_start = [
            0,
            0,
            0,
        ]
        port_1_stop = [
            0,
            self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port_1 = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_1_start,
            port_1_stop,
            "z",
            excite=1,
            priority=6,
            edges2grid="x",
        )
        port_2_start = [
            total_length,
            0,
            0,
        ]
        port_2_stop = [
            total_length,
            self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port_2 = self.FDTD.AddLumpedPort(
            2,
            self.params.charac_imp,
            port_2_start,
            port_2_stop,
            "z",
            excite=0,
            priority=6,
            edges2grid="x",
        )

        return [port_1, port_2]

    def build_band_stop_quarter_wave_filter(self) -> list[LumpedPort]:
        """
        Construct the complete band-stop quarter-wave stub filter geometry.

        Orchestrates the creation of all filter components, including the
        substrate, ground plane, series transmission line sections, open
        shunt stubs, and excitation ports. The mesh must be created
        separately via ``create_mesh``.

        Returns
        -------
        list of LumpedPort
            A list of two LumpedPort objects [port1, port2] for
            S-parameter extraction.
        """
        self.create_substrate()
        self.create_ground()
        self.create_series_line()
        self.create_shunt_line()
        ports = self.create_ports()
        return ports

    def create_mesh(self, manual_mesh: bool = False) -> None:
        """
        Generate an FDTD mesh for the band-stop filter simulation domain.

        Defines mesh lines for x, y, and z directions based on the
        filter geometry, series and shunt line dimensions, and
        simulation box size. Uses SmoothMeshLines for stable grid
        transitions.

        Parameters
        ----------
        manual_mesh : bool, optional
            If True, use manual mesh line definitions instead of the
            automatic Mesh class. Default is False.

        Returns
        -------
        None
        """
        if manual_mesh is True:
            mesh = self.CSX.GetGrid()
            mesh.SetDeltaUnit(self.params.unit)
            total_length = _get_total_length(
                self.params.filter_order,
                self.params.line_length_mm,
                self.params.shunt_line_width_mm,
            )

            mesh.AddLine(
                "x",
                [
                    -self.params.simulation_box[0] - self.params.lambda0 / 2,
                    self.params.simulation_box[0] + self.params.lambda0,
                ],
            )
            mesh.AddLine(
                "y",
                [
                    -self.params.simulation_box[1] - self.params.lambda0 / 2,
                    self.params.simulation_box[1] + self.params.lambda0,
                ],
            )
            mesh.AddLine(
                "z",
                [
                    -self.params.simulation_box[2] / 3,
                    self.params.simulation_box[2] * 2 / 3,
                ],
            )
            # Add mesh lines for substrate
            mesh.AddLine(
                "x",
                [
                    -self.params.substrate_width_mm / 3,
                    self.params.substrate_width_mm * 2 / 3,
                ],
            )
            mesh.AddLine(
                "y",
                [
                    -self.params.substrate_length_mm / 3,
                    self.params.substrate_length_mm * 2 / 3,
                ],
            )
            mesh.AddLine("x", 0)
            mesh.AddLine("x", total_length)
            mesh.AddLine(
                "y",
                np.linspace(
                    (self.params.series_line_width_mm + self.params.line_length_mm)
                    + 2
                    - 0.03,
                    (self.params.series_line_width_mm + self.params.line_length_mm)
                    - 2
                    + 0.06,
                    8,
                ),
            )
            mesh.AddLine(
                "y",
                np.linspace(
                    0 - 0.3,
                    self.params.series_line_width_mm + 0.6,
                    9,
                ),
            )

            mesh.AddLine(
                "z",
                np.linspace(
                    -self.params.copper_thickness_mm / 2,
                    self.params.substrate_thickness_mm
                    + self.params.copper_thickness_mm / 2,
                    9,
                ),
            )

            for i in range(self.params.filter_order):
                mesh.AddLine(
                    "x",
                    np.linspace(
                        (i + 1) * self.params.line_length_mm
                        + sum(self.params.shunt_line_width_mm[:i])
                        - 0.3,
                        (i + 1) * self.params.line_length_mm
                        + sum(self.params.shunt_line_width_mm[:i])
                        + self.params.shunt_line_width_mm[i]
                        + 0.6,
                        6,
                    ),
                )

            mesh.SmoothMeshLines("all", self.params.FDTD_mesh_resolution, 2.5)

        else:
            Mesh(self.CSX, self.params)


class BandPassQuarterWaveFilter(QuarterWaveFilter):
    """
    Band-pass quarter-wave stub filter model.

    Extends `QuarterWaveFilter` with the series transmission line sections,
    short-circuited shunt stubs, excitation ports, and FDTD mesh needed for
    a band-pass response. Each shunt stub is grounded at its far end via a
    vertical strap (see `create_shunt_line_short`), which presents a high
    impedance at the junction near `centre_freq` and lets that band through
    while attenuating frequencies further away. Use
    `build_band_pass_quarter_wave_filter` to construct the full geometry, or
    call the individual ``create_*`` methods directly for finer control.
    """

    def create_series_line(self) -> None:
        """
        Create the series transmission line sections.

        Adds multiple rectangular metal boxes for the series line
        segments of the filter, separated by shunt stub locations.

        Returns
        -------
        None
        """
        first_series_line = self.CSX.AddMetal("series_line_1")
        first_series_line.SetColor("#B87333", 255)
        f_line_start = [0, 0, self.params.substrate_thickness_mm]
        f_line_stop = [
            self.params.line_length_mm,
            self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        first_series_line.AddBox(
            priority=1,
            start=f_line_start,
            stop=f_line_stop,
        )
        for i in range(self.params.filter_order):
            remaining_series_line = self.CSX.AddMetal(f"series_line_{i + 2}")
            remaining_series_line.SetColor("#B87333", 255)
            line_start = [
                (i + 1) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1]),
                0,
                self.params.substrate_thickness_mm,
            ]
            line_stop = [
                (i + 2) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1]),
                self.params.series_line_width_mm,
                self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
            ]
            remaining_series_line.AddBox(
                priority=1,
                start=line_start,
                stop=line_stop,
            )

    def create_shunt_line(self) -> None:
        """
        Create the shunt stub line sections.

        Adds multiple rectangular metal boxes for the shunt stub
        elements of the band-pass filter, connecting from the series
        line to the point where it is grounded by
        ``create_shunt_line_short``.

        Returns
        -------
        None
        """
        first_shunt_line = self.CSX.AddMetal("shunt_line_1")
        first_shunt_line.SetColor("#B87333", 255)
        f_line_start = [
            self.params.line_length_mm,
            0,
            self.params.substrate_thickness_mm,
        ]
        f_line_stop = [
            self.params.line_length_mm + self.params.shunt_line_width_mm[0],
            self.params.shunt_line_length_mm[0] + self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        first_shunt_line.AddBox(
            priority=1,
            start=f_line_start,
            stop=f_line_stop,
        )

        for i in range(self.params.filter_order - 1):
            shunt_line = self.CSX.AddMetal(f"shunt_line_{i + 2}")
            shunt_line.SetColor("#B87333", 255)
            line_start = [
                (i + 2) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1]),
                0,
                self.params.substrate_thickness_mm,
            ]
            line_stop = [
                (i + 2) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1])
                + self.params.shunt_line_width_mm[i + 1],
                self.params.series_line_width_mm
                + self.params.shunt_line_length_mm[i + 1],
                self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
            ]
            shunt_line.AddBox(
                priority=1,
                start=line_start,
                stop=line_stop,
            )

    def create_shunt_line_short(self) -> None:
        """
        Create the grounding shorts at the open end of each shunt stub.

        Adds a vertical metal box at the far end of each shunt stub,
        spanning from the ground plane (z=0) up through the substrate to
        the stub's top copper layer, shorting it to ground. This turns
        each shunt stub into a short-circuited quarter-wave stub, which is
        required for the band-pass response.

        Returns
        -------
        None
        """
        first_shunt_line_short = self.CSX.AddMetal("shunt_line_short_1")
        first_shunt_line_short.SetColor("#B87333", 255)
        f_short_start = [
            self.params.line_length_mm,
            self.params.shunt_line_length_mm[0] + self.params.series_line_width_mm,
            0,
        ]
        f_short_stop = [
            self.params.line_length_mm + self.params.shunt_line_width_mm[0],
            self.params.shunt_line_length_mm[0] + self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        first_shunt_line_short.AddBox(
            priority=1,
            start=f_short_start,
            stop=f_short_stop,
        )

        for i in range(self.params.filter_order - 1):
            shunt_line_short = self.CSX.AddMetal(f"shunt_line_short_{i + 2}")
            shunt_line_short.SetColor("#B87333", 255)
            line_start = [
                (i + 2) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1]),
                self.params.series_line_width_mm
                + self.params.shunt_line_length_mm[i + 1],
                0,
            ]
            line_stop = [
                (i + 2) * self.params.line_length_mm
                + sum(self.params.shunt_line_width_mm[: i + 1])
                + self.params.shunt_line_width_mm[i + 1],
                self.params.series_line_width_mm
                + self.params.shunt_line_length_mm[i + 1],
                self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
            ]
            shunt_line_short.AddBox(
                priority=1,
                start=line_start,
                stop=line_stop,
            )

    def create_ports(self) -> list[LumpedPort]:
        """
        Define the excitation lumped ports at both ends of the filter.

        Creates Port 1 (excited) at the input and Port 2 (unexcited)
        at the output for S-parameter measurement.

        Returns
        -------
        list of LumpedPort
            A list of two LumpedPort objects [port1, port2].
        """
        total_length = _get_total_length(
            self.params.filter_order,
            self.params.line_length_mm,
            self.params.shunt_line_width_mm,
        )
        port_1_start = [
            0,
            0,
            0,
        ]
        port_1_stop = [
            0,
            self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port_1 = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_1_start,
            port_1_stop,
            "z",
            excite=1,
            priority=6,
            edges2grid="x",
        )
        port_2_start = [
            total_length,
            0,
            0,
        ]
        port_2_stop = [
            total_length,
            self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port_2 = self.FDTD.AddLumpedPort(
            2,
            self.params.charac_imp,
            port_2_start,
            port_2_stop,
            "z",
            excite=0,
            priority=6,
            edges2grid="x",
        )

        return [port_1, port_2]

    def build_band_pass_quarter_wave_filter(self) -> list[LumpedPort]:
        """
        Construct the complete band-pass quarter-wave stub filter geometry.

        Orchestrates the creation of all filter components, including the
        substrate, ground plane, series transmission line sections, shunt
        stubs with their grounding shorts, and excitation ports. The mesh
        must be created separately via ``create_mesh``.

        Returns
        -------
        list of LumpedPort
            A list of two LumpedPort objects [port1, port2] for
            S-parameter extraction.
        """
        self.create_substrate()
        self.create_ground()
        self.create_series_line()
        self.create_shunt_line()
        self.create_shunt_line_short()
        ports = self.create_ports()
        return ports

    def create_mesh(self, manual_mesh: bool = False) -> None:
        """
        Generate an FDTD mesh for the band-pass filter simulation domain.

        Defines mesh lines for x, y, and z directions based on the
        filter geometry, series and shunt line dimensions, and
        simulation box size. Uses SmoothMeshLines for stable grid
        transitions.

        Parameters
        ----------
        manual_mesh : bool, optional
            If True, use manual mesh line definitions instead of the
            automatic Mesh class. Default is False.

        Returns
        -------
        None
        """
        if manual_mesh is True:
            mesh = self.CSX.GetGrid()
            mesh.SetDeltaUnit(self.params.unit)
            total_length = _get_total_length(
                self.params.filter_order,
                self.params.line_length_mm,
                self.params.shunt_line_width_mm,
            )

            mesh.AddLine(
                "x",
                [
                    -self.params.simulation_box[0] - self.params.lambda0 / 2,
                    self.params.simulation_box[0] + self.params.lambda0,
                ],
            )
            mesh.AddLine(
                "y",
                [
                    -self.params.simulation_box[1] - self.params.lambda0 / 2,
                    self.params.simulation_box[1] + self.params.lambda0,
                ],
            )
            mesh.AddLine(
                "z",
                [
                    -self.params.simulation_box[2] / 3,
                    self.params.simulation_box[2] * 2 / 3,
                ],
            )
            # Add mesh lines for substrate
            mesh.AddLine(
                "x",
                [
                    -self.params.substrate_width_mm / 3,
                    self.params.substrate_width_mm * 2 / 3,
                ],
            )
            mesh.AddLine(
                "y",
                [
                    -self.params.substrate_length_mm / 3,
                    self.params.substrate_length_mm * 2 / 3,
                ],
            )
            mesh.AddLine("x", 0)
            mesh.AddLine("x", total_length)
            mesh.AddLine(
                "y",
                np.linspace(
                    (self.params.series_line_width_mm + self.params.line_length_mm)
                    + 2
                    - 0.03,
                    (self.params.series_line_width_mm + self.params.line_length_mm)
                    - 2
                    + 0.06,
                    8,
                ),
            )
            mesh.AddLine(
                "y",
                np.linspace(
                    0 - 0.3,
                    self.params.series_line_width_mm + 0.6,
                    9,
                ),
            )

            mesh.AddLine(
                "z",
                np.linspace(
                    -self.params.copper_thickness_mm / 2,
                    self.params.substrate_thickness_mm
                    + self.params.copper_thickness_mm / 2,
                    9,
                ),
            )

            for i in range(self.params.filter_order):
                mesh.AddLine(
                    "x",
                    np.linspace(
                        (i + 1) * self.params.line_length_mm
                        + sum(self.params.shunt_line_width_mm[:i])
                        - 0.3,
                        (i + 1) * self.params.line_length_mm
                        + sum(self.params.shunt_line_width_mm[:i])
                        + self.params.shunt_line_width_mm[i]
                        + 0.6,
                        6,
                    ),
                )

            mesh.SmoothMeshLines("all", self.params.FDTD_mesh_resolution, 2.5)

        else:
            Mesh(self.CSX, self.params)
