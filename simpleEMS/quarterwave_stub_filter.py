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
definition and `BandStopQuarterWaveFilter` for structure construction.
"""

import math
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS
from openEMS.ports import LumpedPort

from simpleEMS.sim_tools import SimTools
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
    """Compute total physical length of the filter structure."""
    total_length = (filter_order + 1) * line_length_mm
    for i in range(filter_order):
        total_length += shunt_line_width_mm[i]
    return total_length


@dataclass
class QuarterWaveFilterParams(SimParams):
    """
    Parameters for a quarter-wave stub filter.

    This class extends `SimParams` and computes derived geometric
    parameters required for filter layout and simulation. Supports
    band-pass and band-stop responses with Bessel, Butterworth, or
    Chebyshev prototypes.

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
        Order of the filter.
    ripple_db : float, optional
        Passband ripple in dB for Chebyshev response. Default is None.
    elec_length_deg : int, optional
        Electrical length of each line section in degrees. Default is 90.

    Attributes
    ----------
    frac_bandwidth : float
        Fractional bandwidth (bandwidth_freq / centre_freq).
    series_line_width_mm : float
        Width of the series transmission line in mm.
    shunt_line_width_mm : list of float
        Widths of each shunt stub in mm.
    line_length_mm : float
        Physical length of each line section in mm.
    """

    min_freq: float
    max_freq: float
    centre_freq: float
    bandwidth_freq: float
    filter_type: str  # lowpass highpass bandpass
    filter_response: str  # butterworth chebychev  bessel
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
            The centre frequency for the filter.
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
        """Compute geometry and validate filter type after init."""
        super().__post_init__()
        if self.filter_type not in ("bandpass", "bandstop"):
            raise ValueError(f"Quarter-wave filter does not support {self.filter_type}")

        self.frac_bandwidth = self.bandwidth_freq / self.centre_freq

        self._compute_geometry()

    def _get_shunt_width_and_length(self, idx: int) -> tuple[float, float]:
        """Calculate the width of a single shunt stub for the given index."""
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
        length_shunt_mm = calculate_electrical_length_mm(
            self.elec_length_deg,
            er_eff_shunt,
            self.centre_freq,
        )
        return width_shunt_mm, length_shunt_mm

    def _compute_geometry(self) -> None:
        """
        Compute all derived geometric parameters for the quarter-wave stub filter.
        This method calculates the series line width, shunt line widths,
        and line length based on the characteristic impedance, substrate
        properties, filter coefficients, and target frequency.

        Returns
        -------
        None
        """
        self.series_line_width_mm, er_eff = microstrip_width_from_impedance(
            self.charac_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
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
            "mesh_resolution",
            "metal_mesh_resolution",
            "thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class QuarterWaveFilter(SimTools):
    """
    Base class for quarter-wave stub filter models using openEMS.

    Provides methods for creating the substrate and ground plane shared
    by all quarter-wave stub filter variants.
    """

    def __init__(
        self,
        params: QuarterWaveFilterParams,
        CSX: ContinuousStructure,
        FDTD: openEMS,
    ) -> None:
        """Initialise the QuarterWaveFilter with parameters and simulation objects."""
        self.params = params
        self.CSX = CSX
        self.FDTD = FDTD

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

    This class implements the geometry representation of a band-stop
    quarter-wave stub filter. It extends `QuarterWaveFilter` by adding
    the series transmission line sections, shunt stubs, ports, and
    FDTD mesh.
    """

    def create_series_line(self) -> None:
        """
        Create the series transmission line sections.

        Adds multiple rectangular metal boxes for the series line
        segments of the band-stop filter, separated by shunt stub
        locations.
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

    def create_mesh(self, manual_mesh: bool = False) -> None:
        """
        Generate an FDTD mesh for the band-stop filter simulation domain.

        Defines mesh lines for x, y, and z directions based on the
        filter geometry, series and shunt line dimensions, and
        simulation box size. Uses SmoothMeshLines for stable grid
        transitions.

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

            mesh.SmoothMeshLines("all", self.params.mesh_resolution, 2.5)

        else:
            Mesh(self.CSX, self.params)


class BandPassQuarterWaveFilter(QuarterWaveFilter):
    """
    Band-pass quarter-wave stub filter model.

    This class implements the geometry representation of a band-stop
    quarter-wave stub filter. It extends `QuarterWaveFilter` by adding
    the series transmission line sections, shunt stubs and shorts, ports, and
    FDTD mesh.
    """

    def create_series_line(self) -> None:
        """
        Create the series transmission line sections.

        Adds multiple rectangular metal boxes for the series line
        segments of the band-pass filter, separated by shunt stub
        locations.
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
        line to the short-circuit end.
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
        Create the shunt stub shorts.

        Adds multiple rectangular metal boxes for the shunt stub shorts
        band-pass filter.
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

    def create_mesh(self, manual_mesh: bool = False) -> None:
        """
        Generate an FDTD mesh for the band-stop filter simulation domain.

        Defines mesh lines for x, y, and z directions based on the
        filter geometry, series and shunt line dimensions, and
        simulation box size. Uses SmoothMeshLines for stable grid
        transitions.

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

            mesh.SmoothMeshLines("all", self.params.mesh_resolution, 2.5)

        else:
            Mesh(self.CSX, self.params)
