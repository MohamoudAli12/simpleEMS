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
import math
import numpy as np
from dataclasses import dataclass, field

from simpleEMS.sim_tools import SimTools
from .sim_params import SimParams
from .calc import microstrip_width_from_impedance, phase_shift_length
from .filter_coefficient import get_filter_coefficient


@dataclass
class QuarterWaveFilterParams(SimParams):
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
    line_length_mm: float = field(init=False)

    @property
    def freq_range(self):
        """
        Return the simulation frequency range.
        Returns
        -------
        tuple of (float, float)
            A tuple containing (min_freq, max_freq).
        """
        return self.min_freq, self.max_freq

    @property
    def main_freq(self):
        """
        Return the primary frequency of interest for analysis.
        Returns
        -------
        float
            The centre frequency for the filter.
        """
        return self.centre_freq

    def __post_init__(self):
        super().__post_init__()
        if self.filter_type not in ("bandpass", "bandstop"):
            raise ValueError(f"Quarter-wave filter does not support {self.filter_type}")

        self.frac_bandwidth = self.bandwidth_freq / self.centre_freq
        self._compute_geometry()
        self._create_simulation_box()

    def _get_shunt_width(self, idx):
        g = get_filter_coefficient(
            idx, self.filter_response, self.filter_order, self.ripple_db
        )
        shunt_imped = 0.0
        if self.filter_type == "bandpass":
            shunt_imped = (math.pi * self.charac_imp * self.frac_bandwidth) / (4 * g)
        if self.filter_type == "bandstop":
            shunt_imped = (4 * self.charac_imp) / (math.pi * self.frac_bandwidth * g)
        width_shunt_mm = microstrip_width_from_impedance(
            shunt_imped,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.centre_freq,
        )
        return width_shunt_mm

    def _compute_geometry(self):
        """
        Compute all derived geometric parameters for the microstrip line.
        This method calculates the microstrip width and length based on
        the characteristic impedance, substrate properties, and target
        frequency. It also computes the substrate dimensions.
        Returns
        -------
        None
        """
        self.series_line_width_mm = microstrip_width_from_impedance(
            self.charac_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )

        self.line_length_mm = phase_shift_length(
            self.elec_length_deg,
            self.substrate_eps_r,
            self.main_freq,
        )
        self.shunt_line_width_mm = []
        for i in range(self.filter_order):
            width_shunt = self._get_shunt_width(i)
            self.shunt_line_width_mm.append(width_shunt)

        self.substrate_width_mm = (self.filter_order + 1) * self.line_length_mm + sum(
            self.shunt_line_width_mm
        )

        self.substrate_length_mm = self.series_line_width_mm + self.line_length_mm

        self._round_outputs()

    def _round_outputs(self):
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
            "frac_bandwidth",
            "substrate_width_mm",
            "substrate_length_mm",
            "lambda0",
            "mesh_resolution",
            "metal_mesh_resolution",
            "thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class QuarterWaveFilter(SimTools):
    def __init__(self, params, CSX, FDTD):
        self.params = params
        self.CSX = CSX
        self.FDTD = FDTD

    def create_substrate(self):
        """
        Define and add the dielectric substrate to the simulation.

        This method creates a material using the permittivity and
        loss tangent (kappa) defined in `self.params`, colors it green,
        and adds a box geometry centered on the XY plane.

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
            -self.params.lambda0 / 4,
            -self.params.lambda0 / 4,
            0,
        ]
        substrate_stop = [
            self.params.substrate_width_mm + self.params.lambda0 / 4,
            self.params.substrate_length_mm + self.params.lambda0 / 4,
            self.params.substrate_thickness_mm,
        ]
        substrate.AddBox(priority=0, start=substrate_start, stop=substrate_stop)

    def create_ground(self):
        """
        Define and add the copper ground plane to the geometry.

        Adds a metallic box (PEC) below the substrate. The ground plane
        thickness is defined by `self.params.copper_thickness_mm` and
        extends to the edges of the substrate.

        Returns
        -------
        None

        Notes
        -----
        The ground plane is assigned a higher priority (2).
        """
        ground = self.CSX.AddMetal("ground")
        ground.SetColor("#B87333", 255)
        ground_start = [
            -self.params.lambda0 / 4,
            -self.params.lambda0 / 4,
            0,
        ]
        ground_stop = [
            self.params.substrate_width_mm + self.params.lambda0 / 4,
            self.params.substrate_length_mm + self.params.lambda0 / 4,
            -self.params.copper_thickness_mm,
        ]
        ground.AddBox(priority=2, start=ground_start, stop=ground_stop)


class BandStopQuarterWaveFilter(QuarterWaveFilter):
    """
    class
    """

    def _get_total_length(self):
        total_length = (self.params.filter_order + 1) * self.params.line_length_mm
        for i in range(self.params.filter_order):
            total_length += self.params.shunt_line_width_mm[i]
        return total_length

    def create_series_line(self):
        series_line = self.CSX.AddMetal("series_line")
        series_line.SetColor("#B87333", 255)
        total_length = self._get_total_length()
        line_start = [
            0,
            0,
            self.params.substrate_thickness_mm,
        ]
        line_stop = [
            total_length,
            self.params.series_line_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        series_line.AddBox(
            priority=1,
            start=line_start,
            stop=line_stop,
        )

    def create_shunt_line(self):
        for i in range(self.params.filter_order):
            shunt_line = self.CSX.AddMetal(f"shunt_line_{i + 1}")
            shunt_line.SetColor("#B87333", 255)
            line_start = [
                (i + 1) * self.params.line_length_mm,
                self.params.series_line_width_mm,
                self.params.substrate_thickness_mm,
            ]
            line_stop = [
                (i + 1) * self.params.line_length_mm
                + self.params.shunt_line_width_mm[i],
                self.params.series_line_width_mm + self.params.line_length_mm,
                self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
            ]
            shunt_line.AddBox(
                priority=1,
                start=line_start,
                stop=line_stop,
            )
