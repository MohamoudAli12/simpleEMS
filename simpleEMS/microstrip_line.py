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

from dataclasses import dataclass, field
import numpy as np
from .calc import microstrip_width_from_impedance, phase_shift_length
from .sim_params import SimParams
from .sim_tools import SimTools

# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "MicrostripLineParams",
    "MicrostripLine",
]


@dataclass
class MicrostripLineParams(SimParams):
    """
    This dataclass represents the parameters for a microstrip line.
    """

    min_freq: float
    max_freq: float
    target_freq:float

    ang_length_deg: int = 90
    microstrip_length_mm: float = field(init=False)
    microstrip_width_mm: float = field(init=False)

    @property
    def freq_range(self):
        return self.min_freq, self.max_freq

    @property
    def main_freq(self):
        return self.target_freq


    def __post_init__(self):
        super().__post_init__()
        self._compute_geometry()
        self._create_simulation_box()

    def _compute_geometry(self):

        self.microstrip_width_mm = microstrip_width_from_impedance(
            self.charac_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )

        self.microstrip_length_mm = phase_shift_length(
            self.ang_length_deg,
            self.substrate_eps_r,
            self.main_freq,
        )

        self.substrate_width_mm = self.microstrip_width_mm + 2 * self.lambda0
        self.substrate_length_mm = self.microstrip_length_mm + 2 * self.lambda0

        self._round_outputs()

    def _round_outputs(self):
        for attr in [
            "microstrip_width_mm",
            "microstrip_length_mm",
            "substrate_width_mm",
            "substrate_length_mm",
            "lambda0",
            "mesh_resolution",
            "metal_mesh_resolution",
            "thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class MicrostripLine(SimTools):
    """
    this builds the microstrip line.


    """

    def __init__(self, params, CSX, FDTD) -> None:

        self.params = params
        self.CSX = CSX
        self.FDTD = FDTD

    def create_substrate(self):
        """
        create the substrate
        """
        substrate = self.CSX.AddMaterial(
            "substrate",
            epsilon=self.params.substrate_eps_r,
            kappa=self.params.substrate_kappa,
        )
        substrate.SetColor("#0F8A00", 100)
        substrate_start = [
            -self.params.substrate_width_mm / 2,
            -self.params.substrate_length_mm / 2,
            0,
        ]
        substrate_stop = [
            self.params.substrate_width_mm / 2,
            self.params.substrate_length_mm / 2,
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
            -self.params.substrate_width_mm / 2,
            -self.params.substrate_length_mm / 2,
            0,
        ]
        ground_stop = [
            self.params.substrate_width_mm / 2,
            self.params.substrate_length_mm / 2,
            -self.params.copper_thickness_mm,
        ]
        ground.AddBox(priority=2, start=ground_start, stop=ground_stop)

    def create_microstrip_line(self):
        """
        create the microstrip line
        """
        microstrip_line = self.CSX.AddMetal("microstrip")
        microstrip_line.SetColor("#B87333", 255)
        microstrip_start = [
            self.params.microstrip_width_mm / 2,
            self.params.microstrip_length_mm / 2,
            self.params.substrate_thickness_mm,
        ]
        microstrip_stop = [
            -self.params.microstrip_width_mm / 2,
            -self.params.microstrip_length_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        microstrip_line.AddBox(priority=6, start=microstrip_start, stop=microstrip_stop)
        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=microstrip_line,
            metal_edge_res=self.params.metal_mesh_resolution,
        )

    def create_ports(self):
        """
        create port
        """
        port = [None, None]
        port_1_start = [
            self.params.microstrip_width_mm / 2,
            -self.params.microstrip_length_mm / 2,
            0,
        ]
        port_1_stop = [
            -self.params.microstrip_width_mm / 2,
            -self.params.microstrip_length_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port[0] = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_1_start,
            port_1_stop,
            "z",
            excite=1,
            priority=6,
            edges2grid="y",
        )
        port_2_start = [
            self.params.microstrip_width_mm / 2,
            self.params.microstrip_length_mm / 2,
            0,
        ]
        port_2_stop = [
            -self.params.microstrip_width_mm / 2,
            self.params.microstrip_length_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port[1] = self.FDTD.AddLumpedPort(
            2,
            self.params.charac_imp,
            port_2_start,
            port_2_stop,
            "z",
            excite=0,
            priority=6,
            edges2grid="y",
        )

        return port

    def create_mesh(self):
        mesh = self.CSX.GetGrid()
        mesh.SetDeltaUnit(self.params.unit)

        mesh.AddLine(
            "x", [-self.params.simulation_box[0] / 2, self.params.simulation_box[0] / 2]
        )
        mesh.AddLine(
            "y", [-self.params.simulation_box[1] / 2, self.params.simulation_box[1] / 2]
        )
        mesh.AddLine(
            "z",
            [-self.params.simulation_box[2] / 3, self.params.simulation_box[2] * 2 / 3],
        )
        # Add mesh lines for substrate
        mesh.AddLine(
            "x",
            [-self.params.substrate_width_mm / 2, self.params.substrate_width_mm / 2],
        )
        mesh.AddLine(
            "y",
            [-self.params.substrate_length_mm / 2, self.params.substrate_length_mm / 2],
        )
        # Add mesh lines for patch and feed
        mesh.AddLine(
            "x", -self.params.microstrip_width_mm / 2 - self.params.thirds_rule
        )
        mesh.AddLine("x", self.params.microstrip_width_mm / 2 + self.params.thirds_rule)
        mesh.AddLine(
            "y", -self.params.microstrip_length_mm / 2 - self.params.thirds_rule
        )
        mesh.AddLine(
            "y", self.params.microstrip_length_mm / 2 + self.params.thirds_rule
        )

        # mesh.AddLine("x", -self.params.feed_width_mm / 2 - self.params.thirds_rule)
        # mesh.AddLine("x", self.params.feed_width_mm / 2 + self.params.thirds_rule)
        # mesh.AddLine("y", -self.params.patch_length_mm / 2 - self.params.feed_length_mm)
        # mesh.AddLine(
        #     "y", -self.params.patch_length_mm / 2 + self.params.inset_length_mm
        # )

        mesh.AddLine(
            "z",
            np.linspace(
                -self.params.copper_thickness_mm / 2,
                self.params.substrate_thickness_mm
                + self.params.copper_thickness_mm / 2,
                self.params.substrate_cells,
            ),
        )

        mesh.SmoothMeshLines("all", self.params.mesh_resolution, 1.5)
