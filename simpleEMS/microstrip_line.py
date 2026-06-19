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
Microstrip transmission line design and simulation.

Provides `MicrostripLineParams` for defining the geometric and
material properties of a microstrip line, and `MicrostripLine`
for constructing the full simulation structure (substrate, ground,
trace, ports, and mesh) using openEMS.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS
from openEMS.ports import LumpedPort

from .calc import (
    microstrip_width_from_impedance,
    calculate_electrical_length_mm,
)
from .sim_params import SimParams
from .sim_tools import SimTools
from .mesh import Mesh

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
    Parameters for a microstrip transmission line.
    This class extends SimParams and computes derived geometric
    parameters required for microstrip line layout and simulation.

    Parameters
    ----------
    min_freq : float
        Minimum frequency of the simulation range in Hz.
    max_freq : float
        Maximum frequency of the simulation range in Hz.
    target_freq : float
        Target frequency used to compute the microstrip dimensions in Hz.
    elec_length_deg : int, optional
        Electrical length (in degrees) used to compute the microstrip
        phase shift length. Default is 90.

    Attributes
    ----------
    microstrip_width_mm : float
        Computed width of the microstrip trace in millimeters.
    microstrip_length_mm : float
        Computed length of the microstrip trace in millimeters.
    substrate_width_mm : float
        Substrate width including margin (lambda0 padding).
    substrate_length_mm : float
        Substrate length including margin (lambda0 padding).
    """

    min_freq: float
    max_freq: float
    target_freq: float

    elec_length_deg: int = 90
    microstrip_length_mm: float = field(init=False)
    microstrip_width_mm: float = field(init=False)

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
            The target frequency for the microstrip line simulation.
        """
        return self.target_freq

    @property
    def substrate_length_mm(self) -> float:
        """
        Return the substrate length including lambda0 padding.

        Returns
        -------
        float
            Substrate length in mm.
        """
        return self.microstrip_length_mm + 2 * self.lambda0

    @property
    def substrate_width_mm(self) -> float:
        """
        Return the substrate width including lambda0 padding.

        Returns
        -------
        float
            Substrate width in mm.
        """
        return self.microstrip_width_mm + 2 * self.lambda0

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
            self.substrate_width_mm + self.lambda0,
            self.substrate_length_mm + self.lambda0,
            self.lambda0 * 2,
        )

    def __post_init__(self) -> None:
        """Perform geometric calculations after dataclass initialisation."""
        super().__post_init__()
        self._compute_geometry()

    def _compute_geometry(self) -> None:
        """
        Compute all derived geometric parameters for the microstrip line.
        This method calculates the microstrip width and length based on
        the characteristic impedance, substrate properties, and target
        frequency. It also computes the substrate dimensions.

        Returns
        -------
        None
        """
        self.microstrip_width_mm, er_eff = microstrip_width_from_impedance(
            self.charac_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )

        if self.microstrip_width_mm < self.min_trace_width_mm:
            raise ValueError(
                f"Microstrip width {self.microstrip_width_mm:.4f} mm "
                f"< minimum trace width {self.min_trace_width_mm} mm"
            )

        self.microstrip_length_mm = calculate_electrical_length_mm(
            self.elec_length_deg,
            er_eff,
            self.main_freq,
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
            "microstrip_width_mm",
            "microstrip_length_mm",
            "lambda0",
            "mesh_resolution",
            "metal_mesh_resolution",
            "thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class MicrostripLine(SimTools):
    """
    A class for modeling and simulating microstrip transmission lines.
    This class extends SimTools and provides methods for creating the
    substrate, ground plane, microstrip trace, ports, and mesh for
    electromagnetic simulation using openEMS.

    Parameters
    ----------
    params : MicrostripLineParams
        Data container containing geometric and material properties.
    CSX : ContinuousStructure
        The CSXCAD geometry container used to define physical objects.
    FDTD : openEMS
        The FDTD simulation engine object.
    """

    def __init__(
        self,
        params: MicrostripLineParams,
        CSX: ContinuousStructure,
        FDTD: openEMS,
    ) -> None:
        """Initialise the MicrostripLine with parameters and simulation objects."""
        self.params = params
        self.CSX = CSX
        self.FDTD = FDTD

    def create_substrate(self) -> None:
        """
        Define and add the dielectric substrate to the simulation.
        This method creates a material using the permittivity and
        loss tangent (kappa) defined in self.params, colors it green,
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

    def create_ground(self) -> None:
        """
        Define and add the copper ground plane to the geometry.

        Adds a metallic box (PEC) below the substrate. The ground plane
        thickness is defined by `self.params.copper_thickness_mm` and
        extends to the edges of the substrate.

        Returns
        -------
        None

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

    def create_microstrip_line(self) -> None:
        """
        Create the microstrip trace element.
        Adds a rectangular metal box representing the microstrip
        transmission line on top of the substrate and registers
        the metal edges with the FDTD grid for accurate field
        calculation.
        Returns
        -------
        None
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

    def create_ports(self) -> list[LumpedPort]:
        """
        Define the excitation lumped ports at both ends of the microstrip line.
        Creates two ports: Port 1 (excited) at one end and Port 2
        (unexcited) at the opposite end, enabling S21 transmission
        measurements.

        Returns
        -------

        list
            A list of two LumpedPort objects [port1, port2] used to
            retrieve S-parameter and impedance results after the simulation.
        """
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
        port_1 = self.FDTD.AddLumpedPort(
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
        port_2 = self.FDTD.AddLumpedPort(
            2,
            self.params.charac_imp,
            port_2_start,
            port_2_stop,
            "z",
            excite=0,
            priority=6,
            edges2grid="y",
        )

        return [port_1, port_2]

    def build_microstrip_line(self) -> list[LumpedPort]:
        """
        Construct the complete microstrip transmission line geometry.

        This method orchestrates the creation of all microstrip
        components including the substrate, ground plane,
        microstrip conductor, and ports.

        Returns
        -------
        list
            A list of two LumpedPort objects [port1, port2] for
            S-parameter extraction.
        """
        self.create_substrate()
        self.create_ground()
        self.create_microstrip_line()
        ports = self.create_ports()
        return ports

    def create_mesh(
        self,
        manual_mesh: bool = False,
        smooth_ratio: float = 1.5,
    ) -> None:
        """
        Generate an FDTD mesh for the microstrip line simulation domain.
        This method defines mesh lines for the x, y, and z directions
        based on the microstrip geometry, substrate thickness, and
        simulation box size. It applies the "thirds rule" for mesh
        refinement near metal edges and uses SmoothMeshLines to
        ensure a stable grid transition.

        Parameters
        ----------
        manual_mesh : bool, optional
            If True, use manual mesh line definitions instead of the
            automatic Mesh class. Default is False.
        smooth_ratio : float, optional
            Maximum ratio between adjacent mesh cells. Default is 1.5.
        Returns
        -------
        None
        """
        if manual_mesh is True:
            mesh = self.CSX.GetGrid()
            mesh.SetDeltaUnit(self.params.unit)

            mesh.AddLine(
                "x",
                [-self.params.simulation_box[0] / 2, self.params.simulation_box[0] / 2],
            )
            mesh.AddLine(
                "y",
                [-self.params.simulation_box[1] / 2, self.params.simulation_box[1] / 2],
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
                    -self.params.substrate_width_mm / 2,
                    self.params.substrate_width_mm / 2,
                ],
            )
            mesh.AddLine(
                "y",
                [
                    -self.params.substrate_length_mm / 2,
                    self.params.substrate_length_mm / 2,
                ],
            )
            # Add mesh lines for patch and feed
            mesh.AddLine(
                "x", -self.params.microstrip_width_mm / 2 - self.params.thirds_rule
            )
            mesh.AddLine(
                "x", self.params.microstrip_width_mm / 2 + self.params.thirds_rule
            )
            mesh.AddLine(
                "y", -self.params.microstrip_length_mm / 2 - self.params.thirds_rule
            )
            mesh.AddLine(
                "y", self.params.microstrip_length_mm / 2 + self.params.thirds_rule
            )

            mesh.AddLine(
                "z",
                np.linspace(
                    -self.params.copper_thickness_mm / 2,
                    self.params.substrate_thickness_mm
                    + self.params.copper_thickness_mm / 2,
                    self.params.substrate_cells,
                ),
            )

            mesh.SmoothMeshLines("all", self.params.mesh_resolution, smooth_ratio)
        else:
            Mesh(self.CSX, self.params)
