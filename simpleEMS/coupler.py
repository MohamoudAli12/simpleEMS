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
Microstrip coupler design and simulation.

Provides `QuadratureBranchLineHybridCouplerParams` for defining the
geometric and material properties of a 3 dB quadrature (branch-line)
hybrid, and `QuadratureBranchLineHybridCoupler` for constructing the full
simulation structure (substrate, ground, branch arms, feed lines, ports,
and mesh) using openEMS.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from openEMS.ports import LumpedPort

from .calc import (
    branch_line_hybrid_impedances,
    calculate_electrical_length_mm,
    microstrip_width_from_impedance,
)
from .sim_params import SimParams
from .sim_tools import SimTools, SimSetup
from .fdtd_mesh import Mesh

# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "QuadratureBranchLineHybridCouplerParams",
    "QuadratureBranchLineHybridCoupler",
]


@dataclass
class QuadratureBranchLineHybridCouplerParams(SimParams):
    """
    Parameters for a 3 dB quadrature (branch-line) hybrid coupler.

    This class extends SimParams and computes the derived geometric
    parameters of the four quarter-wave branch arms and the four feed
    lines, following Pozar (4th ed.), section 7.5.

    The layout matches Pozar Figure 7.21: port 1 (input) top left,
    port 2 (through) top right, port 3 (coupled) bottom right and port 4
    (isolated) bottom left. The series arms run along x and the shunt arms
    along y; arm lengths are measured between junction centrelines.

    Parameters
    ----------
    min_freq : float
        Minimum frequency of the simulation range in Hz.
    max_freq : float
        Maximum frequency of the simulation range in Hz.
    centre_freq : float
        Design frequency of the coupler in Hz.
    series_arm_elec_length_deg : float, optional
        Electrical length of each series arm in degrees. Default is 90.
    shunt_arm_elec_length_deg : float, optional
        Electrical length of each shunt arm in degrees. Default is 90.
        Pozar notes the junction discontinuities may require lengthening
        the shunt arms by 10 to 20 degrees.
    feed_elec_length_deg : float, optional
        Electrical length of each feed line in degrees. The feed lines move
        the ports away from the junctions. Default is 45.

    Attributes
    ----------
    series_arm_width_mm : float
        Width of the ``Z0 / sqrt(2)`` series arms in millimeters.
    series_arm_length_mm : float
        Centreline length of the series arms in millimeters.
    shunt_arm_width_mm : float
        Width of the ``Z0`` shunt arms in millimeters.
    shunt_arm_length_mm : float
        Centreline length of the shunt arms in millimeters.
    feed_line_width_mm : float
        Width of the ``Z0`` feed lines in millimeters.
    feed_line_length_mm : float
        Length of each feed line in millimeters.
    substrate_length_mm : float
        Substrate extent along x, from the port 1 edge to the port 2 edge.
    substrate_width_mm : float
        Substrate extent along y, including a 6 x thickness edge margin.

    Raises
    ------
    ValueError
        If any trace width falls below ``min_trace_width_mm``, or if the gap
        between the two series arms falls below ``min_trace_spacing_mm``.
    """

    min_freq: float
    max_freq: float
    centre_freq: float

    series_arm_elec_length_deg: float = 90
    shunt_arm_elec_length_deg: float = 90
    feed_elec_length_deg: float = 45
    series_arm_width_mm: float = field(init=False)
    series_arm_length_mm: float = field(init=False)
    shunt_arm_width_mm: float = field(init=False)
    shunt_arm_length_mm: float = field(init=False)
    feed_line_width_mm: float = field(init=False)
    feed_line_length_mm: float = field(init=False)

    @property
    def freq_range(self) -> tuple[float, float]:
        """
        Return the simulation frequency range.

        Returns
        -------
        tuple[float, float]
            ``(min_freq, max_freq)`` in Hz.
        """
        return self.min_freq, self.max_freq

    @property
    def main_freq(self) -> float:
        """
        Return the primary frequency of interest for analysis.

        Returns
        -------
        float
            The design frequency of the coupler.
        """
        return self.centre_freq

    @property
    def substrate_length_mm(self) -> float:
        """
        Return the substrate length along x, which the feed lines span.

        The ports sit on the feed line ends, so the board stops there and
        the connectors land on its edges.

        Returns
        -------
        float
            Substrate length in mm.
        """
        return (
            self.series_arm_length_mm
            + self.shunt_arm_width_mm
            + 2 * self.feed_line_length_mm
        )

    @property
    def substrate_width_mm(self) -> float:
        """
        Return the substrate width along y including the board-edge margin.

        The board extends 6 substrate thicknesses beyond the outer edge of
        each series arm, far enough to contain the fringing fields.

        Returns
        -------
        float
            Substrate width in mm.
        """
        return (
            self.shunt_arm_length_mm
            + self.series_arm_width_mm
            + 12 * self.substrate_thickness_mm
        )

    @property
    def _default_simulation_box(self) -> NDArray:
        """
        Return the simulation box the manual mesh uses when none is defined.

        Returns
        -------
        NDArray
            Array of shape (3,) with [x, y, z] dimensions in mm.
        """
        return self._create_simulation_box(
            self.substrate_length_mm + self.lambda0,
            self.substrate_width_mm + self.lambda0,
            self.lambda0 * 2,
        )

    def __post_init__(self) -> None:
        """Perform geometric calculations after dataclass initialisation."""
        super().__post_init__()
        self._compute_geometry()

    def _compute_geometry(self) -> None:
        """
        Compute the width and length of every arm and feed line.

        Takes the arm impedances from ``branch_line_hybrid_impedances``,
        converts each impedance to a microstrip width, and derives each
        length from its electrical length at the effective permittivity
        that width implies. Rounds all outputs to ``fp_precision`` decimal
        places.

        Raises
        ------
        ValueError
            If any computed width is below ``min_trace_width_mm``, or if the
            gap between the series arms is below ``min_trace_spacing_mm``.

        Returns
        -------
        None
        """
        series_arm_imp, shunt_arm_imp = branch_line_hybrid_impedances(self.charac_imp)

        self.series_arm_width_mm, series_arm_er_eff = microstrip_width_from_impedance(
            series_arm_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )
        self.shunt_arm_width_mm, shunt_arm_er_eff = microstrip_width_from_impedance(
            shunt_arm_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )
        self.feed_line_width_mm, feed_line_er_eff = microstrip_width_from_impedance(
            self.charac_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )

        for name, width in [
            ("Series arm", self.series_arm_width_mm),
            ("Shunt arm", self.shunt_arm_width_mm),
            ("Feed line", self.feed_line_width_mm),
        ]:
            if width < self.min_trace_width_mm:
                raise ValueError(
                    f"{name} width {width:.4f} mm "
                    f"< minimum trace width {self.min_trace_width_mm} mm"
                )

        self.series_arm_length_mm = calculate_electrical_length_mm(
            self.series_arm_elec_length_deg,
            series_arm_er_eff,
            self.main_freq,
        )
        self.shunt_arm_length_mm = calculate_electrical_length_mm(
            self.shunt_arm_elec_length_deg,
            shunt_arm_er_eff,
            self.main_freq,
        )
        self.feed_line_length_mm = calculate_electrical_length_mm(
            self.feed_elec_length_deg,
            feed_line_er_eff,
            self.main_freq,
        )

        series_arm_gap_mm = self.shunt_arm_length_mm - self.series_arm_width_mm
        if series_arm_gap_mm < self.min_trace_spacing_mm:
            raise ValueError(
                f"Gap between series arms {series_arm_gap_mm:.4f} mm "
                f"< minimum trace spacing {self.min_trace_spacing_mm} mm"
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
            "series_arm_width_mm",
            "series_arm_length_mm",
            "shunt_arm_width_mm",
            "shunt_arm_length_mm",
            "feed_line_width_mm",
            "feed_line_length_mm",
            "lambda0",
            "FDTD_mesh_resolution",
            "FDTD_metal_mesh_resolution",
            "FDTD_thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class QuadratureBranchLineHybridCoupler(SimTools):
    """
    A class for modeling and simulating a 3 dB quadrature branch-line hybrid.
    This class extends SimTools and provides methods for creating the
    substrate, ground plane, branch arms, feed lines, ports, and mesh for
    electromagnetic simulation using openEMS.

    The coupler is centred on the origin. The series arms run along x at
    ``y = +/- shunt_arm_length_mm / 2`` and the shunt arms run along y at
    ``x = +/- series_arm_length_mm / 2``.

    Parameters
    ----------
    params : QuadratureBranchLineHybridCouplerParams
        Data container containing geometric and material properties.
    sim : SimSetup
        Named tuple containing the CSXCAD geometry and openEMS FDTD engine.
    """

    def __init__(
        self,
        params: QuadratureBranchLineHybridCouplerParams,
        sim: SimSetup,
    ) -> None:
        """Initialise the coupler with parameters and simulation objects."""
        self.params = params
        self.CSX = sim.CSX
        self.FDTD = sim.FDTD

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
            -self.params.substrate_length_mm / 2,
            -self.params.substrate_width_mm / 2,
            0,
        ]
        substrate_stop = [
            self.params.substrate_length_mm / 2,
            self.params.substrate_width_mm / 2,
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
            -self.params.substrate_length_mm / 2,
            -self.params.substrate_width_mm / 2,
            0,
        ]
        ground_stop = [
            self.params.substrate_length_mm / 2,
            self.params.substrate_width_mm / 2,
            -self.params.copper_thickness_mm,
        ]
        ground.AddBox(priority=2, start=ground_start, stop=ground_stop)

    def create_branch_arms(self) -> None:
        """
        Create the four quarter-wave branch arms.

        Adds the two series arms along x, which span the full outer width
        of the square and cover the corner junctions, and the two shunt arms
        along y, which fill the gap between the series arms. Registers the
        metal edges with the FDTD grid.

        Returns
        -------
        None
        """
        branch_arms = self.CSX.AddMetal("branch_arms")
        branch_arms.SetColor("#B87333", 255)
        trace_bottom = self.params.substrate_thickness_mm
        trace_top = self.params.substrate_thickness_mm + self.params.copper_thickness_mm
        outer_half_length = (
            self.params.series_arm_length_mm + self.params.shunt_arm_width_mm
        ) / 2

        for side in (1, -1):
            series_arm_centre_y = side * self.params.shunt_arm_length_mm / 2
            series_arm_start = [
                -outer_half_length,
                series_arm_centre_y - self.params.series_arm_width_mm / 2,
                trace_bottom,
            ]
            series_arm_stop = [
                outer_half_length,
                series_arm_centre_y + self.params.series_arm_width_mm / 2,
                trace_top,
            ]
            branch_arms.AddBox(priority=6, start=series_arm_start, stop=series_arm_stop)

            shunt_arm_centre_x = side * self.params.series_arm_length_mm / 2
            shunt_arm_half_length = (
                self.params.shunt_arm_length_mm - self.params.series_arm_width_mm
            ) / 2
            shunt_arm_start = [
                shunt_arm_centre_x - self.params.shunt_arm_width_mm / 2,
                -shunt_arm_half_length,
                trace_bottom,
            ]
            shunt_arm_stop = [
                shunt_arm_centre_x + self.params.shunt_arm_width_mm / 2,
                shunt_arm_half_length,
                trace_top,
            ]
            branch_arms.AddBox(priority=6, start=shunt_arm_start, stop=shunt_arm_stop)

        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=branch_arms,
            metal_edge_res=self.params.FDTD_metal_mesh_resolution,
        )

    def create_feed_lines(self) -> None:
        """
        Create the four feed lines that connect the corners to the ports.

        Each feed line runs along x from a corner junction out to the
        substrate edge, on the centreline of its series arm.

        Returns
        -------
        None
        """
        feed_lines = self.CSX.AddMetal("feed_lines")
        feed_lines.SetColor("#B87333", 255)
        inner_x = (
            self.params.series_arm_length_mm + self.params.shunt_arm_width_mm
        ) / 2
        outer_x = self.params.substrate_length_mm / 2

        for side_x in (1, -1):
            for side_y in (1, -1):
                feed_centre_y = side_y * self.params.shunt_arm_length_mm / 2
                feed_start = [
                    side_x * inner_x,
                    feed_centre_y - self.params.feed_line_width_mm / 2,
                    self.params.substrate_thickness_mm,
                ]
                feed_stop = [
                    side_x * outer_x,
                    feed_centre_y + self.params.feed_line_width_mm / 2,
                    self.params.substrate_thickness_mm
                    + self.params.copper_thickness_mm,
                ]
                feed_lines.AddBox(priority=6, start=feed_start, stop=feed_stop)

        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=feed_lines,
            metal_edge_res=self.params.FDTD_metal_mesh_resolution,
        )

    def create_ports(self) -> list[LumpedPort]:
        """
        Define the lumped ports at the outer ends of the four feed lines.

        Port 1 (input, excited) sits top left, port 2 (through) top right,
        port 3 (coupled) bottom right and port 4 (isolated) bottom left, as
        in Pozar Figure 7.21.

        Returns
        -------
        list[LumpedPort]
            ``[port_1, port_2, port_3, port_4]``, used to retrieve
            S-parameter and impedance results after the simulation.
        """
        port_x = self.params.substrate_length_mm / 2
        port_y = self.params.shunt_arm_length_mm / 2
        port_positions = [
            (-port_x, port_y),
            (port_x, port_y),
            (port_x, -port_y),
            (-port_x, -port_y),
        ]

        ports = []
        for port_number, (position_x, position_y) in enumerate(port_positions, 1):
            port_start = [
                position_x,
                position_y - self.params.feed_line_width_mm / 2,
                0,
            ]
            port_stop = [
                position_x,
                position_y + self.params.feed_line_width_mm / 2,
                self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
            ]
            port = self.FDTD.AddLumpedPort(
                port_number,
                self.params.charac_imp,
                port_start,
                port_stop,
                "z",
                excite=1 if port_number == 1 else 0,
                priority=6,
                edges2grid="x",
            )
            ports.append(port)

        return ports

    def build_quadrature_branch_line_hybrid_coupler(self) -> list[LumpedPort]:
        """
        Construct the complete branch-line hybrid coupler geometry.

        This method orchestrates the creation of all coupler components
        including the substrate, ground plane, branch arms, feed lines,
        and ports.

        Returns
        -------
        list[LumpedPort]
            ``[port_1, port_2, port_3, port_4]``, for S-parameter
            extraction.
        """
        self.create_substrate()
        self.create_ground()
        self.create_branch_arms()
        self.create_feed_lines()
        ports = self.create_ports()
        return ports

    def create_mesh(
        self,
        manual_mesh: bool = False,
        smooth_ratio: float = 1.5,
    ) -> None:
        """
        Generate an FDTD mesh for the coupler simulation domain.
        This method defines mesh lines for the x, y, and z directions
        based on the coupler geometry, substrate thickness, and
        simulation box size. It applies the "thirds rule" outside every
        metal edge and uses SmoothMeshLines to ensure a stable grid
        transition.

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

            simulation_bounds = self.params.simulation_bounds
            if simulation_bounds is None:
                simulation_box = self.params._default_simulation_box
                simulation_bounds = [
                    [-simulation_box[0] / 2, simulation_box[0] / 2],
                    [-simulation_box[1] / 2, simulation_box[1] / 2],
                    [-simulation_box[2] / 3, simulation_box[2] * 2 / 3],
                ]
            mesh.AddLine("x", list(simulation_bounds[0]))
            mesh.AddLine("y", list(simulation_bounds[1]))
            mesh.AddLine("z", list(simulation_bounds[2]))
            # Add mesh lines for substrate
            mesh.AddLine(
                "x",
                [
                    -self.params.substrate_length_mm / 2,
                    self.params.substrate_length_mm / 2,
                ],
            )
            mesh.AddLine(
                "y",
                [
                    -self.params.substrate_width_mm / 2,
                    self.params.substrate_width_mm / 2,
                ],
            )
            # Add thirds-rule lines outside the shunt arm edges
            thirds_rule = self.params.FDTD_thirds_rule
            shunt_arm_outer_x = (
                self.params.series_arm_length_mm + self.params.shunt_arm_width_mm
            ) / 2 + thirds_rule
            shunt_arm_inner_x = (
                self.params.series_arm_length_mm - self.params.shunt_arm_width_mm
            ) / 2 - thirds_rule
            mesh.AddLine(
                "x",
                np.concatenate(
                    [
                        -shunt_arm_outer_x,
                        -shunt_arm_inner_x,
                        shunt_arm_inner_x,
                        shunt_arm_outer_x,
                    ]
                ),
            )
            # Add thirds-rule lines outside the series arm and feed line edges
            series_arm_outer_y = (
                self.params.shunt_arm_length_mm + self.params.series_arm_width_mm
            ) / 2 + thirds_rule
            series_arm_inner_y = (
                self.params.shunt_arm_length_mm - self.params.series_arm_width_mm
            ) / 2 - thirds_rule
            feed_line_outer_y = (
                self.params.shunt_arm_length_mm + self.params.feed_line_width_mm
            ) / 2 + thirds_rule
            feed_line_inner_y = (
                self.params.shunt_arm_length_mm - self.params.feed_line_width_mm
            ) / 2 - thirds_rule
            mesh.AddLine(
                "y",
                np.concatenate(
                    [
                        -series_arm_outer_y,
                        -series_arm_inner_y,
                        series_arm_inner_y,
                        series_arm_outer_y,
                        -feed_line_outer_y,
                        -feed_line_inner_y,
                        feed_line_inner_y,
                        feed_line_outer_y,
                    ]
                ),
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

            mesh.SmoothMeshLines("all", self.params.FDTD_mesh_resolution, smooth_ratio)
        else:
            Mesh(self.CSX, self.params)
