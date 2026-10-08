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
Printed quarter-wave monopole antenna design and simulation.

Provides `PrintedMonopoleAntennaParams`, which derives the monopole
dimensions from a target resonant frequency, and `PrintedMonopoleAntenna`,
which builds the microstrip feed, the printed radiating strip, the partial
ground plane and the feed port in openEMS.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from openEMS.physical_constants import C0
from openEMS.ports import LumpedPort

from .calc import (
    dipole_resonant_length_mm,
    microstrip_width_from_impedance,
    ungrounded_strip_eps_eff,
)
from .sim_params import SimParams
from .sim_tools import SimTools, SimSetup, m_to_mm
from .fdtd_mesh import Mesh

# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "PrintedMonopoleAntennaParams",
    "PrintedMonopoleAntenna",
]


@dataclass
class PrintedMonopoleAntennaParams(SimParams):
    """
    Parameters for a microstrip-fed printed quarter-wave monopole.

    This class extends SimParams and computes the derived geometry of a
    printed strip monopole fed by a ``Z0`` microstrip line over a partial
    ground plane. By image theory the ground plane stands in for the
    missing half of a half-wave dipole, so the radiating strip is half the
    length of the equivalent printed dipole and presents about half its
    impedance (Balanis, *Antenna Theory* (3rd ed.), section 4.7.4,
    eq. (4-106)).

    The board runs along x, so the xz-plane (phi = 0) is an E-plane of the
    monopole and the yz-plane (phi = 90) its omnidirectional H-plane. The
    ground plane covers the bottom face for ``-ground_length_mm <= x <= 0``;
    the feed line runs on the top face over it, from the port on the board
    edge at ``x = -ground_length_mm`` to the ground edge at ``x = 0``, where
    the radiating strip continues to its tip at ``x = monopole_length_mm``.
    The structure is centred on ``y = 0``.

    Parameters
    ----------
    min_freq : float
        Minimum frequency of the simulation range in Hz.
    max_freq : float
        Maximum frequency of the simulation range in Hz.
    resonant_freq : float
        Design frequency of the monopole in Hz.
    length_wavelengths : float, optional
        Length of the radiating strip, from the ground edge to its tip, as
        a fraction of the guided wavelength. Default is 0.235, half the
        0.47 first guess for a resonant dipole.
    radiator_width_mm : float, optional
        Width of the radiating strip in millimeters. Together with the
        substrate it sets ``eps_eff``. Default is ``None``, which uses the
        feed line width.
    ground_length_wavelengths : float, optional
        Length of the ground plane along x as a fraction of the free-space
        wavelength. Default is 0.25.
    ground_width_wavelengths : float, optional
        Width of the ground plane and board along y as a fraction of the
        free-space wavelength. Default is 0.4.

    Attributes
    ----------
    feed_line_width_mm : float
        Width of the ``Z0`` feed line in millimeters.
    eps_eff : float
        Effective relative permittivity of the radiating strip.
    guided_wavelength_mm : float
        Guided wavelength along the radiating strip at ``resonant_freq``
        in millimeters.
    monopole_length_mm : float
        Length of the radiating strip beyond the ground edge in
        millimeters.
    ground_length_mm : float
        Length of the ground plane, and of the feed line over it, in
        millimeters.
    ground_width_mm : float
        Width of the ground plane and the board in millimeters.
    substrate_length_mm : float
        Substrate extent along x, including a 6 x thickness margin beyond
        the monopole tip.
    substrate_width_mm : float
        Substrate extent along y; equal to ``ground_width_mm``.

    Raises
    ------
    ValueError
        If the feed line or radiator width falls below
        ``min_trace_width_mm``, or if the radiator is wider than the board.
    """

    min_freq: float
    max_freq: float
    resonant_freq: float

    length_wavelengths: float = 0.235
    radiator_width_mm: float | None = None
    ground_length_wavelengths: float = 0.25
    ground_width_wavelengths: float = 0.4
    feed_line_width_mm: float = field(init=False)
    eps_eff: float = field(init=False)
    guided_wavelength_mm: float = field(init=False)
    monopole_length_mm: float = field(init=False)
    ground_length_mm: float = field(init=False)
    ground_width_mm: float = field(init=False)

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
            The resonant frequency of the monopole.
        """
        return self.resonant_freq

    @property
    def substrate_length_mm(self) -> float:
        """
        Return the substrate length along x including the board-edge margin.

        The board starts at the port edge under the ground plane and extends
        6 substrate thicknesses beyond the monopole tip.

        Returns
        -------
        float
            Substrate length in mm.
        """
        return (
            self.ground_length_mm
            + self.monopole_length_mm
            + 6 * self.substrate_thickness_mm
        )

    @property
    def substrate_width_mm(self) -> float:
        """
        Return the substrate width along y, which the ground plane spans.

        Returns
        -------
        float
            Substrate width in mm.
        """
        return self.ground_width_mm

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
        Compute the feed line, radiating strip and ground plane dimensions.

        Takes the feed line width from ``microstrip_width_from_impedance``,
        the effective permittivity of the radiating strip from
        ``ungrounded_strip_eps_eff`` at its own width, and the strip length from
        ``dipole_resonant_length_mm``. Scales the ground plane from the
        free-space wavelength. Rounds all outputs to ``fp_precision``
        decimal places.

        Raises
        ------
        ValueError
            If the feed line or radiator width is below
            ``min_trace_width_mm``, or the radiator is wider than the board.

        Returns
        -------
        None
        """
        self.feed_line_width_mm, _ = microstrip_width_from_impedance(
            self.charac_imp,
            self.substrate_thickness_mm,
            self.copper_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )
        if self.radiator_width_mm is None:
            self.radiator_width_mm = self.feed_line_width_mm

        for name, width in [
            ("Feed line", self.feed_line_width_mm),
            ("Radiator", self.radiator_width_mm),
        ]:
            if width < self.min_trace_width_mm:
                raise ValueError(
                    f"{name} width {width:.4f} mm "
                    f"< minimum trace width {self.min_trace_width_mm} mm"
                )

        self.eps_eff = ungrounded_strip_eps_eff(
            self.radiator_width_mm,
            self.substrate_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )
        self.guided_wavelength_mm = m_to_mm(
            C0 / (self.main_freq * np.sqrt(self.eps_eff))
        )
        self.monopole_length_mm = dipole_resonant_length_mm(
            self.main_freq, self.eps_eff, self.length_wavelengths
        )

        free_space_wavelength_mm = m_to_mm(C0 / self.main_freq)
        self.ground_length_mm = (
            self.ground_length_wavelengths * free_space_wavelength_mm
        )
        self.ground_width_mm = self.ground_width_wavelengths * free_space_wavelength_mm

        if self.radiator_width_mm >= self.ground_width_mm:
            raise ValueError(
                f"Radiator width {self.radiator_width_mm:.4f} mm "
                f">= board width {self.ground_width_mm:.4f} mm"
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
            "feed_line_width_mm",
            "radiator_width_mm",
            "eps_eff",
            "guided_wavelength_mm",
            "monopole_length_mm",
            "ground_length_mm",
            "ground_width_mm",
            "lambda0",
            "FDTD_mesh_resolution",
            "FDTD_metal_mesh_resolution",
            "FDTD_thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class PrintedMonopoleAntenna(SimTools):
    """
    A class for modeling and simulating a microstrip-fed printed monopole.

    This class extends SimTools and provides methods for creating the
    substrate, partial ground plane, feed line, radiating strip, feed port
    and mesh for electromagnetic simulation using openEMS.

    The board runs along x and is centred on ``y = 0``. The ground plane
    edge, and the feed point of the monopole, sit at ``x = 0``.

    Parameters
    ----------
    params : PrintedMonopoleAntennaParams
        Data container containing geometric and material properties.
    sim : SimSetup
        Named tuple containing the CSXCAD geometry and openEMS FDTD engine.
    """

    def __init__(
        self,
        params: PrintedMonopoleAntennaParams,
        sim: SimSetup,
    ) -> None:
        """Initialise the monopole with parameters and simulation objects."""
        self.params = params
        self.CSX = sim.CSX
        self.FDTD = sim.FDTD

    def create_substrate(self) -> None:
        """
        Define and add the dielectric substrate to the simulation.

        This method creates a material using the permittivity and
        loss tangent (kappa) defined in self.params, colors it green,
        and adds a box geometry from the port edge to beyond the monopole
        tip.

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
            -self.params.ground_length_mm,
            -self.params.substrate_width_mm / 2,
            0,
        ]
        substrate_stop = [
            self.params.substrate_length_mm - self.params.ground_length_mm,
            self.params.substrate_width_mm / 2,
            self.params.substrate_thickness_mm,
        ]
        substrate.AddBox(priority=0, start=substrate_start, stop=substrate_stop)

    def create_ground(self) -> None:
        """
        Define and add the partial copper ground plane to the geometry.

        Adds a metallic box (PEC) below the substrate that spans the board
        width and runs from the port edge to ``x = 0``, leaving the board
        under the radiating strip clear.

        Returns
        -------
        None
        """
        ground = self.CSX.AddMetal("ground")
        ground.SetColor("#B87333", 255)
        ground_start = [
            -self.params.ground_length_mm,
            -self.params.ground_width_mm / 2,
            0,
        ]
        ground_stop = [
            0,
            self.params.ground_width_mm / 2,
            -self.params.copper_thickness_mm,
        ]
        ground.AddBox(priority=2, start=ground_start, stop=ground_stop)

    def create_feed_line(self) -> None:
        """
        Create the microstrip feed line over the ground plane.

        The line runs along x from the port edge to the ground edge at
        ``x = 0``, centred on ``y = 0``. Registers the metal edges with the
        FDTD grid.

        Returns
        -------
        None
        """
        feed_line = self.CSX.AddMetal("feed_line")
        feed_line.SetColor("#B87333", 255)
        feed_start = [
            -self.params.ground_length_mm,
            -self.params.feed_line_width_mm / 2,
            self.params.substrate_thickness_mm,
        ]
        feed_stop = [
            0,
            self.params.feed_line_width_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        feed_line.AddBox(priority=6, start=feed_start, stop=feed_stop)

        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=feed_line,
            metal_edge_res=self.params.FDTD_metal_mesh_resolution,
        )

    def create_radiator(self) -> None:
        """
        Create the radiating strip of the monopole.

        The strip continues the feed line along x from the ground edge at
        ``x = 0`` to its tip at ``x = monopole_length_mm``, centred on
        ``y = 0``. Registers the metal edges with the FDTD grid.

        Returns
        -------
        None
        """
        radiator = self.CSX.AddMetal("radiator")
        radiator.SetColor("#B87333", 255)
        radiator_start = [
            0,
            -self.params.radiator_width_mm / 2,
            self.params.substrate_thickness_mm,
        ]
        radiator_stop = [
            self.params.monopole_length_mm,
            self.params.radiator_width_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        radiator.AddBox(priority=6, start=radiator_start, stop=radiator_stop)

        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=radiator,
            metal_edge_res=self.params.FDTD_metal_mesh_resolution,
        )

    def create_port(self) -> LumpedPort:
        """
        Define the excitation lumped port at the outer end of the feed line.

        The port sits on the board edge at ``x = -ground_length_mm``, spans
        the feed line width in y, and runs in z from the ground plane to the
        top of the feed line.

        Returns
        -------
        port : openEMS.ports.LumpedPort
            The created lumped port object, used to retrieve S-parameter
            and impedance results after the simulation.
        """
        port_start = [
            -self.params.ground_length_mm,
            -self.params.feed_line_width_mm / 2,
            0,
        ]
        port_stop = [
            -self.params.ground_length_mm,
            self.params.feed_line_width_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_start,
            port_stop,
            "z",
            excite=1,
            priority=6,
            edges2grid="x",
        )
        return port

    def build_printed_monopole_antenna(self) -> LumpedPort:
        """
        Construct the complete printed monopole geometry.

        This method orchestrates the creation of the substrate, the partial
        ground plane, the feed line, the radiating strip and the feed port.

        Returns
        -------
        port : openEMS.ports.LumpedPort
            The feed port, for S-parameter extraction.
        """
        self.create_substrate()
        self.create_ground()
        self.create_feed_line()
        self.create_radiator()
        port = self.create_port()
        return port

    def create_mesh(
        self,
        manual_mesh: bool = False,
        smooth_ratio: float = 1.5,
    ) -> None:
        """
        Generate an FDTD mesh for the monopole simulation domain.

        This method defines mesh lines for the x, y, and z directions
        based on the monopole geometry, substrate thickness, and simulation
        box size. It applies the "thirds rule" outside the trace edges, the
        ground edge and the monopole tip, and uses SmoothMeshLines to ensure
        a stable grid transition.

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

            substrate_start_x = -self.params.ground_length_mm
            substrate_stop_x = self.params.substrate_length_mm + substrate_start_x
            simulation_bounds = self.params.simulation_bounds
            if simulation_bounds is None:
                simulation_box = self.params._default_simulation_box
                board_centre_x = (substrate_start_x + substrate_stop_x) / 2
                simulation_bounds = [
                    [
                        board_centre_x - simulation_box[0] / 2,
                        board_centre_x + simulation_box[0] / 2,
                    ],
                    [-simulation_box[1] / 2, simulation_box[1] / 2],
                    [-simulation_box[2] / 2, simulation_box[2] / 2],
                ]
            mesh.AddLine("x", list(simulation_bounds[0]))
            mesh.AddLine("y", list(simulation_bounds[1]))
            mesh.AddLine("z", list(simulation_bounds[2]))
            # Add mesh lines for substrate
            mesh.AddLine("x", [substrate_start_x, substrate_stop_x])
            mesh.AddLine(
                "y",
                [
                    -self.params.substrate_width_mm / 2,
                    self.params.substrate_width_mm / 2,
                ],
            )
            # Add thirds-rule lines outside the feed line and radiator edges
            thirds_rule = self.params.FDTD_thirds_rule
            feed_line_edge_y = self.params.feed_line_width_mm / 2 + thirds_rule
            radiator_edge_y = self.params.radiator_width_mm / 2 + thirds_rule
            mesh.AddLine(
                "y",
                np.concatenate(
                    [
                        -feed_line_edge_y,
                        feed_line_edge_y,
                        -radiator_edge_y,
                        radiator_edge_y,
                    ]
                ),
            )
            # Add thirds-rule lines outside the ground edge and the monopole tip
            mesh.AddLine(
                "x",
                np.concatenate(
                    [
                        thirds_rule,
                        self.params.monopole_length_mm + thirds_rule,
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
