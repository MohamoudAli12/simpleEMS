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
Printed half-wave dipole antenna design and simulation.

Provides `PrintedDipoleAntennaParams`, which derives the dipole dimensions
from a target resonant frequency, and `PrintedDipoleAntenna`, which builds
the two printed arms, the substrate and a lumped port across the centre
feed gap in openEMS.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from openEMS.physical_constants import C0
from openEMS.ports import LumpedPort

from .calc import dipole_resonant_length_mm, ungrounded_strip_eps_eff
from .sim_params import SimParams
from .sim_tools import SimTools, SimSetup, m_to_mm
from .fdtd_mesh import Mesh

# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "PrintedDipoleAntennaParams",
    "PrintedDipoleAntenna",
]


@dataclass
class PrintedDipoleAntennaParams(SimParams):
    """
    Parameters for a centre-fed printed half-wave dipole.

    This class extends SimParams and computes the derived geometry of two
    collinear strip arms printed on one side of an ungrounded substrate,
    following R. Zingg, *Printed Dipole Antenna*, section I. The arm width
    and substrate set the effective permittivity ``eps_eff`` of the arms,
    which sets the propagation speed ``v = c / sqrt(eps_eff)`` along them
    and therefore the guided wavelength ``v / f`` the dipole is cut to.

    The dipole runs along x, centred on the origin. A lumped port bridges
    the feed gap between the inner ends of the arms and drives it directly,
    so the model needs no balun. A resonant half-wave dipole presents about
    70 ohms, so set ``charac_imp`` to 70-75 ohms for a matched port.

    Parameters
    ----------
    min_freq : float
        Minimum frequency of the simulation range in Hz.
    max_freq : float
        Maximum frequency of the simulation range in Hz.
    resonant_freq : float
        Design frequency of the dipole in Hz.
    length_wavelengths : float, optional
        Total tip-to-tip length as a fraction of the guided wavelength.
        Default is 0.47.
    arm_width_mm : float, optional
        Width of each arm in millimeters. Together with the substrate it
        sets ``eps_eff``. Default is 1.0.
    feed_gap_wavelengths : float, optional
        Feed gap between the arms as a fraction of the guided wavelength.
        Default is 0.01.

    Attributes
    ----------
    eps_eff : float
        Effective relative permittivity of the arms.
    guided_wavelength_mm : float
        Guided wavelength at ``resonant_freq`` in millimeters.
    dipole_length_mm : float
        Total tip-to-tip length of the dipole in millimeters.
    arm_length_mm : float
        Length of each arm, from the feed gap to its tip, in millimeters.
    feed_gap_mm : float
        Gap between the inner ends of the arms in millimeters.
    substrate_length_mm : float
        Substrate extent along x, including a 6 x thickness edge margin.
    substrate_width_mm : float
        Substrate extent along y, including a 6 x thickness edge margin.

    Raises
    ------
    ValueError
        If the arm width falls below ``min_trace_width_mm``, the feed gap
        falls below ``min_trace_spacing_mm``, or the feed gap leaves no room
        for the arms.
    """

    min_freq: float
    max_freq: float
    resonant_freq: float

    length_wavelengths: float = 0.47
    arm_width_mm: float = 1.0
    feed_gap_wavelengths: float = 0.01
    eps_eff: float = field(init=False)
    guided_wavelength_mm: float = field(init=False)
    dipole_length_mm: float = field(init=False)
    arm_length_mm: float = field(init=False)
    feed_gap_mm: float = field(init=False)

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
            The resonant frequency of the dipole.
        """
        return self.resonant_freq

    @property
    def substrate_length_mm(self) -> float:
        """
        Return the substrate length along x including the board-edge margin.

        The board extends 6 substrate thicknesses beyond each arm tip.

        Returns
        -------
        float
            Substrate length in mm.
        """
        return self.dipole_length_mm + 12 * self.substrate_thickness_mm

    @property
    def substrate_width_mm(self) -> float:
        """
        Return the substrate width along y including the board-edge margin.

        The board extends 6 substrate thicknesses beyond each arm edge.

        Returns
        -------
        float
            Substrate width in mm.
        """
        return self.arm_width_mm + 12 * self.substrate_thickness_mm

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
        Compute the dipole length, arm length, arm width and feed gap.

        Takes the effective permittivity of the arms from
        ``ungrounded_strip_eps_eff``, the tip-to-tip length from
        ``dipole_resonant_length_mm`` (eqs. I-1 and I-2), and scales the
        feed gap from the guided wavelength.
        Rounds all outputs to ``fp_precision`` decimal places.

        Raises
        ------
        ValueError
            If the arm width is below ``min_trace_width_mm``, the feed gap
            is below ``min_trace_spacing_mm``, or the arms have no length.

        Returns
        -------
        None
        """
        self.eps_eff = ungrounded_strip_eps_eff(
            self.arm_width_mm,
            self.substrate_thickness_mm,
            self.substrate_eps_r,
            self.main_freq,
        )
        self.guided_wavelength_mm = m_to_mm(
            C0 / (self.main_freq * np.sqrt(self.eps_eff))
        )

        self.dipole_length_mm = dipole_resonant_length_mm(
            self.main_freq, self.eps_eff, self.length_wavelengths
        )
        self.feed_gap_mm = self.feed_gap_wavelengths * self.guided_wavelength_mm
        self.arm_length_mm = (self.dipole_length_mm - self.feed_gap_mm) / 2

        for name, value, minimum in [
            ("Arm width", self.arm_width_mm, self.min_trace_width_mm),
            ("Feed gap", self.feed_gap_mm, self.min_trace_spacing_mm),
        ]:
            if value < minimum:
                raise ValueError(f"{name} {value:.4f} mm < minimum {minimum} mm")

        if self.arm_length_mm <= 0:
            raise ValueError(
                f"Feed gap {self.feed_gap_mm:.4f} mm leaves no room for the arms "
                f"of a {self.dipole_length_mm:.4f} mm dipole"
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
            "eps_eff",
            "guided_wavelength_mm",
            "dipole_length_mm",
            "arm_length_mm",
            "feed_gap_mm",
            "lambda0",
            "FDTD_mesh_resolution",
            "FDTD_metal_mesh_resolution",
            "FDTD_thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class PrintedDipoleAntenna(SimTools):
    """
    A class for modeling and simulating a centre-fed printed dipole.

    This class extends SimTools and provides methods for creating the
    substrate, dipole arms, feed port and mesh for electromagnetic
    simulation using openEMS.

    The dipole is centred on the origin and runs along x. The arms sit on
    top of the substrate, with no ground plane beneath them.

    Parameters
    ----------
    params : PrintedDipoleAntennaParams
        Data container containing geometric and material properties.
    sim : SimSetup
        Named tuple containing the CSXCAD geometry and openEMS FDTD engine.
    """

    def __init__(
        self,
        params: PrintedDipoleAntennaParams,
        sim: SimSetup,
    ) -> None:
        """Initialise the dipole with parameters and simulation objects."""
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

    def create_arms(self) -> None:
        """
        Create the two dipole arms.

        Each arm runs along x from the edge of the feed gap out to its tip,
        centred on ``y = 0``. Registers the metal edges with the FDTD grid.

        Returns
        -------
        None
        """
        arms = self.CSX.AddMetal("dipole_arms")
        arms.SetColor("#B87333", 255)
        inner_x = self.params.feed_gap_mm / 2
        outer_x = self.params.dipole_length_mm / 2

        for side in (1, -1):
            arm_start = [
                side * inner_x,
                -self.params.arm_width_mm / 2,
                self.params.substrate_thickness_mm,
            ]
            arm_stop = [
                side * outer_x,
                self.params.arm_width_mm / 2,
                self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
            ]
            arms.AddBox(priority=6, start=arm_start, stop=arm_stop)

        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=arms,
            metal_edge_res=self.params.FDTD_metal_mesh_resolution,
        )

    def create_port(self) -> LumpedPort:
        """
        Define the excitation lumped port across the feed gap.

        The port fills the gap between the inner ends of the arms, spans the
        arm width in y and the copper thickness in z, and drives current
        along x from one arm into the other.

        Returns
        -------
        port : openEMS.ports.LumpedPort
            The created lumped port object, used to retrieve S-parameter
            and impedance results after the simulation.
        """
        port_start = [
            -self.params.feed_gap_mm / 2,
            -self.params.arm_width_mm / 2,
            self.params.substrate_thickness_mm,
        ]
        port_stop = [
            self.params.feed_gap_mm / 2,
            self.params.arm_width_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_start,
            port_stop,
            "x",
            excite=1,
            priority=6,
            edges2grid="x",
        )
        return port

    def build_printed_dipole_antenna(self) -> LumpedPort:
        """
        Construct the complete printed dipole geometry.

        This method orchestrates the creation of the substrate, the two
        dipole arms and the feed port.

        Returns
        -------
        port : openEMS.ports.LumpedPort
            The feed port, for S-parameter extraction.
        """
        self.create_substrate()
        self.create_arms()
        port = self.create_port()
        return port

    def create_mesh(
        self,
        manual_mesh: bool = False,
        smooth_ratio: float = 1.5,
    ) -> None:
        """
        Generate an FDTD mesh for the dipole simulation domain.

        This method defines mesh lines for the x, y, and z directions
        based on the dipole geometry, substrate thickness, and simulation
        box size. It applies the "thirds rule" outside the arm tips and
        edges, grids the feed gap evenly, and uses SmoothMeshLines to
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

            simulation_bounds = self.params.simulation_bounds
            if simulation_bounds is None:
                simulation_box = self.params._default_simulation_box
                simulation_bounds = [
                    [-simulation_box[0] / 2, simulation_box[0] / 2],
                    [-simulation_box[1] / 2, simulation_box[1] / 2],
                    [-simulation_box[2] / 2, simulation_box[2] / 2],
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
            # Add thirds-rule lines outside the arm tips and grid the feed gap
            thirds_rule = self.params.FDTD_thirds_rule
            arm_tip_x = self.params.dipole_length_mm / 2 + thirds_rule
            mesh.AddLine(
                "x",
                np.concatenate(
                    [
                        -arm_tip_x,
                        arm_tip_x,
                        np.linspace(
                            -self.params.feed_gap_mm / 2,
                            self.params.feed_gap_mm / 2,
                            5,
                        ),
                    ]
                ),
            )
            # Add thirds-rule lines outside the arm edges
            arm_edge_y = self.params.arm_width_mm / 2 + thirds_rule
            mesh.AddLine("y", np.concatenate([-arm_edge_y, arm_edge_y]))

            mesh.AddLine(
                "z",
                np.linspace(
                    0,
                    self.params.substrate_thickness_mm
                    + self.params.copper_thickness_mm / 2,
                    self.params.substrate_cells,
                ),
            )

            mesh.SmoothMeshLines("all", self.params.FDTD_mesh_resolution, smooth_ratio)
        else:
            Mesh(self.CSX, self.params)
