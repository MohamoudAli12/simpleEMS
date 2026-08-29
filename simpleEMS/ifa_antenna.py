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
Inverted-F antenna (IFA) design and simulation.

Provides `InvertedFAntennaParams`, which derives the antenna dimensions
from a target resonant frequency, and `InvertedFAntenna`, which builds the
printed IFA geometry (shorting leg, radiating tip, feed line, and port)
in openEMS.
"""

from dataclasses import dataclass, field

import numpy as np

from numpy.typing import NDArray
from openEMS.ports import LumpedPort

from openEMS.physical_constants import C0
from .sim_params import SimParams
from .fdtd_mesh import Mesh
from .sim_tools import SimTools, SimSetup, m_to_mm


@dataclass
class InvertedFAntennaParams(SimParams):
    """
    Parameters for a printed inverted-F antenna (IFA).

    Extends `SimParams` and computes the derived geometric parameters
    (radiating tip, shorting leg, and feed line dimensions) required to
    build and simulate a printed IFA.

    Parameters
    ----------
    resonant_freq : float
        Target resonant frequency of the antenna in Hz.
    span_freq : float
        Frequency span used to compute the simulation range around
        the resonant frequency in Hz.

    Attributes
    ----------
    rad_tip_length_mm : float
        Length of the radiating tip (x-direction) in millimeters.
    rad_tip_width_mm : float
        Width of the radiating tip (y-direction) in millimeters.
    short_tip_length_mm : float
        Length of the shorting leg (y-direction) in millimeters.
    short_tip_width_mm : float
        Width of the shorting leg (x-direction) in millimeters; equal to
        `rad_tip_width_mm`.
    feed_spacing_factor : int
        Divisor applied to `rad_tip_length_mm` to derive `feed_spacing_mm`.
    feed_spacing_mm : float
        Distance in x between the shorting leg and the feed arm, in
        millimeters.
    feed_line_length_mm : float
        Length of the feed line, in millimeters; equal to
        `short_tip_length_mm`.
    lambda_eff : float
        Free-space wavelength at `resonant_freq`, in meters.
    rad_tip_total_length_mm : float
        Total arm length (shorting leg, corner, and radiating tip) in
        millimeters, set to a quarter wavelength at `resonant_freq`.
    substrate_width_mm : float
        Substrate width including margin (lambda0 padding).
    substrate_length_mm : float
        Substrate length including margin (lambda0 padding).
    simulation_box : NDArray
        A 1D array of shape (3,) representing the 3D simulation domain
        size: ``[x_size_mm, y_size_mm, z_size_mm]``.

    Notes
    -----
    `short_tip_length_mm`, `rad_tip_width_mm`, and `feed_spacing_mm` are
    fixed starting values; the feed spacing in particular sets the input
    match and is meant to be tuned.

    As a subclass of `SimParams`, this class also accepts all of
    `SimParams`'s constructor arguments (e.g. `substrate_eps_r`,
    `substrate_tand`, `substrate_thickness_mm`, `charac_imp`); see that
    class's docstring for details.
    """

    resonant_freq: float
    span_freq: float
    rad_tip_length_mm: float = field(init=False)
    rad_tip_width_mm: float = field(init=False)
    short_tip_length_mm: float = field(init=False)
    short_tip_width_mm: float = field(init=False)
    feed_spacing_mm: float = field(init=False)
    lambda_eff: float = field(init=False)
    rad_tip_total_length_mm: float = field(init=False)
    feed_spacing_factor: int = field(init=False)
    feed_line_length_mm: float = field(init=False)

    @property
    def freq_range(self) -> tuple[float, float]:
        """
        Compute the simulation frequency range.

        Returns
        -------
        tuple of (float, float)
            A tuple containing (f_min, f_max) calculated as
            (resonant_freq - span_freq, resonant_freq + span_freq).
        """
        return (
            self.resonant_freq - self.span_freq,
            self.resonant_freq + self.span_freq,
        )

    @property
    def main_freq(self) -> float:
        """
        Return the primary frequency of interest for post-processing.

        Returns
        -------
        float
            The resonant frequency of the antenna.
        """
        return self.resonant_freq

    @property
    def substrate_length_mm(self) -> float:
        """
        Return the substrate length including lambda0 padding.

        Returns
        -------
        float
            Substrate length in mm.
        """
        return self.rad_tip_length_mm + 2 * self.lambda0

    @property
    def substrate_width_mm(self) -> float:
        """
        Return the substrate width including lambda0 padding.

        Returns
        -------
        float
            Substrate width in mm.
        """
        return self.rad_tip_length_mm + 2 * self.lambda0

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
        Compute all derived geometric parameters for the IFA.

        Sets the fixed starting values (`rad_tip_width_mm`,
        `short_tip_length_mm`), splits a quarter wavelength at the resonant
        frequency between the shorting leg and the radiating tip, and
        derives `feed_spacing_mm` and `feed_line_length_mm` from those.

        Returns
        -------
        None
        """
        self.lambda_eff = C0 / (self.resonant_freq)
        self.rad_tip_total_length_mm = m_to_mm(self.lambda_eff / 4)
        self.rad_tip_width_mm = 1.0
        self.short_tip_length_mm = 5.0
        self.rad_tip_length_mm = (
            self.rad_tip_total_length_mm
            - self.short_tip_length_mm
            - self.rad_tip_width_mm
        )
        self.feed_spacing_factor = 3
        self.feed_spacing_mm = self.rad_tip_length_mm / self.feed_spacing_factor
        self.short_tip_width_mm = self.rad_tip_width_mm
        self.feed_line_length_mm = self.short_tip_length_mm
        self._round_outputs()

    def _round_outputs(self) -> None:
        """
        Round all geometric parameters to the configured floating-point precision.

        This method iterates over key geometric and mesh attributes and rounds
        them to `self.fp_precision` decimal places for cleaner output and
        consistent simulation behavior.

        Returns
        -------
        None
        """
        for attr in [
            "rad_tip_width_mm",
            "rad_tip_length_mm",
            "short_tip_width_mm",
            "short_tip_length_mm",
            "feed_line_length_mm",
            "feed_spacing_mm",
            "lambda0",
            "FDTD_mesh_resolution",
            "FDTD_metal_mesh_resolution",
            "FDTD_thirds_rule",
        ]:
            setattr(self, attr, np.round(getattr(self, attr), self.fp_precision))


class InvertedFAntenna(SimTools):
    """
    Printed inverted-F antenna (IFA) model.

    Extends `SimTools` and provides methods for creating the substrate,
    ground plane, shorting leg and via, radiating tip, feed line, port, and
    mesh for electromagnetic simulation using openEMS.

    Parameters
    ----------
    params : InvertedFAntennaParams
        Data container containing geometric and material properties.
    sim : SimSetup
        Named tuple containing the CSXCAD geometry and openEMS FDTD engine.

    Notes
    -----
    The antenna sits on the ``y > 0`` half of the substrate; the ground
    plane covers the ``y < 0`` half.
    """

    def __init__(
        self,
        params: InvertedFAntennaParams,
        sim: SimSetup,
    ) -> None:
        self.params = params
        self.CSX = sim.CSX
        self.FDTD = sim.FDTD

    def create_substrate(self) -> None:
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

        Adds a metallic box (PEC) below the substrate, covering the
        ``y < 0`` half of it and leaving the half carrying the antenna
        clear. The ground plane thickness is defined by
        `self.params.copper_thickness_mm`.

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
            0,
            -self.params.copper_thickness_mm,
        ]
        ground.AddBox(priority=2, start=ground_start, stop=ground_stop)

    def create_short_leg(self) -> None:
        """
        Create the shorting leg of the antenna.

        Adds a metal box on top of the substrate running in +y from over the
        ground plane edge to the start of the radiating tip, with width
        `short_tip_width_mm` and length `short_tip_length_mm`.

        Returns
        -------
        None
        """
        short = self.CSX.AddMetal("short_line")
        short.SetColor("#B87333", 255)
        short_start = [
            0,
            -1,
            self.params.substrate_thickness_mm,
        ]
        short_stop = [
            self.params.short_tip_width_mm,
            self.params.short_tip_length_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        short.AddBox(priority=6, start=short_start, stop=short_stop)

    def create_radiating_tip(self) -> None:
        """
        Create the radiating tip of the antenna.

        Adds a metal box on top of the substrate running in +x from the end
        of the shorting leg, with length `rad_tip_length_mm` and width
        `rad_tip_width_mm`. Together with the shorting leg it forms the
        quarter-wave arm.

        Returns
        -------
        None
        """
        rad_tip = self.CSX.AddMetal("rad_line")
        rad_tip.SetColor("#B87333", 255)
        rad_tip_start = [
            0,
            self.params.short_tip_length_mm,
            self.params.substrate_thickness_mm,
        ]
        rad_tip_stop = [
            self.params.rad_tip_length_mm,
            self.params.short_tip_length_mm + self.params.rad_tip_width_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        rad_tip.AddBox(priority=6, start=rad_tip_start, stop=rad_tip_stop)

    def create_short_via(self) -> None:
        """
        Create the via that shorts the antenna arm to ground.

        Adds a metal box through the substrate thickness at the base of the
        shorting leg, connecting it to the ground plane.

        Returns
        -------
        None
        """
        short_via = self.CSX.AddMetal("short_via")
        short_via.SetColor("#B87333", 255)
        via_start = [
            0,
            -1,
            0,
        ]
        via_stop = [
            self.params.short_tip_width_mm,
            0,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        short_via.AddBox(priority=6, start=via_start, stop=via_stop)

    def create_excite_line(self) -> None:
        """
        Create the feed arm that couples the feed line to the antenna.

        Adds a metal box on top of the substrate running in +y parallel to
        the shorting leg, offset from it by `feed_spacing_mm` in x. That
        offset sets the input impedance of the antenna.

        Returns
        -------
        None
        """
        excite_line = self.CSX.AddMetal("excite_line")
        excite_line.SetColor("#B87333", 255)
        excite_start = [
            self.params.feed_spacing_mm,
            0,
            self.params.substrate_thickness_mm,
        ]
        excite_stop = [
            self.params.feed_spacing_mm + self.params.rad_tip_width_mm,
            self.params.feed_line_length_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        excite_line.AddBox(priority=6, start=excite_start, stop=excite_stop)

    def create_port(self) -> LumpedPort:
        """
        Define the excitation lumped port at the end of the feed line.

        The port spans the substrate vertically at the outer end of the feed
        line, from the bottom of the ground plane to the top of the feed
        line metallization.

        Returns
        -------
        port : openEMS.ports.LumpedPort
            The created lumped port object, used to retrieve S-parameter
            and impedance results after the simulation.
        """
        port_start = [
            self.params.feed_spacing_mm,
            0,
            -self.params.copper_thickness_mm,
        ]
        port_stop = [
            self.params.feed_spacing_mm + self.params.rad_tip_width_mm,
            0,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_start,
            port_stop,
            "z",
            1.0,
            priority=6,
            edges2grid="y",
        )
        return port

    def create_mesh(self) -> None:
        """
        Generate the FDTD mesh for the simulation domain.

        Delegates to the automatic `Mesh` class, which meshes the primitives
        already added to `self.CSX`. Call this after all geometry has been
        created.

        Returns
        -------
        None
        """
        Mesh(self.CSX, self.params)

    def build_inverted_f_antenna(self) -> LumpedPort:
        """
        Build the complete IFA geometry and mesh.

        Runs the full build sequence — substrate, ground plane, shorting
        leg and via, feed arm, radiating tip, port, and mesh — in the
        order required by their dependencies.

        Returns
        -------
        port : openEMS.ports.LumpedPort
            The created lumped port object, used to retrieve S-parameter
            and impedance results after the simulation.
        """
        self.create_substrate()
        self.create_ground()
        self.create_short_leg()
        self.create_short_via()
        self.create_excite_line()
        self.create_radiating_tip()
        port = self.create_port()
        self.create_mesh()
        return port
