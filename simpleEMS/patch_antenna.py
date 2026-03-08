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
from .calc import microstrip_width_from_impedance, patch_dims, phase_shift_length
from .sim_params import SimParams
from .sim_tools import SimTools, mm_to_m
from .console import console

# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "InsetPatchParams",
    "InsetFedPatchAntenna",
    "ProbePatchParams",
    "ProbeFedPatchAntenna",
]


@dataclass
class InsetPatchParams(SimParams):
    """
    Parameters for an inset-fed rectangular microstrip patch antenna.

    This class extends `SimParams` and computes derived geometric
    parameters required for antenna layout and simulation.

    Parameters
    ----------
    ang_length : int, optional
        Electrical length (in degrees) used to compute the feed phase
        shift length. Default is 90.

    Attributes
    ----------
    patch_length_mm : float
        Physical length of the patch in millimeters.
    patch_width_mm : float
        Physical width of the patch in millimeters.
    inset_length_mm : float
        Inset depth in millimeters.
    inset_width_mm : float
        Inset gap width in millimeters.
    feed_width_mm : float
        Width of the feed line in millimeters.
    feed_length_mm : float
        Computed feed line length in millimeters based on phase shift.
    substrate_width_mm : float
        Substrate width including margin (lambda0 padding).
    substrate_length_mm : float
        Substrate length including margin (lambda0 padding).
    simulation_box : ndarray
        A 1D array of shape (3,) representing the 3D simulation domain
        size: ``[x_size_mm, y_size_mm, z_size_mm]``.
    """

    ang_length: int = 90
    patch_length_mm: float = field(init=False)
    patch_width_mm: float = field(init=False)
    inset_length_mm: float = field(init=False)
    inset_width_mm: float = field(init=False)
    feed_width_mm: float = field(init=False)
    feed_length_mm: float = field(init=False)

    def __post_init__(self):
        """
        Perform geometric calculations after data class initialization.

        This method triggers the calculation of all physical dimensions
        based on the frequency and material parameters provided.

        Returns
        -------
        None
        """

        super().__post_init__()

        self.feed_length_mm = np.round(
            phase_shift_length(
                self.ang_length, self.substrate_eps_r, self.resonant_freq
            ),
            self.fp_precision,
        )
        self.feed_width_mm = np.round(
            microstrip_width_from_impedance(
                self.charac_imp,
                self.substrate_thickness_mm,
                self.copper_thickness_mm,
                self.substrate_eps_r,
                self.resonant_freq,
            ),
            self.fp_precision,
        )
        self.patch_dims = patch_dims(
            self.resonant_freq,
            self.substrate_eps_r,
            mm_to_m(self.substrate_thickness_mm),
            self.charac_imp,
            self.copper_thickness_mm,
        )

        self.patch_width_mm = np.round(
            self.patch_dims.patch_width_mm, self.fp_precision
        )
        self.patch_length_mm = np.round(
            self.patch_dims.patch_length_mm, self.fp_precision
        )
        self.inset_width_mm = np.round(
            self.patch_dims.inset_width_mm / 2, self.fp_precision
        )

        self.inset_length_mm = np.round(
            self.patch_dims.inset_length_mm, self.fp_precision
        )
        self.substrate_width_mm = np.round(
            self.patch_width_mm + 2 * self.lambda0, self.fp_precision
        )
        self.substrate_length_mm = np.round(
            self.patch_length_mm + 2 * self.lambda0, self.fp_precision
        )

        self.simulation_box = np.round(
            np.array(
                [
                    self.substrate_width_mm + self.lambda0,
                    self.substrate_length_mm + self.lambda0,
                    self.lambda0 * 2,
                ]
            ),
            self.fp_precision,
        )


class PatchAntenna(SimTools):
    """
    Base class for rectangular microstrip patch antenna models.

    This class implements the common physical structures (substrate and
    ground plane) shared by all patch antenna variants. Specific feed
    types (e.g., inset, probe, or edge-fed) should inherit from this
    base class.

    Parameters
    ----------
    params :
        Data container containing geometric and material properties
        (e.g., substrate thickness, permittivity, and dimensions).
    CSX : ContinuousStructure
        The CSXCAD geometry container used to define physical objects.
    FDTD : FDTD
        The FDTD simulation engine object.

    Attributes
    ----------
    params :
        Stored reference to the simulation parameters.
    CSX : ContinuousStructure
        Stored reference to the geometry engine.
    FDTD : FDTD
        Stored reference to the simulation engine.

    Notes
    -----
    All physical dimensions are expected to be in millimeters (mm)
    as defined in the `params` object.
    """

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


class InsetFedPatchAntenna(PatchAntenna):
    """
    Inset-fed rectangular microstrip patch antenna model.

    This class implements the geometry representation
    of an inset-fed microstrip patch antenna. It extends `PatchAntenna` class
    by implementing the inset patch antenna element.
    """

    def create_patch_with_inset(self):
        """
        Create the patch element with a notched inset for feed line.

        This method defines the patch geometry as a linear polygon (LinPoly)
        to account for the cutouts where the feed line enters the patch. It
        also adds the metal edges to the FDTD grid for accurate field
        calculation at the conductor boundaries.

        Returns
        -------
        None

        Notes
        -----
        The method includes a DRC (Design Rule Check) warning if the
        `inset_width_mm` is below 0.089 mm, which may exceed standard
        PCB fabrication capabilities.
        """
        patch_inset = self.CSX.AddMetal("patch_inset")
        patch_inset.SetColor("#B87333", 255)
        if self.params.inset_width_mm / 2 < 0.089:
            console.print(
                f"WARNING: Inset Width {self.params.inset_width_mm / 2} is too small, check minimum trace spacing with your PCB manufacturer",
                style="warning",
            )
        pp = [
            [
                -self.params.patch_width_mm / 2,
                -self.params.feed_width_mm / 2 - self.params.inset_width_mm / 2,
                -self.params.feed_width_mm / 2 - self.params.inset_width_mm / 2,
                self.params.feed_width_mm / 2 + self.params.inset_width_mm / 2,
                self.params.feed_width_mm / 2 + self.params.inset_width_mm / 2,
                self.params.patch_width_mm / 2,
                self.params.patch_width_mm / 2,
                -self.params.patch_width_mm / 2,
            ],
            [
                -self.params.patch_length_mm / 2,
                -self.params.patch_length_mm / 2,
                -self.params.patch_length_mm / 2 + self.params.inset_length_mm,
                -self.params.patch_length_mm / 2 + self.params.inset_length_mm,
                -self.params.patch_length_mm / 2,
                -self.params.patch_length_mm / 2,
                self.params.patch_length_mm / 2,
                self.params.patch_length_mm / 2,
            ],
        ]

        patch_inset.AddLinPoly(
            pp, "z", self.params.substrate_thickness_mm, self.params.copper_thickness_mm
        )
        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=patch_inset,
            metal_edge_res=self.params.metal_mesh_resolution,
        )

    def create_feed(self):
        """
        Create the microstrip feed line.

        Adds a rectangular metal box for feed element.

        Returns
        -------
        None

        Notes
        -----
        The method includes a DRC (Design Rule Check) warning if the
        `feed_width_mm` is below 0.1 mm, which may exceed standard
        PCB fabrication capabilities.
        """
        feed = self.CSX.AddMetal("feed")
        feed.SetColor("#B87333", 255)
        if self.params.feed_width_mm < 0.1:
            console.print(
                f"WARNING: Trace Width {self.params.feed_width_mm} is too small, check minimum trace width with your PCB manufacturer",
                style="warning",
            )
        feed_start = [
            self.params.feed_width_mm / 2,
            -self.params.patch_length_mm / 2 + self.params.inset_length_mm,
            self.params.substrate_thickness_mm,
        ]
        feed_stop = [
            -self.params.feed_width_mm / 2,
            -self.params.patch_length_mm / 2 - self.params.feed_length_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        feed.AddBox(priority=6, start=feed_start, stop=feed_stop)
        self.FDTD.AddEdges2Grid(
            dirs="xy", properties=feed, metal_edge_res=self.params.metal_mesh_resolution
        )

    def create_port(self):
        """
        Define the excitation lumped port at the end of the feed line.

        The port is placed between the ground plane (z=0) and the top of
        the feed line metallization.

        Returns
        -------
        port : openEMS.ports.LumpedPort
            The created lumped port object, used to retrieve S-parameter
            and impedance results after the simulation.
        """
        port_position = -self.params.patch_length_mm / 2 - self.params.feed_length_mm
        port_start = [self.params.feed_width_mm / 2, port_position, 0]
        port_stop = [
            -self.params.feed_width_mm / 2,
            port_position,
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

    def create_mesh(self):
        """
        Generate an FDTD mesh for the simulation domain.

        This method defines the mesh lines for the x, y, and z directions
        based on the antenna geometry, substrate thickness, and
        simulation box size. It applies the "thirds rule" for mesh
        refinement near metal edges and uses `SmoothMeshLines` to
        ensure a stable grid transition.

        Returns
        -------
        None
        """
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
        mesh.AddLine("x", -self.params.patch_width_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("x", self.params.patch_width_mm / 2 + self.params.thirds_rule)
        mesh.AddLine("y", -self.params.patch_length_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("y", self.params.patch_length_mm / 2 + self.params.thirds_rule)

        mesh.AddLine("x", -self.params.feed_width_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("x", self.params.feed_width_mm / 2 + self.params.thirds_rule)
        mesh.AddLine("y", -self.params.patch_length_mm / 2 - self.params.feed_length_mm)
        mesh.AddLine(
            "y", -self.params.patch_length_mm / 2 + self.params.inset_length_mm
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

        mesh.SmoothMeshLines("all", self.params.mesh_resolution, 1.5)


@dataclass
class ProbePatchParams(SimParams):
    """
    Parameters for an probe-fed rectangular microstrip patch antenna.

    This class inherits from `SimParams` and computes derived geometric
    parameters required for antenna layout and simulation.

    Parameters
    ----------
    None.

    Attributes
    ----------
    patch_length_mm : float
        Physical length of the patch in millimeters.
    patch_width_mm : float
        Physical width of the patch in millimeters.
    probe_pos_mm: float
        Probe position in y direction. x is set to 0.
    substrate_width_mm : float
        Substrate width including margin (lambda0 padding).
    substrate_length_mm : float
        Substrate length including margin (lambda0 padding).
    simulation_box : np.ndarray
        3D simulation domain size as:
            [x_size_mm, y_size_mm, z_size_mm]
            Includes lambda0 air padding around the antenna structure.
    """

    patch_length_mm: float = field(init=False)
    patch_width_mm: float = field(init=False)
    probe_pos_mm: float = field(init=False)

    def __post_init__(self):
        """
        Perform geometric calculations after data class initialization.

        This method triggers the calculation of all physical dimensions
        based on the frequency and material parameters provided.

        Returns
        -------
        None
        """

        super().__post_init__()
        self.patch_dims = patch_dims(
            self.resonant_freq,
            self.substrate_eps_r,
            self.substrate_thickness_mm * self.unit,
            self.charac_imp,
            self.copper_thickness_mm,
        )
        self.patch_width_mm = np.round(
            self.patch_dims.patch_width_mm, self.fp_precision
        )
        self.patch_length_mm = np.round(
            self.patch_dims.patch_length_mm, self.fp_precision
        )
        self.probe_pos_mm = np.round((self.patch_dims.probe_pos_mm), self.fp_precision)
        self.substrate_width_mm = np.round(
            self.patch_width_mm + 2 * self.lambda0, self.fp_precision
        )
        self.substrate_length_mm = np.round(
            self.patch_length_mm + 2 * self.lambda0, self.fp_precision
        )

        self.simulation_box = np.round(
            np.array(
                [
                    self.substrate_width_mm + self.lambda0,
                    self.substrate_length_mm + self.lambda0,
                    self.lambda0 * 2,
                ]
            ),
            self.fp_precision,
        )


class ProbeFedPatchAntenna(PatchAntenna):
    """
    Probe-fed rectangular microstrip patch antenna model.

    This class implements the geometry representation
    of a probe-fed microstrip patch antenna. It extends `PatchAntenna` class
    by implementing the patch antenna element.
    """

    def create_probe_fed_patch(self):
        """
        Create the probe fed patch element.

        This method defines the patch geometry as a linear polygon (LinPoly)
        it also adds the metal edges to the FDTD grid for accurate field
        calculation at the conductor boundaries.

        Returns
        -------
        None

        """

        patch_probe = self.CSX.AddMetal("patch_probe")
        patch_probe.SetColor("#B87333", 255)
        patch_start = [
            self.params.patch_width_mm / 2,
            self.params.patch_length_mm / 2,
            self.params.substrate_thickness_mm,
        ]
        patch_stop = [
            -self.params.patch_width_mm / 2,
            -self.params.patch_length_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        patch_probe.AddBox(priority=6, start=patch_start, stop=patch_stop)
        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=patch_probe,
            metal_edge_res=self.params.metal_mesh_resolution,
        )

    def create_port(self):
        """
        Define the excitation lumped port at the probe position.

        The port is placed between the ground plane (z=0) and the top of patch element.

        Returns
        -------
        port : openEMS.ports.LumpedPort
            The created lumped port object, used to retrieve S-parameter
            and impedance results after the simulation.
        """

        port_start = [1, self.params.probe_pos_mm, 0]
        port_stop = [0, self.params.probe_pos_mm, self.params.substrate_thickness_mm]
        port = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_start,
            port_stop,
            "z",
            1.0,
            priority=6,
            edges2grid="xy",
        )
        return port

    def create_mesh(self):
        """
        Generate an FDTD mesh for the simulation domain.

        This method defines the mesh lines for the x, y, and z directions
        based on the antenna geometry, substrate thickness, and
        simulation box size. It applies the "thirds rule" for mesh
        refinement near metal edges and uses `SmoothMeshLines` to
        ensure a stable grid transition.

        Returns
        -------
        None
        """
        mesh = self.CSX.GetGrid()
        mesh.SetDeltaUnit(self.params.unit)

        mesh.AddLine("y", self.params.probe_pos)
        mesh.AddLine(
            "x",
            np.linspace(
                0,
                1,
                6,
            ),
        )
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
        mesh.AddLine("x", -self.params.patch_width_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("x", self.params.patch_width_mm / 2 + self.params.thirds_rule)
        mesh.AddLine("y", -self.params.patch_length_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("y", self.params.patch_length_mm / 2 + self.params.thirds_rule)
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
