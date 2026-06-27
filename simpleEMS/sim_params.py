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
Base simulation parameter definitions for openEMS structures.

Defines the `SimParams` dataclass which serves as the base class
for all structure-specific parameter classes. Handles common
attributes such as substrate properties, frequency range, mesh
resolution, and simulation box computation.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from openEMS.physical_constants import C0, EPS0


@dataclass(kw_only=True)
class SimParams:
    """
    Base class for all openEMS simulation parameters.
    This class stores user-defined parameters required for an OpenEMS
    simulation, such as substrate properties, material settings, and mesh
    controls. Several dependent quantities — such as wavelength, dielectric
    loss (kappa), mesh resolution, and recommended simulation box size —
    are computed automatically in ``__post_init__()`` based on the
    frequency range and inputs provided.
    Subclasses (e.g., InsetFedPatchParams, ProbeFedPatchParams,
    MicrostripLineParams) must define ``freq_range`` and ``main_freq``
    properties, and are responsible for setting substrate dimensions
    before calling ``_create_simulation_box()``.

    Parameters
    ----------

    substrate_eps_r : float
        Relative permittivity of the substrate.
    substrate_tand : float
        tangent loss of the substrate.
    substrate_thickness_mm : float
        Substrate thickness in millimeters.
    substrate_width_mm : float
        Substrate width in millimeters. Must be provided by subclasses.
    substrate_length_mm : float
        Substrate length in millimeters. Must be provided by subclasses.
    substrate_cells : int, optional
        Suggested number of mesh cells along the substrate thickness.
        Default value is ``4``.
    unit : float, optional
        Unit used in the model. Default value is ``1e-3`` which represents millimeters.
        This value should not be changed.
    copper_thickness_mm : float, optional
        Thickness of the copper layer in millimeters. Default is ``0.035``.
    min_trace_width_mm : float, optional
        Minimum manufacturable trace width in millimeters. Default is ``0.1``.
    min_trace_spacing_mm : float, optional
        Minimum manufacturable trace spacing in millimeters. Default is ``0.089``.
    num_points : int
        Number of frequency points for post-processing. Default is ``1000``.
    fp_precision : int, optional
        Floating-point precision used when generating geometric values.
        Default is ``3``.
    charac_imp : float, optional
        Feed/port impedance in ohms. Default is ``50 ohms``.
    timestep : int, optional
        FDTD simulation time-step count. Default is ``90000000``.
    end_criteria : float, optional
        Convergence threshold for the stopping criteria. Default is ``1e-4``.
    mesh_resolution_factor : float, optional
        Division factor used to compute the global mesh resolution
        (``lambda0 / factor``). Default is ``10``.

    metal_mesh_resolution_factor : float, optional
        Division factor used to compute the metal primitives mesh resolution
        (``lambda0 / factor``). Default is ``40``.

    Attributes
    ----------
    substrate_kappa : float
        Conductivity-equivalent dielectric loss term computed from
        loss tangent, frequency, and permittivity.
    lambda0 : float
        Effective wavelength in the substrate, computed as
        ``C0 / (f_ref * sqrt(eps_r) * unit)``.
    simulation_box : NDArray of shape (3,)
        Bounding box dimensions [x, y, z] for the FDTD domain in mm,
        including lambda0 air padding around the structure.
    mesh_resolution : float
        Computed global mesh resolution.
    metal_mesh_resolution : float
        Computed mesh resolution for metal primitives.
    thirds_rule : NDArray
        Small mesh offsets (2/3 and -1/3 of mesh_resolution) applied
        near metal edges for accurate field resolution.
    """

    substrate_eps_r: float
    substrate_tand: float
    substrate_thickness_mm: float
    substrate_kappa: float = field(init=False)

    substrate_cells: int = 4

    unit: float = 1e-3  # mm
    num_points: int = 1000

    copper_thickness_mm: float = 0.035
    min_trace_width_mm: float = 0.1
    min_trace_spacing_mm: float = 0.089
    fp_precision: int = 3
    charac_imp: float = 50
    timestep: int = 90000000
    end_criteria: float = 1e-4

    mesh_resolution_factor: int = 10
    metal_mesh_resolution_factor: int = 40
    mesh_resolution: float = field(init=False)
    metal_mesh_resolution: float = field(init=False)

    lambda0: float = field(init=False)
    thirds_rule: NDArray = field(init=False)
    backend: str = "FDTD"

    @property
    def freq_range(self) -> tuple[float, float]:
        """
        Return the simulation frequency range.
        This property must be implemented by subclasses to define the
        frequency bounds used in the simulation.
        Returns
        -------
        tuple of (float, float)
            A tuple containing (f_min, f_max) for the simulation.
        Raises
        ------
        NotImplementedError
            If the subclass does not define this property.
        """
        raise NotImplementedError("Subclasses must define freq_range")

    @property
    def main_freq(self) -> float:
        """
        Return the primary frequency of interest.
        This property must be implemented by subclasses to define the
        main frequency used for post-processing and analysis.
        Returns
        -------
        float
            The main/target frequency.
        Raises
        ------
        NotImplementedError
            If the subclass does not define this property.
        """
        raise NotImplementedError("Subclasses must define main_freq")

    @property
    def simulation_box(self) -> NDArray:
        """
        Return the 3D simulation bounding box.

        Must be implemented by subclasses.

        Returns
        -------
        NDArray
            Array of shape (3,) with [x, y, z] dimensions in mm.

        Raises
        ------
        NotImplementedError
            If the subclass does not define this property.
        """
        raise NotImplementedError("subclasses must define simulation_box")

    @property
    def substrate_width_mm(self) -> float:
        """
        Return the substrate width in mm.

        Must be implemented by subclasses.

        Returns
        -------
        float
            Substrate width in mm.

        Raises
        ------
        NotImplementedError
            If the subclass does not define this property.
        """
        raise NotImplementedError("subclasses must define substrate width")

    @property
    def substrate_length_mm(self) -> float:
        """
        Return the substrate length in mm.

        Must be implemented by subclasses.

        Returns
        -------
        float
            Substrate length in mm.

        Raises
        ------
        NotImplementedError
            If the subclass does not define this property.
        """
        raise NotImplementedError("subclasses must define substrate length")

    def __post_init__(self) -> None:
        """Perform common parameter computations after dataclass initialisation."""
        self._compute_common()

    def _compute_common(self) -> None:
        """
        Compute common simulation parameters shared across all structure types.
        This method calculates the substrate kappa (dielectric loss),
        free-space wavelength, global mesh resolution, metal mesh resolution,
        and the thirds rule for mesh refinement based on the frequency range
        and material properties.
        Returns
        -------
        None
        """

        self.substrate_kappa = (
            self.substrate_tand
            * 2
            * np.pi
            * self.main_freq
            * EPS0
            * self.substrate_eps_r
        )

        self.lambda0 = C0 / (self.main_freq * np.sqrt(self.substrate_eps_r) * self.unit)

        self.mesh_resolution = self.lambda0 / self.mesh_resolution_factor
        self.metal_mesh_resolution = self.lambda0 / self.metal_mesh_resolution_factor

        self.thirds_rule = (
            np.array([2 * self.mesh_resolution / 3, -self.mesh_resolution / 3]) / 4
        )

    def _create_simulation_box(
        self, x_dir: float, y_dir: float, z_dir: float
    ) -> NDArray:
        """
        Compute the 3D simulation bounding box from substrate dimensions
        and lambda0 padding.

        Parameters
        ----------
        x_dir : float
            X-direction extent of the simulation box in mm.
        y_dir : float
            Y-direction extent of the simulation box in mm.
        z_dir : float
            Z-direction extent of the simulation box in mm.

        Returns
        -------
        NDArray
            A numpy array of shape (3,) containing the [x, y, z]
            simulation box dimensions in mm, rounded to fp_precision.
        """
        simulation_box = np.round(
            np.array(
                [
                    x_dir,
                    y_dir,
                    z_dir,
                ]
            ),
            self.fp_precision,
        )
        return simulation_box
