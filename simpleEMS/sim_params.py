#! /usr/bin/env python3
from dataclasses import dataclass, field
import numpy as np
from openEMS.physical_constants import C0, EPS0


@dataclass
class SimParams:
    """
    This class stores all user-defined parameters required for an OpenEMS
    simulation, such as frequency settings, substrate dimensions, and mesh
    controls. Several dependent quantities—such as wavelength, dielectric
    loss, mesh resolution, and recommended simulation box size—are computed
    automatically in ``__post_init__()`` based on the provided inputs.

    Parameters
    ----------
    resonant_freq : float
        Target resonant frequency in Hz.
    corner_freq : float
         frequency used to compute the upper and lower bounds of the frequency range

    substrate_eps_r : float
        Relative permittivity of the substrate.
    substrate_tand : float
        tangent loss of the substrate.
    substrate_thickness_mm : float
        Substrate thickness in millimeters.
    substrate_width_mm : float
        Substrate width in millimeters.
    substrate_length_mm : float
        Substrate length in millimeters.
    substrate_cells : int, optional
        Suggested number of mesh cells along the substrate thickness.
        Default value is ``4``.

    unit : float, optional
        unit used in the model. Default value is ``1e-3`` which represents millimeters.
        this value should not be changed.
    copper_thickness_mm : float, optional
        Thickness of the copper layer in millimeters. Default is ``0.035``.

    fp_precision : int, optional
        Floating-point precision used when generating geometric values.
        Default is ``3``.
    charac_imp : float, optional
        Feed/port impedance in ohms. Default is ``50 ohms``.
    timestep : int, optional
        FDTD simulation time-step count. Default is ``1000000``.
    end_criteria : float, optional
        Convergence threshold for the stopping criteria. Default is ``1e-4``.

    mesh_resolution_factor : float, optional
        Division factor used to compute the global mesh resolution
        (``lambda0 / factor``). Default is ``20``.

    metal_mesh_resolution_factor : float, optional
        Division factor used to compute the metal primitives mesh resolution
        (``lambda0 / factor``). Default is ``40``.


    Attributes
    ----------
    substrate_kappa : float
        Computed dielectric loss term.
    lambda0 : float
        Free space wavelength.
    simulation_box : ndarray of shape (3,)
        Recommended bounding box for the FDTD domain
        (width, length, height).
    mesh_resolution : float
        Computed global mesh resolution.
    metal_mesh_resolution : float
        Computed mesh resolution for metal primitives.

    thirds_rule : ndarray
        This defines one third two thirds rule required for accurate results.
    """

    resonant_freq: float
    corner_freq: float

    substrate_eps_r: float
    substrate_tand: float
    substrate_thickness_mm: float
    substrate_width_mm: float = 100
    substrate_length_mm: float = 100
    substrate_cells: int = 4

    unit: float = 1e-3  # mm
    num_points: int = 1000

    copper_thickness_mm: float = 0.035
    fp_precision: int = 3
    charac_imp: float = 50
    timestep: int = 1000000
    end_criteria: float = 1e-4
    mesh_resolution_factor: int = 20
    metal_mesh_resolution_factor: int = 40

    substrate_kappa: float = field(init=False)
    lambda0: float = field(init=False)
    simulation_box: np.ndarray = field(init=False)
    mesh_resolution: float = field(init=False)
    thirds_rule: np.ndarray = field(init=False)

    def __post_init__(self):
        self.substrate_kappa = (
            self.substrate_tand
            * 2
            * np.pi
            * self.resonant_freq
            * EPS0
            * self.substrate_eps_r
        )

        self.lambda0 = np.round(
            C0
            / (
                (self.resonant_freq + self.corner_freq)
                * np.sqrt(self.substrate_eps_r)
                * self.unit
            ),
            self.fp_precision,
        )

        # mesh resolution
        self.mesh_resolution = np.round(
            self.lambda0 / self.mesh_resolution_factor, self.fp_precision
        )
        self.metal_mesh_resolution = np.round(
            self.lambda0 / self.metal_mesh_resolution_factor, self.fp_precision
        )

        self.thirds_rule = np.round(
            (
                np.array(
                    [
                        2 * self.mesh_resolution / 3,
                        -self.mesh_resolution / 3,
                    ]
                )
                / 4
            ),
            self.fp_precision,
        )
