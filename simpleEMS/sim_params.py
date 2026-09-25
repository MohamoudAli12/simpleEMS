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

from .fem_backend import FEMOptions

_FEM_DEFAULTS = FEMOptions()  # single source of truth for the FEM_* defaults below


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
    properties and provide ``substrate_width_mm`` and
    ``substrate_length_mm``, as properties or as fields.

    Parameters
    ----------

    substrate_eps_r : float
        Relative permittivity of the substrate.
    substrate_tand : float
        Loss tangent of the substrate material.
    substrate_thickness_mm : float
        Substrate thickness in millimeters.
    substrate_width_mm : float
        Substrate width in millimeters. Must be provided by subclasses.
    substrate_length_mm : float
        Substrate length in millimeters. Must be provided by subclasses.
    substrate_cells : int, optional
        Number of FDTD mesh lines through the substrate thickness in z,
        counting both faces, evenly spaced (``substrate_cells - 1`` cells).
        Must be at least ``2``. Default value is ``7``.
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
        For the FEM backend this is the number of interpolated output points the
        smooth S-parameter curve is evaluated at (not the number of solves).
    backend_engine : str, optional
        Solver backend to use: ``"FDTD"`` (openEMS, default) or ``"FEM"``
        (Gmsh + GetDP finite-element frequency-domain solver).
    simulation_box : array_like, optional
        FDTD backend only. Extent of the simulation domain in mm, either as
        ``[x, y, z]`` sizes centred on the origin (shape ``(3,)``) or as
        ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` bounds (shape
        ``(3, 2)``). The automatic mesher uses it exactly as given and raises
        if the structure does not fit inside. Default is ``None``, which pads
        the structure's own extent by ``max(lambda0, 15% of its span)`` on
        every side. The FEM backend sizes its air box from ``FEM_air_pad_*``
        instead.
    FEM_num_solve_points : int, optional
        Number of full FEM solves the adaptive rational-interpolation sweep is
        allowed to perform (must be ``>= 4``). Ignored by the FDTD backend.
        Default is ``10``.
    FEM_max_solve_points : int, optional
        Ceiling on the solve count when the interpolated curve comes out
        non-passive and the sweep keeps solving to pull it back (must be
        ``>= FEM_num_solve_points``). A passive response never costs more than
        ``FEM_num_solve_points``. Default is ``None``, which allows twice
        ``FEM_num_solve_points``.
    FEM_boundary : str, optional
        FEM backend only. Outer truncation: ``"silver_muller"`` (default),
        ``"pml"`` or ``"pec"``. The first two let the box radiate;
        ``"pec"`` shorts it instead, making the model a shielded enclosure.
        Use ``"pec"`` for a closed structure -- a transmission line, a filter
        -- where nothing is meant to leave the box. It is also the only
        boundary that agrees with what a wave port's own transverse mode solve
        assumes, since that mode is solved on a cross-section whose outline is
        a PEC wall, so it is the one to reach for when a matched line loses
        power between its ports. A PEC box cannot radiate, so
        ``FEMNF2FF.CalcNF2FF`` refuses a far field under it.
    FEM_symmetry : tuple, optional
        FEM backend only. Mirror-symmetry plane ``(axis, kind, at)`` used to
        halve the mesh. ``None`` (default) disables symmetry.
    FEM_fe_order : int, optional
        FEM backend only. Nedelec edge-element order: ``1`` (default) or
        ``2``.
    FEM_air_pad_frac : float, optional
        FEM backend only. Air padding as a fraction of the free-space
        wavelength at ``FEM_mesh_freq``. Defaults to
        :attr:`FEMOptions.air_pad_frac`. Ignored when ``FEM_air_pad_mm`` is
        set.
    FEM_air_pad_mm : float or tuple, optional
        FEM backend only. Explicit air padding in millimetres, added to the
        structure's bounding box in place of the ``FEM_air_pad_frac``
        wavelength formula. Give one value for all six faces, three values
        ``[x, y, z]`` to pad each axis symmetrically, or three
        ``[low, high]`` pairs to set every face on its own -- for example
        ``[[8, 8], [8, 8], [2, 30]]`` for a patch, which needs a deep air
        column above it but almost none below its ground plane. Use this for
        a non-radiating structure (e.g. a filter) whose box has no reason to
        be a quarter-wavelength at all. A face a wave port stands on is padded
        by nothing whatever this says, because the port is the domain wall
        there; that face can therefore carry no far field. If a far-field
        pattern is later requested and this padding is too small for an
        accurate near-to-far-field transform at the requested frequency,
        ``FEMNF2FF.CalcNF2FF`` raises ``ValueError`` naming the face that is
        short and the padding needed. Default is ``None`` (auto, via
        ``FEM_air_pad_frac``).
    FEM_elems_per_wavelength : float, optional
        FEM backend only. Target coarse mesh density, applied per material
        against that material's own wavelength. Default is ``16.0``; see
        :class:`FEMOptions` for why it is not ``8.0``.
    FEM_mesh_freq : float, optional
        FEM backend only. Frequency the mesh is sized at, in Hz -- both the
        element size and the air padding come off its wavelength. Default is
        ``None``, meaning ``main_freq``: ``freq_range`` is the window the
        S-parameters are plotted over, so widening it to see more of a curve
        no longer refines the mesh. Raise it to check the structure at the top
        of a wide band, at the usual cost in elements.
    FEM_mesh_fine_scale : float, optional
        FEM backend only. Multiplier on the near-conductor element size.
        Default is ``1.0``.
    FEM_port_mode_modes : int, optional
        Number of eigenpairs each wave port's transverse mode solve computes.
        Raise it if a port reports finding no guided mode, or to reach a
        higher ``FEM_port_mode_index``. Default is ``6``.
    FEM_port_mode_index : int, optional
        Which guided mode a wave port runs in, counting from ``0`` for the one
        with the largest propagation constant. ``0`` (the default) is right for
        microstrip and any other single-conductor line. A conductor-backed
        coplanar waveguide carries several quasi-TEM modes -- the CPW mode the
        line is meant to run in, and a microstrip-like mode between the trace
        and the backside ground -- and the CPW mode is not the one with the
        largest propagation constant, so it has to be named.
    FEM_port_type : str, optional
        Which port the FEM backend makes of the ports the geometry already
        carries. ``"lumpedport"`` (the default) drives one constant field
        direction across the port sheet -- right for a gap feed such as a
        probe-fed patch, and what the FEM backend has always done.
        ``"waveport"`` instead solves the transverse mode living on the port's
        cross-section and drives the line with that, which is what a
        transmission line needs: a microstrip quasi-TEM mode is not uniform
        across the port face. A wave port terminates the simulation domain, so
        it has to stand on an end of the structure -- run the line out to the
        board edge. The air box then puts its wall on the port plane, and a
        port too far inside to do that is refused. The geometry is unchanged
        either way -- the same ``AddLumpedPort`` or ``create_cpw_port`` call is
        simply read differently. A coplanar waveguide port is always treated as a wave
        port, because its mode is odd and a single constant vector excites the
        even parallel-plate mode instead.
    FEM_port_mode_eps_eff : float, optional
        Effective permittivity naming which guided mode a wave port runs in.
        A conductor-backed coplanar waveguide carries several quasi-TEM modes,
        and this picks the one whose effective permittivity is closest --
        naming the mode by a property of the line rather than by its position
        in a spectrum that shifts with frequency and mesh density. Takes
        precedence over ``FEM_port_mode_index``. Default is ``None``.
    FEM_port_mode_zc : float, optional
        Characteristic impedance in ohms to reference a wave port's
        S-parameters to, overriding the value measured from the solved mode.
        The measured value is good to a few percent on microstrip, but the
        power-voltage impedance of a multi-conductor mode depends on the path
        the voltage is integrated along, so on a coplanar waveguide it is
        better to state the impedance the line was designed for. Default is
        ``None`` (use the measured value).
    FEM_waveport_width_mm : float, optional
        FEM backend only. Width of each wave port's cross-section in
        millimetres, centred on the line. Default is ``None``: ``10 * w`` for
        a trace narrower than the substrate is thick, ``5 * w`` otherwise,
        where ``w`` is the port's width across the line.
    FEM_waveport_height_mm : float, optional
        FEM backend only. Height of each wave port's cross-section in
        millimetres, measured from the ground side of the substrate. Default
        is ``None``: six substrate thicknesses.
    FEM_min_layers : int, optional
        FEM backend only. Element layers through the dielectric thickness.
        Default is ``3``.
    fp_precision : int, optional
        Floating-point precision used when generating geometric values.
        Default is ``3``.
    charac_imp : float, optional
        Feed/port impedance in ohms. Default is ``50``.
    FDTD_timestep : int, optional
        FDTD simulation time-step count. Default is ``90000000``.
    FDTD_end_criteria : float, optional
        Convergence threshold for the stopping criteria. Default is ``1e-4``.
    FDTD_mesh_resolution_factor : int, optional
        Division factor used to compute the global mesh resolution
        (``lambda0 / factor``). Default is ``10``.

    FDTD_metal_mesh_resolution_factor : int, optional
        Division factor used to compute the metal primitives mesh resolution
        (``lambda0 / factor``). Default is ``40``.

    Attributes
    ----------
    substrate_kappa : float
        Conductivity-equivalent dielectric loss term computed from
        loss tangent, frequency, and permittivity.
    lambda0 : float
        Effective wavelength in the substrate, computed as
        ``C0 / (main_freq * sqrt(eps_r) * unit)``.
    FDTD_mesh_resolution : float
        Computed global mesh resolution.
    FDTD_metal_mesh_resolution : float
        Computed mesh resolution for metal primitives.
    FDTD_thirds_rule : NDArray
        Small mesh offsets, ``[2/3, -1/3] * mesh_resolution / 4``, applied
        near metal edges for accurate field resolution.
    """

    substrate_eps_r: float
    substrate_tand: float
    substrate_thickness_mm: float
    substrate_kappa: float = field(init=False)

    substrate_cells: int = 7

    unit: float = 1e-3  # mm
    num_points: int = 1000

    backend_engine: str = "FDTD"
    simulation_box: NDArray | None = None

    FEM_num_solve_points: int = 10
    FEM_max_solve_points: int | None = _FEM_DEFAULTS.max_solve_points
    FEM_boundary: str = _FEM_DEFAULTS.boundary
    FEM_symmetry: tuple | None = _FEM_DEFAULTS.symmetry
    FEM_fe_order: int = _FEM_DEFAULTS.fe_order
    FEM_air_pad_frac: float = _FEM_DEFAULTS.air_pad_frac
    FEM_air_pad_mm: float | tuple | None = _FEM_DEFAULTS.air_pad_mm
    FEM_elems_per_wavelength: float = _FEM_DEFAULTS.elems_per_wavelength
    FEM_mesh_freq: float | None = _FEM_DEFAULTS.mesh_freq
    FEM_mesh_fine_scale: float = _FEM_DEFAULTS.mesh_fine_scale
    FEM_min_layers: int = _FEM_DEFAULTS.min_layers
    FEM_port_type: str = _FEM_DEFAULTS.port_type
    FEM_port_mode_modes: int = _FEM_DEFAULTS.port_mode_modes
    FEM_port_mode_index: int = _FEM_DEFAULTS.port_mode_index
    FEM_port_mode_zc: float | None = _FEM_DEFAULTS.port_mode_zc
    FEM_port_mode_eps_eff: float | None = _FEM_DEFAULTS.port_mode_eps_eff
    FEM_waveport_width_mm: float | None = _FEM_DEFAULTS.waveport_width_mm
    FEM_waveport_height_mm: float | None = _FEM_DEFAULTS.waveport_height_mm

    FDTD_timestep: int = 90000000
    FDTD_end_criteria: float = 1e-4
    FDTD_mesh_resolution_factor: int = 10
    FDTD_metal_mesh_resolution_factor: int = 40
    FDTD_mesh_resolution: float = field(init=False)
    FDTD_metal_mesh_resolution: float = field(init=False)
    FDTD_thirds_rule: NDArray = field(init=False)

    copper_thickness_mm: float = 0.035
    min_trace_width_mm: float = 0.1
    min_trace_spacing_mm: float = 0.089
    fp_precision: int = 3
    charac_imp: float = 50
    lambda0: float = field(init=False)

    @property
    def fem_options(self) -> FEMOptions:
        """Bundle the flat ``FEM_*`` fields into a :class:`FEMOptions` instance."""
        return FEMOptions(
            boundary=self.FEM_boundary,
            symmetry=self.FEM_symmetry,
            fe_order=self.FEM_fe_order,
            air_pad_frac=self.FEM_air_pad_frac,
            air_pad_mm=self.FEM_air_pad_mm,
            elems_per_wavelength=self.FEM_elems_per_wavelength,
            # freq_range is the plot window; the mesh is sized for the
            # frequency the structure was designed at
            mesh_freq=self.FEM_mesh_freq or self.main_freq,
            mesh_fine_scale=self.FEM_mesh_fine_scale,
            min_layers=self.FEM_min_layers,
            num_solve_points=self.FEM_num_solve_points,
            max_solve_points=self.FEM_max_solve_points,
            port_type=self.FEM_port_type,
            port_mode_modes=self.FEM_port_mode_modes,
            port_mode_index=self.FEM_port_mode_index,
            port_mode_zc=self.FEM_port_mode_zc,
            port_mode_eps_eff=self.FEM_port_mode_eps_eff,
            waveport_width_mm=self.FEM_waveport_width_mm,
            waveport_height_mm=self.FEM_waveport_height_mm,
            # substrate_kappa was built at main_freq; read it back there
            kappa_freq=self.main_freq,
        )

    @property
    def freq_range(self) -> tuple[float, float]:
        """
        Return the simulation frequency range.

        Must be implemented by subclasses to define the frequency bounds
        used in the simulation.

        Returns
        -------
        tuple of (float, float)
            A tuple containing (f_min, f_max) in Hz for the simulation.

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

        Must be implemented by subclasses to define the main frequency
        used for post-processing and analysis.

        Returns
        -------
        float
            The main/target frequency in Hz.

        Raises
        ------
        NotImplementedError
            If the subclass does not define this property.
        """
        raise NotImplementedError("Subclasses must define main_freq")

    @property
    def simulation_bounds(self) -> NDArray | None:
        """
        Return ``simulation_box`` as per-axis ``[min, max]`` bounds.

        A ``(3,)`` box of sizes is centred on the origin.

        Returns
        -------
        NDArray or None
            Array of shape (3, 2), ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``
            in mm, or ``None`` when no simulation box is defined.
        """
        if self.simulation_box is None:
            return None
        simulation_box = np.asarray(self.simulation_box, dtype=float)
        if simulation_box.shape == (3,):
            return np.column_stack((-simulation_box / 2, simulation_box / 2))
        return simulation_box

    def __post_init__(self) -> None:
        """Perform common parameter computations after dataclass initialisation."""
        self._validate_backend()
        self._validate_simulation_box()
        self._validate_substrate_cells()
        self._compute_common()

    def _validate_substrate_cells(self) -> None:
        """
        Check that ``substrate_cells`` can place both substrate faces.

        Raises
        ------
        ValueError
            If ``substrate_cells`` is less than ``2``.
        """
        if self.substrate_cells < 2:
            raise ValueError(
                "substrate_cells counts both substrate faces and must be >= 2, "
                f"got {self.substrate_cells}"
            )

    def _validate_simulation_box(self) -> None:
        """
        Normalise ``simulation_box`` to a float array and check its shape.

        Raises
        ------
        ValueError
            If the box is neither ``(3,)`` positive sizes nor ``(3, 2)``
            bounds with ``min < max`` on every axis.
        """
        if self.simulation_box is None:
            return
        simulation_box = np.asarray(self.simulation_box, dtype=float)
        if simulation_box.shape == (3,):
            if np.any(simulation_box <= 0):
                raise ValueError(
                    f"simulation_box sizes must be positive, got {simulation_box}"
                )
        elif simulation_box.shape == (3, 2):
            if np.any(simulation_box[:, 0] >= simulation_box[:, 1]):
                raise ValueError(
                    "simulation_box bounds need min < max on every axis, "
                    f"got {simulation_box.tolist()}"
                )
        else:
            raise ValueError(
                "simulation_box must be [x, y, z] sizes (shape (3,)) or "
                "[[xmin, xmax], [ymin, ymax], [zmin, zmax]] (shape (3, 2)), "
                f"got shape {simulation_box.shape}"
            )
        self.simulation_box = simulation_box

    def _validate_backend(self) -> None:
        """
        Validate the solver backend selection.

        Raises
        ------
        ValueError
            If ``backend_engine`` is not ``"FDTD"`` or ``"FEM"``, if
            ``FEM_port_type`` is not a supported choice, or if
            ``FEM_num_solve_points`` is below the minimum needed for a stable
            rational fit.
        """
        if self.backend_engine not in ("FDTD", "FEM"):
            raise ValueError(
                f"backend_engine must be 'FDTD' or 'FEM', got {self.backend_engine!r}"
            )
        if self.FEM_port_type not in ("lumpedport", "waveport"):
            raise ValueError(
                f"FEM_port_type must be 'lumpedport' or 'waveport', "
                f"got {self.FEM_port_type!r}"
            )
        if self.FEM_num_solve_points < 4:
            raise ValueError(
                f"FEM_num_solve_points must be >= 4 for a stable rational fit, "
                f"got {self.FEM_num_solve_points}"
            )
        if (
            self.FEM_max_solve_points is not None
            and self.FEM_max_solve_points < self.FEM_num_solve_points
        ):
            raise ValueError(
                f"FEM_max_solve_points must be >= FEM_num_solve_points "
                f"({self.FEM_num_solve_points}), got {self.FEM_max_solve_points}"
            )

    def _compute_common(self) -> None:
        """
        Compute common simulation parameters shared across all structure types.

        Computes the substrate kappa (dielectric loss), free-space
        wavelength, global mesh resolution, metal mesh resolution, and the
        thirds rule for mesh refinement, based on the frequency range and
        material properties.

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

        self.FDTD_mesh_resolution = self.lambda0 / self.FDTD_mesh_resolution_factor
        self.FDTD_metal_mesh_resolution = (
            self.lambda0 / self.FDTD_metal_mesh_resolution_factor
        )

        self.FDTD_thirds_rule = (
            np.array(
                [2 * self.FDTD_mesh_resolution / 3, -self.FDTD_mesh_resolution / 3]
            )
            / 4
        )

    def _create_simulation_box(
        self, x_dir: float, y_dir: float, z_dir: float
    ) -> NDArray:
        """
        Assemble the 3D simulation bounding box from its per-axis extents.

        Callers (subclasses) are responsible for computing each extent,
        typically the structure size plus lambda0 air padding, before
        passing it in here.

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
            simulation box dimensions in mm, rounded to ``fp_precision``
            decimal places.
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
