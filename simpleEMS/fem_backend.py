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
FEM backend for the simpleEMS pipeline.

Bridges a CSXCAD geometry to the finite-element solver: assigns each solid an
electromagnetic role, meshes the structure, runs the frequency sweep, and
returns the results as the same :class:`~simpleEMS.sim_tools.SimData` named
tuple the FDTD backend produces, so the plotting and export utilities work
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import AAA

from CSXCAD import ContinuousStructure

from . import (
    fem_formulation,
    fem_geometry,
    fem_materials,
    fem_port_mode,
    fem_solver,
    fem_sweep,
)
from .console import console
from .export_cad import export_step
from .fem_materials import EPS0, Dielectric, guess_role

if TYPE_CHECKING:
    from .sim_tools import SimData

__all__ = ["simulate_step_FEM", "FEMOptions"]

# The three public functions run in the same order as the FDTD pipeline and,
# like it, hand state between stages through FILES in the output directory
# rather than through Python objects (the FDTD path writes structure.xml then
# reads probe files; we write the two files below). This keeps each stage
# stateless and independently re-runnable:
#
#   build_mesh   ->  fem_mesh.json   (mesh path, .pro path, port numbers + Zref)
#   run_sweep    ->  fem_sparams.npz (dense interpolated S, freqs, ref impedances)
#   compute_sim_data reads fem_sparams.npz -> SimData
_AXIS_TO_DIR = {0: "x", 1: "y", 2: "z"}
_MESH_META = "fem_mesh.json"
_SPARAMS = "fem_sparams.npz"


def _air_pad_faces_mm(air_pad_mm: float | tuple) -> tuple[tuple[float, float], ...]:
    """
    Expand an air-pad setting into one value per box face.

    Parameters
    ----------
    air_pad_mm : float or tuple
        One value for every face, three values for the three axes, or three
        ``(low, high)`` pairs.

    Returns
    -------
    tuple[tuple[float, float], ...]
        ``((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))``, in millimetres.

    Raises
    ------
    ValueError
        If the value is neither a single number, nor three numbers, nor three
        ``(low, high)`` pairs.
    """
    if isinstance(air_pad_mm, (int, float)):
        return ((float(air_pad_mm), float(air_pad_mm)),) * 3
    try:
        values = tuple(air_pad_mm)
    except TypeError:
        values = ()
    if len(values) == 3:
        if all(isinstance(value, (int, float)) for value in values):
            return tuple((float(value), float(value)) for value in values)
        if all(
            not isinstance(value, (int, float)) and len(tuple(value)) == 2
            for value in values
        ):
            return tuple((float(low), float(high)) for low, high in values)
    raise ValueError(
        "air_pad_mm must be one value, three values [x, y, z], or three "
        "[low, high] pairs [[xlo, xhi], [ylo, yhi], [zlo, zhi]], got "
        f"{air_pad_mm!r}"
    )


@dataclass
class FEMOptions:
    """
    Solver and mesh options shared by every FEM entry point.

    Bundles the settings into one object so they can be passed around together
    instead of as many separate arguments.

    Parameters
    ----------
    boundary : str
        Outer boundary condition: ``"silver_muller"`` (default), ``"pml"`` or
        ``"pec"``. See :attr:`~simpleEMS.sim_params.SimParams` for what each
        one is for.
    symmetry : tuple, optional
        Mirror-symmetry plane ``(axis, kind, at)``, where ``axis`` is ``"x"``,
        ``"y"``, or ``"z"``, ``kind`` is ``"pec"`` or ``"pmc"``, and ``at`` is
        the plane coordinate in metres (``None`` for the structure centre).
        Only the half of the structure on one side of the plane is meshed.
        Default ``None`` (no symmetry).
    fe_order : int
        Element order, either ``1`` (default) or ``2``. Order ``2`` is more
        accurate and roughly three times as expensive to solve.
    air_pad_frac : float
        Air padding around the structure, as a fraction of the free-space
        wavelength at ``mesh_freq``. Default ``0.25``. Ignored when
        ``air_pad_mm`` is set.
    air_pad_mm : float or tuple, optional
        Air padding around the structure in millimetres, used in place of
        ``air_pad_frac``. Either one value for all six faces, three values
        ``[x, y, z]`` padding each axis symmetrically, or three
        ``[low, high]`` pairs padding every face on its own -- so an antenna
        can carry deep air above the patch and little below the ground plane.
        Default ``None`` (pad by ``air_pad_frac``).
    elems_per_wavelength : float
        Target number of mesh elements per wavelength, applied separately in
        each material. Default ``16.0``.
    mesh_freq : float, optional
        Frequency the mesh is sized at, in Hz. Element size and air padding
        are both derived from its wavelength, so this is the one number that
        sets how expensive the mesh is. Default ``None``, which falls back to
        the top of the sweep; ``SimParams`` fills it with ``main_freq``, so a
        wide plot range no longer refines the mesh.
    mesh_fine_scale : float
        Multiplier on the element size near conductors. Values above ``1``
        coarsen the mesh there. Default ``1.0``.
    min_layers : int
        Number of element layers through the dielectric thickness. Default
        ``3``.
    num_solve_points : int
        Number of frequencies the sweep solves at, from which the full
        S-parameter curve is interpolated. Must be ``>= 4``. Default ``10``.
    max_solve_points : int, optional
        Ceiling on the solve count when the interpolated curve breaks
        passivity and the sweep keeps solving to fix it. Must be
        ``>= num_solve_points``. Default ``None``, which allows twice
        ``num_solve_points``. A passive response costs ``num_solve_points``.
    port_mode_modes : int
        Number of eigenpairs each wave port's mode solve computes. Raise it if
        a port reports that it found no guided mode. Default ``6``.
    port_type : str
        Which port the FEM backend makes of the ports the geometry already
        carries: ``"lumpedport"`` (default) drives one constant field
        direction across the port sheet, which is right for a gap feed;
        ``"waveport"`` solves the transverse mode of the port's cross-section
        and drives that instead, which is what a transmission line needs. A
        coplanar waveguide port is always a wave port -- its mode is odd, and
        a single constant vector excites the wrong one.
    port_mode_eps_eff : float, optional
        Effective permittivity naming which guided mode a wave port runs in:
        the mode whose own effective permittivity is closest to it is used.
        This is the robust way to pick a mode on a multi-conductor line,
        because it names the mode by a property of the line rather than by its
        position in a spectrum that changes with frequency and mesh density.
        Takes precedence over ``port_mode_index``. Default ``None``.
    port_mode_zc : float, optional
        Characteristic impedance in ohms to reference a wave port's
        S-parameters to, overriding the one measured from the solved mode.
        The measured value is good to a few percent on a single-conductor line
        such as microstrip, but the power-voltage impedance of a
        multi-conductor mode depends on which path the voltage is integrated
        along, so on a coplanar waveguide it is better to state the impedance
        the line was designed for. Default ``None`` (use the measured value).
    port_mode_index : int
        Which guided mode a wave port runs in, counting from ``0`` for the one
        with the largest propagation constant. ``0`` (the default) is right for
        a single-conductor line such as microstrip. A cross-section with more
        than one conductor above the ground plane -- conductor-backed coplanar
        waveguide is the common case -- carries more than one quasi-TEM mode,
        and the one with the largest ``beta`` need not be the one the line is
        meant to run in. See :mod:`~simpleEMS.fem_port_mode`.
    waveport_width_mm : float, optional
        Width of each wave port's cross-section in millimetres, centred on the
        line. Default ``None``: ``10 * w`` for a trace narrower than the
        substrate is thick, ``5 * w`` otherwise, where ``w`` is the port's
        width across the line.
    waveport_height_mm : float, optional
        Height of each wave port's cross-section in millimetres, measured from
        the ground side of the substrate. Default ``None``: six substrate
        thicknesses.
    kappa_freq : float, optional
        Frequency at which a CSXCAD material's ``kappa`` was set, in Hz, which
        is where it has to be read back as a loss tangent. openEMS models
        dielectric loss as a conductivity fixed at one frequency --
        ``SimParams.substrate_kappa`` is ``tand*2*pi*main_freq*eps0*eps_r`` --
        so inverting it anywhere else scales the loss by the ratio of the two.
        Default ``None``, which falls back to the sweep's centre frequency.
    """

    boundary: str = "silver_muller"
    # the plane must be a symmetry of the excitation too; 'pmc' is the usual
    # choice for a planar antenna fed on its centreline
    symmetry: tuple | None = None
    # order 1 gets the guided propagation constant noticeably wrong and mesh
    # refinement does not fix it; use 2 when phase or group delay matter
    fe_order: int = 1
    air_pad_frac: float = 0.25
    # for a structure whose box has no reason to be a quarter-wavelength at
    # all, such as a filter; FEMNF2FF.CalcNF2FF raises if this is later too
    # small for a far-field transform at the requested frequency.
    # Kept exactly as given -- air_pad_faces_mm expands it -- because SimParams
    # round-trips every field of this class by equality.
    air_pad_mm: float | tuple | None = None
    elems_per_wavelength: float = 16.0
    # The sweep grid says where to plot S-parameters, not what the structure is
    # for; meshing off its top end silently refines a wide plot window. Sized
    # off the design frequency instead -- SimParams feeds main_freq in.
    mesh_freq: float | None = None
    mesh_fine_scale: float = 1.0
    min_layers: int = 3
    num_solve_points: int = 10
    max_solve_points: int | None = None
    port_type: str = "lumpedport"
    port_mode_modes: int = 6
    port_mode_index: int = 0
    port_mode_zc: float | None = None
    port_mode_eps_eff: float | None = None
    waveport_width_mm: float | None = None
    waveport_height_mm: float | None = None
    kappa_freq: float | None = None

    def __post_init__(self) -> None:
        """
        Validate the option values.

        Raises
        ------
        ValueError
            If ``boundary`` or ``fe_order`` is not one of the supported
            choices, or if ``num_solve_points`` is below the minimum.
        """
        if self.boundary not in ("silver_muller", "pml", "pec"):
            raise ValueError(
                f"boundary must be 'silver_muller', 'pml' or 'pec', "
                f"got {self.boundary!r}"
            )
        if self.fe_order not in (1, 2):
            raise ValueError(f"fe_order must be 1 or 2, got {self.fe_order}")
        if self.mesh_freq is not None and self.mesh_freq <= 0:
            raise ValueError(f"mesh_freq must be > 0 Hz, got {self.mesh_freq}")
        if self.num_solve_points < 4:
            raise ValueError(
                f"num_solve_points must be >= 4 for a stable rational fit, "
                f"got {self.num_solve_points}"
            )
        if (
            self.max_solve_points is not None
            and self.max_solve_points < self.num_solve_points
        ):
            raise ValueError(
                f"max_solve_points must be >= num_solve_points "
                f"({self.num_solve_points}), got {self.max_solve_points}"
            )
        if self.port_type not in ("lumpedport", "waveport"):
            raise ValueError(
                f"port_type must be 'lumpedport' or 'waveport', got {self.port_type!r}"
            )
        if self.port_mode_modes < 1:
            raise ValueError(
                f"port_mode_modes must be >= 1, got {self.port_mode_modes}"
            )
        if self.port_mode_index < 0:
            raise ValueError(
                f"port_mode_index must be >= 0, got {self.port_mode_index}"
            )
        if self.port_mode_eps_eff is not None and self.port_mode_eps_eff < 1:
            raise ValueError(
                "port_mode_eps_eff is an effective permittivity, so it cannot "
                f"be below 1, got {self.port_mode_eps_eff}"
            )
        if self.port_mode_zc is not None and self.port_mode_zc <= 0:
            raise ValueError(f"port_mode_zc must be positive, got {self.port_mode_zc}")
        if self.port_mode_index >= self.port_mode_modes:
            raise ValueError(
                f"port_mode_index {self.port_mode_index} is out of reach of "
                f"port_mode_modes {self.port_mode_modes}; compute at least "
                f"{self.port_mode_index + 1} eigenpairs to select that mode"
            )
        for name in ("waveport_width_mm", "waveport_height_mm"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.air_pad_mm is not None:
            faces = _air_pad_faces_mm(self.air_pad_mm)
            for axis, (low, high) in zip("xyz", faces, strict=True):
                for pad, sign in ((low, "-"), (high, "+")):
                    if pad < 0:
                        raise ValueError(
                            f"air_pad_mm cannot be negative: {pad} on the "
                            f"{axis}{sign} face of {self.air_pad_mm!r}. Zero is "
                            "allowed and puts the domain boundary on that face "
                            "of the structure."
                        )

    @property
    def air_pad_faces_mm(self) -> tuple[tuple[float, float], ...] | None:
        """
        Return the air padding per face, or ``None`` to use ``air_pad_frac``.

        Returns
        -------
        tuple[tuple[float, float], ...] or None
            ``((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))`` in millimetres,
            or ``None`` when no explicit padding is set.
        """
        if self.air_pad_mm is None:
            return None
        return _air_pad_faces_mm(self.air_pad_mm)


_FEM_DEFAULTS = (
    FEMOptions()
)  # single source of truth for simulate_step_FEM's flat defaults


@dataclass
class SolidSpec:
    """
    One CAD solid and the electromagnetic role assigned to it.

    Parameters
    ----------
    name : str
        Solid name, matching the CSXCAD property or STEP product name.
    role : str
        What the solid is treated as: ``"dielectric"``, ``"pec"``,
        ``"lossy_conductor"``, ``"port"``, or ``"ignore"`` to leave it out of
        the simulation. Default ``"ignore"``.
    dielectric : Dielectric, optional
        Material properties, used when ``role`` is ``"dielectric"``. Default
        ``None``.
    sigma : float
        Conductivity in S/m, used when ``role`` is ``"lossy_conductor"``.
        Default ``0.0``.
    """

    name: str
    role: str = "ignore"  # 'dielectric' | 'pec' | 'lossy_conductor' | 'port' | 'ignore'
    dielectric: Dielectric | None = None
    sigma: float = 0.0  # conductivity [S/m] for role == 'lossy_conductor'


@dataclass
class PortSpec:
    """
    A port defined on a named solid.

    Parameters
    ----------
    solid : str
        Name of the solid that becomes the port. A coplanar waveguide port is
        drawn as one solid per gap, so see ``solids`` for the rest of them.
    number : int
        One-based port index, setting the row/column order of the S-parameter
        matrix.
    z0 : float
        Reference impedance in ohms. Default ``50.0``.
    direction : str
        Axis the port is excited along: ``"x"``, ``"y"``, or ``"z"``. For a
        wave port this is instead the axis its voltage is measured along, used
        to report a characteristic impedance. Default ``"z"``.
    kind : str
        ``"lumped"`` (default) drives one constant field direction across the
        port sheet -- right for a gap feed, wrong for a transmission line.
        ``"wave"`` solves the transverse mode of the port's cross-section and
        drives that instead, which is what a microstrip or coplanar waveguide
        needs. See :mod:`~simpleEMS.fem_port_mode`.
    prop_dir : str
        For a wave port, the axis the line runs along; the port face is the
        cross-section normal to it. ``"x"``, ``"y"``, or ``"z"``, or ``""``
        (the default) to derive it from the port's own geometry. Unused by a
        lumped port.
    solids : list[str]
        Every solid making up this port. A lumped port has one; a coplanar
        waveguide port has one per gap, and all of them together define the
        one cross-section its mode is solved on. Defaults to ``[solid]``.
    """

    solid: str
    number: int
    z0: float = 50.0
    direction: str = "z"  # excitation E-field axis: 'x' | 'y' | 'z'
    kind: str = "lumped"  # 'lumped' | 'wave'
    prop_dir: str = ""  # propagation axis of a wave port; "" derives it
    solids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate the port kind and its propagation axis.

        Raises
        ------
        ValueError
            If ``kind`` is not a supported choice, or if a wave port's
            ``prop_dir`` is not an axis, or is the same axis as ``direction``
            (the voltage across a line is measured across it, not along it).
        """
        if self.kind not in ("lumped", "wave"):
            raise ValueError(f"kind must be 'lumped' or 'wave', got {self.kind!r}")
        if not self.solids:
            self.solids = [self.solid]
        if self.prop_dir and self.prop_dir not in ("x", "y", "z"):
            raise ValueError(
                f"prop_dir must be 'x', 'y' or 'z', or '' to derive it, "
                f"got {self.prop_dir!r}"
            )
        if self.kind == "wave" and self.prop_dir == self.direction:
            raise ValueError(
                f"a wave port's prop_dir ({self.prop_dir!r}) is the axis the "
                f"line runs along and direction ({self.direction!r}) the axis "
                "its voltage is measured across, so they cannot be the same"
            )


@dataclass
class Problem:
    """
    A complete problem definition for the FEM backend to solve.

    Parameters
    ----------
    step_file : str
        Path to the STEP file to mesh and solve.
    name : str
        Base name given to the generated mesh and problem files. Default
        ``"structure"``.
    solids : dict[str, SolidSpec]
        Role and material assigned to each solid, keyed by solid name.
    ports : list[PortSpec]
        The ports found among ``solids``, sorted by port number.
    freqs : NDArray
        Frequency points (Hz) the problem is solved over. Default
        ``[2.45e9]``.
    options : FEMOptions
        Solver and mesh options. Each is also readable directly off the
        problem (e.g. ``problem.fe_order``) through the properties below.
    """

    step_file: str
    name: str = "structure"
    solids: dict[str, SolidSpec] = field(default_factory=dict)
    ports: list[PortSpec] = field(default_factory=list)
    freqs: NDArray = field(default_factory=lambda: np.array([2.45e9]))
    options: FEMOptions = field(default_factory=FEMOptions)

    def dielectrics(self) -> list[SolidSpec]:
        """Return the solids assigned the dielectric role."""
        return [s for s in self.solids.values() if s.role == "dielectric"]

    # Read-only pass-throughs onto `options` -- FEMOptions is the only place
    # that declares these defaults; Problem never redeclares them.
    @property
    def boundary(self) -> str:
        """Outer boundary condition: ``"silver_muller"``, ``"pml"`` or ``"pec"``."""
        return self.options.boundary

    @property
    def fe_order(self) -> int:
        """Element order: ``1`` or ``2``."""
        return self.options.fe_order

    @property
    def symmetry(self) -> tuple | None:
        """Mirror-symmetry plane ``(axis, kind, at)``, or ``None``."""
        return self.options.symmetry

    @property
    def air_pad_frac(self) -> float:
        """Air padding as a fraction of the wavelength at ``mesh_freq``."""
        return self.options.air_pad_frac

    @property
    def air_pad_mm(self) -> float | tuple | None:
        """Air padding in millimetres, or ``None`` to use ``air_pad_frac``."""
        return self.options.air_pad_mm

    @property
    def air_pad_faces_mm(self) -> tuple[tuple[float, float], ...] | None:
        """Air padding per face in mm, or ``None`` to use ``air_pad_frac``."""
        return self.options.air_pad_faces_mm

    @property
    def elems_per_wavelength(self) -> float:
        """Target number of mesh elements per wavelength."""
        return self.options.elems_per_wavelength

    @property
    def mesh_freq(self) -> float:
        """Frequency the mesh is sized at, in Hz; the top of the sweep if unset."""
        return self.options.mesh_freq or float(np.max(self.freqs))

    @property
    def mesh_fine_scale(self) -> float:
        """Multiplier on the element size near conductors."""
        return self.options.mesh_fine_scale

    @property
    def min_layers(self) -> int:
        """Number of element layers through the dielectric thickness."""
        return self.options.min_layers

    @property
    def num_solve_points(self) -> int:
        """Number of frequencies the sweep solves at."""
        return self.options.num_solve_points

    @property
    def max_solve_points(self) -> int | None:
        """Ceiling on the solve count when the curve breaks passivity."""
        return self.options.max_solve_points

    @property
    def port_type(self) -> str:
        """Port the FEM backend makes: ``"lumpedport"`` or ``"waveport"``."""
        return self.options.port_type

    @property
    def port_mode_modes(self) -> int:
        """Eigenpairs each wave port's mode solve computes."""
        return self.options.port_mode_modes

    @property
    def port_mode_index(self) -> int:
        """Which guided mode a wave port runs in, largest ``beta`` first."""
        return self.options.port_mode_index

    @property
    def port_mode_eps_eff(self) -> float | None:
        """Effective permittivity naming a wave port's mode, or ``None``."""
        return self.options.port_mode_eps_eff

    @property
    def port_mode_zc(self) -> float | None:
        """Wave-port reference impedance override, in ohms, or ``None``."""
        return self.options.port_mode_zc

    @property
    def waveport_width_mm(self) -> float | None:
        """Wave-port cross-section width in mm, or ``None`` for the default."""
        return self.options.waveport_width_mm

    @property
    def waveport_height_mm(self) -> float | None:
        """Wave-port cross-section height in mm, or ``None`` for the default."""
        return self.options.waveport_height_mm


# ----------------------------
# CSX -> Problem mapping
# ----------------------------
def _port_number(name: str, fallback: int) -> int:
    """Read the port number off the end of ``name``, else use ``fallback``."""
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else fallback


def _register_port(
    prob: Problem,
    seen: set[int],
    solid: str,
    number: int,
    z0: float,
    direction: str = "z",
    kind: str = "lumped",
    prop_dir: str = "",
) -> None:
    """Register ``solid`` as part of port ``number``.

    A coplanar waveguide port is drawn as one solid per gap, so a port number
    can turn up more than once. The extra solids are added to the port that
    already exists rather than dropped -- dropping them is how the second gap
    used to end up meshed with no boundary condition at all.
    """
    if number in seen:
        for existing in prob.ports:
            if existing.number == number and solid not in existing.solids:
                existing.solids.append(solid)
        return
    seen.add(number)
    prob.ports.append(
        PortSpec(
            solid=solid,
            number=number,
            z0=z0,
            direction=direction,
            kind=kind,
            prop_dir=prop_dir,
            solids=[solid],
        )
    )


def _csx_roles(
    csx: ContinuousStructure, kappa_freq: float, port_type: str = "lumpedport"
) -> tuple[dict, dict, dict, dict]:
    """
    Assign an electromagnetic role to each CSXCAD property.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry.
    kappa_freq : float
        Frequency in Hz at which each material's ``kappa`` was set, which is
        where it has to be read back as a loss tangent.
    port_type : str
        ``"lumpedport"`` (default) or ``"waveport"``, deciding what the ports
        already in the geometry become. A coplanar waveguide port is a wave
        port either way.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(role_by_name, dielectric_by_name, port_by_name, sigma_by_name)``,
        each keyed by property name: the role string, the
        :class:`~simpleEMS.fem_materials.Dielectric` of each dielectric, the
        ``(z0, direction, number, kind, is_cpw)`` of each port, and the
        conductivity in S/m of each lossy conductor.
    """
    role_by_name: dict[str, str] = {}
    dielectric_by_name: dict[str, Dielectric] = {}
    port_by_name: dict[str, tuple[float, str, int, str, bool]] = {}
    sigma_by_name: dict[str, float] = {}

    # The EM role of each solid is read straight from its CSXCAD property type,
    # so the same geometry the user built for openEMS maps to the FEM problem
    # with no extra annotation: Metal -> PEC, Material -> dielectric, a lumped
    # element -> a port, and a conducting sheet -> a lossy (surface-impedance)
    # conductor.
    for prop in csx.GetAllProperties():
        type_string = prop.GetTypeString()
        name = prop.GetName()
        if type_string == "Metal":
            role_by_name[name] = "pec"
        elif type_string == "ConductingSheet":
            # A conducting sheet is a lossy conductor: its finite conductivity
            # becomes a surface-impedance (Leontovich) boundary in the .pro.
            role_by_name[name] = "lossy_conductor"
            sigma_by_name[name] = float(prop.GetConductivity())
        elif type_string == "Material":
            eps_r = float(prop.GetMaterialProperty("epsilon"))
            # openEMS stores dielectric loss as an equivalent conductivity
            # kappa [S/m], fixed at the frequency it was defined at:
            # kappa = tan_d * w * eps0 * eps_r. GetDP wants tan_d, so invert it
            # at that same frequency -- anywhere else scales the loss by the
            # ratio of the two.
            kappa = float(prop.GetMaterialProperty("kappa"))
            tan_d = (
                kappa / (2 * math.pi * kappa_freq * EPS0 * eps_r)
                if eps_r > 0 and kappa
                else 0.0
            )
            role_by_name[name] = "dielectric"
            dielectric_by_name[name] = Dielectric(eps_r=eps_r, tan_d=tan_d)
        elif type_string == "LumpedElement":
            # A lumped element is the excitation port: its resistance is the
            # reference impedance z0, its direction the E-field axis, and a
            # trailing number in the name (port_resist_<N>) the port index.
            z0 = float(prop.GetResistance())
            direction = _AXIS_TO_DIR[int(prop.GetDirection())]
            number = _port_number(name, len(port_by_name) + 1)
            role_by_name[name] = "port"

            # How many boxes the property carries says which port openEMS drew.
            # AddLumpedPort adds exactly one; AddCPWPort adds one per gap
            # (openEMS/ports.py), and that is the only trace of the difference
            # a CSXCAD property keeps.
            is_cpw = len(prop.GetAllPrimitives()) > 1
            kind = "wave" if (is_cpw or port_type == "waveport") else "lumped"
            if is_cpw:
                # openEMS puts 2*Feed_R on a CPW port -- "applied to each gap
                # as 2*R" -- so halve it to get back the line impedance the
                # S-parameters should be referenced to.
                z0 = z0 / 2.0
            port_by_name[name] = (z0, direction, number, kind, is_cpw)

    return role_by_name, dielectric_by_name, port_by_name, sigma_by_name


def _apply_FEM_options(prob: Problem, FEM_options: FEMOptions | None) -> None:
    """Set ``prob``'s options to ``FEM_options``, in place, unless it is ``None``."""
    if FEM_options is not None:
        prob.options = FEM_options


def _build_problem(
    csx: ContinuousStructure,
    freqs: NDArray,
    output_path: Path,
    FEM_options: FEMOptions | None = None,
) -> Problem:
    """
    Export ``csx`` to STEP and build a :class:`Problem` from its properties.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry to solve.
    freqs : NDArray
        Frequency points (Hz) to solve over.
    output_path : Path
        Directory to export ``structure.step`` into.
    FEM_options : FEMOptions, optional
        Solver and mesh options. ``None`` keeps the defaults.

    Returns
    -------
    Problem
        The populated FEM problem.
    """
    # Gmsh meshes STEP, not CSXCAD, so we round-trip the geometry through a STEP
    # file. export_step names each solid after its CSXCAD property, which is how
    # we match a meshed solid back to the role read from CSX above.
    output_path.mkdir(parents=True, exist_ok=True)
    export_step(csx, output_path)
    step_file = str(output_path / "structure.step")

    freqs = np.asarray(freqs, dtype=float)
    options = FEM_options or FEMOptions()
    # openEMS stores dielectric loss as a conductivity fixed at one frequency,
    # so kappa has to be read back at that same frequency or the loss comes out
    # scaled by the ratio of the two. The sweep's centre is only a fallback for
    # a geometry that reached here without params behind it.
    kappa_freq = options.kappa_freq or 0.5 * (float(freqs.min()) + float(freqs.max()))
    role_by_name, dielectric_by_name, port_by_name, sigma_by_name = _csx_roles(
        csx, kappa_freq, options.port_type
    )

    prob = Problem(step_file=step_file, name="structure", freqs=freqs)
    seen_ports: set[int] = set()
    for solid_name in fem_geometry.list_solids(step_file):
        # A property with several primitives is exported as name_0, name_1, ...;
        # strip a trailing _<int> so each piece maps back to the parent role.
        base = (
            solid_name
            if solid_name in role_by_name
            else re.sub(r"_\d+$", "", solid_name)
        )
        role = role_by_name.get(base, "ignore")
        prob.solids[solid_name] = SolidSpec(
            name=solid_name,
            role=role,
            dielectric=dielectric_by_name.get(base),
            sigma=sigma_by_name.get(base, 0.0),
        )
        if role == "port":
            z0, direction, number, kind, _is_cpw = port_by_name[base]
            _register_port(prob, seen_ports, solid_name, number, z0, direction, kind)

    prob.ports.sort(key=lambda p: p.number)
    _apply_FEM_options(prob, FEM_options)

    if options.port_type != "waveport":
        promoted = sorted(
            {
                port.number
                for name, (_z, _d, number, kind, is_cpw) in port_by_name.items()
                if is_cpw and kind == "wave"
                for port in prob.ports
                if port.number == number
            }
        )
        if promoted:
            console.print(
                f"[info]port(s) {promoted}: coplanar waveguide, so solved as a "
                "wave port rather than a lumped one -- the CPW mode is odd and "
                "a lumped port would excite the wrong one. Set "
                'FEM_port_type="waveport" to do this everywhere.[/info]'
            )
    return prob


# ----------------------------
# pipeline stages
# ----------------------------
def _module_bytes(mod: object) -> bytes:
    """Source bytes of ``mod``, or its name if the file cannot be read."""
    path = getattr(mod, "__file__", None)
    try:
        return Path(path).read_bytes() if path else str(mod).encode()
    except OSError:
        return str(mod).encode()


def _mesh_fingerprint(
    csx: ContinuousStructure, freqs: NDArray, FEM_options: FEMOptions | None
) -> str:
    """Hash of everything that affects the mesh, so it can be reused or rebuilt.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry.
    freqs : NDArray
        Frequency points (Hz) to solve over.
    FEM_options : FEMOptions, optional
        Solver and mesh options.

    Returns
    -------
    str
        Hex digest covering the geometry, the frequencies, the options, and
        the source of the modules that generate the mesh and problem files.
    """
    # Read straight off the live CSXCAD properties rather than exporting STEP
    # or meshing, so this costs nothing close to a rebuild. The mesh/.pro
    # generating modules are hashed in too: a cache hit skips write_problem as
    # well as the meshing, so without them an edit to the formulation would
    # silently leave a stale .pro in place.
    parts = []
    this_module = sys.modules[__name__]  # fem_backend itself; see the note above
    for mod in (fem_formulation, fem_geometry, fem_materials, this_module):
        parts.append(f"src:{hashlib.sha256(_module_bytes(mod)).hexdigest()}")
    for prop in csx.GetAllProperties():
        parts.append(f"{prop.__class__.__name__}:{prop.GetName()}")
        for prim in prop.GetAllPrimitives():
            cls = prim.__class__.__name__
            if cls == "CSPrimBox":
                parts.append(f"box:{prim.GetStart()}:{prim.GetStop()}")
            elif cls == "CSPrimLinPoly":
                parts.append(
                    f"linpoly:{prim.GetCoords()}:{prim.GetElevation()}:"
                    f"{prim.GetNormDir()}:{prim.GetLength()}"
                )
            elif cls in ("CSPrimCylinder", "CSPrimCylindricalShell"):
                # Vias are cylinders; without their dimensions here, moving a
                # via or resizing it leaves the fingerprint unchanged and a
                # stale mesh is reused.
                width = prim.GetShellWidth() if cls == "CSPrimCylindricalShell" else 0.0
                parts.append(
                    f"cylinder:{prim.GetStart()}:{prim.GetStop()}:"
                    f"{prim.GetRadius()}:{width}"
                )
            else:
                parts.append(cls)
    parts.append(f"freqs:{np.asarray(freqs, dtype=float).tobytes()!r}")
    parts.append(f"FEM_options:{asdict(FEM_options) if FEM_options else None}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _mesh_problem(
    prob: Problem, output_path: Path, verbose: bool = True, fingerprint: str = ""
) -> str:
    """
    Mesh a :class:`Problem`, write its problem file, and record the metadata.

    Parameters
    ----------
    prob : Problem
        The FEM problem to mesh.
    output_path : Path
        Directory to write the mesh, problem, and metadata files into.
    verbose : bool
        Print progress. Default ``True``.
    fingerprint : str
        Hash of the mesh inputs, stored alongside the mesh so a later
        :func:`build_mesh` call can tell whether the mesh is still valid.
        Default ``""``, meaning the mesh is never reused.

    Returns
    -------
    str
        Path to the generated ``.msh`` file.

    Raises
    ------
    RuntimeError
        If the problem has no ports.
    """
    if not prob.ports:
        raise RuntimeError(
            "No ports found in the geometry; a FEM simulation needs at least one port."
        )

    # Mesh once (fem_geometry) and emit the matching GetDP problem file
    # (fem_formulation); both are reused for every frequency in the sweep.
    mesh = fem_geometry.build_mesh(prob, output_path, verbose=verbose)
    pro_path = fem_formulation.write_problem(prob, mesh, output_path)

    # Persist just what the sweep/compute stages need, so they do not have to
    # re-mesh or re-derive the port reference impedances.
    meta = {
        "msh_path": mesh.msh_path,
        "pro_path": pro_path,
        "name": prob.name,
        "bbox": list(mesh.bbox),  # structure extents (m); needed for far-field box
        "boundary": mesh.boundary,  # a "pec" box cannot radiate; see fem_radiation
        "domain_bbox": list(mesh.inner_bbox),  # meshed E/H extents (m); Huygens
        # box must stay strictly inside this or CutBox samples outside the mesh
        "symmetry_axis": (
            {"x": 0, "y": 1, "z": 2}[prob.symmetry[0]] if prob.symmetry else None
        ),  # axis whose min-face sits on the symmetry plane, not open air
        # so the far-field transform can mirror the half model back to whole
        "symmetry_plane": mesh.sym_plane if prob.symmetry else None,
        "symmetry_kind": mesh.sym_kind if prob.symmetry else None,
        "port_numbers": sorted(pm.number for pm in mesh.port_regions.values()),
        "ref_impedances": {
            str(pm.number): pm.ref_impedance for pm in mesh.port_regions.values()
        },
        # gap lengths, for the wave-amplitude scaling in _sweep_from_meta
        "port_gaps": {str(pm.number): pm.gap for pm in mesh.port_regions.values()},
        # A wave port's cross-section and its 2D problem file depend only on the
        # geometry, so they are cut and written here, once, and the sweep only
        # re-solves the mode at each of its frequencies.
        "port_modes": {
            str(pm.number): fem_port_mode.prepare_port_mode(
                prob, mesh, pm, output_path
            ).to_dict()
            for pm in sorted(mesh.port_regions.values(), key=lambda p: p.number)
            if pm.is_wave
        },
        "fingerprint": fingerprint,
    }
    (output_path / _MESH_META).write_text(json.dumps(meta, indent=2))
    return mesh.msh_path


def build_mesh(
    csx: ContinuousStructure,
    freqs: NDArray,
    output_path: str | Path,
    verbose: bool = True,
    FEM_options: FEMOptions | None = None,
) -> str:
    """
    Mesh a CSXCAD geometry for the FEM backend.

    Exports the geometry to STEP, meshes it, writes the problem file, and
    records the port metadata the later stages read. A mesh already in
    ``output_path`` is reused instead if nothing affecting it has changed, so
    this is cheap to call before every simulation.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry.
    freqs : NDArray
        Frequency points (Hz) to solve over.
    output_path : str | Path
        Directory for the mesh and problem files.
    verbose : bool
        Print progress. Default ``True``.
    FEM_options : FEMOptions, optional
        Solver and mesh options. ``None`` keeps the defaults.

    Returns
    -------
    str
        Path to the ``.msh`` file, whether reused or freshly built.

    Raises
    ------
    RuntimeError
        If the geometry contains no ports.
    """
    output_path = Path(output_path)
    fingerprint = _mesh_fingerprint(csx, freqs, FEM_options)
    meta_file = output_path / _MESH_META
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        msh_path = meta.get("msh_path")
        if (
            meta.get("fingerprint") == fingerprint
            and msh_path is not None
            and Path(msh_path).exists()
        ):
            if verbose:
                console.print(f"[info]reusing unchanged FEM mesh: {msh_path}[/info]")
            return msh_path

    prob = _build_problem(csx, freqs, output_path, FEM_options)
    return _mesh_problem(prob, output_path, verbose, fingerprint)


def _renormalise(s: NDArray, z_old: NDArray, z_new: NDArray) -> NDArray:
    """
    Re-reference an S-matrix from one set of port impedances to another.

    A wave port's S-parameters come out referenced to the port's *own* modal
    impedance, because that is what a modal wave amplitude is measured
    against, and that impedance is whatever the line happens to have at that
    frequency. Everything downstream -- ``z11``, VSWR, a Touchstone export --
    expects them referenced to the impedance the user asked for, so they are
    converted here.

    Parameters
    ----------
    s : NDArray
        The ``(n, n)`` S-matrix as solved.
    z_old : NDArray
        Reference impedance of each port as solved, in ohms.
    z_new : NDArray
        Reference impedance to convert to, in ohms.

    Returns
    -------
    NDArray
        The re-referenced S-matrix.
    """
    z_old = np.asarray(z_old, dtype=float)
    z_new = np.asarray(z_new, dtype=float)
    if np.allclose(z_old, z_new):
        return s
    r = (z_new - z_old) / (z_new + z_old)
    gamma = np.diag(r)
    a = np.diag(np.sqrt(1.0 - r**2) / (1.0 - r))
    n = s.shape[0]
    return np.linalg.solve(a.T, (s - gamma) @ np.linalg.solve(np.eye(n) - gamma @ s, a))


def _sweep_from_meta(
    freqs: NDArray,
    num_solve_points: int,
    output_path: Path,
    verbose: bool = True,
    max_solve_points: int | None = None,
) -> None:
    """
    Run the frequency sweep on the mesh recorded in ``fem_mesh.json``.

    Parameters
    ----------
    freqs : NDArray
        Frequency points (Hz) to report results at.
    num_solve_points : int
        Number of frequencies the sweep solves at.
    max_solve_points : int, optional
        Ceiling on the solve count when the curve breaks passivity. Default
        ``None``, which allows twice ``num_solve_points``.
    output_path : Path
        Directory holding the mesh metadata written by :func:`build_mesh`, and
        receiving the ``fem_sparams.npz`` results.
    verbose : bool
        Print progress. Default ``True``.
    """
    meta = json.loads((output_path / _MESH_META).read_text())

    pro_path = meta["pro_path"]
    msh_path = meta["msh_path"]
    port_numbers = list(meta["port_numbers"])
    outdir = output_path.absolute() / "output"
    idx = {n: i for i, n in enumerate(port_numbers)}  # port number -> matrix index
    npt = len(port_numbers)
    ref_port = port_numbers[0]  # SimData reports V/I/input_power for this port only

    # The .pro reports a field overlap; an S-parameter is a ratio of wave
    # amplitudes, so scale by (gap_n/gap_k)*sqrt(Z_k/Z_n). This is 1 when the
    # ports match. Applied here so the stored matrix is a true S-matrix, which
    # the rational sweep and its passivity check both assume.
    ref_z = meta["ref_impedances"]
    ref_impedances = [ref_z[str(n)] for n in port_numbers]
    gaps = meta["port_gaps"]
    mode_setups = {
        int(n): fem_port_mode.PortModeSetup.from_dict(d)
        for n, d in meta.get("port_modes", {}).items()
    }

    def _amp_scale(n: int, k: int) -> float:
        """Overlap-to-wave-amplitude factor for the S_nk entry."""
        # A wave port's overlap is already divided by its mode's self-overlap,
        # so it is a wave amplitude as it stands and needs no rescaling. Only
        # the lumped ports' field projections do.
        if n in mode_setups or k in mode_setups:
            return 1.0
        return (gaps[str(n)] / gaps[str(k)]) * math.sqrt(ref_z[str(k)] / ref_z[str(n)])

    wave_norm = np.array(
        [[_amp_scale(n, k) for k in port_numbers] for n in port_numbers]
    )

    # GetDP appends to these Format Table files (write_problem's xs_file =
    # "File >") instead of overwriting, so a sweep's per-solve rows accumulate
    # in one file. Clear any left over from a previous sweep/run first, or
    # this run's rows would land after stale ones. Rows are read by position,
    # so leftovers would misalign the whole S-matrix.
    stale_patterns = [
        "intPort.txt",
        "xS_*.txt",
        "V_*.txt",
        "I_*.txt",
        "Ploss.txt",
        "Pcond_*.txt",
        "Vdrv_*.txt",
        "Idrv_*.txt",
        "mode_*.pos",
        "et_*.pos",
    ]
    for pattern in stale_patterns:
        for stale in outdir.glob(pattern):
            stale.unlink()

    # freq -> (V, I) of the reference port's own total voltage/current, sampled
    # only at the frequencies the adaptive S-sweep below actually solves (no
    # extra getdp runs: V_<n>/I_<n> are written by the same Get_SParameters call
    # that already produces xS_<n> at each (freq, active-port) solve).
    vi_solved: dict[float, tuple[complex, complex]] = {}

    # One full FEM solve at a single frequency, producing the whole S-matrix.
    # A single getdp launch drives every port, reusing one factorisation, and
    # appends a row per port to xS_<n>.txt in the resolution's port order.
    def solve_at(freq: float) -> NDArray:
        s = np.zeros((npt, npt), dtype=complex)
        # Each wave port's mode is re-solved here rather than once for the whole
        # sweep: a line's effective permittivity disperses, so both the profile
        # the 3D solve is driven with and the modal index its port term is
        # scaled by belong to this frequency.
        setnumbers: dict[str, float] = {"FREQ": freq}
        modal_z: dict[int, float] = {}
        for number, setup in sorted(mode_setups.items()):
            # Quiet: the mode solve is a step inside this frequency's solve,
            # not a solve of its own, and the line below reports what it found.
            mode = fem_port_mode.solve_port_mode(
                setup, freq, output_path, verbose=False
            )
            # Both parts: the modal admittance of a lossy line is complex, and
            # GetDP's -setnumber only carries reals.
            setnumbers[f"NEFF_RE_{number}"] = float(mode.n_eff.real)
            setnumbers[f"NEFF_IM_{number}"] = float(mode.n_eff.imag)
            setnumbers[f"ZC_{number}"] = float(mode.zc)
            modal_z[number] = float(mode.zc)
            if verbose:
                console.print(
                    f"[info]port {number} mode @ {freq / 1e9:.4g} GHz: "
                    f"eps_eff={mode.eps_eff.real:.4g}, Zc={mode.zc:.4g} ohm, "
                    f"alpha={mode.alpha_db_per_m:.4g} dB/m[/info]"
                )
        fem_solver.run_getdp(
            pro_path,
            msh_path,
            output_path,
            setnumbers,
            None,  # Analysis runs Get_SParameters itself, once per port
            label=f"S-params @ {freq / 1e9:.4f} GHz",
            verbose=False,  # the sweep reports this solve on its own line
        )
        for n in port_numbers:
            row = fem_solver.read_complex_rows(outdir / f"xS_{n}.txt", npt)
            for active, value in zip(port_numbers, row, strict=True):
                s[idx[n], idx[active]] = value * wave_norm[idx[n], idx[active]]
        # A wave port measured against its own mode, so its S is referenced to
        # that mode's impedance; convert to the impedance the user asked for.
        if mode_setups:
            z_solved = np.array(
                [modal_z.get(n, ref_z[str(n)]) for n in port_numbers], dtype=float
            )
            s = _renormalise(s, z_solved, np.array(ref_impedances, dtype=float))
        # V/I of the reference port while it is the driven one.
        v_rows = fem_solver.read_complex_rows(outdir / f"V_{ref_port}.txt", npt)
        i_rows = fem_solver.read_complex_rows(outdir / f"I_{ref_port}.txt", npt)
        vi_solved[freq] = (v_rows[idx[ref_port]], i_rows[idx[ref_port]])
        return s

    # The expensive part is each solve_at() call, so let the adaptive sweep pick
    # a handful of frequencies (<= num_solve_points), then rational-interpolate
    # S(f) onto the full dense grid.
    freqs = np.asarray(freqs, dtype=float)
    s_dense = fem_sweep.rational_sweep(
        freqs,
        port_numbers,
        solve_at,
        num_solve_points,
        max_solves=max_solve_points,
        verbose=verbose,
    )

    # V/I are linear in the same field solution S was fit from, so the same
    # sparse solve points suffice -- rational-interpolate them the same way
    # (AAA is much better conditioned on data scaled to [-1, 1] than raw Hz).
    fmin, fmax = float(freqs[0]), float(freqs[-1])
    span = (fmax - fmin) or 1.0

    def _zof(f: NDArray) -> NDArray:
        return (2 * np.asarray(f) - (fmin + fmax)) / span

    vi_freqs = np.array(sorted(vi_solved))
    v_samples = np.array([vi_solved[f][0] for f in vi_freqs])
    i_samples = np.array([vi_solved[f][1] for f in vi_freqs])
    v_dense = AAA(_zof(vi_freqs), v_samples)(_zof(freqs))
    i_dense = AAA(_zof(vi_freqs), i_samples)(_zof(freqs))

    np.savez(
        output_path / _SPARAMS,
        freqs=freqs,
        S=s_dense,
        port_numbers=np.asarray(port_numbers),
        ref_impedances=np.asarray(ref_impedances),
        port_voltage=v_dense,
        port_current=i_dense,
    )
    if verbose:
        console.print(
            f"[success]FEM S-parameters written to {output_path / _SPARAMS}[/success]"
        )


def run_sweep(
    csx: ContinuousStructure,
    freqs: NDArray,
    output_path: str | Path,
    verbose: bool = True,
    FEM_options: FEMOptions | None = None,
) -> None:
    """
    Run the FEM frequency sweep on a CSXCAD geometry.

    Meshes the geometry first (reusing an existing mesh where possible), then
    solves at ``FEM_options.num_solve_points`` frequencies, interpolates the
    S-parameters onto ``freqs``, and writes them to ``fem_sparams.npz`` in
    ``output_path``.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry.
    freqs : NDArray
        Frequency points (Hz) to report results at.
    output_path : str | Path
        Directory holding the mesh and problem files, and receiving the
        results.
    verbose : bool
        Print progress. Default ``True``.
    FEM_options : FEMOptions, optional
        Solver and mesh options, including ``num_solve_points``. ``None``
        keeps the defaults.

    Raises
    ------
    RuntimeError
        If the geometry contains no ports.
    """
    output_path = Path(output_path)
    build_mesh(csx, freqs, output_path, verbose=verbose, FEM_options=FEM_options)
    options = FEM_options or FEMOptions()
    _sweep_from_meta(
        freqs,
        options.num_solve_points,
        output_path,
        verbose,
        max_solve_points=options.max_solve_points,
    )


def compute_sim_data(
    freqs: NDArray, charac_imp: float, output_path: str | Path
) -> SimData:
    """
    Read the FEM sweep results into a :class:`~simpleEMS.sim_tools.SimData`.

    Parameters
    ----------
    freqs : NDArray
        Frequency points. Unused -- the frequencies stored with the results
        are used instead; accepted so this matches the FDTD equivalent.
    charac_imp : float
        Reference impedance in ohms, used only if none was stored with the
        results.
    output_path : str | Path
        Directory containing the ``fem_sparams.npz`` results written by
        :func:`run_sweep`.

    Returns
    -------
    SimData
        Named tuple of ``freqs``, ``s11``, ``s21`` (``None`` for a single-port
        problem), ``z11``, ``vswr``, ``input_power``, ``port_voltage``,
        ``port_current``, and ``ref_impedance``. See
        :class:`~simpleEMS.sim_tools.SimData`.

    Raises
    ------
    RuntimeError
        If the sweep results file is missing.
    """
    from .sim_tools import SimData

    output_path = Path(output_path)
    npz_path = output_path / _SPARAMS
    if not npz_path.exists():
        raise RuntimeError(
            f"FEM results not found at {npz_path}; run the simulation first."
        )

    data = np.load(npz_path)
    s = data["S"]
    port_voltage = data["port_voltage"]
    port_current = data["port_current"]
    freqs_out = data["freqs"]
    ref_impedances = data["ref_impedances"]
    nports = s.shape[1]

    s11 = s[:, 0, 0]
    s21 = s[:, 1, 0] if nports >= 2 else None
    z0 = float(ref_impedances[0]) if ref_impedances.size else charac_imp
    z11 = z0 * (1 + s11) / (1 - s11)  # input impedance from the reflection coeff
    s11_mag = np.clip(np.abs(s11), 0, 0.999)  # prevent division by zero error
    vswr = (1 + s11_mag) / (1 - s11_mag)
    # accepted power at the driven (reference) port, from its total voltage and
    # current (see fem_formulation.write_problem's V_n/I_n post-quantities).
    input_power = 0.5 * np.real(port_voltage * np.conj(port_current))

    return SimData(
        freqs_out, s11, s21, z11, vswr, input_power, port_voltage, port_current, z0
    )


# ----------------------------
# standalone STEP entry point
# ----------------------------
def _problem_from_step(
    step_file: str,
    freqs: NDArray,
    dielectrics: dict | None,
    pec: list | None,
    lossy_conductor: dict | None,
    ports: dict | None,
    charac_imp: float,
    FEM_options: FEMOptions | None,
) -> Problem:
    """Build a :class:`Problem` from a STEP file's solids, without CSXCAD."""
    step_file = str(Path(step_file).expanduser())
    freqs = np.asarray(freqs, dtype=float)
    dielectrics = dielectrics or {}
    pec_names = set(pec or [])
    lossy_conductor = lossy_conductor or {}
    ports = ports or {}

    prob = Problem(step_file=step_file, name="structure", freqs=freqs)
    # There is no CSXCAD here to read a port kind off, so FEM_port_type is the
    # default and the per-port dict overrides it.
    default_kind = (
        "wave" if (FEM_options or FEMOptions()).port_type == "waveport" else "lumped"
    )
    seen_ports: set[int] = set()
    for name in fem_geometry.list_solids(step_file):
        # explicit overrides win; otherwise guess the role from the solid name
        if name in dielectrics:
            eps_r, tan_d = dielectrics[name]
            prob.solids[name] = SolidSpec(
                name=name,
                role="dielectric",
                dielectric=Dielectric(eps_r=eps_r, tan_d=tan_d),
            )
        elif name in lossy_conductor:
            prob.solids[name] = SolidSpec(
                name=name, role="lossy_conductor", sigma=float(lossy_conductor[name])
            )
        elif name in pec_names:
            prob.solids[name] = SolidSpec(name=name, role="pec")
        elif name in ports:
            spec = ports[name]
            prob.solids[name] = SolidSpec(name=name, role="port")
            number = int(spec.get("number", len(prob.ports) + 1))
            _register_port(
                prob,
                seen_ports,
                name,
                number,
                float(spec.get("z0", charac_imp)),
                spec.get("direction", "z"),
                spec.get("kind", default_kind),
                spec.get("prop_dir", ""),
            )
        else:
            role = guess_role(name) or "ignore"
            solid = SolidSpec(name=name, role=role)
            if role == "dielectric":
                # placeholder eps_r=1; override via the `dielectrics` argument
                solid.dielectric = Dielectric()
            prob.solids[name] = solid
            if role == "port":
                _register_port(
                    prob,
                    seen_ports,
                    name,
                    len(prob.ports) + 1,
                    charac_imp,
                    kind=default_kind,
                )

    prob.ports.sort(key=lambda p: p.number)
    _apply_FEM_options(prob, FEM_options)
    return prob


def simulate_step_FEM(
    step_file: str,
    freqs: NDArray,
    *,
    unit: str = "mm",
    dielectrics: dict | None = None,
    pec: list | None = None,
    lossy_conductor: dict | None = None,
    ports: dict | None = None,
    FEM_boundary: str = _FEM_DEFAULTS.boundary,
    FEM_symmetry: tuple | None = _FEM_DEFAULTS.symmetry,
    FEM_fe_order: int = _FEM_DEFAULTS.fe_order,
    FEM_air_pad_frac: float = _FEM_DEFAULTS.air_pad_frac,
    FEM_air_pad_mm: float | tuple | None = _FEM_DEFAULTS.air_pad_mm,
    FEM_elems_per_wavelength: float = _FEM_DEFAULTS.elems_per_wavelength,
    FEM_mesh_freq: float | None = _FEM_DEFAULTS.mesh_freq,
    FEM_mesh_fine_scale: float = _FEM_DEFAULTS.mesh_fine_scale,
    FEM_min_layers: int = _FEM_DEFAULTS.min_layers,
    FEM_num_solve_points: int = _FEM_DEFAULTS.num_solve_points,
    FEM_max_solve_points: int | None = _FEM_DEFAULTS.max_solve_points,
    charac_imp: float = 50.0,
    output_path: str | Path = "Sim_Path",
    run: bool = True,
    show_mesh: bool = True,
    mesh_style: str = "wireframe",
    verbose: bool = True,
) -> SimData:
    """
    Simulate a STEP geometry with the FEM backend and return the results.

    A standalone entry point that needs no CSXCAD geometry: it meshes the STEP
    file, runs the frequency sweep, and returns the same results bundle the
    rest of the pipeline uses, so all ``SimTools`` post-processing applies.

    Parameters
    ----------
    step_file : str
        Path to the ``.step`` file to simulate.
    freqs : NDArray
        Frequency points (Hz) to report results at.
    unit : str
        Advisory only -- the unit declared in the STEP file itself is what
        gets used. Default ``"mm"``.
    dielectrics : dict, optional
        Dielectric solids, as ``{solid_name: (eps_r, tan_d)}``.
    pec : list, optional
        Names of solids to treat as perfect electric conductors.
    lossy_conductor : dict, optional
        Lossy conductors, as ``{solid_name: sigma}`` with ``sigma`` in S/m.
    ports : dict, optional
        Ports, as ``{solid_name: {"z0": ..., "direction": "x|y|z",
        "number": ..., "kind": "lumped|wave", "prop_dir": "x|y|z"}}``. Only
        ``kind: "wave"`` reads ``prop_dir``, the axis the line runs along; such
        a port's solid must be the line's cross-section, normal to it. If
        ``ports`` is omitted, solids whose names look like ports are used
        instead.
    FEM_boundary : str
        Outer boundary condition: ``"silver_muller"`` (default), ``"pml"`` or
        ``"pec"``.
    FEM_symmetry : tuple, optional
        Mirror-symmetry plane ``(axis, kind, at)``, so only half the structure
        is meshed. See :class:`FEMOptions`. Default ``None``.
    FEM_fe_order : int
        Element order: ``1`` (default) or ``2``.
    FEM_air_pad_frac : float
        Air padding around the structure, as a fraction of the longest
        wavelength. Default ``0.25``. Ignored when ``FEM_air_pad_mm`` is set.
    FEM_air_pad_mm : float or tuple, optional
        Air padding around the structure in millimetres, used in place of
        ``FEM_air_pad_frac``. Either one value for all six faces, three
        values ``[x, y, z]``, or three ``[low, high]`` pairs. Default
        ``None``.
    FEM_elems_per_wavelength : float
        Target number of mesh elements per wavelength. Default ``16.0``.
    FEM_mesh_freq : float, optional
        Frequency the mesh is sized at, in Hz -- both the element size and
        the air padding come off its wavelength. Default ``None``, which
        sizes the mesh at the top of ``freqs``. Name the frequency the
        structure was designed at to stop a wide sweep refining the mesh.
    FEM_mesh_fine_scale : float
        Multiplier on the element size near conductors. Default ``1.0``.
    FEM_min_layers : int
        Number of element layers through the dielectric thickness. Default
        ``3``.
    FEM_num_solve_points : int
        Number of frequencies the sweep solves at (must be ``>= 4``). Default
        ``10``.
    FEM_max_solve_points : int, optional
        Ceiling on the solve count when the curve breaks passivity. Default
        ``None``, which allows twice ``FEM_num_solve_points``.
    charac_imp : float
        Port reference impedance in ohms, used for any port that does not set
        its own. Default ``50.0``.
    output_path : str | Path
        Working directory for the mesh, problem, and result files. Default
        ``"Sim_Path"``.
    run : bool
        Whether to solve. If ``False``, the geometry is only meshed and the
        results are read back from an earlier run in ``output_path``. Default
        ``True``.
    show_mesh:bool
        Whether to show the created mesh. Default is ``True``.
    mesh_style: str
        mesh style to be used for visualisation. default is "wireframe".
        other options are: "surface" | "wireframe" | "points",
    verbose : bool
        Print progress. Default ``True``.

    Returns
    -------
    SimData
        Named tuple of ``freqs``, ``s11``, ``s21`` (``None`` for a single-port
        problem), ``z11``, ``vswr``, ``input_power``, ``port_voltage``,
        ``port_current``, and ``ref_impedance``. See
        :class:`~simpleEMS.sim_tools.SimData`.

    Raises
    ------
    RuntimeError
        If no ports are found or specified, or if ``run`` is ``False`` and
        ``output_path`` holds no results from an earlier run.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(freqs, dtype=float)

    FEM_options = FEMOptions(
        boundary=FEM_boundary,
        symmetry=FEM_symmetry,
        fe_order=FEM_fe_order,
        air_pad_frac=FEM_air_pad_frac,
        air_pad_mm=FEM_air_pad_mm,
        elems_per_wavelength=FEM_elems_per_wavelength,
        mesh_freq=FEM_mesh_freq,
        mesh_fine_scale=FEM_mesh_fine_scale,
        min_layers=FEM_min_layers,
        num_solve_points=FEM_num_solve_points,
        max_solve_points=FEM_max_solve_points,
    )
    prob = _problem_from_step(
        step_file,
        freqs,
        dielectrics,
        pec,
        lossy_conductor,
        ports,
        charac_imp,
        FEM_options,
    )
    if not prob.ports:
        raise RuntimeError(
            "No ports found or specified for the STEP geometry; pass `ports=...` "
            "or name a port solid so it can be auto-detected."
        )

    msh_path = _mesh_problem(prob, output_path, verbose)
    if show_mesh:
        import pyvista as pv

        vtk_path = Path(msh_path).with_suffix(".vtk")
        grid = pv.read(str(vtk_path if vtk_path.exists() else msh_path))
        plotter = pv.Plotter()
        plotter.add_mesh(
            grid,
            style=mesh_style,
            show_edges=mesh_style != "wireframe",
            opacity=0.5,
            scalars="CellEntityIds",
            cmap="tab20",
            show_scalar_bar=False,
        )
        plotter.add_axes()
        plotter.view_xy()
        plotter.show()
    if run:
        _sweep_from_meta(
            freqs,
            prob.num_solve_points,
            output_path,
            verbose,
            max_solve_points=prob.max_solve_points,
        )
    return compute_sim_data(freqs, charac_imp, output_path)
