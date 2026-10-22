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
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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
from .sim_params import SimParams

if TYPE_CHECKING:
    from .sim_tools import SimData

__all__ = ["simulate_step_FEM"]

# The three public functions run in the same order as the FDTD pipeline and,
# like it, hand state between stages through FILES in the output directory
# rather than through Python objects (the FDTD path writes structure.xml then
# reads probe files; we write the two files below). This keeps each stage
# stateless and independently re-runnable:
#
#   build_mesh   ->  fem_mesh.json   (mesh path, .pro path, port numbers + Zref)
#   run_sweep    ->  fem_sparams.npz (dense interpolated S, freqs, ref impedances)
#   compute_sim_data reads fem_sparams.npz -> SimData
#
# fem_solver names both files, so the far-field and field-dump modules can
# read them without importing this one.
_AXIS_TO_DIR = {0: "x", 1: "y", 2: "z"}


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
        "FEM_air_pad_mm must be one value, three values [x, y, z], or three "
        "[low, high] pairs [[xlo, xhi], [ylo, yhi], [zlo, zhi]], got "
        f"{air_pad_mm!r}"
    )


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

    The solver and mesh settings each default to the ``FEM_*`` field of the
    same name on :class:`~simpleEMS.sim_params.SimParams`, which documents
    them.

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
    mesh_freq : float, optional
        Frequency the mesh is sized at, in Hz. Default ``None``, which falls
        back to the top of ``freqs``; a structure built from ``SimParams``
        passes its ``main_freq``.
    boundary, symmetry, fe_order, air_pad_frac, air_pad_mm : optional
        See the ``FEM_*`` field of the same name on ``SimParams``.
    elems_per_wavelength, mesh_fine_scale, min_layers : optional
        See the ``FEM_*`` field of the same name on ``SimParams``.
    num_solve_points, max_solve_points : optional
        See the ``FEM_*`` field of the same name on ``SimParams``.
    port_type, port_mode_modes, port_mode_index, port_mode_zc : optional
        See the ``FEM_*`` field of the same name on ``SimParams``.
    port_mode_eps_eff, waveport_width_mm, waveport_height_mm : optional
        See the ``FEM_*`` field of the same name on ``SimParams``.
    """

    step_file: str
    name: str = "structure"
    solids: dict[str, SolidSpec] = field(default_factory=dict)
    ports: list[PortSpec] = field(default_factory=list)
    freqs: NDArray = field(default_factory=lambda: np.array([2.45e9]))
    mesh_freq: float | None = None
    boundary: str = SimParams.FEM_boundary
    symmetry: tuple | None = SimParams.FEM_symmetry
    fe_order: int = SimParams.FEM_fe_order
    air_pad_frac: float = SimParams.FEM_air_pad_frac
    # Kept exactly as given; air_pad_faces_mm expands it
    air_pad_mm: float | tuple | None = SimParams.FEM_air_pad_mm
    elems_per_wavelength: float = SimParams.FEM_elems_per_wavelength
    mesh_fine_scale: float = SimParams.FEM_mesh_fine_scale
    min_layers: int = SimParams.FEM_min_layers
    num_solve_points: int = SimParams.FEM_num_solve_points
    max_solve_points: int | None = SimParams.FEM_max_solve_points
    port_type: str = SimParams.FEM_port_type
    port_mode_modes: int = SimParams.FEM_port_mode_modes
    port_mode_index: int = SimParams.FEM_port_mode_index
    port_mode_zc: float | None = SimParams.FEM_port_mode_zc
    port_mode_eps_eff: float | None = SimParams.FEM_port_mode_eps_eff
    waveport_width_mm: float | None = SimParams.FEM_waveport_width_mm
    waveport_height_mm: float | None = SimParams.FEM_waveport_height_mm

    def __post_init__(self) -> None:
        """
        Fill in ``mesh_freq`` and validate the solver and mesh settings.

        Raises
        ------
        ValueError
            If a setting is not one of its supported choices or is out of its
            range. The message names the ``FEM_*`` setting the user set.
        """
        if self.mesh_freq is None:
            self.mesh_freq = float(np.max(self.freqs))
        if self.boundary not in ("silver_muller", "pml", "pec"):
            raise ValueError(
                f"FEM_boundary must be 'silver_muller', 'pml' or 'pec', "
                f"got {self.boundary!r}"
            )
        if self.fe_order not in (1, 2):
            raise ValueError(f"FEM_fe_order must be 1 or 2, got {self.fe_order}")
        if self.mesh_freq <= 0:
            raise ValueError(f"FEM_mesh_freq must be > 0 Hz, got {self.mesh_freq}")
        if self.num_solve_points < 4:
            raise ValueError(
                f"FEM_num_solve_points must be >= 4 for a stable rational fit, "
                f"got {self.num_solve_points}"
            )
        if (
            self.max_solve_points is not None
            and self.max_solve_points < self.num_solve_points
        ):
            raise ValueError(
                f"FEM_max_solve_points must be >= FEM_num_solve_points "
                f"({self.num_solve_points}), got {self.max_solve_points}"
            )
        if self.port_type not in ("lumpedport", "waveport"):
            raise ValueError(
                f"FEM_port_type must be 'lumpedport' or 'waveport', "
                f"got {self.port_type!r}"
            )
        if self.port_mode_modes < 1:
            raise ValueError(
                f"FEM_port_mode_modes must be >= 1, got {self.port_mode_modes}"
            )
        if self.port_mode_index < 0:
            raise ValueError(
                f"FEM_port_mode_index must be >= 0, got {self.port_mode_index}"
            )
        if self.port_mode_eps_eff is not None and self.port_mode_eps_eff < 1:
            raise ValueError(
                "FEM_port_mode_eps_eff is an effective permittivity, so it "
                f"cannot be below 1, got {self.port_mode_eps_eff}"
            )
        if self.port_mode_zc is not None and self.port_mode_zc <= 0:
            raise ValueError(
                f"FEM_port_mode_zc must be positive, got {self.port_mode_zc}"
            )
        if self.port_mode_index >= self.port_mode_modes:
            raise ValueError(
                f"FEM_port_mode_index {self.port_mode_index} is out of reach of "
                f"FEM_port_mode_modes {self.port_mode_modes}; compute at least "
                f"{self.port_mode_index + 1} eigenpairs to select that mode"
            )
        for name in ("waveport_width_mm", "waveport_height_mm"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"FEM_{name} must be positive, got {value}")
        if self.air_pad_mm is not None:
            faces = _air_pad_faces_mm(self.air_pad_mm)
            for axis, (low, high) in zip("xyz", faces, strict=True):
                for pad, sign in ((low, "-"), (high, "+")):
                    if pad < 0:
                        raise ValueError(
                            f"FEM_air_pad_mm cannot be negative: {pad} on the "
                            f"{axis}{sign} face of {self.air_pad_mm!r}. Zero is "
                            "allowed and puts the domain boundary on that face "
                            "of the structure."
                        )

    def dielectrics(self) -> list[SolidSpec]:
        """Return the solids assigned the dielectric role."""
        return [s for s in self.solids.values() if s.role == "dielectric"]

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


# ----------------------------
# CSX -> Problem mapping
# ----------------------------
def _port_number(name: str, fallback: int) -> int:
    """Read the port number off the end of ``name``, else use ``fallback``."""
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else fallback


def _register_port(prob: Problem, port: PortSpec) -> None:
    """Add ``port`` to ``prob``, or its solid to the port of the same number.

    A coplanar waveguide port is drawn as one solid per gap, so a port number
    can turn up more than once. The extra solids are added to the port that
    already exists rather than dropped -- dropping them is how the second gap
    used to end up meshed with no boundary condition at all.
    """
    for existing in prob.ports:
        if existing.number == port.number:
            if port.solid not in existing.solids:
                existing.solids.append(port.solid)
            return
    prob.ports.append(port)


def _csx_roles(
    csx: ContinuousStructure, kappa_freq: float, port_type: str = "lumpedport"
) -> tuple[dict[str, SolidSpec], dict[str, PortSpec]]:
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
    tuple[dict[str, SolidSpec], dict[str, PortSpec]]
        ``(solid_by_name, port_by_name)``, keyed by property name: the
        :class:`SolidSpec` of every property with a role, and the
        :class:`PortSpec` of every port. Both name the property itself, so
        each solid exported from it takes a copy under its own name.
    """
    solid_by_name: dict[str, SolidSpec] = {}
    port_by_name: dict[str, PortSpec] = {}
    cpw_port_numbers: list[int] = []

    # The EM role of each solid is read straight from its CSXCAD property type,
    # so the same geometry the user built for openEMS maps to the FEM problem
    # with no extra annotation: Metal -> PEC, Material -> dielectric, a lumped
    # element -> a port, and a conducting sheet -> a lossy (surface-impedance)
    # conductor.
    for prop in csx.GetAllProperties():
        type_string = prop.GetTypeString()
        name = prop.GetName()
        if type_string == "Metal":
            solid_by_name[name] = SolidSpec(name=name, role="pec")
        elif type_string == "ConductingSheet":
            # A conducting sheet is a lossy conductor: its finite conductivity
            # becomes a surface-impedance (Leontovich) boundary in the .pro.
            solid_by_name[name] = SolidSpec(
                name=name,
                role="lossy_conductor",
                sigma=float(prop.GetConductivity()),
            )
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
            solid_by_name[name] = SolidSpec(
                name=name,
                role="dielectric",
                dielectric=Dielectric(eps_r=eps_r, tan_d=tan_d),
            )
        elif type_string == "LumpedElement":
            # A lumped element is the excitation port: its resistance is the
            # reference impedance z0, its direction the E-field axis, and a
            # trailing number in the name (port_resist_<N>) the port index.
            z0 = float(prop.GetResistance())
            direction = _AXIS_TO_DIR[int(prop.GetDirection())]
            number = _port_number(name, len(port_by_name) + 1)
            solid_by_name[name] = SolidSpec(name=name, role="port")

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
                cpw_port_numbers.append(number)
            port_by_name[name] = PortSpec(
                solid=name, number=number, z0=z0, direction=direction, kind=kind
            )

    if cpw_port_numbers and port_type != "waveport":
        console.print(
            f"[info]port(s) {sorted(cpw_port_numbers)}: coplanar waveguide, so "
            "solved as a wave port rather than a lumped one -- the CPW mode is "
            "odd and a lumped port would excite the wrong one. Set "
            'FEM_port_type="waveport" to do this everywhere.[/info]'
        )
    return solid_by_name, port_by_name


def _fem_settings(params: SimParams) -> dict:
    """
    Read the ``FEM_*`` fields of ``params`` as :class:`Problem` keyword arguments.

    Parameters
    ----------
    params : SimParams
        The structure's parameters.

    Returns
    -------
    dict
        Each ``FEM_<name>`` field under ``<name>``, with ``mesh_freq`` resolved
        to a frequency.
    """
    settings = {
        param_field.name.removeprefix("FEM_"): getattr(params, param_field.name)
        for param_field in fields(params)
        if param_field.name.startswith("FEM_")
    }
    # freq_range is the window the S-parameters are plotted over; the mesh is
    # sized for the frequency the structure was designed at
    settings["mesh_freq"] = params.FEM_mesh_freq or params.main_freq
    return settings


def _build_problem(
    csx: ContinuousStructure,
    freqs: NDArray,
    output_path: Path,
    params: SimParams,
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
    params : SimParams
        The structure's parameters, whose ``FEM_*`` fields set the solver and
        mesh settings.

    Returns
    -------
    Problem
        The populated FEM problem.

    Raises
    ------
    ValueError
        If a ``FEM_*`` setting is out of range.
    """
    # Gmsh meshes STEP, not CSXCAD, so we round-trip the geometry through a STEP
    # file. export_step names each solid after its CSXCAD property, which is how
    # we match a meshed solid back to the role read from CSX below.
    output_path.mkdir(parents=True, exist_ok=True)
    step_file = str(output_path / "structure.step")
    # Created before the export so a bad FEM_* setting is refused first
    prob = Problem(
        step_file=step_file,
        name="structure",
        freqs=np.asarray(freqs, dtype=float),
        **_fem_settings(params),
    )
    export_step(csx, output_path)

    # openEMS stores dielectric loss as a conductivity fixed at main_freq (see
    # SimParams.substrate_kappa), so kappa has to be read back there or the
    # loss comes out scaled by the ratio of the two frequencies.
    solid_by_name, port_by_name = _csx_roles(csx, params.main_freq, prob.port_type)

    for solid_name in fem_geometry.list_solids(step_file):
        # A property with several primitives is exported as name_0, name_1, ...;
        # strip a trailing _<int> so each piece maps back to the parent role.
        base = (
            solid_name
            if solid_name in solid_by_name
            else re.sub(r"_\d+$", "", solid_name)
        )
        if base in solid_by_name:
            prob.solids[solid_name] = replace(solid_by_name[base], name=solid_name)
        else:
            prob.solids[solid_name] = SolidSpec(name=solid_name)
        if base in port_by_name:
            _register_port(
                prob,
                replace(port_by_name[base], solid=solid_name, solids=[solid_name]),
            )

    prob.ports.sort(key=lambda p: p.number)
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
    csx: ContinuousStructure, freqs: NDArray, params: SimParams
) -> str:
    """Hash of everything that affects the mesh, so it can be reused or rebuilt.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry.
    freqs : NDArray
        Frequency points (Hz) to solve over.
    params : SimParams
        The structure's parameters.

    Returns
    -------
    str
        Hex digest covering the geometry, the frequencies, the settings, and
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
    parts.append(f"settings:{_fem_settings(params)}")
    # The dielectric loss tangents are read back at main_freq
    parts.append(f"main_freq:{params.main_freq}")
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
    (output_path / fem_solver.MESH_META).write_text(json.dumps(meta, indent=2))
    return mesh.msh_path


def build_mesh(
    csx: ContinuousStructure,
    freqs: NDArray,
    output_path: str | Path,
    params: SimParams,
    verbose: bool = True,
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
    params : SimParams
        The structure's parameters, whose ``FEM_*`` fields set the solver and
        mesh settings.
    verbose : bool
        Print progress. Default ``True``.

    Returns
    -------
    str
        Path to the ``.msh`` file, whether reused or freshly built.

    Raises
    ------
    RuntimeError
        If the geometry contains no ports.
    ValueError
        If a ``FEM_*`` setting is out of range.
    """
    output_path = Path(output_path)
    fingerprint = _mesh_fingerprint(csx, freqs, params)
    meta_file = output_path / fem_solver.MESH_META
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

    prob = _build_problem(csx, freqs, output_path, params)
    return _mesh_problem(prob, output_path, verbose, fingerprint)


def show_fem_mesh(msh_path: str | Path) -> None:
    """
    Show a Gmsh mesh in PyVista, one coloured, switchable actor per region.

    Each region (dielectric, air, PEC, port, absorbing boundary, ...) gets a
    fixed colour, a legend entry and a checkbox that hides or shows it; air and
    the absorbing boundary start hidden so the structure is visible. A corner
    table lists the node and tetrahedron counts, the shortest and longest edge,
    the worst scaled Jacobian and the tetrahedra in each volume region.

    Parameters
    ----------
    msh_path : str | Path
        The Gmsh ``.msh`` file. Its ``.vtk`` sibling, written alongside by the
        mesher, is read when present.

    Returns
    -------
    None
    """
    import pyvista as pv

    vtk_path = Path(msh_path).with_suffix(".vtk")
    grid = pv.read(str(vtk_path if vtk_path.exists() else msh_path))
    grid.points *= 1e3  # metres -> millimetres, so the axes read in mm
    region_ids = grid.cell_data["CellEntityIds"]
    volume_cells = grid.celltypes == pv.CellType.TETRA
    tetrahedra = grid.extract_cells(np.flatnonzero(volume_cells))

    plotter = pv.Plotter(window_size=(1200, 850), title="simpleEMS FEM mesh")
    legend = []
    volume_counts = []
    for row, region_id in enumerate(np.unique(region_ids)):
        if fem_materials.dielectric_region(0) <= region_id < fem_materials.AIR:
            name = f"Substrate {region_id - fem_materials.dielectric_region(0) + 1}"
            color = "#3a9e5c"
        elif fem_materials.port_region(0) < region_id < fem_materials.ABC:
            name = f"Port {region_id - fem_materials.port_region(0)}"
            color = "#e0383b"
        else:
            name, color = {
                fem_materials.AIR: ("Air", "#7fb3ff"),
                fem_materials.PML: ("PML", "#b38fff"),
                fem_materials.PEC: ("PEC", "#d98c3a"),
                fem_materials.LOSSY_CONDUCTOR: ("Lossy conductor", "#a0703a"),
                fem_materials.ABC: ("ABC", "#9aa4b2"),
                fem_materials.SYM: ("Symmetry", "#f2c94c"),
            }.get(int(region_id), (f"Region {region_id}", "#ffffff"))
        in_region = region_ids == region_id
        is_volume = bool(np.any(in_region & volume_cells))
        if is_volume:
            volume_counts.append((name, int(np.sum(in_region & volume_cells))))
        actor = plotter.add_mesh(
            grid.extract_cells(np.flatnonzero(in_region & (volume_cells == is_volume))),
            color=color,
            show_edges=True,
            edge_color="black",
            line_width=0.3,
            opacity=0.25 if is_volume else 1.0,
        )
        visible = region_id not in (fem_materials.AIR, fem_materials.ABC)
        actor.SetVisibility(visible)
        plotter.add_checkbox_button_widget(
            lambda state, actor=actor: actor.SetVisibility(state),
            value=visible,
            position=(10, 10 + row * 45),
            size=35,
            color_on=color,
        )
        plotter.add_text(name, position=(55, 18 + row * 45), font_size=10)
        legend.append([name, color])
    plotter.add_legend(legend, size=(0.18, 0.2))

    rows = [("Nodes", f"{grid.n_points:,}"), ("Tetrahedra", f"{tetrahedra.n_cells:,}")]
    if tetrahedra.n_cells:
        quality = tetrahedra.cell_quality("scaled_jacobian")["scaled_jacobian"]
        edge_lengths = (
            tetrahedra.extract_all_edges()
            .compute_cell_sizes(area=False, volume=False)
            .cell_data["Length"]
        )
        rows += [
            ("Shortest edge", f"{edge_lengths.min() * 1e3:.2f} um"),
            ("Longest edge", f"{edge_lengths.max():.2f} mm"),
            ("Worst quality", f"{quality.min():.3f}"),
        ]
    rows += [(f"  {name}", f"{count:,}") for name, count in volume_counts]
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    stats_actor = plotter.add_text(
        "\n".join(
            f" {label:<{label_width}}  {value:>{value_width}} " for label, value in rows
        ),
        position="upper_left",
        font_size=12,
        font="courier",
        color="#f2f2f2",
    )
    stats_property = stats_actor.GetTextProperty()
    stats_property.SetBackgroundColor(0.08, 0.08, 0.1)
    stats_property.SetBackgroundOpacity(0.75)
    stats_property.SetFrame(True)
    stats_property.SetFrameColor(0.45, 0.45, 0.5)

    plotter.add_axes(viewport=(0.86, 0.0, 1.0, 0.16))
    plotter.view_isometric()
    plotter.show()


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
    a = np.diag(np.sqrt(1.0 - r**2))
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
    meta = json.loads((output_path / fem_solver.MESH_META).read_text())

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
        for n, d in meta["port_modes"].items()
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
    # freq -> radiation efficiency of the reference port's solve. The same
    # launch prints its losses (Get_Power), so this costs no solve either.
    efficiency_solved: dict[float, float] = {}

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
        lost_power, radiated_power = fem_solver.read_powers(outdir, ref_port)
        efficiency_solved[freq] = (
            radiated_power / (radiated_power + lost_power)
            if radiated_power > 0
            else 0.0
        )
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
    # sparse solve points suffice -- rational-interpolate them the same way.
    vi_freqs = np.array(sorted(vi_solved))
    v_samples = np.array([vi_solved[f][0] for f in vi_freqs])
    i_samples = np.array([vi_solved[f][1] for f in vi_freqs])
    v_dense = fem_sweep.interpolate(vi_freqs, v_samples, freqs)
    i_dense = fem_sweep.interpolate(vi_freqs, i_samples, freqs)

    np.savez(
        output_path / fem_solver.SWEEP_RESULTS,
        freqs=freqs,
        S=s_dense,
        port_numbers=np.asarray(port_numbers),
        ref_impedances=np.asarray(ref_impedances),
        port_voltage=v_dense,
        port_current=i_dense,
        # The raw solve points, not a dense curve: plot_efficiency interpolates
        # them onto whatever frequencies it is asked for.
        efficiency_freqs=np.array(sorted(efficiency_solved)),
        efficiency=np.array([efficiency_solved[f] for f in sorted(efficiency_solved)]),
    )
    if verbose:
        console.print(
            "[success]FEM S-parameters written to "
            f"{output_path / fem_solver.SWEEP_RESULTS}[/success]"
        )


def run_sweep(
    csx: ContinuousStructure,
    freqs: NDArray,
    output_path: str | Path,
    params: SimParams,
    verbose: bool = True,
) -> None:
    """
    Run the FEM frequency sweep on a CSXCAD geometry.

    Meshes the geometry first (reusing an existing mesh where possible), then
    solves at ``params.FEM_num_solve_points`` frequencies, interpolates the
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
    params : SimParams
        The structure's parameters, whose ``FEM_*`` fields set the solver and
        mesh settings.
    verbose : bool
        Print progress. Default ``True``.

    Raises
    ------
    RuntimeError
        If the geometry contains no ports.
    ValueError
        If a ``FEM_*`` setting is out of range.
    """
    output_path = Path(output_path)
    build_mesh(csx, freqs, output_path, params, verbose=verbose)
    _sweep_from_meta(
        freqs,
        params.FEM_num_solve_points,
        output_path,
        verbose,
        max_solve_points=params.FEM_max_solve_points,
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
        problem), ``s31`` / ``s41`` (``None`` with fewer than three / four
        ports), ``z11``, ``vswr``, ``input_power``, ``port_voltage``,
        ``port_current``, and ``ref_impedance``. See
        :class:`~simpleEMS.sim_tools.SimData`.

    Raises
    ------
    RuntimeError
        If the sweep results file is missing.
    """
    from .sim_tools import SimData

    output_path = Path(output_path)
    npz_path = output_path / fem_solver.SWEEP_RESULTS
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
    s31 = s[:, 2, 0] if nports >= 3 else None
    s41 = s[:, 3, 0] if nports >= 4 else None
    z0 = float(ref_impedances[0]) if ref_impedances.size else charac_imp
    z11 = z0 * (1 + s11) / (1 - s11)  # input impedance from the reflection coeff
    s11_mag = np.clip(np.abs(s11), 0, 0.999)  # prevent division by zero error
    vswr = (1 + s11_mag) / (1 - s11_mag)
    # accepted power at the driven (reference) port, from its total voltage and
    # current (see fem_formulation.write_problem's V_n/I_n post-quantities).
    input_power = 0.5 * np.real(port_voltage * np.conj(port_current))

    return SimData(
        freqs_out,
        s11,
        s21,
        s31,
        s41,
        z11,
        vswr,
        input_power,
        port_voltage,
        port_current,
        z0,
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
    settings: dict,
) -> Problem:
    """Build a :class:`Problem` from a STEP file's solids, without CSXCAD.

    ``settings`` holds the :class:`Problem` solver and mesh keyword arguments.
    """
    step_file = str(Path(step_file).expanduser())
    freqs = np.asarray(freqs, dtype=float)
    dielectrics = dielectrics or {}
    pec_names = set(pec or [])
    lossy_conductor = lossy_conductor or {}
    ports = ports or {}

    prob = Problem(step_file=step_file, name="structure", freqs=freqs, **settings)
    # There is no CSXCAD here to read a port kind off, so FEM_port_type is the
    # default and the per-port dict overrides it.
    default_kind = "wave" if prob.port_type == "waveport" else "lumped"
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
            _register_port(
                prob,
                PortSpec(
                    solid=name,
                    number=int(spec.get("number", len(prob.ports) + 1)),
                    z0=float(spec.get("z0", charac_imp)),
                    direction=spec.get("direction", "z"),
                    kind=spec.get("kind", default_kind),
                    prop_dir=spec.get("prop_dir", ""),
                ),
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
                    PortSpec(
                        solid=name,
                        number=len(prob.ports) + 1,
                        z0=charac_imp,
                        kind=default_kind,
                    ),
                )

    prob.ports.sort(key=lambda p: p.number)
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
    FEM_boundary: str = SimParams.FEM_boundary,
    FEM_symmetry: tuple | None = SimParams.FEM_symmetry,
    FEM_fe_order: int = SimParams.FEM_fe_order,
    FEM_air_pad_frac: float = SimParams.FEM_air_pad_frac,
    FEM_air_pad_mm: float | tuple | None = SimParams.FEM_air_pad_mm,
    FEM_elems_per_wavelength: float = SimParams.FEM_elems_per_wavelength,
    FEM_mesh_freq: float | None = SimParams.FEM_mesh_freq,
    FEM_mesh_fine_scale: float = SimParams.FEM_mesh_fine_scale,
    FEM_min_layers: int = SimParams.FEM_min_layers,
    FEM_num_solve_points: int = SimParams.FEM_num_solve_points,
    FEM_max_solve_points: int | None = SimParams.FEM_max_solve_points,
    FEM_port_type: str = SimParams.FEM_port_type,
    FEM_port_mode_modes: int = SimParams.FEM_port_mode_modes,
    FEM_port_mode_index: int = SimParams.FEM_port_mode_index,
    FEM_port_mode_zc: float | None = SimParams.FEM_port_mode_zc,
    FEM_port_mode_eps_eff: float | None = SimParams.FEM_port_mode_eps_eff,
    FEM_waveport_width_mm: float | None = SimParams.FEM_waveport_width_mm,
    FEM_waveport_height_mm: float | None = SimParams.FEM_waveport_height_mm,
    charac_imp: float = 50.0,
    output_path: str | Path = "Sim_Path",
    run: bool = True,
    show_mesh: bool = True,
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
        is meshed. See :class:`~simpleEMS.sim_params.SimParams`. Default
        ``None``.
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
    FEM_port_type : str
        Kind of every port whose entry in ``ports`` does not set ``"kind"``,
        and of every port found by name: ``"lumpedport"`` (default) or
        ``"waveport"``. A wave port drives the mode of the line's
        cross-section, which a transmission-line feed needs, and becomes the
        domain wall, so its solid must stand on an end of the structure.
    FEM_port_mode_modes : int
        Number of modes the wave-port eigensolver computes. Default ``6``.
    FEM_port_mode_index : int
        Which of those modes a wave port runs in, counting from ``0``, the
        default.
    FEM_port_mode_zc : float, optional
        Characteristic impedance in ohms to reference a wave port's
        S-parameters to, in place of the one measured from its mode. Default
        ``None``.
    FEM_port_mode_eps_eff : float, optional
        Effective permittivity naming a wave port's mode, which takes
        precedence over ``FEM_port_mode_index``. Default ``None``.
    FEM_waveport_width_mm, FEM_waveport_height_mm : float, optional
        Size of each wave port's cross-section in millimetres. Default
        ``None``, which sizes it from the trace width and substrate height.
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
    show_mesh : bool
        Whether to show the created mesh with :func:`show_fem_mesh` before the
        sweep. Default ``True``.
    verbose : bool
        Print progress. Default ``True``.

    Returns
    -------
    SimData
        Named tuple of ``freqs``, ``s11``, ``s21`` (``None`` for a single-port
        problem), ``s31`` / ``s41`` (``None`` with fewer than three / four
        ports), ``z11``, ``vswr``, ``input_power``, ``port_voltage``,
        ``port_current``, and ``ref_impedance``. See
        :class:`~simpleEMS.sim_tools.SimData`.

    Raises
    ------
    RuntimeError
        If no ports are found or specified, or if ``run`` is ``False`` and
        ``output_path`` holds no results from an earlier run.
    ValueError
        If a ``FEM_*`` setting is out of range.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(freqs, dtype=float)

    settings = {
        "boundary": FEM_boundary,
        "symmetry": FEM_symmetry,
        "fe_order": FEM_fe_order,
        "air_pad_frac": FEM_air_pad_frac,
        "air_pad_mm": FEM_air_pad_mm,
        "elems_per_wavelength": FEM_elems_per_wavelength,
        "mesh_freq": FEM_mesh_freq,
        "mesh_fine_scale": FEM_mesh_fine_scale,
        "min_layers": FEM_min_layers,
        "num_solve_points": FEM_num_solve_points,
        "max_solve_points": FEM_max_solve_points,
        "port_type": FEM_port_type,
        "port_mode_modes": FEM_port_mode_modes,
        "port_mode_index": FEM_port_mode_index,
        "port_mode_zc": FEM_port_mode_zc,
        "port_mode_eps_eff": FEM_port_mode_eps_eff,
        "waveport_width_mm": FEM_waveport_width_mm,
        "waveport_height_mm": FEM_waveport_height_mm,
    }
    prob = _problem_from_step(
        step_file,
        freqs,
        dielectrics,
        pec,
        lossy_conductor,
        ports,
        charac_imp,
        settings,
    )
    if not prob.ports:
        raise RuntimeError(
            "No ports found or specified for the STEP geometry; pass `ports=...` "
            "or name a port solid so it can be auto-detected."
        )

    msh_path = _mesh_problem(prob, output_path, verbose)
    if show_mesh:
        show_fem_mesh(msh_path)
    if run:
        _sweep_from_meta(
            freqs,
            prob.num_solve_points,
            output_path,
            verbose,
            max_solve_points=prob.max_solve_points,
        )
    return compute_sim_data(freqs, charac_imp, output_path)
