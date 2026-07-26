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

Bridges a CSXCAD geometry to the finite-element solver: maps CSX properties to
material/PEC/port roles, meshes the structure, drives the adaptive
rational-interpolation frequency sweep, and returns results as the standard
:class:`~simpleEMS.sim_tools.SimData` named tuple so existing plotting and
export utilities work unchanged.

"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import AAA

from CSXCAD import ContinuousStructure

from . import fem_formulation, fem_geometry, fem_solver, fem_sweep
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


@dataclass
class FEMOptions:
    """
    Global FEM solver/mesh options, forwarded to the GetDP problem.

    Bundles the ``Problem``-level knobs so the simpleEMS entry can carry them
    without threading many arguments. ``None``/defaults reproduce today's
    behaviour.

    Parameters
    ----------
    boundary : str
        Outer truncation: ``"silver_muller"`` (default) or ``"pml"``.
    symmetry : tuple | None
        Optional mirror-symmetry plane ``(axis, kind, at)`` with ``axis`` in
        ``x/y/z``, ``kind`` in ``pec/pmc``, ``at`` the plane coordinate (m) or
        ``None`` for the structure centre. Halves the mesh.
    fe_order : int
        Nedelec edge-element order: ``1`` (default) or ``2``.
    air_pad_frac : float
        Air padding as a fraction of the longest wavelength. Default ``0.2``.
        Ignored when ``air_pad_mm`` is set.
    air_pad_mm : float | None
        Explicit air padding in millimetres, added to every face of the
        structure's bounding box in place of the ``air_pad_frac`` wavelength
        formula. Use this for non-radiating structures (e.g. filters) whose
        box shouldn't scale with a wide S-parameter sweep's lowest frequency.
        If a far-field pattern is later requested (``CalcNF2FF``) and this
        padding is too small for an accurate near-to-far-field transform at
        the requested frequency, ``FEMNF2FF.CalcNF2FF`` raises ``ValueError``
        naming the minimum padding needed. Default ``None`` (auto, via
        ``air_pad_frac``).
    elems_per_wavelength : float
        Target coarse mesh density in open air. Default ``8.0``.
    mesh_fine_scale : float
        Multiplier on the near-conductor element size. Default ``1.0``.
    min_layers : int
        Element layers through the dielectric thickness. Default ``3``.
    port_type : str
        ``"lumped"`` (impedance z0, default) or ``"wave"`` (matched to the line
        characteristic impedance) for all ports.
    """

    boundary: str = "silver_muller"
    symmetry: tuple | None = None
    fe_order: int = 1
    air_pad_frac: float = 0.2
    air_pad_mm: float | None = None
    elems_per_wavelength: float = 8.0
    mesh_fine_scale: float = 1.0
    min_layers: int = 3
    port_type: str = "lumped"

    def __post_init__(self) -> None:
        """
        Validate the option choices.

        Raises
        ------
        ValueError
            If ``boundary``, ``fe_order``, or ``port_type`` is not one of the
            supported choices.
        """
        if self.boundary not in ("silver_muller", "pml"):
            raise ValueError(
                f"boundary must be 'silver_muller' or 'pml', got {self.boundary!r}"
            )
        if self.fe_order not in (1, 2):
            raise ValueError(f"fe_order must be 1 or 2, got {self.fe_order}")
        if self.port_type not in ("lumped", "wave"):
            raise ValueError(
                f"port_type must be 'lumped' or 'wave', got {self.port_type!r}"
            )


@dataclass
class SolidSpec:
    """
    One CAD solid and the electromagnetic role assigned to it.

    Parameters
    ----------
    name : str
        Solid name (matches the CSXCAD property name / STEP product name).
    role : str
        One of ``"dielectric"``, ``"pec"``, ``"lossy_conductor"``, ``"port"``, or
        ``"ignore"`` (excluded from the EM problem). Default ``"ignore"``.
    dielectric : Dielectric | None
        Material properties, set when ``role == "dielectric"``. Default ``None``.
    sigma : float
        Conductivity in S/m, used when ``role == "lossy_conductor"``. Default ``0.0``.
    """

    name: str
    role: str = "ignore"  # 'dielectric' | 'pec' | 'lossy_conductor' | 'port' | 'ignore'
    dielectric: Dielectric | None = None
    sigma: float = 0.0  # conductivity [S/m] for role == 'lossy_conductor'


@dataclass
class PortSpec:
    """
    A port on a named solid's terminal sheet.

    Parameters
    ----------
    solid : str
        Name of the solid whose footprint sheet becomes the port.
    number : int
        1-based port index; sets the row/column order of the S-parameter matrix.
    z0 : float
        Reference impedance in ohms. Default ``50.0``.
    direction : str
        Excitation E-field axis: ``"x"``, ``"y"``, or ``"z"``. Default ``"z"``.
    port_type : str
        ``"lumped"`` (referenced to ``z0``) or ``"wave"`` (referenced to the
        computed line characteristic impedance). Default ``"lumped"``.
    """

    solid: str
    number: int
    z0: float = 50.0
    direction: str = "z"  # excitation E-field axis: 'x' | 'y' | 'z'
    port_type: str = "lumped"  # 'lumped' (impedance z0) or 'wave' (matched to Zc)


@dataclass
class Problem:
    """
    A finite-element problem definition consumed by the FEM backend.

    Parameters
    ----------
    step_file : str
        Path to the STEP file to mesh and solve.
    name : str
        Base name used for the generated mesh/problem files. Default
        ``"structure"``.
    boundary : str
        Outer domain truncation: ``"silver_muller"`` (default, first-order
        absorbing boundary) or ``"pml"``.
    solids : dict[str, SolidSpec]
        Solid name -> role/material assignment.
    ports : list[PortSpec]
        Ports found among ``solids``, sorted by port number.
    freqs : NDArray
        Frequency grid (Hz) the problem is solved/interpolated over. Default
        ``[2.45e9]``.
    fe_order : int
        Nedelec edge-element order: ``1`` (default) or ``2``.
    symmetry : tuple | None
        Optional mirror-symmetry plane ``(axis, kind, at)`` -- see
        :class:`FEMOptions`. ``None`` (default) means no symmetry is applied.
    air_pad_frac : float
        Air padding as a fraction of the longest free-space wavelength.
        Default ``0.2``. Ignored when ``air_pad_mm`` is set.
    air_pad_mm : float | None
        Explicit air padding in millimetres -- see :class:`FEMOptions`.
        Default ``None``.
    elems_per_wavelength : float
        Target coarse mesh density in open air. Default ``8.0``.
    mesh_fine_scale : float
        Multiplier on the near-conductor element size. Default ``1.0``.
    min_layers : int
        Minimum element layers through the dielectric thickness. Default ``3``.
    """

    step_file: str
    name: str = "structure"
    boundary: str = "silver_muller"  # or "pml"
    solids: dict[str, SolidSpec] = field(default_factory=dict)
    ports: list[PortSpec] = field(default_factory=list)
    freqs: NDArray = field(default_factory=lambda: np.array([2.45e9]))
    fe_order: int = 1
    symmetry: tuple | None = None
    # mesh controls
    air_pad_frac: float = 0.2  # air padding as a fraction of the longest wavelength
    air_pad_mm: float | None = None  # explicit air padding (mm), overrides air_pad_frac
    elems_per_wavelength: float = 8.0  # coarse mesh density in open air
    mesh_fine_scale: float = 1.0  # multiplier on the near-conductor element size
    min_layers: int = 3  # element layers through the dielectric

    def dielectrics(self) -> list[SolidSpec]:
        """Return the solids assigned the dielectric role."""
        return [s for s in self.solids.values() if s.role == "dielectric"]


# ----------------------------
# CSX -> Problem mapping
# ----------------------------
def _port_number(name: str, fallback: int) -> int:
    """Extract a trailing integer from a port name, else use ``fallback``."""
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else fallback


def _register_port(
    prob: Problem,
    seen: set[int],
    solid: str,
    number: int,
    z0: float,
    direction: str = "z",
) -> None:
    """Append a :class:`PortSpec` for ``number`` unless already registered."""
    if number in seen:
        return
    seen.add(number)
    prob.ports.append(PortSpec(solid=solid, number=number, z0=z0, direction=direction))


def _csx_roles(
    csx: ContinuousStructure, centre_freq: float
) -> tuple[dict, dict, dict, dict]:
    """
    Classify CSXCAD properties into FEM roles.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry.
    centre_freq : float
        Frequency (Hz) used to convert conductivity to a loss tangent.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(role_by_name, dielectric_by_name, port_by_name, sigma_by_name)``:
        ``role_by_name`` maps a property name to its role string (``"pec"``,
        ``"dielectric"``, ``"lossy_conductor"``, or ``"port"``); ``dielectric_by_name``
        maps a dielectric property's name to its
        :class:`~simpleEMS.fem_materials.Dielectric`;
        ``port_by_name`` maps a port property's name to ``(z0, direction, number)``;
        ``sigma_by_name`` maps a lossy conductor's name to its conductivity in S/m.
    """
    role_by_name: dict[str, str] = {}
    dielectric_by_name: dict[str, Dielectric] = {}
    port_by_name: dict[str, tuple[float, str, int]] = {}
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
            # openEMS stores dielectric loss as an equivalent conductivity kappa
            # [S/m]; convert it back to a loss tangent at the band centre via
            # kappa = tan_d * w * eps0 * eps_r (GetDP wants tan_d, not kappa).
            kappa = float(prop.GetMaterialProperty("kappa"))
            tan_d = (
                kappa / (2 * math.pi * centre_freq * EPS0 * eps_r)
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
            port_by_name[name] = (z0, direction, number)

    return role_by_name, dielectric_by_name, port_by_name, sigma_by_name


def _apply_FEM_options(prob: Problem, FEM_options: FEMOptions | None) -> None:
    """Apply global :class:`FEMOptions` to a :class:`Problem` (in place)."""
    if FEM_options is None:
        return
    prob.boundary = FEM_options.boundary
    prob.symmetry = FEM_options.symmetry
    prob.fe_order = FEM_options.fe_order
    prob.air_pad_frac = FEM_options.air_pad_frac
    prob.air_pad_mm = FEM_options.air_pad_mm
    prob.elems_per_wavelength = FEM_options.elems_per_wavelength
    prob.mesh_fine_scale = FEM_options.mesh_fine_scale
    prob.min_layers = FEM_options.min_layers
    for port in prob.ports:
        port.port_type = FEM_options.port_type


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
        Output frequency grid (also sets fmin/fmax for meshing).
    output_path : Path
        Directory to export ``structure.step`` into.
    FEM_options : FEMOptions | None
        Global solver/mesh options to apply. ``None`` keeps the defaults.

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
    centre_freq = 0.5 * (float(freqs.min()) + float(freqs.max()))
    role_by_name, dielectric_by_name, port_by_name, sigma_by_name = _csx_roles(
        csx, centre_freq
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
            z0, direction, number = port_by_name[base]
            _register_port(prob, seen_ports, solid_name, number, z0, direction)

    prob.ports.sort(key=lambda p: p.number)
    _apply_FEM_options(prob, FEM_options)
    return prob


# ----------------------------
# pipeline stages
# ----------------------------
def _mesh_problem(prob: Problem, output_path: Path, verbose: bool = True) -> str:
    """
    Mesh a :class:`Problem`, emit the GetDP ``.pro``, and persist the metadata.

    Shared Problem-first core of the CSX and STEP entry points.

    Parameters
    ----------
    prob : Problem
        The FEM problem to mesh.
    output_path : Path
        Directory to write the mesh, problem, and metadata files into.
    verbose : bool
        Print progress. Default ``True``.

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
        "domain_bbox": list(mesh.inner_bbox),  # meshed E/H extents (m); Huygens
        # box must stay strictly inside this or CutBox samples outside the mesh
        "symmetry_axis": (
            {"x": 0, "y": 1, "z": 2}[prob.symmetry[0]] if prob.symmetry else None
        ),  # axis whose min-face sits on the symmetry plane, not open air
        "port_numbers": sorted(pm.number for pm in mesh.port_regions.values()),
        "ref_impedances": {
            str(pm.number): pm.ref_impedance for pm in mesh.port_regions.values()
        },
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

    Exports the geometry to STEP, builds the tagged tetrahedral mesh and the
    GetDP problem file, and records the port metadata for later stages.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry.
    freqs : NDArray
        Output frequency grid.
    output_path : str | Path
        Directory for the mesh and problem files.
    verbose : bool
        Print progress. Default ``True``.
    FEM_options : FEMOptions | None
        Global solver/mesh options. ``None`` keeps the defaults.

    Returns
    -------
    str
        Path to the generated ``.msh`` file.

    Raises
    ------
    RuntimeError
        If the geometry contains no ports.
    """
    output_path = Path(output_path)
    prob = _build_problem(csx, freqs, output_path, FEM_options)
    return _mesh_problem(prob, output_path, verbose)


def _sweep_from_meta(
    freqs: NDArray,
    num_solve_points: int,
    output_path: Path,
    verbose: bool = True,
) -> None:
    """
    Run the adaptive GetDP sweep from an existing ``fem_mesh.json``.

    Shared sweep core (needs no CSX): reads the mesh/problem metadata, drives
    the adaptive rational sweep, and writes ``fem_sparams.npz``.

    Parameters
    ----------
    freqs : NDArray
        Dense output frequency grid (Hz).
    num_solve_points : int
        Number of full FEM solves the adaptive sweep may perform.
    output_path : Path
        Directory containing ``fem_mesh.json`` (from :func:`build_mesh`) and
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

    # GetDP appends to these Format Table files (write_problem's xs_file =
    # "File >") instead of overwriting, so a sweep's per-solve rows accumulate
    # in one file. Clear any left over from a previous sweep/run first, or
    # this run's rows would land after stale ones from before.
    stale_patterns = [
        "intPort.txt",
        "xS_*.txt",
        "V_*.txt",
        "I_*.txt",
        "Ploss.txt",
        "Prad.txt",
    ]
    for pattern in stale_patterns:
        for stale in outdir.glob(pattern):
            stale.unlink()

    # freq -> (V, I) of the reference port's own total voltage/current, sampled
    # only at the frequencies the adaptive S-sweep below actually solves (no
    # extra getdp runs: V_<n>/I_<n> are written by the same Get_SParameters call
    # that already produces xS_<n> at each (freq, active-port) solve).
    vi_solved: dict[float, tuple[complex, complex]] = {}

    # One full FEM solve at a single frequency: drive each port in turn and read
    # back the column of the S-matrix it produces (S[:, active]).
    def solve_at(freq: float) -> NDArray:
        s = np.zeros((npt, npt), dtype=complex)
        for active in port_numbers:
            fem_solver.run_getdp(
                pro_path,
                msh_path,
                output_path,
                {"FREQ": freq, "ACTIVE_PORT": active},
                "Get_SParameters",
            )
            for n in port_numbers:
                s[idx[n], idx[active]] = fem_solver.read_complex(
                    outdir / f"xS_{n * 10 + active}.txt"
                )
            if active == ref_port:
                v = fem_solver.read_complex(outdir / f"V_{active * 10 + active}.txt")
                i = fem_solver.read_complex(outdir / f"I_{active * 10 + active}.txt")
                vi_solved[freq] = (v, i)
        return s

    # The expensive part is each solve_at() call, so let the adaptive sweep pick
    # a handful of frequencies (<= num_solve_points), then rational-interpolate
    # S(f) onto the full dense grid.
    freqs = np.asarray(freqs, dtype=float)
    s_dense = fem_sweep.rational_sweep(
        freqs, port_numbers, solve_at, num_solve_points, verbose=verbose
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

    ref_impedances = [meta["ref_impedances"][str(n)] for n in port_numbers]
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
    num_solve_points: int,
    output_path: str | Path,
    verbose: bool = True,
    FEM_options: FEMOptions | None = None,
) -> None:
    """
    Run the adaptive FEM frequency sweep on a CSXCAD geometry.

    (Re-)meshes the geometry from the current ``csx``/``FEM_options`` --
    ``run_sweep`` can be called on its own (``write_and_show_structure`` may
    have been skipped) and must not silently reuse a mesh from a previous,
    differently configured call -- then performs ``num_solve_points`` full
    GetDP solves, rational-interpolates the dense S-parameter curve, and
    writes it to ``fem_sparams.npz`` in ``output_path``.

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry.
    freqs : NDArray
        Dense output frequency grid.
    num_solve_points : int
        Number of full FEM solves the adaptive sweep may perform.
    output_path : str | Path
        Directory holding the mesh/problem files and receiving the results.
    verbose : bool
        Print progress. Default ``True``.
    FEM_options : FEMOptions | None
        Global solver/mesh options.

    Raises
    ------
    RuntimeError
        If the geometry contains no ports.
    """
    output_path = Path(output_path)
    build_mesh(csx, freqs, output_path, verbose=verbose, FEM_options=FEM_options)
    _sweep_from_meta(freqs, num_solve_points, output_path, verbose)


def compute_sim_data(
    freqs: NDArray, charac_imp: float, output_path: str | Path
) -> SimData:
    """
    Read the FEM sweep results into a :class:`~simpleEMS.sim_tools.SimData`.

    Parameters
    ----------
    freqs : NDArray
        Frequency grid (unused; kept for signature parity -- the stored grid is
        authoritative).
    charac_imp : float
        Fallback reference impedance if none was stored.
    output_path : str | Path
        Directory containing ``fem_sparams.npz`` from :func:`run_sweep`.

    Returns
    -------
    SimData
        ``freqs``, ``s11``, ``s21`` (``None`` for a single-port problem),
        ``z11``, ``vswr``, ``input_power``, ``port_voltage``, ``port_current``,
        and ``ref_impedance`` -- see :class:`~simpleEMS.sim_tools.SimData`.

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
    # GetDP solves in the physics time convention exp(-i w t), while openEMS
    # (and thus the rest of simpleEMS / the Smith chart) uses the engineering
    # convention exp(+j w t). The two differ by a complex conjugate, so
    # conjugate here to report S/Z in the same convention as the FDTD backend
    # (otherwise the reactance sign flips: Smith-chart points mirror across the
    # real axis, i.e. inductive <-> capacitive / top <-> bottom). port_voltage/
    # port_current are linear in the same field solution, so they need the same
    # conjugation.
    s = np.conj(data["S"])
    port_voltage = np.conj(data["port_voltage"])
    port_current = np.conj(data["port_current"])
    freqs_out = data["freqs"]
    ref_impedances = data["ref_impedances"]
    nports = s.shape[1]

    # Derive the same engineering quantities the FDTD path exposes, so the
    # existing SimTools plotting/exports work unchanged. Formulas are the
    # standard 1-port relations; s21 is only meaningful for multi-port runs.
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
    """Build a :class:`Problem` directly from a STEP file (no CSXCAD)."""
    step_file = str(Path(step_file).expanduser())
    freqs = np.asarray(freqs, dtype=float)
    dielectrics = dielectrics or {}
    pec_names = set(pec or [])
    lossy_conductor = lossy_conductor or {}
    ports = ports or {}

    prob = Problem(step_file=step_file, name="structure", freqs=freqs)
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
            )
        else:
            role = guess_role(name) or "ignore"
            solid = SolidSpec(name=name, role=role)
            if role == "dielectric":
                # placeholder eps_r=1; override via the `dielectrics` argument
                solid.dielectric = Dielectric()
            prob.solids[name] = solid
            if role == "port":
                _register_port(prob, seen_ports, name, len(prob.ports) + 1, charac_imp)

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
    num_solve_points: int = 10,
    FEM_options: FEMOptions | None = None,
    charac_imp: float = 50.0,
    output_path: str | Path = "fem_out",
    run: bool = True,
    verbose: bool = True,
) -> SimData:
    """
    Solve a raw STEP geometry with the FEM backend, returning ``SimData``.

    A CSX-free entry point mirroring :func:`standalone_model.simulate_model`:
    it meshes the STEP, generates the GetDP problem, runs the adaptive sweep,
    and returns the standard results bundle so all ``SimTools`` post-processing
    applies.

    Parameters
    ----------
    step_file : str
        Path to the ``.step`` file.
    freqs : NDArray
        Dense output frequency grid (Hz).
    unit : str
        Advisory only -- the STEP file's own declared unit is authoritative
        (Gmsh converts it to metres). Default ``"mm"``.
    dielectrics : dict | None
        ``{solid_name: (eps_r, tan_d)}`` overrides for dielectric solids.
    pec : list | None
        Solid names to force to perfect electric conductors.
    lossy_conductor : dict | None
        ``{solid_name: sigma}`` lossy (surface-impedance) conductors.
    ports : dict | None
        ``{solid_name: {"z0": ..., "direction": "x|y|z", "number": ...}}``.
        If omitted, solids whose names look like ports are auto-registered.
    num_solve_points : int
        Number of full FEM solves the adaptive sweep may perform.
    FEM_options : FEMOptions | None
        Global solver/mesh options (boundary, symmetry, fe_order, mesh tuning,
        port type).
    charac_imp : float
        Default/fallback port reference impedance in ohms.
    output_path : str | Path
        Working directory for the mesh/problem/results.
    run : bool
        If ``False``, mesh and generate the GetDP problem but skip the solve.
        The final result is still read back via :func:`compute_sim_data`, so
        this only succeeds if ``fem_sparams.npz`` already exists in
        ``output_path`` from an earlier solved run. Default ``True``.
    verbose : bool
        Print progress. Default ``True``.

    Returns
    -------
    SimData
        ``freqs``, ``s11``, ``s21`` (``None`` for a single-port problem),
        ``z11``, ``vswr``, ``input_power``, ``port_voltage``, ``port_current``,
        and ``ref_impedance`` -- see :class:`~simpleEMS.sim_tools.SimData`.

    Raises
    ------
    RuntimeError
        If no ports are found or specified, or (when ``run=False`` with no
        prior results) if the sweep results file is missing.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(freqs, dtype=float)

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

    _mesh_problem(prob, output_path, verbose)
    if run:
        _sweep_from_meta(freqs, num_solve_points, output_path, verbose)
    return compute_sim_data(freqs, charac_imp, output_path)
