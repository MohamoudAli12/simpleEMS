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
PROTOTYPE: STEP -> CSXCAD -> openEMS (FDTD) import pipeline.

Mirrors the STEP-to-FEM entry point (:func:`simpleEMS.fem_backend.simulate_step_FEM`)
but drives the FDTD backend instead: each named solid in the STEP file is
tessellated (via CadQuery, already a project dependency through
:mod:`simpleEMS.export_cad`) and imported into CSXCAD as a
``CSPrimPolyhedronReader`` primitive on a Material/Metal property, so the
existing FDTD machinery (:class:`~simpleEMS.fdtd_mesh.Mesh` auto-meshing,
:func:`~simpleEMS.sim_tools.setup_simulation`, ``SimTools``) runs unchanged.

Not wired into ``simpleEMS.__init__`` yet -- import directly:
``from simpleEMS.fdtd_import_step import simulate_step_FDTD``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
from numpy.typing import NDArray
from openEMS.physical_constants import EPS0

from .console import console
from .fdtd_mesh import Mesh
from .sim_params import SimParams
from .sim_tools import SimData, SimTools, setup_simulation

__all__ = ["simulate_step_FDTD", "StepFDTDParams"]

# STEP can't represent a degenerate (zero-volume) solid, so export_cad.py's
# `_make_box` floors every box dimension to 1e-3 mm. A port that started life
# as a genuine zero-thickness CSXCAD sheet (as in patch_antenna.py's native
# ports: `port_start[1] == port_stop[1]`) comes back from STEP with a real
# 1e-3 mm box instead -- e.g. port_resist_1's y-extent is exactly 0.001 mm.
# That's still wide enough to be a real (non-floating-point-noise) interval,
# so Mesh's "at least min_lines" rule over-refines it same as a real feature
# would. Anything under this threshold on a port's non-excitation axes is
# treated as that padding artifact, not real geometry, and collapsed to a
# single point -- comfortably above the 1e-3 mm floor (with margin) and
# comfortably below any real trace dimension in this project (e.g.
# `min_trace_width_mm` defaults to 0.1 mm).
_DEGENERATE_PORT_THICKNESS_MM = 1e-2


def _quantize_f32(value: float) -> float:
    """
    Round-trip ``value`` through a 32-bit float.

    Binary STL (what ``cq.exporters.export`` writes and
    ``CSPrimPolyhedronReader`` reads) stores every vertex as a 32-bit float,
    so any PEC/dielectric edge imported via :func:`_tessellate_to_stl` is
    only ever as precise as the nearest float32 -- e.g. the nominal
    ``-0.1375`` mm feed edge actually lands in CSXCAD as
    ``-0.13750000298023224``.

    A port box built straight from CadQuery's ``.BoundingBox()`` stays at
    full float64 precision instead (and that query is itself a loose/padded
    bound, not the exact geometry, so it may not even be
    ``-0.1375`` to begin with). Since ``Mesh`` (mesh.py) only merges
    coordinates that match to (near enough) float64 precision, an edge meant
    to sit flush against a tessellated trace -- but off by a few nanometers
    -- gets treated as a distinct boundary and pathologically over-refined
    (``Mesh`` forces at least ``min_lines`` mesh lines across any x/y
    interval, however small).

    Quantizing a port coordinate through the same float32 round-trip lands
    it on the exact bit-identical value the STL path already produced for
    the same nominal design coordinate, so ``Mesh`` sees one boundary
    instead of two.

    Parameters
    ----------
    value : float
        Coordinate value, in mm.

    Returns
    -------
    float
        ``value`` rounded to its nearest float32 representation.
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _port_box(bb: object, direction: str) -> tuple[list[float], list[float]]:
    """
    Build ``(start, stop)`` coordinates for a lumped port from its solid's
    bounding box.

    Any axis other than the excitation ``direction`` whose extent is under
    :data:`_DEGENERATE_PORT_THICKNESS_MM` is collapsed to its midpoint
    (true zero thickness) -- undoing the STEP export's forced minimum box
    size on what was originally a zero-thickness CSXCAD port sheet (see
    :data:`_DEGENERATE_PORT_THICKNESS_MM`). The excitation axis and any
    genuinely wide transverse axis (e.g. the trace width) are left as-is.
    Every resulting coordinate is then quantized to float32
    (:func:`_quantize_f32`) to match the tessellated PEC/dielectric edges.

    Parameters
    ----------
    bb : cadquery.occ_impl.geom.BoundBox
        The port solid's bounding box.
    direction : str
        Excitation E-field axis: ``"x"``, ``"y"``, or ``"z"``.

    Returns
    -------
    tuple[list[float], list[float]]
        ``(start, stop)``, each ``[x, y, z]``.
    """
    axes = {"x": (bb.xmin, bb.xmax), "y": (bb.ymin, bb.ymax), "z": (bb.zmin, bb.zmax)}
    for axis, (lo, hi) in axes.items():
        if axis != direction and (hi - lo) < _DEGENERATE_PORT_THICKNESS_MM:
            mid = 0.5 * (lo + hi)
            axes[axis] = (mid, mid)
    start = [_quantize_f32(axes[a][0]) for a in ("x", "y", "z")]
    stop = [_quantize_f32(axes[a][1]) for a in ("x", "y", "z")]
    return start, stop


@dataclass(kw_only=True)
class StepFDTDParams(SimParams):
    """
    Minimal :class:`~simpleEMS.sim_params.SimParams` adapter for a
    standalone STEP file.

    The real dielectrics are imported as their own CSXCAD ``Material``
    properties (see :func:`simulate_step_FDTD`), so this class does not
    model a substrate itself -- ``substrate_tand`` stays at its air default
    and only exists to satisfy the base class. ``substrate_eps_r`` *is*
    meaningful here even though no substrate is modeled directly: the base
    class derives ``lambda0`` (and from it, ``FDTD_mesh_resolution``,
    ``FDTD_metal_mesh_resolution``, and the automatic air padding around the
    structure) from ``substrate_eps_r``. Leaving it at the air default of
    ``1.0`` would compute a vacuum wavelength instead of the dielectric-
    loaded one the imported material(s) actually produce -- overpadding the
    simulation box and, worse, under-resolving the mesh right at the metal
    edges. :func:`simulate_step_FDTD` sets it to the largest ``eps_r`` among
    the imported ``dielectrics`` so ``lambda0`` matches what a native
    (non-STEP) build of the same structure would compute.
    ``simulation_box`` is seeded with the STEP geometry's own bounding box;
    :class:`~simpleEMS.fdtd_mesh.Mesh` grows it by ``lambda0`` automatically.

    Parameters
    ----------
    freqs : NDArray
        Frequency sweep (Hz) driving ``freq_range``/``main_freq``.
    struct_bbox_mm : tuple[float, float, float, float, float, float]
        Combined bounding box of every imported solid, in mm:
        ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
    """

    freqs: NDArray
    struct_bbox_mm: tuple[float, float, float, float, float, float]
    substrate_eps_r: float = 1.0
    substrate_tand: float = 0.0
    substrate_thickness_mm: float = 0.0

    @property
    def freq_range(self) -> tuple[float, float]:
        """``(fmin, fmax)`` from ``freqs``."""
        return float(self.freqs.min()), float(self.freqs.max())

    @property
    def main_freq(self) -> float:
        """Band-centre frequency, used for the loss-tangent -> kappa conversion."""
        fmin, fmax = self.freq_range
        return 0.5 * (fmin + fmax)

    @property
    def substrate_width_mm(self) -> float:
        """X-extent of the imported geometry."""
        return self.struct_bbox_mm[1] - self.struct_bbox_mm[0]

    @property
    def substrate_length_mm(self) -> float:
        """Y-extent of the imported geometry."""
        return self.struct_bbox_mm[3] - self.struct_bbox_mm[2]

    @property
    def simulation_box(self) -> NDArray:
        """Geometry's own bounding box;
        :class:`~simpleEMS.fdtd_mesh.Mesh` pads it further.
        """
        return self._create_simulation_box(
            self.struct_bbox_mm[1] - self.struct_bbox_mm[0],
            self.struct_bbox_mm[3] - self.struct_bbox_mm[2],
            self.struct_bbox_mm[5] - self.struct_bbox_mm[4],
        )


def _load_named_solids(step_file: Path) -> dict[str, cq.Solid]:
    """
    Load a STEP file's top-level named solids via CadQuery.

    ``cq.Assembly.load`` reads the STEP AP242 product/assembly structure
    (the same one :func:`simpleEMS.export_cad.export_step` writes), so each
    solid comes back keyed by the CSXCAD property name it was exported
    under.

    Parameters
    ----------
    step_file : Path
        Path to the STEP file.

    Returns
    -------
    dict[str, cq.Solid]
        Solid name -> CadQuery solid.
    """
    asm = cq.Assembly.load(str(step_file))
    return {child.name: child.obj for child in asm.children}


def _combined_bbox_mm(
    solids: list[cq.Solid],
) -> tuple[float, float, float, float, float, float]:
    """Combined axis-aligned bounding box of ``solids``, in the STEP file's units."""
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    for solid in solids:
        bb = solid.BoundingBox()
        xmin, xmax = min(xmin, bb.xmin), max(xmax, bb.xmax)
        ymin, ymax = min(ymin, bb.ymin), max(ymax, bb.ymax)
        zmin, zmax = min(zmin, bb.zmin), max(zmax, bb.zmax)
    return (xmin, xmax, ymin, ymax, zmin, zmax)


def _tessellate_to_stl(
    solid: cq.Solid,
    workdir: Path,
    name: str,
    tolerance: float = 1e-3,
    angular_tolerance: float = 0.2,
) -> Path:
    """
    Tessellate a solid to an STL file for :meth:`CSProperties.AddPolyhedronReader`.

    CSXCAD has no STEP reader; it can only build geometry from primitives
    (box/polygon/...) or read a triangulated mesh (STL/PLY) via
    ``CSPrimPolyhedronReader``. Coordinates are written verbatim (no unit
    conversion), matching the STEP file's own units against the CSXCAD grid
    unit set in :func:`~simpleEMS.sim_tools.setup_simulation`.

    Every vertex this writes is float32 (see :func:`_quantize_f32`) --
    nothing to fix here; callers that need a *different* coordinate (e.g. a
    port box) to land on the exact same value as an edge tessellated by this
    function must quantize that other coordinate themselves.

    Parameters
    ----------
    solid : cq.Solid
        The solid to tessellate.
    workdir : Path
        Directory to write ``<name>.stl`` into.
    name : str
        Base filename (matches the solid/property name).
    tolerance : float
        Linear tessellation tolerance, in the STEP file's units. Default ``1e-3``.
    angular_tolerance : float
        Angular tessellation tolerance in radians. Default ``0.2``.

    Returns
    -------
    Path
        Path to the written STL file.
    """
    stl_path = workdir / f"{name}.stl"
    cq.exporters.export(
        solid, str(stl_path), tolerance=tolerance, angularTolerance=angular_tolerance
    )
    return stl_path


def simulate_step_FDTD(
    step_file: str | Path,
    freqs: NDArray,
    dielectrics: dict[str, tuple[float, float]],
    pec: list[str],
    ports: dict[str, dict],
    output_path: str | Path | None = None,
    charac_imp: float = 50.0,
    FDTD_boundary: list[str] | None = None,
    FDTD_timestep: int = 90000000,
    FDTD_end_criteria: float = 1e-4,
    FDTD_mesh_resolution_factor: int = 10,
    FDTD_metal_mesh_resolution_factor: int = 40,
    num_points: int = 1000,
    show_structure: bool = True,
    run: bool = True,
) -> tuple[SimData, object]:
    """
    Import a STEP file into CSXCAD and simulate it with the openEMS FDTD backend.

    Every named solid in ``step_file`` (matched by :func:`_load_named_solids`)
    is routed by name into one of three roles -- the same split
    :func:`simpleEMS.fem_backend.simulate_step_FEM` uses for the FEM
    backend -- and reconstructed as CSXCAD geometry:

    - ``dielectrics`` : tessellated onto a ``CSPropMaterial`` (``epsilon``,
      ``kappa`` derived from the given loss tangent at the sweep's centre
      frequency).
    - ``pec`` : tessellated onto a ``CSPropMetal``.
    - ``ports`` : *not* tessellated. Each port solid's bounding box instead
      defines a real ``FDTD.AddLumpedPort`` (thin volumes don't mesh well
      as a polyhedron; openEMS needs an actual lumped-element box+excitation
      there anyway).

    The mesh is then generated automatically from the resulting CSXCAD
    primitives via the existing :class:`~simpleEMS.fdtd_mesh.Mesh` engine --
    identical to every other FDTD structure in this project -- rather than
    anything STEP-specific.

    Parameters
    ----------
    step_file : str | Path
        Path to the STEP file to import.
    freqs : NDArray
        Frequency sweep (Hz) for post-processing and the Gaussian excitation
        (``[freqs.min(), freqs.max()]`` sets the excite band).
    dielectrics : dict[str, tuple[float, float]]
        Dielectric solid name -> ``(eps_r, tan_d)``.
    pec : list[str]
        Perfect-electric-conductor solid names.
    ports : dict[str, dict]
        Port solid name -> kwargs for :meth:`openEMS.openEMS.AddLumpedPort`:
        ``number`` (1-based, default assigned by dict order), ``z0``
        (default ``charac_imp``), ``direction`` (``"x"``/``"y"``/``"z"``,
        default ``"z"``), ``excite`` (default ``1.0`` for the first port,
        ``0.0`` for the rest).
    output_path : str | Path, optional
        Directory for the tessellated STL cache and simulation results.
        Defaults to ``cwd / "Sim_Path"``.
    charac_imp : float
        Default port reference impedance in ohms. Default ``50.0``.
    FDTD_boundary : list[str], optional
        Six openEMS boundary-condition strings
        (``[xmin, xmax, ymin, ymax, zmin, zmax]``). Defaults to
        ``["PML_8"] * 6``.
    FDTD_timestep : int
        Maximum FDTD time-step count. Default ``90000000``.
    FDTD_end_criteria : float
        Energy decay stopping criterion. Default ``1e-4``.
    FDTD_mesh_resolution_factor : int
        Divides ``lambda0`` for the global mesh resolution. Default ``10``.
    FDTD_metal_mesh_resolution_factor : int
        Divides ``lambda0`` for the near-metal mesh resolution. Default ``40``.
    num_points : int
        Number of frequency points for post-processing. Default ``1000``.
    show_structure : bool
        Open the geometry in AppCSXCAD before running. Default ``True``.
    run : bool
        Execute the FDTD solver. Default ``True``; set ``False`` to only
        build/inspect the structure.

    Returns
    -------
    tuple[SimData, SimSetup, object]
        ``(sim_data, sim, nf2ff)`` -- S-parameters/impedance/VSWR, the
        underlying CSX/FDTD objects, and the near-field-to-far-field
        recording box (created before the solver runs, so far-field dumps
        are actually captured; pass it straight to
        ``SimTools.compute_nf2ff_3d``/``plot_*`` helpers).

    Raises
    ------
    KeyError
        If a name in ``dielectrics``, ``pec``, or ``ports`` is not found
        among the STEP file's solids.
    RuntimeError
        If ``ports`` is empty.
    """
    if not ports:
        raise RuntimeError("simulate_step_FDTD needs at least one entry in `ports`")

    step_file = Path(step_file)
    output_path = Path(output_path) if output_path else Path.cwd() / "Sim_Path"
    output_path.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(freqs, dtype=float)

    console.print("-------------------------------------------", style="info")
    console.print(f"Importing STEP geometry: {step_file}", style="info")
    console.print("-------------------------------------------", style="info")

    solids = _load_named_solids(step_file)
    missing = (set(dielectrics) | set(pec) | set(ports)) - solids.keys()
    if missing:
        raise KeyError(f"solid(s) not found in {step_file.name}: {sorted(missing)}")

    bbox_mm = _combined_bbox_mm(list(solids.values()))
    # lambda0 (and everything derived from it: mesh resolution, air padding)
    # is computed by the base class from substrate_eps_r -- use the most
    # dielectrically-loaded imported material so it matches a native build
    # of the same structure instead of assuming vacuum (see StepFDTDParams).
    substrate_eps_r = max((eps_r for eps_r, _ in dielectrics.values()), default=1.0)
    params = StepFDTDParams(
        freqs=freqs,
        struct_bbox_mm=bbox_mm,
        substrate_eps_r=substrate_eps_r,
        charac_imp=charac_imp,
        num_points=num_points,
        FDTD_timestep=FDTD_timestep,
        FDTD_end_criteria=FDTD_end_criteria,
        FDTD_mesh_resolution_factor=FDTD_mesh_resolution_factor,
        FDTD_metal_mesh_resolution_factor=FDTD_metal_mesh_resolution_factor,
    )
    sim = setup_simulation(params, FDTD_boundary=FDTD_boundary)
    CSX, FDTD = sim.CSX, sim.FDTD

    stl_dir = output_path / "step_stl"
    stl_dir.mkdir(parents=True, exist_ok=True)

    for name, (eps_r, tan_d) in dielectrics.items():
        kappa = tan_d * 2 * np.pi * params.main_freq * EPS0 * eps_r
        material = CSX.AddMaterial(name, epsilon=eps_r, kappa=kappa)
        stl_path = _tessellate_to_stl(solids[name], stl_dir, name)
        material.AddPolyhedronReader(str(stl_path), priority=1).ReadFile()
        console.print(f"[info]  material: {name} (eps_r={eps_r}, tan_d={tan_d})[/info]")

    for name in pec:
        metal = CSX.AddMetal(name)
        stl_path = _tessellate_to_stl(solids[name], stl_dir, name)
        metal.AddPolyhedronReader(str(stl_path), priority=10).ReadFile()
        console.print(f"[info]  pec: {name}[/info]")

    port_objs = []
    for i, (name, spec) in enumerate(ports.items()):
        bb = solids[name].BoundingBox()
        number = spec.get("number", i + 1)
        z0 = spec.get("z0", charac_imp)
        direction = spec.get("direction", "z")
        excite = spec.get("excite", 1.0 if i == 0 else 0.0)
        start, stop = _port_box(bb, direction)
        port = FDTD.AddLumpedPort(
            number,
            z0,
            start,
            stop,
            direction,
            excite,
            priority=20,
        )
        port_objs.append((number, port))
        console.print(
            f"[info]  port {number}: {name} (z0={z0}, dir={direction})[/info]"
        )

    port_objs.sort(key=lambda t: t[0])
    ports_list = [p for _, p in port_objs]

    Mesh(CSX, params)

    # Must exist before FDTD.Run() -- it registers the dump boxes the solver
    # records far-field data into while running.
    nf2ff = SimTools.create_nf2ff(sim)

    if show_structure:
        SimTools.write_and_show_structure(sim, output_path)

    if run:
        SimTools.run_simulation(sim, output_path)

    port_arg = ports_list if len(ports_list) > 1 else ports_list[0]
    sim_data = SimTools.compute_sim_data(sim, port_arg, output_path)

    return sim_data, nf2ff
