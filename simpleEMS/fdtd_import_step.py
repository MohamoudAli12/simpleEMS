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
PROTOTYPE: simulate a STEP file with the FDTD backend.

Rebuilds each named solid in a STEP file as CSXCAD geometry and simulates it
with openEMS, so a CAD file can be simulated without building the geometry by
hand first. This is the FDTD counterpart of
:func:`simpleEMS.fem_backend.simulate_step_FEM`.
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

__all__ = ["simulate_step_FDTD"]

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

# A solid whose volume matches its bounding box to within this relative
# tolerance (and which has exactly six faces) is an axis-aligned box, so it can
# be rebuilt as a native CSXCAD box instead of a tessellated polyhedron -- see
# :func:`_axis_aligned_box`. The tolerance only has to absorb OCC's own
# volume-integration error, hence the tight value.
_BOX_VOLUME_RTOL = 1e-9


def _quantize_f32(value: float) -> float:
    """
    Round a coordinate to the precision a tessellated solid's edges land on.

    Parameters
    ----------
    value : float
        Coordinate value, in mm.

    Returns
    -------
    float
        ``value`` rounded to the nearest 32-bit float.
    """
    # STL stores every vertex as a 32-bit float, so a tessellated edge is only
    # ever as precise as that. A coordinate built straight from CadQuery stays
    # at full float64, and Mesh only merges coordinates that match to near
    # float64 precision -- so an edge meant to sit flush against a tessellated
    # trace, but off by a few nanometres, becomes a second boundary and gets
    # pathologically over-refined. Rounding it the same way lands it on the
    # bit-identical value, and Mesh sees one boundary instead of two.
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _axis_aligned_box(solid: cq.Solid) -> tuple[list[float], list[float]] | None:
    """
    Measure ``solid`` as a box, if it is one.

    A solid counts as a box when it has six faces and fills its own extents.
    A rotated or chamfered box does not, nor does a cylinder.

    Parameters
    ----------
    solid : cq.Solid
        The solid to measure.

    Returns
    -------
    tuple[list[float], list[float]] | None
        The opposite corners of the box, each ``[x, y, z]``, or ``None`` if
        the solid is not a box.
    """
    # Rebuilding a box as a real box rather than a tessellated shape matters
    # beyond fidelity: fdtd_mesh treats every tessellated solid as possibly
    # concave, so an unnotched metal imported that way can cancel the notch
    # verdict of a genuinely notched neighbour overlapping it along the queried
    # axis. That suppresses the refinement at the notched metal's edge and
    # shifts the modelled resonance.
    try:
        if len(solid.Faces()) != 6:
            return None
        bb = solid.BoundingBox()
        bbox_volume = (bb.xmax - bb.xmin) * (bb.ymax - bb.ymin) * (bb.zmax - bb.zmin)
        if bbox_volume <= 0:
            return None
        if abs(solid.Volume() - bbox_volume) > _BOX_VOLUME_RTOL * bbox_volume:
            return None
    except Exception:
        return None
    start = [_quantize_f32(v) for v in (bb.xmin, bb.ymin, bb.zmin)]
    stop = [_quantize_f32(v) for v in (bb.xmax, bb.ymax, bb.zmax)]
    return start, stop


def _port_box(bb: object, direction: str) -> tuple[list[float], list[float]]:
    """
    Place a lumped port on the extents of the solid that marks it.

    An axis thinner than a port can really be is flattened to nothing, which
    undoes the minimum thickness a STEP file has to give it. The axis the port
    is excited along, and any genuinely wide axis such as the trace width, are
    kept as they are.

    Parameters
    ----------
    bb : cadquery.occ_impl.geom.BoundBox
        Extents of the solid that marks the port.
    direction : str
        Axis the port is excited along: ``"x"``, ``"y"``, or ``"z"``.

    Returns
    -------
    tuple[list[float], list[float]]
        The opposite corners of the port, each ``[x, y, z]``.
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
    The simulation settings for a structure imported from a STEP file.

    Takes the place of the :class:`~simpleEMS.sim_params.SimParams` a
    hand-built structure would supply. The imported solids carry their own
    materials, so the substrate fields here only describe the structure well
    enough to size the mesh and the simulation box around it.

    Parameters
    ----------
    freqs : NDArray
        Frequency points (Hz), which set the frequency range and the centre
        frequency.
    struct_bbox_mm : tuple[float, float, float, float, float, float]
        Extents of all the imported solids together, in mm, as
        ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
    """

    # substrate_tand is unused and only satisfies the base class, but
    # substrate_eps_r matters: the base class derives lambda0 from it, and with
    # it the mesh resolution and the air padding. simulate_step_FDTD sets it to
    # the largest eps_r imported, so lambda0 matches what a hand-built version
    # of the same structure would give. Left at the air default it would
    # compute a vacuum wavelength -- overpadding the box and, worse,
    # under-resolving the mesh at the metal edges.

    freqs: NDArray
    struct_bbox_mm: tuple[float, float, float, float, float, float]
    substrate_eps_r: float = 1.0
    substrate_tand: float = 0.0
    substrate_thickness_mm: float = 0.0

    @property
    def freq_range(self) -> tuple[float, float]:
        """Lowest and highest frequency to simulate, in Hz."""
        return float(self.freqs.min()), float(self.freqs.max())

    @property
    def main_freq(self) -> float:
        """Centre frequency of the range, in Hz."""
        fmin, fmax = self.freq_range
        return 0.5 * (fmin + fmax)

    @property
    def substrate_width_mm(self) -> float:
        """Width of the imported structure, in mm."""
        return self.struct_bbox_mm[1] - self.struct_bbox_mm[0]

    @property
    def substrate_length_mm(self) -> float:
        """Length of the imported structure, in mm."""
        return self.struct_bbox_mm[3] - self.struct_bbox_mm[2]

    @property
    def simulation_box(self) -> NDArray:
        """Extents of the imported structure, which the mesh pads further."""
        return self._create_simulation_box(
            self.struct_bbox_mm[1] - self.struct_bbox_mm[0],
            self.struct_bbox_mm[3] - self.struct_bbox_mm[2],
            self.struct_bbox_mm[5] - self.struct_bbox_mm[4],
        )


def _load_named_solids(step_file: Path) -> dict[str, cq.Solid]:
    """
    Read the named solids out of a STEP file.

    Parameters
    ----------
    step_file : Path
        Path to the STEP file to read.

    Returns
    -------
    dict[str, cq.Solid]
        Each solid, keyed by its name in the file. A file written by
        :func:`simpleEMS.export_cad.export_step` names each solid after the
        CSXCAD property it came from.
    """
    asm = cq.Assembly.load(str(step_file))
    return {child.name: child.obj for child in asm.children}


def _combined_bbox_mm(
    solids: list[cq.Solid],
) -> tuple[float, float, float, float, float, float]:
    """Extents of all of ``solids`` together, in the STEP file's own units."""
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
    Write a solid to an STL file, which CSXCAD can read.

    Used for the solids that are not boxes, since CSXCAD cannot read STEP.
    Coordinates are written as they are, in the STEP file's own units.

    Parameters
    ----------
    solid : cq.Solid
        The solid to write.
    workdir : Path
        Directory to write ``<name>.stl`` into.
    name : str
        Name of the file to write, matching the solid's own name.
    tolerance : float
        How far the written shape may depart from the solid, in the STEP
        file's units. Default ``1e-3``.
    angular_tolerance : float
        How far it may depart in angle, in radians. Default ``0.2``.

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
    Simulate a STEP file with the FDTD backend and return the results.

    Rebuilds each named solid as CSXCAD geometry, meshes it, runs openEMS,
    and reads the results back, all in one call. Name each solid you want in
    the simulation in ``dielectrics``, ``pec``, or ``ports``; any solid none
    of them names is left out.

    Parameters
    ----------
    step_file : str | Path
        Path to the STEP file to simulate.
    freqs : NDArray
        Frequency points (Hz) to report results at. The lowest and highest of
        them also set the band the structure is excited over.
    dielectrics : dict[str, tuple[float, float]]
        Dielectric solids, as ``{solid_name: (eps_r, tan_d)}``.
    pec : list[str]
        Names of the solids to treat as perfect electric conductors.
    ports : dict[str, dict]
        Ports, as ``{solid_name: {...}}``, where each port accepts
        ``number`` (defaults to its position in the dict), ``z0`` (defaults
        to ``charac_imp``), ``direction`` (``"x"``, ``"y"``, or ``"z"``,
        defaults to ``"z"``), and ``excite`` (defaults to ``1.0`` for the
        first port and ``0.0`` for the rest).
    output_path : str | Path, optional
        Directory for the simulation results and intermediate files.
        Defaults to ``cwd / "Sim_Path"``.
    charac_imp : float
        Port reference impedance in ohms, used for any port that does not set
        its own. Default ``50.0``.
    FDTD_boundary : list[str], optional
        Six boundary conditions, one per face of the simulation box, in the
        order ``[xmin, xmax, ymin, ymax, zmin, zmax]``. Defaults to
        ``["PML_8"] * 6``.
    FDTD_timestep : int
        Largest number of time steps to run. Default ``90000000``.
    FDTD_end_criteria : float
        How far the energy must decay before the solver stops. Smaller values
        are more accurate and take longer. Default ``1e-4``.
    FDTD_mesh_resolution_factor : int
        How fine the mesh is overall. Higher is finer. Default ``10``.
    FDTD_metal_mesh_resolution_factor : int
        How fine the mesh is near metal. Higher is finer. Default ``40``.
    num_points : int
        Number of frequency points to report results at. Default ``1000``.
    show_structure : bool
        Show the geometry in AppCSXCAD before the solver runs. Default
        ``True``.
    run : bool
        Whether to run the solver. Set to ``False`` to build and inspect the
        structure only. Default ``True``.

    Returns
    -------
    tuple[SimData, object]
        ``(sim_data, nf2ff)`` -- the S-parameters, impedance, VSWR and port
        power, and the object that records the far-field data. Pass the
        second straight to the ``SimTools`` radiation plots.

    Raises
    ------
    KeyError
        If a name given in ``dielectrics``, ``pec``, or ``ports`` is not one
        of the STEP file's solids.
    RuntimeError
        If no ports are given.
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

    def _add_solid(prop: object, name: str, priority: int) -> str:
        """Add the solid ``name`` to the CSXCAD property ``prop``, as a box
        where it is one and a tessellated shape otherwise. Returns which of
        the two it was, for the progress output."""
        box = _axis_aligned_box(solids[name])
        if box is not None:
            prop.AddBox(priority=priority, start=box[0], stop=box[1])
            return "box"
        stl_path = _tessellate_to_stl(solids[name], stl_dir, name)
        prop.AddPolyhedronReader(str(stl_path), priority=priority).ReadFile()
        return "polyhedron"

    for name, (eps_r, tan_d) in dielectrics.items():
        kappa = tan_d * 2 * np.pi * params.main_freq * EPS0 * eps_r
        material = CSX.AddMaterial(name, epsilon=eps_r, kappa=kappa)
        kind = _add_solid(material, name, priority=1)
        console.print(
            f"[info]  material: {name} (eps_r={eps_r}, tan_d={tan_d}, {kind})[/info]"
        )

    for name in pec:
        metal = CSX.AddMetal(name)
        kind = _add_solid(metal, name, priority=10)
        console.print(f"[info]  pec: {name} ({kind})[/info]")

    # A port solid is not rebuilt as geometry like the others: openEMS needs a
    # real lumped element with an excitation there, so the solid only supplies
    # the extents to place one on.
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
