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
Turn a STEP file into a tagged 3D mesh for the FEM backend, using Gmsh.

Reduces every conductor and port to a zero-thickness sheet, wraps the
structure in an air box, meshes the result with the sheets embedded in it, and
labels each volume and surface so the generated problem file can refer to
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

from .console import console
from .fem_materials import (
    ABC,
    AIR,
    C0,
    LOSSY_CONDUCTOR,
    PEC,
    PML,
    SYM,
    dielectric_region,
    port_region,
)

if TYPE_CHECKING:
    from .fem_backend import Problem


@dataclass
class PortMesh:
    """
    Description of one port as it appears in the mesh.

    Parameters
    ----------
    number : int
        One-based port index, matching
        :class:`~simpleEMS.fem_backend.PortSpec.number`.
    region : int
        Tag identifying the port in the mesh and problem files.
    direction : str
        Axis the port is excited along: ``"x"``, ``"y"``, or ``"z"``.
    z0 : float
        Port reference impedance in ohms.
    gap : float
        Length of the port along ``direction``, in metres.
    width : float
        Width of the port across ``direction``, in metres.
    center : tuple[float, float, float]
        Centre of the port, in metres.
    width_meshed : float
        Width of the port actually present in the mesh, in metres. Differs
        from ``width`` only when a symmetry plane cuts the port. Default
        ``0.0``, meaning the same as ``width``.
    kind : str
        ``"lumped"`` (default) for a resistive-sheet port driven by one
        constant field direction, or ``"wave"`` for a port whose transverse
        mode is solved on its cross-section by
        :mod:`~simpleEMS.fem_port_mode`.
    prop_dir : str
        For a wave port, the axis the line runs along -- the axis the port
        face is normal to: ``"x"``, ``"y"``, or ``"z"``. Unused by a lumped
        port. Default ``"y"``.
    """

    number: int
    region: int
    direction: str
    z0: float
    gap: float  # electrical gap length along `direction` (m)
    width: float  # transverse width of the port sheet (m)
    center: tuple[float, float, float]
    width_meshed: float = 0.0  # meshed width; differs under symmetry
    kind: str = "lumped"  # 'lumped' | 'wave'
    prop_dir: str = "y"  # propagation axis of a wave port

    @property
    def is_wave(self) -> bool:
        """Whether this port's excitation comes from a solved transverse mode."""
        return self.kind == "wave"

    @property
    def ref_impedance(self) -> float:
        """Impedance the S-parameters are referenced to, in ohms."""
        # always the full structure's z0, even for a half model: a symmetry
        # plane is a modelling device, not a change to the device under test
        return self.z0

    @property
    def meshed_impedance(self) -> float:
        """Impedance the port presents in the mesh, in ohms.

        The same as ``z0`` unless a symmetry plane cuts the port, which
        doubles it.
        """
        # port current must be referenced to this, not to z0, or the accepted
        # power comes out scaled while the volume-integrated loss does not
        meshed = self.width_meshed or self.width
        # Bounding boxes carry a modelling tolerance, so an uncut port measures
        # a hair narrower than its marker; snap that to no correction at all.
        if abs(meshed - self.width) <= 0.01 * self.width:
            return self.z0
        return self.z0 * self.width / meshed

    @property
    def sheet_impedance(self) -> float:
        """Impedance per square, in ohms, that makes the port present ``z0``.

        A lumped-port quantity only. A wave port terminates into its own mode
        rather than into a resistive sheet, so its boundary term is scaled by
        the modal index ``beta / k0`` instead -- see
        :func:`~simpleEMS.fem_port_mode.solve_port_mode`.
        """
        return self.z0 * self.width / self.gap


@dataclass
class Mesh:
    """
    A generated mesh and the region tags that identify its parts.

    Parameters
    ----------
    msh_path : str
        Path to the written ``.msh`` file.
    dielectric_regions : dict[str, int]
        Region tag of each dielectric volume, keyed by solid name.
    air_region : int
        Region tag of the air volume.
    pec_region : int
        Region tag of the perfect-conductor surfaces.
    port_regions : dict[int, PortMesh]
        Description of each port, keyed by port number.
    abc_region : int
        Region tag of the absorbing boundary surfaces.
    boundary : str
        Outer boundary condition used: ``"silver_muller"`` or ``"pml"``.
    bbox : tuple[float, float, float, float, float, float]
        Extents of the structure as ``(x0, y0, z0, x1, y1, z1)``, in metres.
    box_bbox : tuple[float, float, float, float, float, float]
        Extents of the whole meshed domain, in metres.
    lambda_min : float
        Shortest wavelength anywhere in the problem, in metres. Default
        ``0.0``.
    impedance_regions : list[tuple[int, float]]
        ``(region_tag, sigma)`` for each distinct lossy-conductor
        conductivity. Default ``[]``.
    pml_region : int
        Region tag of the PML volume, or ``0`` if no PML is used. Default
        ``0``.
    inner_bbox : tuple[float, float, float, float, float, float]
        Extents of the air volume inside the PML, in metres. Empty when there
        is no PML. Default ``()``.
    pml_thick : float
        Thickness of the PML, in metres. Default ``0.0``.
    sym_region : int
        Region tag of the symmetry plane, or ``0`` if none is applied.
        Default ``0``.
    sym_kind : str
        Symmetry plane type: ``"pec"`` or ``"pmc"``, empty if none is applied.
        Default ``""``.
    sym_axis : int
        Axis the structure is mirrored across (``0``, ``1``, or ``2``), or
        ``-1`` if none is applied. Default ``-1``.
    sym_plane : float
        Position of the symmetry plane along ``sym_axis``, in metres; the
        meshed half lies above it. Default ``0.0``.
    dielectric_bboxes : dict[str, tuple]
        Extents of each dielectric solid, keyed by solid name. A wave port's
        mode solve reads these to give each triangle of its cross-section the
        right permittivity. Default ``{}``.
    """

    msh_path: str
    dielectric_regions: dict[str, int]  # solid name -> region id
    air_region: int
    pec_region: int
    port_regions: dict[int, PortMesh]  # port number -> info
    abc_region: int
    boundary: str
    bbox: tuple[float, float, float, float, float, float]  # structure extents (m)
    box_bbox: tuple[float, float, float, float, float, float]  # air-box extents (m)
    lambda_min: float = field(default=0.0)
    impedance_regions: list = field(default_factory=list)  # [(region_id, sigma), ...]
    pml_region: int = 0  # 0 if no PML
    inner_bbox: tuple = ()  # air/PML interface extents (m); PML damping starts here
    pml_thick: float = 0.0  # PML shell thickness (m)
    sym_region: int = 0  # symmetry-plane surface region (0 if none)
    sym_kind: str = ""  # 'pec' or 'pmc'
    sym_axis: int = -1  # mirrored axis index, -1 if none
    sym_plane: float = 0.0  # symmetry plane coordinate (m); half kept is >= this
    dielectric_bboxes: dict = field(default_factory=dict)  # solid name -> bbox


# ----------------------------
# helpers
# ----------------------------
def _short_name(entity_name: str) -> str:
    """Return the solid name at the end of a full Gmsh entity name, e.g.
    ``'Shapes/<uuid>/patch_inset/patch_inset'`` -> ``'patch_inset'``."""
    return entity_name.rstrip("/").split("/")[-1] if entity_name else ""


def _bbox_inside(inner: tuple, outer: tuple, tol: float) -> bool:
    """True if the box ``inner`` sits inside the box ``outer``, within ``tol``."""
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] >= outer[2] - tol
        and inner[3] <= outer[3] + tol
        and inner[4] <= outer[4] + tol
        and inner[5] <= outer[5] + tol
    )


def _bbox_volume(bb: tuple) -> float:
    """Volume enclosed by the box ``bb``, given as ``(x0, y0, z0, x1, y1, z1)``."""
    return max(bb[3] - bb[0], 0) * max(bb[4] - bb[1], 0) * max(bb[5] - bb[2], 0)


def _dist_point_bbox(bb: tuple, p: tuple) -> float:
    """Shortest distance from the point ``p`` to the box ``bb``, ``0`` if inside."""
    dx = max(bb[0] - p[0], 0.0, p[0] - bb[3])
    dy = max(bb[1] - p[1], 0.0, p[1] - bb[4])
    dz = max(bb[2] - p[2], 0.0, p[2] - bb[5])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _dielectric_bbox(originals: list) -> tuple:
    """Extents of all the dielectric solids together, i.e. the board itself."""
    bb = [1e30, 1e30, 1e30, -1e30, -1e30, -1e30]
    for role, _name, obb, _v in originals:
        if role == "dielectric":
            for i in range(3):
                bb[i] = min(bb[i], obb[i])
                bb[i + 3] = max(bb[i + 3], obb[i + 3])
    return tuple(bb)


def _footprint_face(solid_tag: int, diel_bbox: tuple) -> int:
    """The face of a thin solid that becomes its sheet: the largest one, or on
    a tie the one nearest the dielectric so the sheet lands on its surface."""
    faces = [t for _, t in gmsh.model.getBoundary([(3, solid_tag)], oriented=False)]
    info = []
    for ft in faces:
        area = gmsh.model.occ.getMass(2, ft)
        com = gmsh.model.occ.getCenterOfMass(2, ft)
        info.append((area, _dist_point_bbox(diel_bbox, com), ft))
    max_area = max(a for a, _d, _t in info)
    plate = [(d, t) for a, d, t in info if a >= 0.99 * max_area]
    plate.sort()  # by distance to dielectric, ascending
    return plate[0][1]


def _clip_sheet_to_dielectric(face_tag: int, direction: str, diel_bbox: tuple) -> int:
    """Trim a port sheet to the dielectric's extent along ``direction``, so the
    port sits flush between the conductors instead of protruding past them."""
    ai = {"x": 0, "y": 1, "z": 2}[direction]
    big = 0.1
    lo = [diel_bbox[0] - big, diel_bbox[1] - big, diel_bbox[2] - big]
    hi = [diel_bbox[3] + big, diel_bbox[4] + big, diel_bbox[5] + big]
    lo[ai], hi[ai] = diel_bbox[ai], diel_bbox[ai + 3]  # exact span along direction
    box = gmsh.model.occ.addBox(
        lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    )
    out, _ = gmsh.model.occ.intersect(
        [(2, face_tag)], [(3, box)], removeObject=True, removeTool=True
    )
    faces = [t for d, t in out if d == 2]
    return faces[0] if faces else face_tag


# export_cad floors every box dimension at 1e-3 drawing units, so a port solid
# that was drawn flat is not exactly flat by the time it reaches gmsh.
_FLAT = 2e-6  # metres; the 1e-3 mm floor with room to spare

# A feed drawn all the way to the board edge leaves the port sheet exactly on
# the dielectric's own end face. The two fragment into coincident surfaces and
# gmsh rejects the result as overlapping facets, so the sheet is stepped inside.
_PORT_EDGE_STEP = 1e-5  # metres; 8e-5 wavelengths at 2.45 GHz


def _port_prop_axis(bb: tuple, diel_bbox: tuple) -> int:
    """Work out which axis a port's line runs along.

    The port face is normal to it, so the port solid is flat along it. A lumped
    port box is flat along exactly one axis -- it spans the trace width and the
    gap, and nothing along the run -- which settles it outright. A coplanar
    waveguide port is drawn as one thin box per gap and so is flat along *two*:
    the run, and the substrate normal. The substrate normal is the thinnest
    axis of the dielectric, which tells the two apart.

    Parameters
    ----------
    bb : tuple
        Extents of the port, as ``(x0, y0, z0, x1, y1, z1)``.
    diel_bbox : tuple
        Extents of the dielectrics, whose thinnest axis is the board normal.

    Returns
    -------
    int
        Axis index ``0``, ``1`` or ``2``.

    Raises
    ------
    RuntimeError
        If the port's extents do not single one out.
    """
    span = [bb[i + 3] - bb[i] for i in range(3)]
    flat = [i for i in range(3) if span[i] <= _FLAT]
    if len(flat) == 1:
        return flat[0]
    if len(flat) == 2:
        diel_span = [diel_bbox[i + 3] - diel_bbox[i] for i in range(3)]
        normal = min(range(3), key=lambda i: diel_span[i])
        remaining = [i for i in flat if i != normal]
        if len(remaining) == 1:
            return remaining[0]
    raise RuntimeError(
        f"cannot tell which axis this port's line runs along: its extents are "
        f"{tuple(round(v, 9) for v in span)} m, which leaves {len(flat)} flat "
        "axis/axes. A wave port has to be a plane cutting across the line. "
        "Give the axis explicitly with prop_dir on the port."
    )


def _step_sheet_off_the_dielectric_edge(face_tag: int, diel_bbox: tuple) -> None:
    """Move a port sheet that lies on the dielectric's end face inside it.

    A feed drawn to the board edge -- an edge-launch connector -- leaves the
    sheet coplanar with that face, which fragments into coincident surfaces.
    The sheet is stepped inside along the axis it is flat in, the axis the line
    runs along, which moves the reference plane by well under a thousandth of a
    wavelength. A sheet that already sits inside the board is left alone.

    Parameters
    ----------
    face_tag : int
        The port sheet to place.
    diel_bbox : tuple
        Extents of the dielectrics, as ``(x0, y0, z0, x1, y1, z1)``.

    Returns
    -------
    None
    """
    sheet_bbox = gmsh.model.getBoundingBox(2, face_tag)
    axis = _port_prop_axis(sheet_bbox, diel_bbox)
    position = sheet_bbox[axis]
    for face, inward in ((diel_bbox[axis], 1.0), (diel_bbox[axis + 3], -1.0)):
        if abs(position - face) < _FLAT:
            step = [0.0, 0.0, 0.0]
            step[axis] = inward * _PORT_EDGE_STEP
            gmsh.model.occ.translate([(2, face_tag)], *step)
            gmsh.model.occ.synchronize()
            return


# Ansys' microstrip wave-port guidance: wide enough and tall enough that the
# fringing field has died away before it reaches the port's PEC outline.
_WAVEPORT_WIDTH_NARROW = 10.0  # port widths per trace width, trace narrower than h
_WAVEPORT_WIDTH_WIDE = 5.0  # port widths per trace width, otherwise
_WAVEPORT_HEIGHT = 6.0  # port heights per substrate thickness

# A wave port terminates the domain, so its plane has to be an end of the
# structure. A CPW port's plane is snapped onto the FDTD grid before the FEM
# backend ever sees it, which lands it a fraction of a millimetre inside the
# board, so a plane this close to the end is pulled out to it.
_PORT_SNAP_FRAC = 0.01  # of the shortest wavelength in the densest dielectric


def _structure_bbox(originals: list) -> tuple:
    """Extents of the structure proper: everything but the ports.

    A port solid is measured and then deleted, and the exporter gives one drawn
    as a sheet a micron of thickness, so it reaches half a micron past the board
    it sits on. That overhang must not decide where the board ends.

    Parameters
    ----------
    originals : list
        One ``(role, name, bbox, bbox_volume)`` entry per solid.

    Returns
    -------
    tuple
        Extents as ``(x0, y0, z0, x1, y1, z1)``, empty-safe: a geometry of
        nothing but ports returns zeros.
    """
    boxes = [bb for role, _n, bb, _v in originals if role not in ("ignore", "port")]
    if not boxes:
        return (0.0,) * 6
    return tuple(
        [min(bb[i] for bb in boxes) for i in range(3)]
        + [max(bb[i + 3] for bb in boxes) for i in range(3)]
    )


def _wave_port_faces(
    problem: Problem,
    port_geo: dict,
    originals: list,
    diel_bbox: tuple,
    lambda_min: float,
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Find which faces of the structure the wave ports stand on.

    A wave port is a terminating boundary, not a sheet in the middle of the air:
    the mode it launches has nothing behind it. So the air box must put its wall
    on the port plane, and this reports which faces those are and where they go.

    The plane is the end of the structure proper, not of the port solid, and not
    of the padded box: it has to cut the board, or the cross-section comes out
    all air and the mode solve finds nothing guided.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying the ports and their propagation axes.
    port_geo : dict
        Extents of each port solid, keyed by name.
    originals : list
        One ``(role, name, bbox, bbox_volume)`` entry per solid, which
        :func:`_structure_bbox` measures the structure proper from.
    diel_bbox : tuple
        Extents of the dielectrics, used to derive an unset propagation axis.
    lambda_min : float
        Shortest wavelength in the densest dielectric, in metres, which sets how
        far a port may be snapped.

    Returns
    -------
    tuple[dict[tuple[int, int], float], dict[int, float]]
        The plane to put each terminated face on, keyed by ``(axis, side)`` with
        ``side`` ``0`` for the low face and ``1`` for the high one; and the plane
        each wave port sits on, keyed by port number.

    Raises
    ------
    RuntimeError
        If a wave port's plane is too far inside the structure to be an end of
        it, so no domain wall can be placed on it.
    """
    struct = _structure_bbox(originals)
    snap = _PORT_SNAP_FRAC * lambda_min
    faces: dict[tuple[int, int], float] = {}
    planes: dict[int, float] = {}
    for pspec in problem.ports:
        if pspec.kind != "wave":
            continue
        boxes = [port_geo[name] for name in pspec.solids if name in port_geo]
        if not boxes:
            continue
        merged = tuple(
            [min(box[i] for box in boxes) for i in range(3)]
            + [max(box[i + 3] for box in boxes) for i in range(3)]
        )
        axis = (
            {"x": 0, "y": 1, "z": 2}[pspec.prop_dir]
            if pspec.prop_dir
            else _port_prop_axis(merged, diel_bbox)
        )
        plane = 0.5 * (merged[axis] + merged[axis + 3])
        ends = ((struct[axis], 0), (struct[axis + 3], 1))
        end, side = min(ends, key=lambda pair: abs(plane - pair[0]))
        offset = abs(plane - end)
        if offset > snap:
            raise RuntimeError(
                f"wave port {pspec.number} sits {offset * 1e3:.4f} mm inside the "
                f"structure along '{'xyz'[axis]}', which spans "
                f"{struct[axis] * 1e3:.4f} to {struct[axis + 3] * 1e3:.4f} mm. A "
                "wave port terminates the simulation domain, so it has to stand "
                "on an end of the structure -- draw the line out to the board "
                "edge, or use FEM_port_type='lumpedport' for a feed inside the "
                "board."
            )
        if offset > 0.0:
            console.print(
                f"[info]FEM: wave port {pspec.number} moved "
                f"{offset * 1e3:.4f} mm onto the {'xyz'[axis]}{'-+'[side]} end "
                "of the structure, so it terminates the domain.[/info]"
            )
        faces[(axis, side)] = end
        planes[pspec.number] = end
    return faces, planes


def _wave_port_span(
    port_bbox: tuple,
    axis: int,
    diel_bbox: tuple,
    domain_bbox: tuple,
    width_mm: float | None = None,
    height_mm: float | None = None,
    plane: float | None = None,
) -> tuple:
    """Size a wave port's cross-section around the line it terminates.

    Parameters
    ----------
    port_bbox : tuple
        Extents of the port, all its solids together, as
        ``(x0, y0, z0, x1, y1, z1)`` in metres.
    axis : int
        Axis the line runs along, which the cross-section is normal to.
    diel_bbox : tuple
        Extents of the dielectrics; their thinnest axis is the board normal
        and their span along it the substrate height.
    domain_bbox : tuple
        Extents of the domain, which the cross-section is clipped to.
    width_mm : float, optional
        Width across the line in millimetres, centred on the port. ``None``
        derives it from the trace width.
    height_mm : float, optional
        Height along the board normal in millimetres, from the ground side of
        the substrate. ``None`` derives it from the substrate height.
    plane : float, optional
        Position along ``axis`` to place the cross-section on, which is both
        the domain wall and the S-parameters' reference plane. ``None`` (the
        default) keeps the plane the port solid was drawn on.

    Returns
    -------
    tuple
        Extents of the cross-section, flat along ``axis`` at the port plane.

    Raises
    ------
    RuntimeError
        If the line runs along the board normal, so no cross-section fits.
    """
    diel_span = [diel_bbox[i + 3] - diel_bbox[i] for i in range(3)]
    normal = min(range(3), key=lambda i: diel_span[i])
    if normal == axis:
        raise RuntimeError(
            "a wave port's line cannot run along the board normal; give the "
            "axis explicitly with prop_dir on the port"
        )
    across = 3 - axis - normal
    trace_width = port_bbox[across + 3] - port_bbox[across]
    substrate_height = diel_span[normal]

    if width_mm is not None:
        width = width_mm * 1e-3
    elif trace_width < substrate_height:
        width = _WAVEPORT_WIDTH_NARROW * trace_width
    else:
        width = _WAVEPORT_WIDTH_WIDE * trace_width
    height = (
        height_mm * 1e-3
        if height_mm is not None
        else _WAVEPORT_HEIGHT * substrate_height
    )

    span = list(domain_bbox)
    # the port plane, which is both the domain wall and the S-parameters'
    # reference plane
    at = plane if plane is not None else 0.5 * (port_bbox[axis] + port_bbox[axis + 3])
    span[axis] = at
    span[axis + 3] = at

    centre = 0.5 * (port_bbox[across] + port_bbox[across + 3])
    span[across] = max(domain_bbox[across], centre - width / 2)
    span[across + 3] = min(domain_bbox[across + 3], centre + width / 2)

    # the ground sits on the substrate face farther from the trace
    line_level = 0.5 * (port_bbox[normal] + port_bbox[normal + 3])
    diel_mid = 0.5 * (diel_bbox[normal] + diel_bbox[normal + 3])
    if line_level >= diel_mid:
        span[normal] = diel_bbox[normal]
        span[normal + 3] = min(domain_bbox[normal + 3], diel_bbox[normal] + height)
    else:
        span[normal + 3] = diel_bbox[normal + 3]
        span[normal] = max(domain_bbox[normal], diel_bbox[normal + 3] - height)
    return tuple(span)


def _wave_port_sheet(bb: tuple, prop_dir: str) -> int:
    """Build the cross-section rectangle of a wave port.

    A lumped port's sheet is the footprint of its own solid, pressed against
    the dielectric. A wave port is not a footprint at all: it is the plane the
    line's mode lives on, cutting straight across the substrate and the air
    above it. So the sheet is rebuilt from the solid's extents rather than
    copied off one of its faces, which keeps it exactly planar and exactly
    normal to the direction of propagation however the solid was drawn.

    Parameters
    ----------
    bb : tuple
        Extents of the port solid, as ``(x0, y0, z0, x1, y1, z1)``.
    prop_dir : str
        Axis the line runs along, which the sheet is normal to.

    Returns
    -------
    int
        Tag of the created surface.
    """
    p = {"x": 0, "y": 1, "z": 2}[prop_dir]
    u, v = (p + 1) % 3, (p + 2) % 3
    at = 0.5 * (bb[p] + bb[p + 3])

    corners = []
    for cu, cv in ((0, 0), (1, 0), (1, 1), (0, 1)):
        xyz = [0.0, 0.0, 0.0]
        xyz[p] = at
        xyz[u] = bb[u + 3] if cu else bb[u]
        xyz[v] = bb[v + 3] if cv else bb[v]
        corners.append(gmsh.model.occ.addPoint(*xyz))
    lines = [gmsh.model.occ.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    loop = gmsh.model.occ.addCurveLoop(lines)
    return gmsh.model.occ.addPlaneSurface([loop])


def _init() -> None:
    """Initialise a fresh, quiet Gmsh session."""
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)


@dataclass
class _Faces:
    """The sheet and boundary faces passed between :func:`build_mesh`'s stages."""

    pec: set[int]
    imped_by_sigma: dict[float, set[int]]
    port_by_name: dict[str, set[int]]
    all_port: set[int]
    all_imped: set[int]
    ground: set[int]
    abc: set[int]
    sym: set[int]


def list_solids(step_file: str) -> list[str]:
    """
    List the names of the solids in a STEP file, without meshing it.

    Parameters
    ----------
    step_file : str
        Path to the STEP file to read.

    Returns
    -------
    list[str]
        The solid names.
    """
    _init()
    try:
        gmsh.model.add("probe")
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        names = []
        for dim, tag in gmsh.model.getEntities(3):
            names.append(_short_name(gmsh.model.getEntityName(dim, tag)))
        return names
    finally:
        gmsh.finalize()


# ----------------------------
# build_mesh stages
# ----------------------------
# Planar microwave conductors are electrically thin (tens of microns), so
# meshing them as 3D volumes forces microscopic elements. Instead every
# conductor and port is reduced to a zero-thickness sheet (its footprint face)
# that is imprinted into the dielectric/air mesh:
#
# 1. Import the STEP solids (converted to metres) and identify each by name.
# 2. For every metal/port solid, copy its footprint face (largest face, snapped
#    to the nearest dielectric surface) and delete the solid.
# 3. Wrap the structure in an air box padded by ~lambda/4; the outer faces
#    become the Silver-Muller absorbing boundary (or PML volumes).
# 4. `fragment` the dielectric volumes + air box with the conductor sheets so
#    the sheets become conforming interior faces.
# 5. Classify each resulting volume back to its origin solid.
# 6. The outer absorbing boundary is the domain shell minus the PEC/impedance/
#    port faces.
# 7. Assign integer physical groups, refine near conductors, mesh, and write.
def _snapshot_solids(problem: Problem) -> tuple[list, tuple, tuple, dict]:
    """Record each imported solid's role and extents before the geometry is cut up.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying each solid's role.

    Returns
    -------
    tuple[list, tuple, tuple, dict]
        ``(originals, struct, diel_bbox, port_geo)`` -- one
        ``(role, name, bbox, bbox_volume)`` entry per solid, the extents of the
        whole structure, the extents of the dielectrics, and the extents of
        each port solid keyed by name.
    """
    # the boolean operations in the later stages destroy and renumber solids,
    # so this has to happen while their identities are still known
    originals = []  # (role, name, bbox, bbox_volume)
    struct = [1e30, 1e30, 1e30, -1e30, -1e30, -1e30]
    for dim, tag in gmsh.model.getEntities(3):
        short = _short_name(gmsh.model.getEntityName(dim, tag))
        spec = problem.solids.get(short)
        role = spec.role if spec else "ignore"
        bb = gmsh.model.getBoundingBox(dim, tag)
        originals.append((role, short, bb, _bbox_volume(bb)))
        if role != "ignore":
            for i in range(3):
                struct[i] = min(struct[i], bb[i])
                struct[i + 3] = max(struct[i + 3], bb[i + 3])
    struct = tuple(struct)

    diel_bbox = _dielectric_bbox(originals)
    port_geo = {name: bb for role, name, bb, _v in originals if role == "port"}
    return originals, struct, diel_bbox, port_geo


def _vertex_bounds(dim: int, tag: int) -> list[float]:
    """Extents of an entity from its vertices, as ``[x0, y0, z0, x1, y1, z1]``.

    ``getBoundingBox`` pads by the geometry tolerance, so a sheet rebuilt from
    it overhangs its neighbours by a fraction of a micron and fragments into
    slivers. The vertices give the exact corners of a flat rectangle.
    """
    points = gmsh.model.getBoundary([(dim, tag)], recursive=True)
    coords = [gmsh.model.getValue(0, point, []) for _, point in points]
    return [min(c[i] for c in coords) for i in range(3)] + [
        max(c[i] for c in coords) for i in range(3)
    ]


def _conductor_slabs(problem: Problem, diel_bbox: tuple) -> list:
    """Record where each conductor is, and where its sheet will be.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying each solid's role.
    diel_bbox : tuple
        Extents of the dielectrics, which pick each conductor's sheet face.

    Returns
    -------
    list
        One ``(solid_bbox, sheet_bounds)`` entry per conductor: the extents of
        the solid, and the exact extents of the face that becomes its sheet.
    """
    slabs = []
    for dim, tag in gmsh.model.getEntities(3):
        spec = problem.solids.get(_short_name(gmsh.model.getEntityName(dim, tag)))
        if spec and spec.role in ("pec", "lossy_conductor"):
            face = _footprint_face(tag, diel_bbox)
            slabs.append((gmsh.model.getBoundingBox(dim, tag), _vertex_bounds(2, face)))
    return slabs


def _snap_port_to_conductor_sheets(
    face_tag: int, direction: str, conductor_slabs: list
) -> int:
    """Stretch a lumped port's ends onto the sheets its conductors became.

    A conductor turns into a sheet on one face of its solid, so a port drawn to
    the conductor's other face stops a copper thickness short and connects to
    nothing. An inner plane has dielectric on both faces, so which face it
    keeps is a tie-break, and a port on either side has to reach it. Only an
    end inside a conductor's own thickness moves, so a port is never pulled
    across a dielectric onto a different conductor.

    Parameters
    ----------
    face_tag : int
        The port sheet.
    direction : str
        Axis the port's gap runs along, ``"x"``, ``"y"`` or ``"z"``.
    conductor_slabs : list
        One ``(solid_bbox, sheet_bounds)`` entry per conductor, from
        :func:`_conductor_slabs`.

    Returns
    -------
    int
        Tag of the port sheet: the same one if no end moved, else its
        replacement.
    """
    axis = {"x": 0, "y": 1, "z": 2}[direction]
    bounds = _vertex_bounds(2, face_tag)
    lateral = [index for index in range(3) if index != axis]
    moved = False
    for solid_bbox, sheet_bounds in conductor_slabs:
        if sheet_bounds[axis + 3] - sheet_bounds[axis] > _FLAT:
            continue  # this sheet does not lie across the port's gap
        overlaps = all(
            bounds[index] <= sheet_bounds[index + 3] + _FLAT
            and bounds[index + 3] >= sheet_bounds[index] - _FLAT
            for index in lateral
        )
        if not overlaps:
            continue
        sheet_position = sheet_bounds[axis]
        for end in (axis, axis + 3):
            inside_slab = (
                solid_bbox[axis] - _FLAT <= bounds[end] <= solid_bbox[axis + 3] + _FLAT
            )
            if inside_slab and abs(bounds[end] - sheet_position) > _FLAT:
                bounds[end] = sheet_position
                moved = True
    if not moved:
        return face_tag
    flat_axis = min(range(3), key=lambda index: bounds[index + 3] - bounds[index])
    gmsh.model.occ.remove([(2, face_tag)], recursive=True)
    return _wave_port_sheet(tuple(bounds), "xyz"[flat_axis])


def _build_footprint_sheets(problem: Problem, diel_bbox: tuple, port_geo: dict) -> list:
    """Replace each thin conductor and port solid with a zero-thickness sheet.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying each solid's role and each port's direction.
        A wave port's solids are only measured and removed here; their sheet is
        built later, by :func:`_build_wave_port_sheets`.
    diel_bbox : tuple
        Extents of the dielectrics, which the sheets are placed against.
    port_geo : dict
        Extents of each port solid by name, updated in place with the extents
        of the sheet that replaced it.

    Returns
    -------
    list
        One ``(role, name, face_tag)`` entry per sheet created.
    """
    port_direction = {}
    wave_solids: set[str] = set()
    for pspec in problem.ports:
        for name in pspec.solids:
            port_direction[name] = pspec.direction
            if pspec.kind == "wave":
                wave_solids.add(name)
    sheets = []  # (role, name, face_tag)
    metal_solids = []
    conductor_slabs = _conductor_slabs(problem, diel_bbox)
    for dim, tag in gmsh.model.getEntities(3):
        short = _short_name(gmsh.model.getEntityName(dim, tag))
        spec = problem.solids.get(short)
        role = spec.role if spec else "ignore"
        if role in ("pec", "lossy_conductor", "port"):
            if role == "port" and short in wave_solids:
                # A wave port's plane *is* the domain wall, and its rectangle is
                # clipped to the domain -- but the air box does not exist yet; it
                # is built after this. So only the plane is recorded here, and
                # _build_wave_port_sheets makes the sheet once there is a domain.
                port_geo[short] = gmsh.model.getBoundingBox(dim, tag)
                metal_solids.append((3, tag))
                continue
            face = _footprint_face(tag, diel_bbox)
            cp = gmsh.model.occ.copy([(2, face)])
            face_tag = cp[0][1]
            if role == "port":
                # clip flush to the dielectric so the port doesn't protrude
                face_tag = _clip_sheet_to_dielectric(
                    face_tag, port_direction.get(short, "z"), diel_bbox
                )
                gmsh.model.occ.synchronize()
                _step_sheet_off_the_dielectric_edge(face_tag, diel_bbox)
                face_tag = _snap_port_to_conductor_sheets(
                    face_tag, port_direction.get(short, "z"), conductor_slabs
                )
                gmsh.model.occ.synchronize()
                port_geo[short] = gmsh.model.getBoundingBox(2, face_tag)
            sheets.append((role, short, face_tag))
            metal_solids.append((3, tag))
    # delete the metal/port solids; dielectric solids remain and are fragmented
    if metal_solids:
        gmsh.model.occ.remove(metal_solids, recursive=True)
    gmsh.model.occ.synchronize()
    return sheets


def _build_wave_port_sheets(
    problem: Problem,
    sheets: list,
    port_geo: dict,
    diel_bbox: tuple,
    domain_bbox: tuple,
    port_planes: dict[int, float] | None = None,
) -> list:
    """Build each wave port's cross-section, once the domain extents are known.

    One sheet per port, not per solid: a coplanar waveguide port is drawn as
    one box per gap, and the mode lives on the single plane that cuts across
    both of them together. The sheet is a rectangle on that plane, centred on
    the line and standing on the ground side of the substrate. Its size comes
    from ``problem.waveport_width_mm`` and ``problem.waveport_height_mm``, or
    from the trace width and substrate height when those are ``None``, and is
    clipped to the domain.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying the ports, their propagation axes, and the
        wave-port size.
    sheets : list
        The ``(role, name, face_tag)`` entries so far, appended to.
    port_geo : dict
        Extents of each port solid by name; the entry for each port's first
        solid is replaced with the extents of the sheet built for it.
    diel_bbox : tuple
        Extents of the dielectrics, used to find the board normal and the
        substrate height.
    domain_bbox : tuple
        Extents the sheet is clipped to, as ``(x0, y0, z0, x1, y1, z1)``.
    port_planes : dict, optional
        Plane each port is placed on, keyed by port number, from
        :func:`_wave_port_faces`. A port missing from it keeps the plane its
        solid was drawn on.

    Returns
    -------
    list
        ``sheets``, with one entry added per wave port.

    Raises
    ------
    RuntimeError
        If a port's solids were not measured, or its axis cannot be worked out.
    """
    for pspec in problem.ports:
        if pspec.kind != "wave":
            continue
        boxes = [port_geo[n] for n in pspec.solids if n in port_geo]
        if not boxes:
            raise RuntimeError(
                f"port {pspec.number}: none of its solids {pspec.solids} were "
                "found in the geometry"
            )
        # the union of every gap, which is the port as a whole
        merged = tuple(
            [min(b[i] for b in boxes) for i in range(3)]
            + [max(b[i + 3] for b in boxes) for i in range(3)]
        )
        axis = (
            {"x": 0, "y": 1, "z": 2}[pspec.prop_dir]
            if pspec.prop_dir
            else _port_prop_axis(merged, diel_bbox)
        )
        span = _wave_port_span(
            merged,
            axis,
            diel_bbox,
            domain_bbox,
            problem.waveport_width_mm,
            problem.waveport_height_mm,
            (port_planes or {}).get(pspec.number),
        )
        face_tag = _wave_port_sheet(span, "xyz"[axis])
        gmsh.model.occ.synchronize()
        port_geo[pspec.solid] = gmsh.model.getBoundingBox(2, face_tag)
        sheets.append(("port", pspec.solid, face_tag))
    gmsh.model.occ.synchronize()
    return sheets


def _build_air_box(
    problem: Problem,
    struct: tuple,
    lambda0_mesh: float,
    port_faces: dict[tuple[int, int], float] | None = None,
) -> tuple[bool, tuple, tuple, float]:
    """Wrap the structure in an air box, plus an outer PML shell if requested.

    Each of the six faces stands off the structure by its own padding, so a
    radiator can carry a deep air column on the side it radiates into without
    paying for it on the other five.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying the boundary condition and the air padding.
    struct : tuple
        Extents of the structure to wrap, in metres.
    lambda0_mesh : float
        Free-space wavelength at the mesh frequency, in metres, which sets the
        padding unless ``problem.air_pad_mm`` gives it directly.
    port_faces : dict, optional
        Where to put each face a wave port terminates, keyed by ``(axis, side)``,
        from :func:`_wave_port_faces`. Those faces go on the port plane rather
        than being padded, and carry no PML shell: the port is the boundary.

    Returns
    -------
    tuple[bool, tuple, tuple, float]
        ``(is_pml, inner_bbox, box_bbox, pml_thick)`` -- whether a PML was
        added, the extents of the air box, the extents of the whole domain,
        and the PML thickness in metres.
    """
    is_pml = problem.boundary == "pml"
    faces_mm = problem.air_pad_faces_mm
    if faces_mm is None:
        auto = max(problem.air_pad_frac * lambda0_mesh, 3.0 * (struct[5] - struct[2]))
        faces = [[auto, auto] for _ in range(3)]
    else:
        faces = [[low * 1e-3, high * 1e-3] for low, high in faces_mm]
    port_faces = port_faces or {}
    # a wave port is the domain wall, so its face goes on the port plane
    for axis, side in sorted(port_faces):
        if faces[axis][side] > 0.0:
            console.print(
                f"[info]FEM: dropping the {'xyz'[axis]}{'-+'[side]} air padding "
                f"({faces[axis][side] * 1e3:.3f} mm): a wave port terminates the "
                "domain on that face.[/info]"
            )
        faces[axis][side] = 0.0
    # inner box: each face stands off the structure by its own padding, except
    # the ones a wave port terminates, which sit exactly on its plane
    low = [struct[axis] - faces[axis][0] for axis in range(3)]
    high = [struct[3 + axis] + faces[axis][1] for axis in range(3)]
    for (axis, side), plane in port_faces.items():
        (low if side == 0 else high)[axis] = plane
    ax0, ay0, az0 = low
    ax1, ay1, az1 = high
    inner_bbox = (ax0, ay0, az0, ax1, ay1, az1)
    gmsh.model.occ.addBox(ax0, ay0, az0, ax1 - ax0, ay1 - ay0, az1 - az0)
    pml_thick = 0.0
    if is_pml:
        # one uniform shell, sized off the tightest face it actually covers so
        # it never outgrows the smallest air gap; the .pro template carries a
        # single PmlDelta. A wave-port face carries no shell -- the port is the
        # boundary there, and a shell behind it would bury it in the domain.
        padded = [
            faces[axis][side]
            for axis in range(3)
            for side in range(2)
            if (axis, side) not in port_faces
        ]
        pml_thick = max(0.2 * lambda0_mesh, 2.0 * min(padded) / 3.0) if padded else 0.0
        outer_low = list(low)
        outer_high = list(high)
        for axis in range(3):
            if (axis, 0) not in port_faces:
                outer_low[axis] -= pml_thick
            if (axis, 1) not in port_faces:
                outer_high[axis] += pml_thick
        gmsh.model.occ.addBox(
            *outer_low, *(outer_high[i] - outer_low[i] for i in range(3))
        )
        box_bbox = (*outer_low, *outer_high)
    else:
        box_bbox = inner_bbox
    gmsh.model.occ.synchronize()
    return is_pml, inner_bbox, box_bbox, pml_thick


def _apply_symmetry_cut(
    problem: Problem, struct: tuple, box_bbox: tuple, sheets: list
) -> tuple[int | None, float | None, list]:
    """Discard everything below the symmetry plane, if one is set.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying the symmetry plane.
    struct : tuple
        Extents of the structure, used to place a plane given as ``None``.
    box_bbox : tuple
        Extents of the whole domain to cut.
    sheets : list
        The conductor and port sheets, as ``(role, name, face_tag)``.

    Returns
    -------
    tuple[int | None, float | None, list]
        ``(sym_axis_i, sym_plane, sheets)`` -- the mirrored axis, the plane
        position in metres, and the cut sheets. All three are unchanged and
        the first two are ``None`` if no symmetry plane is set.
    """
    if not problem.symmetry:
        return None, None, sheets

    s_axis, _s_kind, s_at = problem.symmetry
    sym_axis_i = {"x": 0, "y": 1, "z": 2}[s_axis]
    sym_plane = (
        s_at
        if s_at is not None
        else 0.5 * (struct[sym_axis_i] + struct[sym_axis_i + 3])
    )
    bb = box_bbox
    lo = [bb[0] - 1.0, bb[1] - 1.0, bb[2] - 1.0]
    hi = [bb[3] + 1.0, bb[4] + 1.0, bb[5] + 1.0]
    lo[sym_axis_i] = sym_plane  # keep coord >= plane
    halfbox = gmsh.model.occ.addBox(
        lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    )
    gmsh.model.occ.synchronize()
    vol_targets = [(3, t) for _, t in gmsh.model.getEntities(3) if t != halfbox]
    sheet_targets = [(2, f) for _, _, f in sheets]
    _o, omap = gmsh.model.occ.intersect(
        vol_targets + sheet_targets,
        [(3, halfbox)],
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    new_sheets = []  # sheets were clipped -> pick up new tags from the map
    for i, (role, name, _f) in enumerate(sheets):
        faces = [t for d, t in omap[len(vol_targets) + i] if d == 2]
        if faces:
            new_sheets.append((role, name, faces[0]))
    return sym_axis_i, sym_plane, new_sheets


def _fragment_and_classify(
    sheets: list, originals: list, inner_bbox: tuple, is_pml: bool
) -> tuple[list, dict, list, list]:
    """Embed the sheets in the volumes, then identify what each volume is.

    Parameters
    ----------
    sheets : list
        The conductor and port sheets, as ``(role, name, face_tag)``.
    originals : list
        Each original solid as ``(role, name, bbox, bbox_volume)``, used to
        identify the volumes afterwards.
    inner_bbox : tuple
        Extents of the air box, which separate air volumes from PML ones.
    is_pml : bool
        Whether the domain has a PML shell.

    Returns
    -------
    tuple[list, dict, list, list]
        ``(sheet_faces, diel_vols, air_vols, pml_vols)`` -- the faces of each
        sheet in the same order as ``sheets``, the dielectric volume tags by
        solid name, and the air and PML volume tags.
    """
    # `fragment` splits the volumes along the sheets so the sheets become
    # shared interior faces while the volumes stay watertight; this is what
    # lets zero-thickness metals be meshed at all. A volume is dielectric iff
    # its own bounding box fits inside a dielectric solid's box -- a centroid
    # test fails, since the enclosing air shell is centred on the structure too.
    vols = [(3, t) for _, t in gmsh.model.getEntities(3)]  # dielectrics + box
    sheet_dimtags = [(2, f) for _, _, f in sheets]
    _out, outmap = gmsh.model.occ.fragment(vols, sheet_dimtags)
    gmsh.model.occ.synchronize()

    # fragment map: first len(vols) entries are volumes, then the sheets
    sheet_faces = []  # aligned with `sheets`
    for i in range(len(sheets)):
        entry = outmap[len(vols) + i]
        sheet_faces.append({t for (d, t) in entry if d == 2})

    diel_vols: dict[str, list[int]] = {}
    air_vols: list[int] = []
    pml_vols: list[int] = []
    inner_diag = (
        (inner_bbox[3] - inner_bbox[0]) ** 2
        + (inner_bbox[4] - inner_bbox[1]) ** 2
        + (inner_bbox[5] - inner_bbox[2]) ** 2
    ) ** 0.5
    for dim, tag in gmsh.model.getEntities(3):
        vbb = gmsh.model.getBoundingBox(dim, tag)
        best = None  # (bbox_volume, name)
        for role, name, bb, bvol in originals:
            if role != "dielectric":
                continue
            diag = (
                (bb[3] - bb[0]) ** 2 + (bb[4] - bb[1]) ** 2 + (bb[5] - bb[2]) ** 2
            ) ** 0.5
            tol = max(1e-6, 1e-3 * diag)
            if _bbox_inside(vbb, bb, tol) and (best is None or bvol < best[0]):
                best = (bvol, name)
        if best is not None:
            diel_vols.setdefault(best[1], []).append(tag)
        elif is_pml and not _bbox_inside(vbb, inner_bbox, 1e-3 * inner_diag):
            pml_vols.append(tag)  # extends beyond the inner air box -> PML shell
        else:
            air_vols.append(tag)

    return sheet_faces, diel_vols, air_vols, pml_vols


def _collect_faces(
    problem: Problem,
    sheets: list,
    sheet_faces: list,
    box_bbox: tuple,
    sym_axis_i: int | None,
    sym_plane: float | None,
) -> _Faces:
    """Sort the meshed faces into the sets each gets a boundary condition from.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying each solid's role and conductivity.
    sheets : list
        The conductor and port sheets, as ``(role, name, face_tag)``.
    sheet_faces : list
        The faces of each sheet, in the same order as ``sheets``.
    box_bbox : tuple
        Extents of the whole domain, used to find the symmetry-plane faces.
    sym_axis_i : int | None
        Axis the structure is mirrored across, or ``None`` for no symmetry.
    sym_plane : float | None
        Position of the symmetry plane in metres, or ``None`` for no symmetry.

    Returns
    -------
    _Faces
        The conductor, port, ground-plane, absorbing-boundary, and
        symmetry-plane face sets.
    """
    existing_faces = {t for _, t in gmsh.model.getEntities(2)}
    pec_faces: set[int] = set()
    conductor_faces_by_name: dict[str, set[int]] = {}  # pec + lossy_conductor
    imped_faces_by_sigma: dict[float, set[int]] = {}  # lossy conductors by sigma
    port_faces_by_name: dict[str, set[int]] = {}
    for (role, name, _f), faces in zip(sheets, sheet_faces, strict=True):
        faces = faces & existing_faces
        if role == "pec":
            pec_faces |= faces
            conductor_faces_by_name.setdefault(name, set()).update(faces)
        elif role == "lossy_conductor":
            sig = problem.solids[name].sigma
            imped_faces_by_sigma.setdefault(sig, set()).update(faces)
            conductor_faces_by_name.setdefault(name, set()).update(faces)
        else:
            port_faces_by_name.setdefault(name, set()).update(faces)
    all_port_faces = (
        set().union(*port_faces_by_name.values()) if port_faces_by_name else set()
    )
    all_imped_faces = (
        set().union(*imped_faces_by_sigma.values()) if imped_faces_by_sigma else set()
    )

    # Identify the ground plane (largest-area conductor sheet). Excluding it from
    # the mesh-refinement set localises the fine mesh to the signal conductors.
    ground_faces: set[int] = set()
    if len(conductor_faces_by_name) > 1:

        def _area(fs: set) -> float:
            """Total area, in m^2, of the faces ``fs``."""
            return sum(gmsh.model.occ.getMass(2, f) for f in fs)

        ground_name = max(
            conductor_faces_by_name,
            key=lambda n: _area(conductor_faces_by_name[n]),
        )
        ground_faces = conductor_faces_by_name[ground_name]

    # ---- outer absorbing shell ----------------------------------------------
    domain_dimtags = gmsh.model.getEntities(3)
    shell = gmsh.model.getBoundary(domain_dimtags, combined=True, oriented=False)
    shell_faces = {t for _, t in shell}
    abc_faces = shell_faces - pec_faces - all_port_faces - all_imped_faces

    # symmetry-plane faces are boundary faces at the cut plane -> not radiating
    sym_faces: set[int] = set()
    if problem.symmetry:
        eps = max(1e-6, 1e-4 * (box_bbox[3 + sym_axis_i] - box_bbox[sym_axis_i]))
        lo = list(box_bbox[:3])
        hi = list(box_bbox[3:])
        lo[sym_axis_i] = sym_plane - eps
        hi[sym_axis_i] = sym_plane + eps
        at_plane = {
            t
            for _, t in gmsh.model.getEntitiesInBoundingBox(
                lo[0] - eps,
                lo[1] - eps,
                lo[2] - eps,
                hi[0] + eps,
                hi[1] + eps,
                hi[2] + eps,
                2,
            )
        }
        sym_faces = at_plane & abc_faces
        abc_faces -= sym_faces

    return _Faces(
        pec=pec_faces,
        imped_by_sigma=imped_faces_by_sigma,
        port_by_name=port_faces_by_name,
        all_port=all_port_faces,
        all_imped=all_imped_faces,
        ground=ground_faces,
        abc=abc_faces,
        sym=sym_faces,
    )


def _assign_physical_groups(
    problem: Problem,
    diel_vols: dict,
    air_vols: list,
    pml_vols: list,
    faces: _Faces,
    port_geo: dict,
) -> tuple[dict, int, list, dict, int]:
    """Tag every volume and face with the region number the solver refers to it by.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying the ports.
    diel_vols : dict
        Dielectric volume tags, keyed by solid name.
    air_vols : list
        Air volume tags.
    pml_vols : list
        PML volume tags.
    faces : _Faces
        The conductor, port, absorbing-boundary, and symmetry-plane faces.
    port_geo : dict
        Extents of each port sheet by solid name, used to measure the ports.

    Returns
    -------
    tuple[dict, int, list, dict, int]
        ``(diel_regions, pml_region, impedance_regions, port_regions,
        sym_region)`` -- the region tag assigned to each part of the mesh.

    Raises
    ------
    RuntimeError
        If a port solid produced no faces to tag.
    """
    # These tags are the only contract between the mesh and the solver, which
    # refers to each region purely by number. They come from fem_materials so
    # that fem_formulation emits exactly the same ones.
    diel_regions = {}
    for i, (name, tags) in enumerate(sorted(diel_vols.items())):
        rid = dielectric_region(i)
        gmsh.model.addPhysicalGroup(3, tags, rid)
        gmsh.model.setPhysicalName(3, rid, name)
        diel_regions[name] = rid

    if air_vols:
        gmsh.model.addPhysicalGroup(3, air_vols, AIR)
        gmsh.model.setPhysicalName(3, AIR, "air")

    pml_region = 0
    if pml_vols:
        pml_region = PML
        gmsh.model.addPhysicalGroup(3, pml_vols, pml_region)
        gmsh.model.setPhysicalName(3, pml_region, "pml")

    if faces.pec:
        gmsh.model.addPhysicalGroup(2, sorted(faces.pec), PEC)
        gmsh.model.setPhysicalName(2, PEC, "pec")

    impedance_regions = []  # [(region_id, sigma), ...]
    for i, (sig, fs) in enumerate(sorted(faces.imped_by_sigma.items())):
        rid = LOSSY_CONDUCTOR + i
        gmsh.model.addPhysicalGroup(2, sorted(fs), rid)
        gmsh.model.setPhysicalName(2, rid, f"impedance_{i}")
        impedance_regions.append((rid, sig))

    port_regions: dict[int, PortMesh] = {}
    for pspec in problem.ports:
        # every solid of the port, so a coplanar waveguide port's two gaps land
        # in one region instead of the second going untagged
        pfaces: set[int] = set()
        for name in pspec.solids:
            pfaces |= faces.port_by_name.get(name, set())
        if not pfaces:
            raise RuntimeError(
                f"port {pspec.number} ({', '.join(pspec.solids)}) produced no "
                "boundary faces"
            )
        rid = port_region(pspec.number)
        gmsh.model.addPhysicalGroup(2, sorted(pfaces), rid)
        gmsh.model.setPhysicalName(2, rid, f"port_{pspec.number}")
        bb = port_geo[pspec.solid]
        ax = {"x": 0, "y": 1, "z": 2}[pspec.direction]
        extents = [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
        gap = extents[ax]
        width = max(extents[i] for i in range(3) if i != ax)  # transverse width
        center = ((bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2)
        # `bb` predates the symmetry cut, so measure what actually got meshed.
        fb = [1e30, 1e30, 1e30, -1e30, -1e30, -1e30]
        for face in pfaces:
            b = gmsh.model.getBoundingBox(2, face)
            for i in range(3):
                fb[i] = min(fb[i], b[i])
                fb[i + 3] = max(fb[i + 3], b[i + 3])
        width_meshed = max(fb[i + 3] - fb[i] for i in range(3) if i != ax)
        port_regions[pspec.number] = PortMesh(
            number=pspec.number,
            region=rid,
            direction=pspec.direction,
            z0=pspec.z0,
            gap=gap,
            width=width,
            center=center,
            width_meshed=width_meshed,
            kind=pspec.kind,
            # By now a wave port's sheet is a plain rectangle, so its flat axis
            # is the propagation axis outright -- no need to re-derive it from
            # the original solids the way _build_wave_port_sheets had to.
            prop_dir=(
                pspec.prop_dir
                if pspec.prop_dir
                else (
                    "xyz"[min(range(3), key=lambda i: extents[i])]
                    if pspec.kind == "wave"
                    else ""
                )
            ),
        )

    if faces.abc:
        gmsh.model.addPhysicalGroup(2, sorted(faces.abc), ABC)
        gmsh.model.setPhysicalName(2, ABC, "abc")

    sym_region = 0
    if faces.sym:
        sym_region = SYM
        gmsh.model.addPhysicalGroup(2, sorted(faces.sym), sym_region)
        gmsh.model.setPhysicalName(2, sym_region, "sym")

    return diel_regions, pml_region, impedance_regions, port_regions, sym_region


def _mesh_and_write(
    problem: Problem,
    originals: list,
    struct: tuple,
    lambda_min: float,
    faces: _Faces,
    workdir: str | Path,
    verbose: bool,
    diel_regions: dict,
    port_regions: dict,
    diel_vols: dict | None = None,
    lambda_air: float = 0.0,
) -> str:
    """Size and generate the mesh, then write it out.

    Elements are small near the signal conductors and grow towards the target
    density of whichever material they sit in.

    Parameters
    ----------
    problem : Problem
        The FEM problem, supplying the mesh density settings.
    originals : list
        Each original solid as ``(role, name, bbox, bbox_volume)``, used to
        measure the dielectric thickness.
    struct : tuple
        Extents of the structure, in metres.
    lambda_min : float
        Shortest wavelength inside the dielectrics, in metres.
    faces : _Faces
        The faces to refine the mesh towards.
    workdir : str | Path
        Directory to write the mesh files into.
    verbose : bool
        Print the mesh size and region tags once written.
    diel_regions : dict
        Dielectric region tags by name, for the progress output.
    port_regions : dict
        Port descriptions by number, for the progress output.
    diel_vols : dict, optional
        Dielectric volume tags by name, sized against ``lambda_min``. Default
        ``None``.
    lambda_air : float
        Shortest wavelength in air, in metres, which sizes everything outside
        the dielectrics. Default ``0.0``, meaning use ``lambda_min``.

    Returns
    -------
    str
        Path to the written ``.msh`` file.
    """
    diel_min_thick = min(
        [
            min(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
            for role, _n, bb, _v in originals
            if role == "dielectric"
        ]
        or [struct[5] - struct[2]]
    )
    # element size that puts `min_layers` elements through the dielectric;
    # `mesh_fine_scale` (>=1) coarsens for cost, but never below one layer.
    lc_fine = max(
        diel_min_thick / max(problem.min_layers, 1) * problem.mesh_fine_scale, 20e-6
    )
    lc_fine = min(lc_fine, diel_min_thick)  # keep at least ~1 layer even if coarsened
    # same target density in each material, each against its own wavelength
    lc_diel = lambda_min / problem.elems_per_wavelength
    lc_air = (lambda_air or lambda_min) / problem.elems_per_wavelength
    # Refine near the signal conductors + ports only (exclude the ground plane);
    # extend the fine zone through the substrate so trace-to-ground is resolved.
    refine_faces = ((faces.pec | faces.all_imped) - faces.ground) | faces.all_port
    dist_max = max(15.0 * lc_fine, 4.0 * diel_min_thick)
    diel_tags = sorted({t for tags in (diel_vols or {}).values() for t in tags})
    _apply_size_field(refine_faces, lc_fine, lc_air, dist_max, diel_tags, lc_diel)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max(lc_air, lc_diel))
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc_fine / 5.0)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    _mesh_cylinders_with_meshadapt()

    gmsh.model.mesh.generate(3)
    msh_path = Path(workdir).absolute() / f"{problem.name}.msh"
    gmsh.write(str(msh_path))
    # also write a legacy VTK for pyvista display (no meshio dependency needed)
    gmsh.write(str(msh_path.with_suffix(".vtk")))

    if verbose:
        nn = len(gmsh.model.mesh.getNodes()[0])
        console.print(f"[info]mesh written: {msh_path}  ({nn} nodes)[/info]")
        console.print(
            f"[info]dielectrics={diel_regions} air={AIR} pec={PEC} "
            f"ports={list(port_regions)} abc={ABC}[/info]"
        )
    gmsh.finalize()
    return str(msh_path)


def _mesh_cylinders_with_meshadapt() -> None:
    """Mesh every cylindrical face with Gmsh's MeshAdapt algorithm.

    Gmsh meshes a cylinder in its unrolled parameter space, where the seam
    appears twice, once on each side. On a thin, finely sized barrel -- a via
    -- the default Frontal-Delaunay algorithm can join three nodes of that seam
    into one triangle. Rolled back up, the triangle is flat, and getdp fails on
    its zero Jacobian with ``Null determinant in 'ChangeOfCoord_Form2'``.
    MeshAdapt meshes those faces cleanly; every other face keeps the default.
    """
    for dim, tag in gmsh.model.getEntities(2):
        if gmsh.model.getType(dim, tag) == "Cylinder":
            gmsh.model.mesh.setAlgorithm(dim, tag, 1)


# ----------------------------
# main entry point
# ----------------------------
def build_mesh(problem: Problem, workdir: str | Path, verbose: bool = True) -> Mesh:
    """
    Mesh the STEP geometry of a :class:`~simpleEMS.fem_backend.Problem`.

    Parameters
    ----------
    problem : Problem
        The problem to mesh, supplying the solids, ports, boundary condition,
        and mesh settings.
    workdir : str | Path
        Directory to write the mesh files into.
    verbose : bool
        Print progress. Default ``True``.

    Returns
    -------
    Mesh
        The mesh and the region tags identifying its parts.

    Raises
    ------
    RuntimeError
        If a port solid produces no faces, e.g. because it does not touch the
        rest of the geometry.
    """
    fmesh = problem.mesh_freq
    eps_max = max([d.dielectric.eps_r for d in problem.dielectrics()] + [1.0])
    # free-space wavelength at the mesh frequency: sizes the air elements and
    # the padding around the structure
    lambda0_mesh = C0 / fmesh
    # smallest wavelength anywhere (inside the highest-eps dielectric)
    lambda_min = lambda0_mesh / (eps_max**0.5)

    Path(workdir).mkdir(parents=True, exist_ok=True)
    _init()
    if verbose:
        gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(problem.name)
    # STEP may be in mm; OCCTargetUnit converts everything to metres so all the
    # geometry below (and the wavelengths above) share one unit system.
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.model.occ.importShapes(problem.step_file)
    gmsh.model.occ.synchronize()

    originals, struct, diel_bbox, port_geo = _snapshot_solids(problem)
    # kept for the wave-port mode solve, which needs to know which part of a
    # port's cross-section is substrate and which is air
    diel_bboxes = {name: bb for role, name, bb, _v in originals if role == "dielectric"}
    # A wave port terminates the domain, so the air box must not pad the face it
    # stands on. That has to be settled before the box is built.
    port_faces, port_planes = _wave_port_faces(
        problem, port_geo, originals, diel_bbox, lambda_min
    )
    sheets = _build_footprint_sheets(problem, diel_bbox, port_geo)
    is_pml, inner_bbox, box_bbox, pml_thick = _build_air_box(
        problem, struct, lambda0_mesh, port_faces
    )
    # Now the domain exists, so a wave port's cross-section can be clipped to
    # it. Under a PML it is clipped to the inner box, so the sheet does not cut
    # into the absorbing shell.
    sheets = _build_wave_port_sheets(
        problem,
        sheets,
        port_geo,
        diel_bbox,
        inner_bbox if is_pml else box_bbox,
        port_planes,
    )
    sym_axis_i, sym_plane, sheets = _apply_symmetry_cut(
        problem, struct, box_bbox, sheets
    )
    sheet_faces, diel_vols, air_vols, pml_vols = _fragment_and_classify(
        sheets, originals, inner_bbox, is_pml
    )
    faces = _collect_faces(
        problem, sheets, sheet_faces, box_bbox, sym_axis_i, sym_plane
    )
    diel_regions, pml_region, impedance_regions, port_regions, sym_region = (
        _assign_physical_groups(problem, diel_vols, air_vols, pml_vols, faces, port_geo)
    )
    msh_path = _mesh_and_write(
        problem,
        originals,
        struct,
        lambda_min,
        faces,
        workdir,
        verbose,
        diel_regions,
        port_regions,
        diel_vols,
        lambda0_mesh,  # free-space wavelength -> the air size target
    )

    return Mesh(
        msh_path=msh_path,
        dielectric_regions=diel_regions,
        air_region=AIR,
        pec_region=PEC,
        port_regions=port_regions,
        abc_region=ABC,
        dielectric_bboxes=diel_bboxes,
        boundary=problem.boundary,
        bbox=struct,
        box_bbox=box_bbox,
        lambda_min=lambda_min,
        impedance_regions=impedance_regions,
        pml_region=pml_region,
        inner_bbox=inner_bbox,
        pml_thick=pml_thick,
        sym_region=sym_region,
        sym_kind=(problem.symmetry[1] if problem.symmetry else ""),
        sym_axis=(sym_axis_i if sym_axis_i is not None else -1),
        sym_plane=(sym_plane if sym_plane is not None else 0.0),
    )


def _apply_size_field(
    refine_faces: set,
    lc_fine: float,
    lc_coarse: float,
    dist_max: float,
    diel_vols: list | None = None,
    lc_diel: float = 0.0,
) -> None:
    """Set the element size everywhere in the mesh.

    Parameters
    ----------
    refine_faces : set
        Conductor and port faces to refine the mesh towards.
    lc_fine : float
        Element size on those faces, in metres.
    lc_coarse : float
        Element size away from them, in metres.
    dist_max : float
        Distance, in metres, over which ``lc_fine`` grows to ``lc_coarse``.
    diel_vols : list, optional
        Dielectric volume tags to hold at ``lc_diel``. Default ``None``,
        which sizes them like everything else.
    lc_diel : float
        Element size inside those volumes, in metres. Ignored unless smaller
        than ``lc_coarse``. Default ``0.0``.
    """
    # The two requirements are combined so the smaller wins at any point: a
    # distance threshold off the conductors, where the fields vary fastest,
    # and a fixed size in the dielectrics, where the wavelength is shorter
    # than in air by sqrt(eps_r).
    fields = []
    if refine_faces:
        # Distance field: distance from any point to the nearest refine surface.
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "SurfacesList", sorted(refine_faces))
        # Threshold: map that distance to a size (fine near, coarse far).
        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", lc_fine)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", lc_coarse)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", lc_fine)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", dist_max)
        fields.append(thr)
    if diel_vols and 0.0 < lc_diel < lc_coarse:
        const = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumbers(const, "VolumesList", list(diel_vols))
        gmsh.model.mesh.field.setNumber(const, "VIn", lc_diel)
        gmsh.model.mesh.field.setNumber(const, "VOut", lc_coarse)
        fields.append(const)
    if not fields:
        return
    if len(fields) == 1:
        background = fields[0]
    else:
        background = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(background, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(background)
    # Disable Gmsh's other size heuristics so this field alone drives the sizing.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
