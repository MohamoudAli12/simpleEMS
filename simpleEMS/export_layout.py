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
Layout export of CSXCAD geometries: Gerber, DXF and KiCad footprints.

CSXCAD has no notion of PCB layers, so :func:`infer_layers` works them out
from the Z position of the metal. Planar metal primitives whose Z ranges
overlap form one copper layer; the highest layer is the top (``F_Cu``), the
lowest the bottom (``B_Cu``), and anything in between an inner layer
(``In1_Cu``, ...). Z-axis cylinders (via barrels) become drills, material
primitives lying inside a copper layer (antipads) become clearances, and the
substrate footprint becomes the board outline.

Three writers share that layout:

- :func:`export_gerber` writes RS-274X copper, outline and Excellon drill
  files for fabrication. File names, the ``4.5`` millimetre coordinate format
  and the X2 attributes follow KiCad's Gerber plot output.
- :func:`export_dxf` writes one DXF drawing with a layer per copper layer,
  for importing the shape into PCB or mechanical CAD.
- :func:`export_kicad_footprint` writes a KiCad ``.kicad_mod`` footprint
  whose copper joins the feed pad's net, for placing the design on a board.

Box, polygon and extruded-polygon primitives are exported as copper;
Z-axis cylinders as drills; Z-axis cylindrical shells as round clearances.
Other primitive types are skipped, with a warning naming the type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from CSXCAD import ContinuousStructure
from CSXCAD.CSPrimitives import (
    CSPrimBox,
    CSPrimCylindricalShell,
    CSPrimLinPoly,
    CSPrimPolygon,
)
from openEMS.ports import LumpedPort

from .console import console

__all__ = [
    "BoardLayout",
    "CopperLayer",
    "Drill",
    "export_dxf",
    "export_gerber",
    "export_kicad_footprint",
    "gerber_coord",
    "infer_layers",
    "primitive_box",
    "primitive_cylindrical_shell",
    "primitive_outline",
    "primitive_polygon",
    "write_copper_layer",
    "write_drill",
    "write_outline",
]

# Tolerance (drawing units) for deciding two Z ranges touch.
_EPS = 1e-6

# Vertices per circle when a round clearance is written as a region. The
# sagitta of a 60-gon on a 1 mm antipad is under a micron -- far inside any
# fab tolerance, though not below the file's own 1e-5 mm coordinate step.
_CIRCLE_SEGMENTS = 60

# Copper ring (mm) the KiCad footprint puts around each via hole. Drill.diameter
# is the plated conductor, so the pad needs a land of its own around it.
_VIA_ANNULAR_RING = 0.15

# Side (mm) of the feed pad when the port has no width in the XY plane.
_DEFAULT_FEED_PAD = 0.2


# ---------------------------------------------------------------------
# Layout model
# ---------------------------------------------------------------------
@dataclass
class CopperLayer:
    """
    One copper layer inferred from the Z position of the metal.

    Parameters
    ----------
    index : int
        1-based layer number, counted from the top.
    count : int
        Total number of copper layers on the board.
    z_min, z_max : float
        Z range covered by the layer's primitives.
    regions : list of (str, primitive)
        Copper primitives on this layer, tagged with their property name.
    clearances : list of (str, primitive)
        Material primitives cut out of this layer (e.g. via antipads).
    """

    index: int
    count: int
    z_min: float
    z_max: float
    regions: list[tuple[str, Any]] = field(default_factory=list)
    clearances: list[tuple[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        """KiCad layer name: ``F_Cu``, ``In<n>_Cu`` or ``B_Cu``."""
        if self.index == 1:
            return "F_Cu"
        if self.index == self.count:
            return "B_Cu"
        return f"In{self.index - 1}_Cu"

    @property
    def extension(self) -> str:
        """Protel file extension KiCad uses for this layer."""
        if self.index == 1:
            return "gtl"
        if self.index == self.count:
            return "gbl"
        return f"g{self.index}"

    @property
    def file_function(self) -> str:
        """Gerber X2 ``.FileFunction`` attribute value."""
        if self.index == 1:
            side = "Top"
        elif self.index == self.count:
            side = "Bot"
        else:
            side = "Inr"
        return f"Copper,L{self.index},{side}"


@dataclass
class Drill:
    """
    A plated hole taken from a Z-axis cylinder.

    Parameters
    ----------
    x, y : float
        Centre of the hole.
    diameter : float
        Cylinder diameter, i.e. the finished conductor diameter as
        :meth:`simpleEMS.components.GenericStructure.create_via` defines it.
    span : tuple of (int, int)
        First and last copper layer index the barrel connects.
    """

    x: float
    y: float
    diameter: float
    span: tuple[int, int]


@dataclass
class BoardLayout:
    """
    Copper layers, drills and outline inferred from a CSXCAD structure.

    Parameters
    ----------
    layers : list of CopperLayer
        Copper layers ordered top (highest Z) to bottom.
    drills : list of Drill
        Plated holes.
    outline : tuple of float or None
        ``(x_min, y_min, x_max, y_max)`` of the board, ``None`` when the
        structure has nothing to export.
    """

    layers: list[CopperLayer]
    drills: list[Drill]
    outline: tuple[float, float, float, float] | None


# ---------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------
def gerber_coord(vertices_xy_pos: tuple[float, float]) -> str:
    """
    Convert a 2D coordinate into Gerber RS-274X format.

    The files declare ``%FSLAX45Y45*%`` (leading zeros omitted, absolute,
    4 integer and 5 decimal digits, millimetres), so a coordinate is an
    integer count of 1e-5 mm with no padding and no ``+`` sign.

    Parameters
    ----------
    vertices_xy_pos : tuple of (float, float)
        The (x, y) position in millimetres.

    Returns
    -------
    str
        A Gerber coordinate string, e.g. ``'X150000Y-250000'``.
    """
    x_int = int(round(vertices_xy_pos[0] * 1e5))
    y_int = int(round(vertices_xy_pos[1] * 1e5))
    return f"X{x_int}Y{y_int}"


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def _circle_points(
    cx: float, cy: float, radius: float, segments: int = _CIRCLE_SEGMENTS
) -> list[tuple[float, float]]:
    """Vertices of a regular polygon approximating a circle, CCW from +x."""
    step = 2 * math.pi / segments
    return [
        (cx + radius * math.cos(index * step), cy + radius * math.sin(index * step))
        for index in range(segments)
    ]


def _corner_points(prim: object) -> list[list[float]] | None:
    """
    Return the transformed 3D points that bound a primitive.

    Returns ``None`` for primitives with no exportable XY footprint
    (unsupported types, polygons not facing +Z).
    """
    cls = prim.__class__.__name__

    if cls == "CSPrimBox":
        start, stop = prim.GetStart(), prim.GetStop()
        points = [
            [x, y, z]
            for x in (start[0], stop[0])
            for y in (start[1], stop[1])
            for z in (start[2], stop[2])
        ]
    elif cls in ("CSPrimPolygon", "CSPrimLinPoly"):
        if prim.GetNormDir() != 2:
            return None
        x0, x1 = prim.GetCoords()
        if len(x0) == 0:
            return None
        z0 = prim.GetElevation()
        z1 = z0 + prim.GetLength() if cls == "CSPrimLinPoly" else z0
        points = [[x, y, z] for x, y in zip(x0, x1, strict=True) for z in (z0, z1)]
    elif cls == "CSPrimCylinder":
        points = [list(prim.GetStart()), list(prim.GetStop())]
    # A shell subclasses CSPrimCylinder, so the exact-name dispatch needs it
    # named separately -- and unlike a barrel, what lands on the layer is the
    # disc it clears, not the axis it runs along, so it needs a real footprint.
    elif cls == "CSPrimCylindricalShell":
        start, stop = prim.GetStart(), prim.GetStop()
        if abs(stop[0] - start[0]) > _EPS or abs(stop[1] - start[1]) > _EPS:
            return None  # not along Z: no disc to clear
        outer_radius = prim.GetRadius() + prim.GetShellWidth() / 2
        points = [
            [start[0] + sx * outer_radius, start[1] + sy * outer_radius, z]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for z in (start[2], stop[2])
        ]
    else:
        return None

    if prim.HasTransform():
        transform = prim.GetTransform()
        points = [list(transform.Transform(point)) for point in points]
    return points


def _bounds(points: list[list[float]]) -> tuple[list[float], list[float]]:
    """Axis-aligned (min, max) corners of a point list."""
    lo = [min(point[i] for point in points) for i in range(3)]
    hi = [max(point[i] for point in points) for i in range(3)]
    return lo, hi


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[list[float]]:
    """Merge overlapping or touching Z intervals, lowest first."""
    merged: list[list[float]] = []
    for z_min, z_max in sorted(intervals):
        if merged and z_min <= merged[-1][1] + _EPS:
            merged[-1][1] = max(merged[-1][1], z_max)
        else:
            merged.append([z_min, z_max])
    return merged


# ---------------------------------------------------------------------
# Layer inference
# ---------------------------------------------------------------------
def infer_layers(
    CSX: ContinuousStructure, ignore: list[str] | tuple[str, ...] = ()
) -> BoardLayout:
    """
    Infer copper layers, drills and the board outline from Z positions.

    Each metal primitive is classified on its own:

    - a cylinder along Z is a drill;
    - a primitive taller than it is narrow (a wall, a side-facing sheet) or
      a cylinder along another axis is skipped with a warning;
    - everything else is planar copper.

    Planar copper whose Z ranges overlap or touch forms one layer. A
    material primitive lying entirely within a copper layer's Z range is a
    clearance on that layer; the remaining materials define the outline.

    Parameters
    ----------
    CSX : ContinuousStructure
        The CSXCAD geometry.
    ignore : list of str, optional
        Property names to leave out.

    Returns
    -------
    BoardLayout
        The inferred layers (top first), drills and outline.
    """
    planar: list[tuple[float, float, str, Any]] = []
    barrels: list[tuple[float, float, float, float, float]] = []
    materials: list[tuple[list[float], list[float], str, Any]] = []
    copper_boxes: list[tuple[list[float], list[float]]] = []

    for prop in CSX.GetAllProperties():
        prop_cls = prop.__class__.__name__
        if prop_cls not in ("CSPropMetal", "CSPropMaterial"):
            continue
        name = prop.GetName()
        if name in ignore:
            console.print(f"[info]omitting {name}[/info]")
            continue

        for prim in prop.GetAllPrimitives():
            prim_cls = prim.__class__.__name__
            points = _corner_points(prim)
            if points is None:
                # Warn for materials too, not just metal: an antipad the
                # exporter cannot draw is a plane with no clearance in it,
                # which is exactly the kind of silence worth breaking.
                console.print(
                    f"[warning]skipping {prim_cls} in {name}: "
                    "no XY footprint to export[/warning]"
                )
                continue
            lo, hi = _bounds(points)

            if prop_cls == "CSPropMaterial":
                materials.append((lo, hi, name, prim))
                continue

            if prim_cls == "CSPrimCylinder":
                if hi[0] - lo[0] > _EPS or hi[1] - lo[1] > _EPS:
                    console.print(
                        f"[warning]skipping cylinder in {name}: "
                        "not along the Z axis[/warning]"
                    )
                    continue
                barrels.append((lo[0], lo[1], 2 * prim.GetRadius(), lo[2], hi[2]))
                continue

            # Taller than it is narrow: a wall or a side-facing sheet.
            if hi[2] - lo[2] > min(hi[0] - lo[0], hi[1] - lo[1]) + _EPS:
                console.print(
                    f"[warning]skipping {prim_cls} in {name}: "
                    "vertical geometry has no single layer[/warning]"
                )
                continue

            planar.append((lo[2], hi[2], name, prim))
            copper_boxes.append((lo, hi))

    # Highest Z first, so index 1 is the top layer.
    intervals = _merge_intervals([(z_min, z_max) for z_min, z_max, _, _ in planar])
    intervals.reverse()
    layers = [
        CopperLayer(index=i + 1, count=len(intervals), z_min=z_min, z_max=z_max)
        for i, (z_min, z_max) in enumerate(intervals)
    ]

    # Assign in property order so each layer file keeps the CSX ordering.
    for z_min, _z_max, name, prim in planar:
        layer = next(
            layer
            for layer in layers
            if layer.z_min - _EPS <= z_min <= layer.z_max + _EPS
        )
        layer.regions.append((name, prim))

    board_boxes: list[tuple[list[float], list[float]]] = []
    for lo, hi, name, prim in materials:
        for layer in layers:
            if layer.z_min - _EPS <= lo[2] and hi[2] <= layer.z_max + _EPS:
                layer.clearances.append((name, prim))
                break
        else:
            board_boxes.append((lo, hi))

    drills: list[Drill] = []
    for x, y, diameter, z_min, z_max in barrels:
        touched = [
            layer.index
            for layer in layers
            if z_min <= layer.z_max + _EPS and z_max >= layer.z_min - _EPS
        ]
        if not touched:
            console.print(
                f"[warning]skipping via at ({x:g}, {y:g}): "
                "it touches no copper layer[/warning]"
            )
            continue
        drills.append(Drill(x, y, diameter, (min(touched), max(touched))))

    boxes = board_boxes or copper_boxes
    outline = None
    if boxes:
        outline = (
            min(lo[0] for lo, _ in boxes),
            min(lo[1] for lo, _ in boxes),
            max(hi[0] for _, hi in boxes),
            max(hi[1] for _, hi in boxes),
        )

    return BoardLayout(layers=layers, drills=drills, outline=outline)


# ---------------------------------------------------------------------
# Primitive outlines
# ---------------------------------------------------------------------
def primitive_outline(
    prim: CSPrimBox | CSPrimCylindricalShell | CSPrimPolygon | CSPrimLinPoly,
) -> list[tuple[float, float]]:
    """
    Return the closed XY outline of a copper or clearance primitive.

    Every layout writer draws from this outline, so Gerber, DXF and KiCad
    see the same shape. The last vertex does not repeat the first.

    Parameters
    ----------
    prim : CSPrimBox, CSPrimCylindricalShell, CSPrimPolygon or CSPrimLinPoly
        The primitive. A box gives its four corners; a Z-axis cylindrical
        shell gives its outer disc as a 60-sided polygon (see
        :func:`primitive_cylindrical_shell` for why the whole disc); a
        polygon gives its vertices.

    Returns
    -------
    list of (float, float)
        Outline vertices in drawing units, with the primitive's transform
        applied.
    """
    cls = prim.__class__.__name__
    if cls == "CSPrimBox":
        start = prim.GetStart()
        stop = prim.GetStop()
        elevation = start[2]

        # (start.x, stop.y)  ┌───────────────┐ (stop.x, stop.y)
        #                    │               │
        #                    │               │
        # (start.x,start.y)  └───────────────┘ (stop.x,start.y)
        corners = [
            (start[0], start[1]),
            (stop[0], start[1]),
            (stop[0], stop[1]),
            (start[0], stop[1]),
        ]
    elif cls == "CSPrimCylindricalShell":
        start = prim.GetStart()
        elevation = start[2]
        outer_radius = prim.GetRadius() + prim.GetShellWidth() / 2
        corners = _circle_points(start[0], start[1], outer_radius)
    else:
        x_coords, y_coords = prim.GetCoords()
        elevation = prim.GetElevation()
        corners = list(zip(x_coords, y_coords, strict=True))

    if prim.HasTransform():
        transform = prim.GetTransform()
        corners = [
            tuple(transform.Transform([x, y, elevation])[:2]) for x, y in corners
        ]
    return corners


# ---------------------------------------------------------------------
# Gerber primitive exporters
# ---------------------------------------------------------------------
def _write_gerber_region(file: TextIO, corners: list[tuple[float, float]]) -> None:
    """Write a closed outline as one Gerber ``G36``/``G37`` region."""
    file.write("G36*\n")
    file.write(gerber_coord(corners[0]) + "D02*\n")
    for corner in (*corners[1:], corners[0]):
        file.write(gerber_coord(corner) + "D01*\n")
    file.write("G37*\n")


def primitive_box(file: TextIO, box: CSPrimBox) -> None:
    """
    Export a CSXCAD box primitive as a closed Gerber region.

    Parameters
    ----------
    file : file object
        Open file handle for writing Gerber output.
    box : CSPrimBox
        The box primitive object containing start and stop coordinates.

    Returns
    -------
    None
    """
    _write_gerber_region(file, primitive_outline(box))


def primitive_cylindrical_shell(file: TextIO, shell: CSPrimCylindricalShell) -> None:
    """
    Export a Z-axis cylindrical shell as a closed Gerber region.

    The region is the shell's **outer** disc, filled -- not the annulus the
    primitive describes. A shell is how
    :meth:`simpleEMS.components.GenericStructure.create_via` draws an antipad,
    and in the 3D model the hole in it is filled by the via barrel. Gerber has
    no barrel on this layer: the hole is a line in the drill file. Clearing
    only the annulus would leave the copper inside it drawn, as an island the
    plated barrel then bonds to the plane -- shorting the via to the very
    plane the antipad exists to keep it clear of. So the void is the whole
    disc.

    Parameters
    ----------
    file : file object
        Open file handle for writing Gerber output.
    shell : CSPrimCylindricalShell
        The shell primitive, whose axis must run along Z.

    Returns
    -------
    None
    """
    _write_gerber_region(file, primitive_outline(shell))


def primitive_polygon(file: TextIO, poly: CSPrimPolygon | CSPrimLinPoly) -> None:
    """
    Export a CSXCAD polygon primitive as a closed Gerber region.

    Only exports polygons with a +Z normal direction (XY-plane); polygons
    with any other normal direction, or fewer than 3 vertices, are skipped
    with a message printed to stdout.

    Parameters
    ----------
    file : file object
        Open file handle for writing Gerber output.
    poly : CSPrimPolygon or CSPrimLinPoly
        The polygon primitive object containing vertex coordinates.

    Returns
    -------
    None
    """
    if poly.GetNormDir() != 2:
        print("Skipping polygon: normal direction is not +Z")
        return

    corners = primitive_outline(poly)

    if len(corners) < 3:
        print("Skipping polygon: not enough points")
        return

    _write_gerber_region(file, corners)


# ---------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------
def _write_header(file: TextIO, file_function: str) -> None:
    """KiCad-style RS-274X header with X2 attributes (no date: reproducible)."""
    file.write("G04 gerber RS274X-file exported by simpleEMS*\n")
    file.write("%TF.GenerationSoftware,simpleEMS,export_gerber*%\n")
    file.write("%TF.SameCoordinates,Original*%\n")
    file.write(f"%TF.FileFunction,{file_function}*%\n")
    file.write("%TF.FilePolarity,Positive*%\n")
    file.write("%FSLAX45Y45*%\n")
    file.write("%MOMM*%\n")
    file.write("G01*\n")


def _write_regions(file: TextIO, items: list[tuple[str, Any]]) -> None:
    """Write primitives as regions, with a ``%LN`` label per property."""
    current = None
    for name, prim in items:
        if name != current:
            file.write(f"%LN{name}*%\n")
            current = name
        cls = prim.__class__.__name__
        if cls == "CSPrimBox":
            primitive_box(file, prim)
        elif cls == "CSPrimCylindricalShell":
            primitive_cylindrical_shell(file, prim)
        else:
            primitive_polygon(file, prim)


def write_copper_layer(file: TextIO, layer: CopperLayer) -> None:
    """
    Write one copper layer: dark regions, then clearances.

    Parameters
    ----------
    file : file object
        Open file handle for writing Gerber output.
    layer : CopperLayer
        The layer to write.

    Returns
    -------
    None
    """
    _write_header(file, layer.file_function)
    file.write("%LPD*%\n")
    _write_regions(file, layer.regions)
    if layer.clearances:
        # Clear polarity erases the copper already drawn above it.
        file.write("%LPC*%\n")
        _write_regions(file, layer.clearances)
        file.write("%LPD*%\n")
    file.write("M02*\n")


def write_outline(file: TextIO, outline: tuple[float, float, float, float]) -> None:
    """
    Write the board outline (``Edge_Cuts``) as a closed rectangle.

    Parameters
    ----------
    file : file object
        Open file handle for writing Gerber output.
    outline : tuple of float
        ``(x_min, y_min, x_max, y_max)`` of the board.

    Returns
    -------
    None
    """
    x0, y0, x1, y1 = outline
    _write_header(file, "Profile,NP")
    file.write("%ADD10C,0.100000*%\n")
    file.write("D10*\n")
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    file.write(gerber_coord(corners[0]) + "D02*\n")
    for corner in corners[1:]:
        file.write(gerber_coord(corner) + "D01*\n")
    file.write("M02*\n")


def write_drill(file: TextIO, drills: list[Drill], span: tuple[int, int]) -> None:
    """
    Write plated holes as a KiCad-style Excellon drill file.

    Parameters
    ----------
    file : file object
        Open file handle for writing Excellon output.
    drills : list of Drill
        Holes sharing the same layer span.
    span : tuple of (int, int)
        First and last copper layer the holes connect.

    Returns
    -------
    None
    """
    diameters = sorted({round(drill.diameter, 3) for drill in drills})

    file.write("M48\n")
    file.write("; DRILL file exported by simpleEMS\n")
    file.write("; FORMAT={-:-/ absolute / metric / decimal}\n")
    file.write(f"; #@! TF.FileFunction,Plated,{span[0]},{span[1]},PTH\n")
    file.write("FMAT,2\n")
    file.write("METRIC\n")
    for tool, diameter in enumerate(diameters, start=1):
        file.write(f"T{tool}C{diameter:.3f}\n")
    file.write("%\n")
    file.write("G90\n")
    file.write("G05\n")
    for tool, diameter in enumerate(diameters, start=1):
        file.write(f"T{tool}\n")
        for drill in drills:
            if round(drill.diameter, 3) == diameter:
                file.write(f"X{drill.x:.4f}Y{drill.y:.4f}\n")
    file.write("T0\n")
    file.write("M30\n")


# ---------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------
def export_gerber(
    CSX: ContinuousStructure,
    output_path: Path,
    options: dict[str, list[str]],
    prefix: str = "layout",
) -> list[Path]:
    """
    Export CSX geometry to one Gerber file per inferred copper layer.

    Layers are inferred from the Z position of the metal (see
    :func:`infer_layers`). Writes, KiCad style:

    - ``<prefix>-F_Cu.gtl``, ``<prefix>-In<n>_Cu.g<n+1>``,
      ``<prefix>-B_Cu.gbl`` -- one file per copper layer;
    - ``<prefix>-Edge_Cuts.gm1`` -- the board outline;
    - ``<prefix>-PTH.drl`` -- through vias, when there are any; blind or
      buried spans get ``<prefix>-PTH-L<a>-L<b>.drl``.

    Parameters
    ----------
    CSX : ContinuousStructure
        The CSXCAD geometry object containing the simulation structure.
    output_path : Path
        Directory where the files will be saved.
    options : dict
        Dictionary of export options. Supported keys:
        - "ignore" : list of property names to skip during export.
    prefix : str, optional
        File name prefix. Default ``"layout"``.

    Returns
    -------
    list of Path
        The files written.
    """
    console.print("-------------------------------------------", style="info")
    console.print("Exporting Geometry to Gerber", style="info")
    console.print("-------------------------------------------", style="info")

    layout = infer_layers(CSX, options.get("ignore", []))
    written: list[Path] = []

    for layer in layout.layers:
        path = output_path / f"{prefix}-{layer.name}.{layer.extension}"
        with open(path, "w") as file:
            write_copper_layer(file, layer)
        names = ", ".join(dict.fromkeys(name for name, _ in layer.regions))
        console.print(
            f"[info]{layer.name}: z = {layer.z_min:g} .. {layer.z_max:g} "
            f"({names})[/info]"
        )
        written.append(path)

    if layout.outline is not None:
        path = output_path / f"{prefix}-Edge_Cuts.gm1"
        with open(path, "w") as file:
            write_outline(file, layout.outline)
        written.append(path)

    spans: dict[tuple[int, int], list[Drill]] = {}
    for drill in layout.drills:
        spans.setdefault(drill.span, []).append(drill)

    through = (1, len(layout.layers))
    for span, drills in sorted(spans.items()):
        suffix = "PTH" if span == through else f"PTH-L{span[0]}-L{span[1]}"
        path = output_path / f"{prefix}-{suffix}.drl"
        with open(path, "w") as file:
            write_drill(file, drills, span)
        console.print(f"[info]{suffix}: {len(drills)} hole(s)[/info]")
        written.append(path)

    return written


def export_dxf(
    CSX: ContinuousStructure,
    output_path: Path,
    options: dict[str, list[str]],
    filename: str = "layout",
) -> Path:
    """
    Export CSX geometry to a DXF drawing, one DXF layer per copper layer.

    Writes ``<filename>.dxf`` in the DXF R12 format, which every CAD tool
    reads, in millimetres. Layers are inferred from the Z position of the
    metal (see :func:`infer_layers`) and named as in KiCad:

    - ``F_Cu``, ``In<n>_Cu``, ``B_Cu`` -- copper, as closed polylines;
    - ``<layer>_Clearance`` -- antipads cut out of that copper layer. DXF
      cannot subtract one shape from another, so they get a layer of their
      own to remove after import;
    - ``Edge_Cuts`` -- the board outline;
    - ``Drills`` -- one circle per via, at the plated conductor diameter.

    Parameters
    ----------
    CSX : ContinuousStructure
        The CSXCAD geometry object containing the simulation structure.
    output_path : Path
        Directory where the file will be saved.
    options : dict
        Dictionary of export options. Supported keys:
        - "ignore" : list of property names to skip during export.
    filename : str, optional
        Base name of the file (``.dxf`` is added). Default ``"layout"``.

    Returns
    -------
    Path
        The file written.
    """
    console.print("-------------------------------------------", style="info")
    console.print("Exporting Geometry to DXF", style="info")
    console.print("-------------------------------------------", style="info")

    layout = infer_layers(CSX, options.get("ignore", []))

    # Each entity is (DXF layer, outline vertices, (x, y, radius) or None).
    entities: list[tuple[str, list[tuple[float, float]], tuple | None]] = []
    for layer in layout.layers:
        for _name, prim in layer.regions:
            entities.append((layer.name, primitive_outline(prim), None))
        for _name, prim in layer.clearances:
            entities.append((f"{layer.name}_Clearance", primitive_outline(prim), None))
    if layout.outline is not None:
        x_min, y_min, x_max, y_max = layout.outline
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        entities.append(("Edge_Cuts", corners, None))
    for drill in layout.drills:
        entities.append(("Drills", [], (drill.x, drill.y, drill.diameter / 2)))

    layer_names = list(dict.fromkeys(name for name, _, _ in entities))

    # DXF is a list of (group code, value) pairs, one per line each.
    pairs: list[tuple[int, str]] = [
        (0, "SECTION"),
        (2, "HEADER"),
        (9, "$ACADVER"),
        (1, "AC1009"),
        (9, "$INSUNITS"),
        (70, "4"),  # millimetres
        (0, "ENDSEC"),
        (0, "SECTION"),
        (2, "TABLES"),
        (0, "TABLE"),
        (2, "LAYER"),
        (70, str(len(layer_names))),
    ]
    for color, name in enumerate(layer_names, start=1):
        pairs += [
            (0, "LAYER"),
            (2, name),
            (70, "0"),
            (62, str(color)),
            (6, "CONTINUOUS"),
        ]
    pairs += [(0, "ENDTAB"), (0, "ENDSEC"), (0, "SECTION"), (2, "ENTITIES")]

    for layer_name, corners, circle in entities:
        if circle is not None:
            x, y, radius = circle
            pairs += [
                (0, "CIRCLE"),
                (8, layer_name),
                (10, f"{x:.6f}"),
                (20, f"{y:.6f}"),
                (30, "0.0"),
                (40, f"{radius:.6f}"),
            ]
            continue
        if len(corners) < 3:
            continue
        # A closed (flag 1) polyline: its vertices follow, then SEQEND.
        pairs += [(0, "POLYLINE"), (8, layer_name), (66, "1"), (70, "1")]
        pairs += [(10, "0.0"), (20, "0.0"), (30, "0.0")]
        for x, y in corners:
            pairs += [
                (0, "VERTEX"),
                (8, layer_name),
                (10, f"{x:.6f}"),
                (20, f"{y:.6f}"),
                (30, "0.0"),
            ]
        pairs += [(0, "SEQEND"), (8, layer_name)]

    pairs += [(0, "ENDSEC"), (0, "EOF")]

    path = output_path / f"{filename}.dxf"
    path.write_text(
        "".join(f"{code}\n{value}\n" for code, value in pairs), newline="\n"
    )
    console.print(f"[info]DXF layers: {', '.join(layer_names)}[/info]")
    return path


def export_kicad_footprint(
    CSX: ContinuousStructure,
    port: LumpedPort,
    output_path: Path,
    options: dict[str, list[str]],
    name: str = "antenna",
    ground: list[str] | tuple[str, ...] = ("ground",),
) -> Path:
    """
    Export CSX geometry as a KiCad footprint (``.kicad_mod``).

    The footprint's origin sits at the feed point, the centre of ``port`` in
    the XY plane, so it snaps onto the end of the feed trace. Copper turns
    into custom-shape pads, so it belongs to a net and DRC treats it as
    connected:

    - pad ``1`` (the feed) holds every copper region not named in
      ``ground``. On the feed's own layer it is anchored at the origin with
      a square the width of the port;
    - pad ``2`` (ground) holds the copper named in ``ground`` and a plated
      through hole for each via, with a 0.15 mm annular ring.

    The board outline goes to ``F.Fab`` and ``F.CrtYd``. KiCad's Y axis
    points down, so Y is mirrored relative to the simulation.

    Parameters
    ----------
    CSX : ContinuousStructure
        The CSXCAD geometry object containing the simulation structure.
    port : openEMS.ports.LumpedPort
        The excitation port the structure's build method returned. Its
        ``start`` and ``stop`` set the feed point and the feed pad size.
    output_path : Path
        Directory where the file will be saved.
    options : dict
        Dictionary of export options. Supported keys:
        - "ignore" : list of property names to skip during export.
    name : str, optional
        Footprint name, also the file name. Default ``"antenna"``.
    ground : list of str, optional
        Property names of the copper that belongs to ground. Default
        ``("ground",)``, the name every simpleEMS structure uses.

    Returns
    -------
    Path
        The file written.

    Notes
    -----
    A pad cannot cut a hole in another pad, so antipads are not written; a
    warning names any that were dropped. Every via becomes a through hole
    on the ground pad, whatever layers it spans in the simulation. Each pad
    other than the feed is anchored on a corner of its first region with a
    10 um square, which KiCad requires and no fab can resolve.
    """
    console.print("-------------------------------------------", style="info")
    console.print("Exporting Geometry to KiCad footprint", style="info")
    console.print("-------------------------------------------", style="info")

    layout = infer_layers(CSX, options.get("ignore", []))

    port_start = np.asarray(port.start, dtype=float)
    port_stop = np.asarray(port.stop, dtype=float)
    feed_x, feed_y = (port_start[:2] + port_stop[:2]) / 2
    port_spans = [
        span for span in np.abs(port_stop[:2] - port_start[:2]) if span > _EPS
    ]
    feed_pad_size = min(port_spans) if port_spans else _DEFAULT_FEED_PAD
    anchor_size = 0.01

    def to_kicad(point: tuple[float, float]) -> tuple[float, float]:
        """Simulation XY to footprint XY: origin at the feed, Y down."""
        # Adding 0.0 turns a -0.0 left by the rounding into 0.0.
        return (
            round(point[0] - feed_x, 6) + 0.0,
            round(feed_y - point[1], 6) + 0.0,
        )

    lines = [
        f'(footprint "{name}"',
        "  (version 20221018)",
        '  (generator "simpleEMS")',
        '  (layer "F.Cu")',
        f'  (descr "{name}, exported by simpleEMS")',
        "  (attr smd)",
    ]

    if layout.outline is not None:
        x_min, y_min, x_max, y_max = layout.outline
        left, top = to_kicad((x_min, y_max))
        right, bottom = to_kicad((x_max, y_min))
    else:
        left = top = -feed_pad_size
        right = bottom = feed_pad_size
    lines += [
        f'  (fp_text reference "REF**" (at 0 {top - 1:.6f}) (layer "F.SilkS")',
        "    (effects (font (size 1 1) (thickness 0.15))))",
        f'  (fp_text value "{name}" (at 0 {bottom + 1:.6f}) (layer "F.Fab")',
        "    (effects (font (size 1 1) (thickness 0.15))))",
    ]
    for fab_layer, width in (("F.Fab", 0.1), ("F.CrtYd", 0.05)):
        lines.append(
            f"  (fp_rect (start {left:.6f} {top:.6f}) (end {right:.6f} {bottom:.6f})"
            f' (stroke (width {width}) (type solid)) (fill none) (layer "{fab_layer}"))'
        )

    dropped_clearances = []
    for layer in layout.layers:
        kicad_layer = layer.name.replace("_", ".")
        dropped_clearances += [clearance for clearance, _ in layer.clearances]
        for pad_number in ("1", "2"):
            outlines = [
                [to_kicad(corner) for corner in primitive_outline(prim)]
                for region_name, prim in layer.regions
                if (region_name in ground) == (pad_number == "2")
            ]
            outlines = [outline for outline in outlines if len(outline) >= 3]
            if not outlines:
                continue

            if pad_number == "1" and layer.index == 1:
                pad_x, pad_y, size = 0.0, 0.0, feed_pad_size
            else:
                (pad_x, pad_y), size = outlines[0][0], anchor_size

            lines += [
                f'  (pad "{pad_number}" smd custom (at {pad_x:.6f} {pad_y:.6f})'
                f' (size {size:.6f} {size:.6f}) (layers "{kicad_layer}")',
                "    (options (clearance outline) (anchor rect))",
                "    (primitives",
            ]
            for outline in outlines:
                points = " ".join(
                    f"(xy {x - pad_x:.6f} {y - pad_y:.6f})" for x, y in outline
                )
                lines.append(f"      (gr_poly (pts {points}) (width 0) (fill yes))")
            lines += ["    )", "  )"]

    for drill in layout.drills:
        x, y = to_kicad((drill.x, drill.y))
        land = drill.diameter + 2 * _VIA_ANNULAR_RING
        lines.append(
            f'  (pad "2" thru_hole circle (at {x:.6f} {y:.6f})'
            f" (size {land:.6f} {land:.6f}) (drill {drill.diameter:.6f})"
            ' (layers "*.Cu" "*.Mask"))'
        )

    lines.append(")")

    if dropped_clearances:
        console.print(
            "[warning]antipads are not written to the footprint: "
            f"{', '.join(dict.fromkeys(dropped_clearances))}[/warning]"
        )

    path = output_path / f"{name}.kicad_mod"
    path.write_text("\n".join(lines) + "\n", newline="\n")
    return path
