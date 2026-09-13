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
Gerber RS-274X export utilities for CSXCAD geometries.

CSXCAD has no notion of PCB layers, so they are inferred from the Z position
of the metal. Planar metal primitives whose Z ranges overlap form one copper
layer; the highest layer is the top (``F_Cu``), the lowest the bottom
(``B_Cu``), and anything in between an inner layer (``In1_Cu``, ...).
Z-axis cylinders (via barrels) go to an Excellon drill file, material
primitives lying inside a copper layer (antipads) are written as clearances,
and the substrate footprint becomes the board outline. File names, the
``4.5`` millimetre coordinate format and the X2 attributes follow KiCad's
Gerber plot output.

Box, polygon and extruded-polygon primitives are exported as copper;
Z-axis cylinders as drills. Other primitive types are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from CSXCAD import ContinuousStructure
from CSXCAD.CSPrimitives import CSPrimBox, CSPrimLinPoly, CSPrimPolygon

from .console import console

__all__ = [
    "BoardLayout",
    "CopperLayer",
    "Drill",
    "export_gerber",
    "gerber_coord",
    "infer_layers",
    "primitive_box",
    "primitive_polygon",
    "write_copper_layer",
    "write_drill",
    "write_outline",
]

# Tolerance (drawing units) for deciding two Z ranges touch.
_EPS = 1e-6


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
                if prop_cls == "CSPropMetal":
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
# Primitive exporters
# ---------------------------------------------------------------------
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
    start = box.GetStart()
    stop = box.GetStop()

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

    if box.HasTransform():
        transform = box.GetTransform()
        corners = [transform.Transform([x, y, start[2]])[:2] for x, y in corners]

    file.write("G36*\n")
    file.write(gerber_coord(corners[0]) + "D02*\n")
    for corner in (*corners[1:], corners[0]):
        file.write(gerber_coord(corner) + "D01*\n")
    file.write("G37*\n")


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

    x0, x1 = poly.GetCoords()

    if len(x0) < 3:
        print("Skipping polygon: not enough points")
        return
    if poly.HasTransform():
        transform = poly.GetTransform()
        elevation = poly.GetElevation()
        transformed = [
            transform.Transform([x, y, elevation])[:2]
            for x, y in zip(x0, x1, strict=True)
        ]
        x0, x1 = zip(*transformed, strict=True)

    file.write("G36*\n")
    file.write(gerber_coord((x0[0], x1[0])) + "D02*\n")
    for x, y in zip(x0[1:], x1[1:], strict=True):
        file.write(gerber_coord((x, y)) + "D01*\n")
    file.write(gerber_coord((x0[0], x1[0])) + "D01*\n")
    file.write("G37*\n")


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
        if prim.__class__.__name__ == "CSPrimBox":
            primitive_box(file, prim)
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
