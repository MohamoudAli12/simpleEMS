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
STEP AP242 export utilities for CSXCAD geometries using CadQuery.

Converts openEMS CSXCAD structures into STEP format for CAD interoperability.
Supports box, linear polygon, and cylinder (solid or shell) primitives; any
other primitive type is reported and skipped.

CSXCAD settles overlapping primitives by priority, which a STEP file has no
way to express, so a dielectric that outranks the metal it sits in -- a via
antipad -- is cut out of that conductor for real. See
:func:`_dielectric_cutters`.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path


import cadquery as cq

from CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS

from .console import console


def _normal_dir(
    normdir: int, elevation: float
) -> tuple[str, tuple[float, float, float]]:
    mapping = {
        0: ("YZ", (elevation, 0.0, 0.0)),
        1: ("XZ", (0.0, elevation, 0.0)),
        2: ("XY", (0.0, 0.0, elevation)),
    }
    return mapping[normdir]


def _make_box(
    start: tuple[float, float, float], stop: tuple[float, float, float]
) -> cq.Workplane:
    x1, y1, z1 = start
    x2, y2, z2 = stop
    w = max(abs(x2 - x1), 1e-3)
    h = max(abs(y2 - y1), 1e-3)
    d = max(abs(z2 - z1), 1e-3)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cz = (z1 + z2) / 2
    return cq.Workplane("XY").box(w, h, d).translate((cx, cy, cz))


def _make_linpoly(
    x_coords: list[float],
    y_coords: list[float],
    elevation: float,
    normdir: int,
    length: float,
) -> cq.Workplane:
    pts = [(x, y) for x, y in zip(x_coords, y_coords, strict=True)]
    wp_str, offset = _normal_dir(normdir, elevation)
    return (
        cq.Workplane(wp_str)
        .polyline(pts)
        .close()
        .extrude(max(length, 1e-3))
        .translate(offset)
    )


def _make_cylinder(
    start: tuple[float, float, float],
    stop: tuple[float, float, float],
    radius: float,
    inner_radius: float = 0.0,
) -> cq.Workplane:
    """Build the cylinder running from ``start`` to ``stop``.

    This is what a via barrel is: ``GenericStructure.create_via`` adds one
    ``AddCylinder`` per via, so without this the whole via property exports
    empty and the FEM mesh -- which only ever sees the STEP file -- has no
    conductor through the board.

    ``start``/``stop`` are the centres of the two end faces, so the axis is
    taken from them directly rather than assumed to be z; ``makeCylinder``
    accepts an arbitrary direction, which keeps a tilted or lying-down
    cylinder correct without a separate rotation.  A non-zero
    ``inner_radius`` cuts a coaxial bore, which is how a
    ``CSPrimCylindricalShell`` (a plated hole rather than a solid barrel) is
    built.  Degenerate inputs are floored at 1e-3 the way ``_make_box`` and
    ``_make_linpoly`` floor theirs, since OCC cannot export a solid with no
    volume.
    """
    axis = [b - a for a, b in zip(start, stop, strict=True)]
    length = math.sqrt(sum(c * c for c in axis))
    if length < 1e-9:  # no axis to point along; pick one rather than fail
        axis, length = [0.0, 0.0, 1.0], 1e-3
    length = max(length, 1e-3)

    direction = cq.Vector(*axis)
    origin = cq.Vector(*start)
    solid = cq.Solid.makeCylinder(max(radius, 1e-3), length, origin, direction)
    if inner_radius > 0:
        solid = solid.cut(
            cq.Solid.makeCylinder(inner_radius, length, origin, direction)
        )
    return cq.Workplane(obj=solid)


def _apply_transform(solid: cq.Workplane, prim: object) -> cq.Workplane:
    """Apply prim's CSXCAD transform, if any, to solid.

    CSXCAD builds primitives in a local frame and applies AddTransform
    (translate/rotate/scale) on top via CSTransform; export_cad must do
    the same or a transformed primitive (e.g. a rotated stub or taper
    from components.py) is exported, meshed, and plotted untransformed.
    transformGeometry (not transformShape) is used because a CSTransform
    matrix may carry scale, which transformShape's rigid gp_Trsf rejects.
    """
    if not prim.HasTransform():
        return solid
    matrix = cq.Matrix(prim.GetTransform().GetMatrix().tolist())
    return cq.Workplane(obj=solid.val().transformGeometry(matrix))


def _process_property(
    prop: object,
) -> list[tuple[cq.Workplane, object]]:
    """Build every primitive of ``prop`` as ``(solid, primitive)`` pairs.

    The primitive rides along with its solid because the caller needs its
    priority and its type to resolve overlaps (see
    :func:`_dielectric_cutters`), and a skipped primitive would otherwise
    break any attempt to line the two lists up again afterwards.
    """
    name = prop.GetName()

    primitives = prop.GetAllPrimitives()
    if not primitives:
        return []

    solids: list[tuple[cq.Workplane, object]] = []
    for prim in primitives:
        cls = prim.__class__.__name__

        if cls == "CSPrimBox":
            start = prim.GetStart()
            stop = prim.GetStop()
            solids.append((_apply_transform(_make_box(start, stop), prim), prim))
            console.print(f"[info]  box: {name} ({start} → {stop})[/info]")

        elif cls == "CSPrimLinPoly":
            x_coords, y_coords = prim.GetCoords()
            elevation = prim.GetElevation()
            normdir = prim.GetNormDir()
            length = prim.GetLength()
            solid = _make_linpoly(x_coords, y_coords, elevation, normdir, length)
            solids.append((_apply_transform(solid, prim), prim))
            nv = len(x_coords)
            console.print(
                f"[info]  polygon: {name} ({nv} verts, elev={elevation}, "
                f"norm={normdir})[/info]"
            )

        # CSPrimCylindricalShell subclasses CSPrimCylinder, so it has to be
        # named here as well -- the dispatch is on the exact class name.
        elif cls in ("CSPrimCylinder", "CSPrimCylindricalShell"):
            start = prim.GetStart()
            stop = prim.GetStop()
            radius = prim.GetRadius()
            if cls == "CSPrimCylindricalShell":
                # CSXCAD gives a shell its mid-wall radius plus a wall width.
                half = prim.GetShellWidth() / 2
                solid = _make_cylinder(
                    start, stop, radius + half, max(radius - half, 0.0)
                )
            else:
                solid = _make_cylinder(start, stop, radius)
            solids.append((_apply_transform(solid, prim), prim))
            console.print(
                f"[info]  cylinder: {name} ({start} → {stop}, r={radius})[/info]"
            )

        else:
            # Ten of CSXCAD's thirteen primitive types still have no branch
            # here. Say so: a silently dropped primitive is how the missing
            # cylinder support went unnoticed, since the property simply
            # exported empty.
            console.print(
                f"[warning]  skipped: {name} ({cls} is not exported to CAD)[/warning]"
            )

    return solids


def _bounds_overlap(first: cq.BoundBox, second: cq.BoundBox) -> bool:
    """Return True if two CadQuery bounding boxes intersect.

    A cheap gate in front of the boolean: OCC cuts are expensive, and on a
    real board almost every metal/dielectric pair is nowhere near touching.
    """
    return (
        first.xmin <= second.xmax
        and second.xmin <= first.xmax
        and first.ymin <= second.ymax
        and second.ymin <= first.ymax
        and first.zmin <= second.zmax
        and second.zmin <= first.zmax
    )


def _cut_shape(solid: cq.Workplane, prim: object) -> cq.Workplane:
    """The shape a dielectric primitive removes from the metal it outranks.

    For everything but a shell that is the primitive's own solid. A shell is
    how :meth:`simpleEMS.components.GenericStructure.create_via` draws an
    antipad, and there the void is the **whole outer disc**, not the ring: the
    ring's bore is filled by the via barrel, so cutting only the ring would
    leave a collar of plane copper hugging the barrel -- which is the short
    the antipad exists to prevent. The Gerber exporter clears the same disc
    for the same reason (see
    :func:`simpleEMS.export_gerber.primitive_cylindrical_shell`).
    """
    if prim.__class__.__name__ != "CSPrimCylindricalShell":
        return solid
    outer_radius = prim.GetRadius() + prim.GetShellWidth() / 2
    filled = _make_cylinder(prim.GetStart(), prim.GetStop(), outer_radius)
    return _apply_transform(filled, prim)


def _dielectric_cutters(
    built: list[tuple[object, list[tuple[cq.Workplane, object]]]],
) -> list[tuple[int, cq.Workplane]]:
    """Collect every dielectric solid that can outrank a conductor.

    CSXCAD settles two overlapping primitives by priority: the higher one
    owns the shared volume. STEP carries no priorities, so an antipad -- a
    dielectric deliberately placed inside a plane -- would export as a solid
    intersecting that plane, and the plane would still be whole. Meshed, the
    copper runs right up to the barrel and the via shorts to the very plane
    the antipad was there to clear it of.

    Only materials are collected, and only metal is cut by them below. A
    conductor overlapping a dielectric is the ordinary case -- a trace on a
    substrate, a barrel through a core -- and is left alone.
    """
    cutters: list[tuple[int, cq.Workplane]] = []
    for prop, solids in built:
        if prop.__class__.__name__ != "CSPropMaterial":
            continue
        for solid, prim in solids:
            cutters.append((prim.GetPriority(), _cut_shape(solid, prim)))
    return cutters


def _apply_cutters(
    solid: cq.Workplane,
    prim: object,
    cutters: list[tuple[int, cq.Workplane]],
    name: str,
) -> cq.Workplane | None:
    """Remove every higher-priority dielectric that overlaps ``solid``.

    Returns ``None`` if nothing survives, which means a clearance swallowed
    the conductor whole -- worth saying out loud rather than exporting an
    empty part.
    """
    priority = prim.GetPriority()
    for cutter_priority, cutter in cutters:
        if cutter_priority <= priority:
            continue
        if not _bounds_overlap(solid.val().BoundingBox(), cutter.val().BoundingBox()):
            continue
        solid = solid.cut(cutter)
        if not solid.solids().vals():
            console.print(
                f"[warning]  {name}: a clearance removed the whole solid[/warning]"
            )
            return None
    return solid


def _unique_label(name: str, used: set[str]) -> str:
    """Return ``name``, or ``name_1``/``name_2``/... if already taken.

    CSXCAD allows two properties to share a name -- ``AddMetal("via")`` builds
    a new property every call rather than returning the existing one -- but
    CadQuery rejects a duplicate assembly part label, so a structure with two
    vias would otherwise abort the export. Collisions are resolved here so the
    common case (one property per name) keeps its plain name.
    """
    if name not in used:
        used.add(name)
        return name
    i = 1
    while f"{name}_{i}" in used:
        i += 1
    used.add(f"{name}_{i}")
    return f"{name}_{i}"


def _build_assembly(CSX: ContinuousStructure) -> tuple[cq.Assembly, int]:
    """Collect every physical property of ``CSX`` into a coloured assembly.

    Shared by the STEP and STL writers, which differ only in the file they
    save this assembly to.

    Parameters
    ----------
    CSX : ContinuousStructure
        The CSXCAD geometry object containing the simulation structure.

    Returns
    -------
    tuple[cq.Assembly, int]
        The assembly and the number of parts added to it.
    """
    physical_types = {"CSPropMetal", "CSPropMaterial", "CSPropLumpedElement"}
    assy = cq.Assembly()
    used: set[str] = set()
    part_count = 0

    # Build everything first: a conductor cannot be cut by a clearance that
    # has not been built yet, and CSXCAD gives no guaranteed property order.
    built: list[tuple[object, list[tuple[cq.Workplane, object]]]] = []
    for prop in CSX.GetAllProperties():
        if prop.__class__.__name__ not in physical_types:
            continue
        console.print(f"[info]processing {prop.GetName()}[/info]")
        built.append((prop, _process_property(prop)))

    cutters = _dielectric_cutters(built)

    for prop, solids in built:
        if not solids:
            continue

        name = prop.GetName()
        is_metal = prop.__class__.__name__ == "CSPropMetal"
        r, g, b, a = prop.GetFillColor()
        color = cq.Color(r / 255, g / 255, b / 255, min(a / 255, 1.0))

        for solid, prim in solids:
            if is_metal and cutters:
                solid = _apply_cutters(solid, prim, cutters, name)
                if solid is None:
                    continue
            label = _unique_label(name, used)
            assy.add(solid, name=label, color=color)
            part_count += 1

    return assy, part_count


def _write_assembly(CSX: ContinuousStructure, filename: Path, export_type: str) -> None:
    """Build ``CSX``'s assembly and save it as ``export_type`` to ``filename``."""
    console.print("-------------------------------------------", style="info")
    console.print(f"Exporting Geometry to {export_type}", style="info")
    console.print("-------------------------------------------", style="info")

    assy, part_count = _build_assembly(CSX)

    if part_count == 0:
        console.print("[warning]No physical geometry found to export[/warning]")
        return

    console.print(f"[info]Writing {part_count} part(s) to {export_type}...[/info]")
    assy.save(str(filename), exportType=export_type)
    console.print(f"[success]{export_type} file written to {filename}[/success]")


def export_step(
    CSX: ContinuousStructure,
    output_path: Path,
) -> None:
    """
    Export CSXCAD geometry to a coloured, multi-layer STEP AP242 file.

    Extracts all Material and Metal properties from the CSXCAD
    structure and writes them as separate coloured bodies in a
    single ``.step`` file.

    Parameters
    ----------
    CSX : ContinuousStructure
        The CSXCAD geometry object containing the simulation structure.
    output_path : Path
        Directory where the STEP file (``structure.step``) will be saved.
    """
    _write_assembly(CSX, output_path / "structure.step", "STEP")


def export_stl(
    CSX: ContinuousStructure,
    output_path: Path,
) -> None:
    """
    Export CSXCAD geometry to a coloured, multi-layer STL file.

    Extracts all Material and Metal properties from the CSXCAD
    structure and writes them as separate coloured bodies in a
    single ``.stl`` file.

    Parameters
    ----------
    CSX : ContinuousStructure
        The CSXCAD geometry object containing the simulation structure.
    output_path : Path
        Directory where the STL file (``structure.stl``) will be saved.
    """
    _write_assembly(CSX, output_path / "structure.stl", "STL")


def export_csxcad_xml_to_step(
    structure_xml_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Load a ``structure.xml`` and export it to a STEP file.

    Parameters
    ----------
    structure_xml_path : str | Path
        Path to the ``structure.xml`` file exported by CSXCAD / openEMS.
    output_path : str | Path
        Directory where the STEP file will be saved. Created if it does
        not already exist.
    """
    console.print(f"[info]Loading {structure_xml_path}...[/info]")

    FDTD = openEMS()
    FDTD.ReadFromXML(str(structure_xml_path))
    _CSX = FDTD.GetCSX()

    with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
        _CSX.Write2XML(tmp.name)
        CSX = ContinuousStructure()
        CSX.ReadFromXML(tmp.name)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    export_step(CSX, output_path)
