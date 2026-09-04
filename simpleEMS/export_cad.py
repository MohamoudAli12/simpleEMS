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
Supports box and linear polygon primitives.
"""

from __future__ import annotations

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
) -> list[cq.Workplane]:
    name = prop.GetName()

    primitives = prop.GetAllPrimitives()
    if not primitives:
        return []

    solids: list[cq.Workplane] = []
    for prim in primitives:
        cls = prim.__class__.__name__

        if cls == "CSPrimBox":
            start = prim.GetStart()
            stop = prim.GetStop()
            solids.append(_apply_transform(_make_box(start, stop), prim))
            console.print(f"[info]  box: {name} ({start} → {stop})[/info]")

        elif cls == "CSPrimLinPoly":
            x_coords, y_coords = prim.GetCoords()
            elevation = prim.GetElevation()
            normdir = prim.GetNormDir()
            length = prim.GetLength()
            solid = _make_linpoly(x_coords, y_coords, elevation, normdir, length)
            solids.append(_apply_transform(solid, prim))
            nv = len(x_coords)
            console.print(
                f"[info]  polygon: {name} ({nv} verts, elev={elevation}, "
                f"norm={normdir})[/info]"
            )

    return solids


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
    console.print("-------------------------------------------", style="info")
    console.print("Exporting Geometry to STEP", style="info")
    console.print("-------------------------------------------", style="info")

    filename = output_path / "structure.step"

    all_props = CSX.GetAllProperties()

    physical_types = {"CSPropMetal", "CSPropMaterial", "CSPropLumpedElement"}
    assy = cq.Assembly()
    used: set[str] = set()
    part_count = 0

    for prop in all_props:
        cls = prop.__class__.__name__
        if cls not in physical_types:
            continue

        name = prop.GetName()
        console.print(f"[info]processing {name}[/info]")

        solids = _process_property(prop)
        if not solids:
            continue

        r, g, b, a = prop.GetFillColor()
        color = cq.Color(r / 255, g / 255, b / 255, min(a / 255, 1.0))

        for solid in solids:
            label = _unique_label(name, used)
            assy.add(solid, name=label, color=color)
            part_count += 1

    if part_count == 0:
        console.print("[warning]No physical geometry found to export[/warning]")
        return

    console.print(f"[info]Writing {part_count} part(s) to STEP...[/info]")
    assy.save(str(filename), exportType="STEP")
    console.print(f"[success]STEP file written to {filename}[/success]")


def export_stl(
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
    console.print("-------------------------------------------", style="info")
    console.print("Exporting Geometry to STL", style="info")
    console.print("-------------------------------------------", style="info")

    filename = output_path / "structure.stl"

    all_props = CSX.GetAllProperties()

    physical_types = {"CSPropMetal", "CSPropMaterial", "CSPropLumpedElement"}
    assy = cq.Assembly()
    used: set[str] = set()
    part_count = 0

    for prop in all_props:
        cls = prop.__class__.__name__
        if cls not in physical_types:
            continue

        name = prop.GetName()
        console.print(f"[info]processing {name}[/info]")

        solids = _process_property(prop)
        if not solids:
            continue

        r, g, b, a = prop.GetFillColor()
        color = cq.Color(r / 255, g / 255, b / 255, min(a / 255, 1.0))

        for solid in solids:
            label = _unique_label(name, used)
            assy.add(solid, name=label, color=color)
            part_count += 1

    if part_count == 0:
        console.print("[warning]No physical geometry found to export[/warning]")
        return

    console.print(f"[info]Writing {part_count} part(s) to STL...[/info]")
    assy.save(str(filename), exportType="STL")
    console.print(f"[success]STL file written to {filename}[/success]")


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
