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

Provides functions to convert openEMS CSXCAD structures into Gerber
format for PCB fabrication. Currently supports box and polygon
primitives in the XY-plane.
"""

# TODO: only box and polygon primitives are working. add support of all primitives.

from .console import console


def gerber_coord(vertices_xy_pos):
    """
    Convert a 2D coordinate into Gerber RS-274X format.

    Parameters
    ----------
    vertices_xy_pos : tuple of (float, float)
        The (x, y) position to convert.

    Returns
    -------
    str
        A Gerber coordinate string in the format
        'X+000001234567Y+000005678901'.
    """
    x = vertices_xy_pos[0]
    y = vertices_xy_pos[1]

    # -------------------------------------------------------
    # Convert to integer Gerber units
    # -------------------------------------------------------
    x_int = int(round(x * 1e6))
    y_int = int(round(y * 1e6))

    # -------------------------------------------------------
    # Format with sign and zero-padding: total width 13
    # -------------------------------------------------------
    x_str = f"{x_int:+013d}"
    y_str = f"{y_int:+013d}"

    # -------------------------------------------------------
    # Return Gerber coordinate string
    # -------------------------------------------------------
    return f"X{x_str}Y{y_str}"


# ---------------------------------------------------------------------
# Primitive exporters
# ---------------------------------------------------------------------
def primitive_box(file, box):
    """
    Export a CSXCAD box primitive to Gerber RS-274X format.
    Writes a rectangular aperture as a closed polygon contour
    to the Gerber output file.
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

    file.write("G36*\n")
    file.write(gerber_coord(start) + "D02*\n")
    file.write(gerber_coord([stop[0], start[1]]) + "D01*\n")
    file.write(gerber_coord(stop) + "D01*\n")
    file.write(gerber_coord([start[0], stop[1]]) + "D01*\n")
    file.write(gerber_coord(start) + "D01*\n")
    file.write("G37*\n")


def primitive_polygon(file, poly):
    """
    Export a CSXCAD polygon primitive to Gerber RS-274X format.
    Only exports polygons with a +Z normal direction (XY-plane).
    Writes the polygon vertices as a closed contour in Gerber format.
    Parameters
    ----------
    file : file object
        Open file handle for writing Gerber output.
    poly : CSPrimPoly or CSPrimLinPoly
        The polygon primitive object containing vertex coordinates.
    Returns
    -------
    None
    """
    # Only export XY-plane polygons (+Z normal)
    if poly.GetNormDir() != 2:
        print("Skipping polygon: normal direction is not +Z")
        return

    x0, x1 = poly.GetCoords()

    if len(x0) < 3:
        print("Skipping polygon: not enough points")
        return

    file.write("G36*\n")

    # Move to first vertex
    file.write(gerber_coord((x0[0], x1[0])) + "D02*\n")

    # Draw edges
    for x, y in zip(x0[1:], x1[1:], strict=True):
        file.write(gerber_coord((x, y)) + "D01*\n")

    # Close polygon explicitly
    file.write(gerber_coord((x0[0], x1[0])) + "D01*\n")
    file.write("G37*\n")


# ---------------------------------------------------------------------
# Process CSX properties
# ---------------------------------------------------------------------
def process_primitives(file, prop_list, options):
    """
    Process and export CSXCAD property primitives to Gerber format.
    This function iterates through a list of CSXCAD properties, filters
    out ignored ones based on options, and exports supported primitives
    (boxes and polygons) to a Gerber file.
    Parameters
    ----------
    file : file object
        Open file handle for writing Gerber output.
    prop_list : list
        List of CSXCAD property objects to process.
    options : dict
        Dictionary of export options. Supported keys:
        - "ignore" : list of property names to skip.
    Returns
    -------
    None
    """
    ignore = options.get("ignore", [])

    for prop in prop_list:
        name = prop.GetName()
        if name in ignore:
            console.print(f"[info]omitting {name}[/info]")
            continue

        # Material filtering: skip if no Kappa
        # if prop_type == "Material":  # and not hasattr(prop, "Kappa"):
        #     print(f"omitting {name} ")
        #     continue

        console.print(f"[info]processing {name} [/info]")
        file.write(f"%LN{name}*%\n")

        # primitives
        primitives = prop.GetAllPrimitives()
        if not primitives:
            console.print("  no primitives found")
            break

        for prim in primitives:
            cls = prim.__class__.__name__

            if cls == "CSPrimBox":
                primitive_box(file, prim)
            elif cls == "CSPrimPoly" or cls == "CSPrimLinPoly":
                primitive_polygon(file, prim)


# ---------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------
def export_gerber(CSX, output_path, options=None):
    """
    Export openEMS CSX geometry to Gerber RS-274X format (XY-plane only).
    Extracts all metal properties from the CSXCAD structure and exports
    their box and polygon primitives to a Gerber layout file.
    Parameters
    ----------
    CSX : ContinuousStructure
        The CSXCAD geometry object containing the simulation structure.
    output_path : Path
        Directory where the Gerber file will be saved.
    options : dict, optional
        Dictionary of export options. Supported keys:
        - "ignore" : list of property names to skip during export.
        Default is None.
    Returns
    -------
    None
    """
    console.print("-------------------------------------------", style="info")
    console.print("Exporting Geometry to Gerber", style="info")
    console.print("-------------------------------------------", style="info")
    # grid = CSX.GetGrid()

    # drawing_unit = grid.GetDeltaUnit()
    # coord_system = grid.GetMeshType()
    filename = output_path / "gerber_layout.gbr"

    with open(filename, "w") as file:
        # Gerber header
        file.write("G04 gerber RS274X-file exported by simpleEMS*\n")
        file.write("G04 Author : Mohamoud Ali*\n")
        file.write("%FSLAX66Y66*%\n")
        file.write("%MOMM*%\n")
        file.write("%INsimpleEMS export*%\n")
        file.write("%ADD10C,0.00100*%\n")

        all_props = CSX.GetAllProperties()

        metals = []

        for prob in all_props:
            cls = prob.__class__.__name__

            if cls == "CSPropMetal":
                metals.append(prob)

        if metals:
            process_primitives(file, metals, options)

        file.write("M02*\n")
