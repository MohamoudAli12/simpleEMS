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

# TODO: only box and polygon primitives are working. add support of all primitives.

import numpy as np
from .console import console


def gerber_coord(vertices_xy_pos):
    """
    Convert a 2D or cylindrical coordinate into Gerber RS-274X format.
    Uses global drawingunit and CoordSystem.
    Returns a string like 'X+000001234567Y+000005678901'.
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
    for x, y in zip(x0[1:], x1[1:]):
        file.write(gerber_coord((x, y)) + "D01*\n")

    # Close polygon explicitly
    file.write(gerber_coord((x0[0], x1[0])) + "D01*\n")
    file.write("G37*\n")


# ---------------------------------------------------------------------
# Process CSX properties
# ---------------------------------------------------------------------
def process_primitives(file,  prop_list, options):
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
            elif cls == "CSPrimPoly":
                primitive_polygon(file, prim)
            elif cls == "CSPrimLinPoly":
                primitive_polygon(file, prim)


# ---------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------
def export_gerber(CSX, output_path, options=None):
    """
    Export openEMS CSX geometry to Gerber RS-274X (XY-plane only).
    """
    console.print("-------------------------------------------", style="info")
    console.print("Exporting Geometry to Gerber", style="info")
    console.print("-------------------------------------------", style="info")
    grid = CSX.GetGrid()

    drawing_unit = grid.GetDeltaUnit()
    coord_system = grid.GetMeshType()
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
            process_primitives(file,  metals, options)

        file.write("M02*\n")
