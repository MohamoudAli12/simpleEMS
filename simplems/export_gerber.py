import math

drawing_unit = None
coord_system = None


def gerber_coord(v, prim_type=None):
    """
    Convert a 2D or cylindrical coordinate into Gerber RS-274X format.
    Uses global drawingunit and CoordSystem.
    Returns a string like 'X+000001234567Y+000005678901'.
    """
    global drawing_unit, coord_system

    # -------------------------------------------------------
    # Convert coordinates depending on coordinate system
    # -------------------------------------------------------
    if coord_system == 1 and prim_type == "CSPrimCylinder":  # cylindrical
        r, a = v[0], v[1]
        x = r * math.cos(a)
        y = r * math.sin(a)
    else:  # Cartesian
        x = v[0]
        y = v[1]

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
def primitive_box(
    file,
    box,
):
    start = box.GetStart()
    stop = box.GetStop()

    file.write("G36*\n")
    file.write(gerber_coord(start) + "D02*\n")
    file.write(gerber_coord([stop[0], start[1]]) + "D01*\n")
    file.write(
        gerber_coord(
            stop,
        )
        + "D01*\n"
    )
    file.write(
        gerber_coord(
            [start[0], stop[1]],
        )
        + "D01*\n"
    )
    file.write(
        gerber_coord(
            start,
        )
        + "D01*\n"
    )
    file.write("G37*\n")


def primitive_cylinder(
    file,
    cyl,
):
    start = cyl.GetStart()
    stop = cyl.GetStop()
    radius = cyl.GetRadius()

    # Only circular projection in XY-plane
    if start == stop:
        file.write(f"%ADD10C,{radius * 2 * 1e3:.6f}*%\n")
        file.write("G54D10*\n")
        file.write(gerber_coord(start, prim_type="CSPrimCylinder") + "D03*\n")
    else:
        print("Skipping cylinder: projection is not circular")


def primitive_polygon(
    file,
    poly,
):
    # Only export XY-plane polygons (+Z normal)
    if poly.GetNormDir() != 2:
        print("Skipping polygon: normal direction is not +Z")
        return

    xs, ys = poly.GetCoords()

    if len(xs) < 3:
        print("Skipping polygon: not enough points")
        return

    file.write("G36*\n")

    # Move to first vertex
    file.write(gerber_coord((xs[0], ys[0])) + "D02*\n")

    # Draw edges
    for x, y in zip(xs[1:], ys[1:]):
        file.write(gerber_coord((x, y)) + "D01*\n")

    # Close polygon explicitly
    file.write(gerber_coord((xs[0], ys[0])) + "D01*\n")
    file.write("G37*\n")


# ---------------------------------------------------------------------
# Process CSX properties
# ---------------------------------------------------------------------
def process_primitives(file, prop_list, options):
    ignore = options.get("ignore", [])

    for prop in prop_list:
        name = prop.GetName()
        if name in ignore:
            print(f"omitting {name}")
            continue

        # Material filtering: skip if no Kappa
        # if prop_type == "Material":  # and not hasattr(prop, "Kappa"):
        #     print(f"omitting {name} ")
        #     continue

        print(f"processing {name}")
        file.write(f"%LN{name}*%\n")

        # primitives
        primitives = prop.GetAllPrimitives()
        if not primitives:
            print("  no primitives found")
            break

        for prim in primitives:
            cls = prim.__class__.__name__

            if cls == "CSPrimBox":
                primitive_box(
                    file,
                    prim,
                )
            elif cls == "CSPrimCylinder":
                primitive_cylinder(
                    file,
                    prim,
                )
            elif cls == "CSPrimPoly":
                primitive_polygon(
                    file,
                    prim,
                )
            elif cls == "CSPrimLinPoly":
                primitive_polygon(
                    file,
                    prim,
                )


# ---------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------
def export_gerber(CSX, output_path, options=None):
    """
    Export openEMS CSX geometry to Gerber RS-274X (XY-plane only).
    """
    global drawing_unit, coord_system
    print("-------------------------------------------")
    print("Exporting Geometry to Gerber")
    print("-------------------------------------------")
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
            process_primitives(
                file,
                metals,
                options,
            )

        file.write("M02*\n")
