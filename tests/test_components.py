"""Unit tests for the primitives built by :mod:`simpleEMS.components`."""

import pytest


@pytest.fixture
def generic_structure(fr4, sim_for):
    from simpleEMS.components import GenericParams, GenericStructure

    params = GenericParams(
        min_freq=2e9,
        max_freq=3e9,
        target_freq=2.45e9,
        substrate_eps_r=fr4["substrate_eps_r"],
        substrate_tand=0.001,
        substrate_thickness_mm=fr4["substrate_thickness_mm"],
        substrate_width_mm=20.0,
        substrate_length_mm=30.0,
    )
    return GenericStructure(params, sim_for(params)), params


def _primitives_named(csx, name):
    return [
        primitive
        for primitive in csx.GetAllPrimitives()
        if primitive.GetProperty().GetName() == name
    ]


def _build_via(structure, **kwargs):
    return structure.create_via(
        position=(0.0, 0.0),
        via_diameter_mm=0.6,
        z_bottom_mm=-0.035,
        z_top_mm=1.6,
        antipad_diameter_mm=1.2,
        antipad_layers=[0.8],
        **kwargs,
    )


@pytest.mark.needs_csxcad
class TestViaAntipad:
    def test_the_antipad_fills_with_the_substrate_by_default(self, generic_structure):
        structure, params = generic_structure
        _build_via(structure)

        (antipad,) = _primitives_named(structure.CSX, "via_antipad")
        material = antipad.GetProperty()
        assert material.GetMaterialProperty("epsilon") == pytest.approx(
            params.substrate_eps_r
        )
        assert material.GetMaterialProperty("kappa") == pytest.approx(
            params.substrate_kappa
        )

    def test_an_explicit_permittivity_fills_the_antipad_with_air(
        self, generic_structure
    ):
        structure, _ = generic_structure
        _build_via(structure, antipad_eps_r=1.0)

        (antipad,) = _primitives_named(structure.CSX, "via_antipad")
        material = antipad.GetProperty()
        assert material.GetMaterialProperty("epsilon") == pytest.approx(1.0)
        assert material.GetMaterialProperty("kappa") == pytest.approx(0.0)

    def test_the_barrel_outranks_the_antipad_so_the_via_stays_continuous(
        self, generic_structure
    ):
        structure, _ = generic_structure
        ground = structure.create_ground()
        barrel = _build_via(structure)

        (antipad,) = _primitives_named(structure.CSX, "via_antipad")
        assert ground.GetPriority() < antipad.GetPriority() < barrel.GetPriority()

    def test_explicit_priorities_override_the_defaults(self, generic_structure):
        structure, _ = generic_structure
        barrel = _build_via(structure, priority=12, antipad_priority=9)

        (antipad,) = _primitives_named(structure.CSX, "via_antipad")
        assert (barrel.GetPriority(), antipad.GetPriority()) == (12, 9)

    def test_the_antipad_survives_the_gerber_export(self, generic_structure, tmp_path):
        """The antipad is a ring in the 3D model but has to reach the plane's
        Gerber file as a void, or the fab gets a plane shorted to the via.

        It did not, once: ``export_gerber`` dispatches on exact class names
        and had no branch for the shell, so the clearance was dropped without
        even a warning -- the antipad is a material, and the warning was
        gated on metal.
        """
        from simpleEMS.export_gerber import export_gerber

        structure, params = generic_structure
        copper = params.copper_thickness_mm
        structure.create_substrate()
        structure.create_ground()
        structure.create_ground(
            "inner_plane",
            start=[-10.0, -15.0, 0.8],
            stop=[10.0, 15.0, 0.8 + copper],
        )
        # A top trace, so the plane the antipad cuts is genuinely an inner
        # layer rather than the topmost copper on the board.
        structure.create_microstrip(
            "trace", position=(0.0, -10.0), width_mm=3.0, length_mm=20.0
        )
        _build_via(structure)

        files = {
            path.name: path.read_text()
            for path in export_gerber(structure.CSX, tmp_path, {})
        }
        inner = files["layout-In1_Cu.g2"]

        assert "%LNvia_antipad*%" in inner
        assert inner.index("%LPC*%") < inner.index("%LNvia_antipad*%")


def _degenerate_elements(msh_path):
    """Count the zero-area triangles and zero-volume tets in a mesh."""
    import gmsh
    import numpy as np

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.open(str(msh_path))
        degenerate = 0
        for dim, quality in ((2, "minSJ"), (3, "minSICN")):
            for _, tag in gmsh.model.getEntities(dim):
                for element_tags in gmsh.model.mesh.getElements(dim, tag)[1]:
                    if len(element_tags):
                        qualities = gmsh.model.mesh.getElementQualities(
                            element_tags, quality
                        )
                        degenerate += int(np.sum(np.abs(qualities) < 1e-6))
        return degenerate
    finally:
        gmsh.finalize()


def _port_ends_touching_pec(msh_path):
    """For each port, whether its bottom and top edges share nodes with PEC.

    Returns ``{port_name: (bottom_touches, top_touches)}``, the ends taken
    along z, the axis every port on the via board runs along.
    """
    import gmsh
    import numpy as np

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.open(str(msh_path))
        groups = {
            gmsh.model.getPhysicalName(dim, tag): tag
            for dim, tag in gmsh.model.getPhysicalGroups(2)
        }
        pec_nodes = set(gmsh.model.mesh.getNodesForPhysicalGroup(2, groups["pec"])[0])
        touching = {}
        for name, tag in groups.items():
            if not name.startswith("port"):
                continue
            node_tags, coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, tag)
            heights = np.asarray(coords).reshape(-1, 3)[:, 2]
            shared = np.array([node in pec_nodes for node in node_tags])
            touching[name] = tuple(
                bool(np.any(shared & np.isclose(heights, end, rtol=0, atol=1e-9)))
                for end in (heights.min(), heights.max())
            )
        return touching
    finally:
        gmsh.finalize()


@pytest.fixture(scope="class")
def via_board_mesh(tmp_path_factory):
    """Mesh a board with one inner plane and a via through its antipad.

    Two prepregs around the plane; the via takes a trace on the top copper down
    through the plane's antipad to the bottom copper. Port 1 is drawn up from
    the plane's top face and port 2 down from its bottom face, so whichever
    face the plane collapses onto, one port has to be snapped to reach it.
    Returns the path to the ``.msh`` file.
    """
    from simpleEMS import fem_backend
    from simpleEMS.components import GenericParams, GenericStructure
    from simpleEMS.sim_tools import setup_simulation

    params = GenericParams(
        min_freq=0.5e9,
        max_freq=10.0e9,
        target_freq=5.0e9,
        substrate_eps_r=4.4,
        substrate_tand=0.02,
        substrate_thickness_mm=1.07,
        substrate_width_mm=6.0,
        substrate_length_mm=12.0,
        substrate_cells=4,
        charac_imp=50,
        backend_engine="FEM",
    )
    sim = setup_simulation(params)
    structure = GenericStructure(params, sim)

    copper = params.copper_thickness_mm
    prepreg = 0.5
    z_plane = copper + prepreg
    z_top = z_plane + copper + prepreg
    half_width = params.substrate_width_mm / 2
    half_length = params.substrate_length_mm / 2
    line_width = 0.93
    structure.create_substrate(
        "lower",
        start=[-half_width, -half_length, copper],
        stop=[half_width, half_length, z_plane],
    )
    structure.create_substrate(
        "upper",
        start=[-half_width, -half_length, z_plane + copper],
        stop=[half_width, half_length, z_top],
    )
    structure.create_ground(
        "plane",
        start=[-half_width, -half_length, z_plane],
        stop=[half_width, half_length, z_plane + copper],
    )
    structure.create_microstrip(
        "top_line",
        position=(0, -half_length),
        width_mm=line_width,
        length_mm=half_length + 0.6,
        z_elevation_mm=z_top,
    )
    structure.create_microstrip(
        "bottom_line",
        position=(0, -0.6),
        width_mm=line_width,
        length_mm=half_length + 0.6,
        z_elevation_mm=0.0,
    )
    structure.create_via(
        "via",
        position=(0.0, 0.0),
        via_diameter_mm=0.3,
        z_bottom_mm=0.0,
        z_top_mm=z_top + copper,
        antipad_diameter_mm=1.0,
        antipad_layers=[z_plane],
    )
    structure.create_lumped_port(
        port_nr=1,
        start=[line_width / 2, -half_length, z_plane + copper],
        stop=[-line_width / 2, -half_length, z_top],
        excite=1,
        edges2grid="y",
    )
    structure.create_lumped_port(
        port_nr=2,
        start=[line_width / 2, half_length, z_plane],
        stop=[-line_width / 2, half_length, copper],
        excite=0,
        edges2grid="y",
    )

    return fem_backend.build_mesh(
        sim.CSX,
        sim.freqs,
        tmp_path_factory.mktemp("via_board"),
        verbose=False,
        FEM_options=params.fem_options,
    )


@pytest.mark.slow
@pytest.mark.needs_csxcad
@pytest.mark.needs_cadquery
class TestViaFEMMesh:
    def test_a_via_through_an_antipad_meshes_without_flat_elements(
        self, via_board_mesh
    ):
        """A via's barrel is a thin cylinder, and the 35 um antipad pulls the
        mesh size down to 20 um around it. Gmsh's default surface algorithm
        then joined three nodes of the barrel's seam into one flat triangle:
        getdp failed on it with ``Null determinant in 'ChangeOfCoord_Form2'``,
        and on a smaller board the 3D mesher rejected it outright.
        """
        assert _degenerate_elements(via_board_mesh) == 0

    def test_ports_on_either_side_of_an_inner_plane_reach_it(self, via_board_mesh):
        """An inner plane collapses onto one of its two faces, and a port drawn
        to the other face stopped a copper thickness short of it. It connected
        to nothing, so the line behind it ran into an open circuit: the
        multilayer via example read |S11| near 0 dB where FDTD showed a match.
        """
        touching = _port_ends_touching_pec(via_board_mesh)

        assert touching == {"port_1": (True, True), "port_2": (True, True)}


@pytest.mark.needs_csxcad
class TestPrimitivePriority:
    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("create_substrate", {}),
            ("create_ground", {}),
            (
                "create_microstrip",
                {"position": (0.0, 0.0), "width_mm": 3.0, "length_mm": 10.0},
            ),
            (
                "create_taper",
                {
                    "position": (0.0, 0.0),
                    "width1_mm": 3.0,
                    "width2_mm": 1.0,
                    "length_mm": 5.0,
                },
            ),
            (
                "create_miter",
                {"position": (0.0, 0.0), "width_mm": 3.0, "miter_distance_mm": 1.5},
            ),
            (
                "create_curved_bend",
                {"position": (0.0, 0.0), "width_mm": 3.0, "bend_radius_mm": 5.0},
            ),
            (
                "create_radial_stub",
                {
                    "position": (0.0, 0.0),
                    "inner_radius_mm": 1.0,
                    "outer_radius_mm": 5.0,
                    "angle_start": 60,
                    "angle_end": 120,
                },
            ),
        ],
        ids=[
            "substrate",
            "ground",
            "microstrip",
            "taper",
            "miter",
            "curved_bend",
            "radial_stub",
        ],
    )
    def test_the_priority_argument_reaches_the_primitive(
        self, generic_structure, method, kwargs
    ):
        structure, _ = generic_structure
        primitive = getattr(structure, method)(priority=11, **kwargs)

        assert primitive.GetPriority() == 11

    def test_the_cpw_applies_its_priority_to_every_conductor(self, generic_structure):
        structure, _ = generic_structure
        primitives = structure.create_cpw(
            position=(0.0, 0.0),
            trace_width_mm=1.0,
            gap_mm=0.2,
            ground_width_mm=3.0,
            length_mm=10.0,
            priority=11,
        )

        assert [primitive.GetPriority() for primitive in primitives] == [11, 11, 11]


# ---------------------------------------------------------------------
# Helpers and parameters
# ---------------------------------------------------------------------
@pytest.mark.needs_csxcad
class TestCirclePoints:
    def test_returns_the_requested_number_of_points(self):
        from simpleEMS.components import circle_points

        x_coords, y_coords = circle_points(0.0, 0.0, 1.0, num_faces=8)

        assert len(x_coords) == len(y_coords) == 8

    def test_points_lie_on_the_circle(self):
        import numpy as np

        from simpleEMS.components import circle_points

        x_coords, y_coords = circle_points(2.0, -1.0, 3.0)

        radii = np.hypot(np.array(x_coords) - 2.0, np.array(y_coords) + 1.0)
        assert radii == pytest.approx(3.0)

    def test_the_first_point_is_on_the_positive_x_axis(self):
        from simpleEMS.components import circle_points

        x_coords, y_coords = circle_points(1.0, 1.0, 2.0)

        assert (x_coords[0], y_coords[0]) == pytest.approx((3.0, 1.0))


@pytest.mark.needs_csxcad
class TestGenericParams:
    def test_frequency_range_is_min_to_max(self, generic_structure):
        _structure, params = generic_structure

        assert params.freq_range == (2e9, 3e9)

    def test_main_freq_is_the_target_frequency(self, generic_structure):
        _structure, params = generic_structure

        assert params.main_freq == 2.45e9


# ---------------------------------------------------------------------
# Placement: rotation, elevation and thickness
# ---------------------------------------------------------------------
@pytest.mark.needs_csxcad
class TestPlacement:
    def test_rotate_point_turns_counterclockwise_about_the_centre(self):
        from simpleEMS.components import GenericStructure

        rotated = GenericStructure._rotate_point(1.0, 12.0, 1.0, 2.0, 90)

        assert rotated == pytest.approx((-9.0, 2.0))

    def test_rotation_pivots_the_primitive_about_its_position(self, generic_structure):
        structure, _ = generic_structure

        primitive = structure.create_microstrip(
            position=(1.0, 2.0), width_mm=2.0, length_mm=10.0, rotation=90
        )

        assert primitive.HasTransform()
        far_end = primitive.GetTransform().Transform([1.0, 12.0, 0.0])
        assert far_end == pytest.approx([-9.0, 2.0, 0.0])

    def test_no_rotation_adds_no_transform(self, generic_structure):
        structure, _ = generic_structure

        primitive = structure.create_microstrip(
            position=(0.0, 0.0), width_mm=2.0, length_mm=10.0
        )

        assert not primitive.HasTransform()

    def test_metal_sits_on_the_substrate_by_default(self, generic_structure):
        structure, params = generic_structure

        primitive = structure.create_microstrip(
            position=(0.0, 0.0), width_mm=2.0, length_mm=10.0
        )

        bound_box = primitive.GetBoundBox()
        assert bound_box[0][2] == pytest.approx(params.substrate_thickness_mm)
        assert bound_box[1][2] - bound_box[0][2] == pytest.approx(
            params.copper_thickness_mm
        )

    def test_an_explicit_elevation_and_thickness_are_used(self, generic_structure):
        structure, _ = generic_structure

        primitive = structure.create_microstrip(
            position=(0.0, 0.0),
            width_mm=2.0,
            length_mm=10.0,
            z_elevation_mm=0.8,
            copper_thickness_mm=0.07,
        )

        bound_box = primitive.GetBoundBox()
        assert bound_box[0][2] == pytest.approx(0.8)
        assert bound_box[1][2] == pytest.approx(0.87)


# ---------------------------------------------------------------------
# Input validation of the shape builders
# ---------------------------------------------------------------------
@pytest.mark.needs_csxcad
class TestShapeValidation:
    def test_antipad_layers_need_a_diameter(self, generic_structure):
        structure, _ = generic_structure

        with pytest.raises(ValueError, match="without an antipad_diameter_mm"):
            structure.create_via(
                position=(0.0, 0.0),
                via_diameter_mm=0.6,
                z_bottom_mm=0.0,
                z_top_mm=1.6,
                antipad_layers=[0.8],
            )

    @pytest.mark.parametrize("antipad_diameter_mm", [0.6, 0.4], ids=["equal", "less"])
    def test_the_antipad_must_be_wider_than_the_via(
        self, generic_structure, antipad_diameter_mm
    ):
        structure, _ = generic_structure

        with pytest.raises(ValueError, match="must exceed via_diameter_mm"):
            structure.create_via(
                position=(0.0, 0.0),
                via_diameter_mm=0.6,
                z_bottom_mm=0.0,
                z_top_mm=1.6,
                antipad_diameter_mm=antipad_diameter_mm,
                antipad_layers=[0.8],
            )

    @pytest.mark.parametrize("miter_distance_mm", [-0.1, 3.1], ids=["negative", "wide"])
    def test_the_miter_cut_must_fit_the_trace(
        self, generic_structure, miter_distance_mm
    ):
        structure, _ = generic_structure

        with pytest.raises(ValueError, match="miter_distance_mm"):
            structure.create_miter(
                position=(0.0, 0.0), width_mm=3.0, miter_distance_mm=miter_distance_mm
            )

    @pytest.mark.parametrize("method", ["create_miter", "create_curved_bend"])
    def test_a_turn_is_left_or_right(self, generic_structure, method):
        structure, _ = generic_structure
        shape = (
            {"miter_distance_mm": 1.0}
            if method == "create_miter"
            else {"bend_radius_mm": 5.0}
        )

        with pytest.raises(ValueError, match="turn must be"):
            getattr(structure, method)(
                position=(0.0, 0.0), width_mm=3.0, turn="up", **shape
            )

    def test_the_bend_radius_must_clear_the_inner_edge(self, generic_structure):
        structure, _ = generic_structure

        with pytest.raises(ValueError, match="bend_radius_mm"):
            structure.create_curved_bend(
                position=(0.0, 0.0), width_mm=3.0, bend_radius_mm=1.5
            )

    @pytest.mark.parametrize("bend_angle", [0, 361], ids=["zero", "over-a-turn"])
    def test_the_bend_angle_is_within_one_turn(self, generic_structure, bend_angle):
        structure, _ = generic_structure

        with pytest.raises(ValueError, match="bend_angle"):
            structure.create_curved_bend(
                position=(0.0, 0.0),
                width_mm=3.0,
                bend_radius_mm=5.0,
                bend_angle=bend_angle,
            )


# ---------------------------------------------------------------------
# Miter and curved bend shapes
# ---------------------------------------------------------------------
@pytest.mark.needs_csxcad
class TestBends:
    @staticmethod
    def coordinates(primitive):
        import numpy as np

        return np.asarray(primitive.GetCoords())

    def test_a_right_miter_mirrors_the_left_one(self, generic_structure):
        import numpy as np

        structure, _ = generic_structure
        left = structure.create_miter(
            "left", position=(0.0, 0.0), width_mm=3.0, miter_distance_mm=1.0
        )
        right = structure.create_miter(
            "right",
            position=(0.0, 0.0),
            width_mm=3.0,
            miter_distance_mm=1.0,
            turn="right",
        )

        left_points = self.coordinates(left)
        right_points = self.coordinates(right)
        assert np.sort(-left_points[0]) == pytest.approx(np.sort(right_points[0]))
        assert np.sort(left_points[1]) == pytest.approx(np.sort(right_points[1]))

    @pytest.mark.parametrize(
        ("miter_distance_mm", "vertex_count"),
        [(0.0, 4), (1.0, 5), (3.0, 3)],
        ids=["square", "chamfer", "full-diagonal"],
    )
    def test_a_collapsed_chamfer_drops_its_vertex(
        self, generic_structure, miter_distance_mm, vertex_count
    ):
        structure, _ = generic_structure

        miter = structure.create_miter(
            position=(0.0, 0.0), width_mm=3.0, miter_distance_mm=miter_distance_mm
        )

        assert self.coordinates(miter).shape[1] == vertex_count

    def test_a_right_bend_mirrors_the_left_one(self, generic_structure):
        import numpy as np

        structure, _ = generic_structure
        left = structure.create_curved_bend(
            "left", position=(0.0, 0.0), width_mm=2.0, bend_radius_mm=5.0
        )
        right = structure.create_curved_bend(
            "right",
            position=(0.0, 0.0),
            width_mm=2.0,
            bend_radius_mm=5.0,
            turn="right",
        )

        left_points = self.coordinates(left)
        right_points = self.coordinates(right)
        assert np.sort(-left_points[0]) == pytest.approx(np.sort(right_points[0]))
        assert np.sort(left_points[1]) == pytest.approx(np.sort(right_points[1]))

    def test_the_bend_spans_the_trace_width_radially(self, generic_structure):
        import numpy as np

        structure, _ = generic_structure

        bend = structure.create_curved_bend(
            position=(0.0, 0.0), width_mm=2.0, bend_radius_mm=5.0
        )

        x_coords, y_coords = self.coordinates(bend)
        radii = np.hypot(x_coords + 5.0, y_coords)
        assert radii.min() == pytest.approx(4.0)
        assert radii.max() == pytest.approx(6.0)


# ---------------------------------------------------------------------
# Coplanar waveguides
# ---------------------------------------------------------------------
CPW = {"trace_width_mm": 1.0, "gap_mm": 0.3, "ground_width_mm": 3.0}


@pytest.mark.needs_csxcad
class TestCoplanarWaveguide:
    def test_registers_three_lines_across_each_slot(self, generic_structure):
        structure, _ = generic_structure

        structure.create_cpw(position=(0.0, 0.0), length_mm=10.0, **CPW)

        assert structure._requested_mesh_lines[0] == pytest.approx(
            [-0.8, -0.65, -0.5, 0.5, 0.65, 0.8]
        )

    @pytest.mark.parametrize("rotation", [90, 270], ids=["90", "270"])
    def test_a_quarter_turn_registers_the_slots_along_y(
        self, generic_structure, rotation
    ):
        structure, _ = generic_structure

        structure.create_cpw(
            position=(0.0, 2.0), length_mm=10.0, rotation=rotation, **CPW
        )

        assert structure._requested_mesh_lines[0] == []
        assert sorted(structure._requested_mesh_lines[1]) == pytest.approx(
            [1.2, 1.35, 1.5, 2.5, 2.65, 2.8]
        )

    def test_an_oblique_cpw_registers_no_slot_lines(self, generic_structure):
        structure, _ = generic_structure

        structure.create_cpw(position=(0.0, 0.0), length_mm=10.0, rotation=45, **CPW)

        assert structure._requested_mesh_lines[:2] == [[], []]

    def test_gcpw_stitches_both_grounds_with_vias(self, generic_structure):
        structure, _ = generic_structure

        structure.create_gcpw(
            position=(0.0, 0.0),
            length_mm=10.0,
            via_diameter_mm=0.5,
            via_pitch_mm=2.0,
            via_z_bottom_mm=0.0,
            via_z_top_mm=1.6,
            **CPW,
        )

        vias = [
            primitive
            for primitive in structure.CSX.GetAllPrimitives()
            if primitive.GetProperty().GetName().startswith("gcpw_via_")
        ]
        x_centres = sorted({round(via.GetStart()[0], 6) for via in vias})
        y_centres = sorted({round(via.GetStart()[1], 6) for via in vias})
        assert len(vias) == 2 * 6
        assert x_centres == pytest.approx([-2.3, 2.3])
        assert y_centres[0] == pytest.approx(0.25)
        assert y_centres[-1] == pytest.approx(9.75)

    def test_gcpw_vias_follow_the_rotation(self, generic_structure):
        structure, _ = generic_structure

        structure.create_gcpw(
            position=(0.0, 0.0),
            length_mm=10.0,
            via_diameter_mm=0.5,
            via_pitch_mm=20.0,
            via_z_bottom_mm=0.0,
            via_z_top_mm=1.6,
            rotation=90,
            **CPW,
        )

        (first_via,) = _primitives_named(structure.CSX, "gcpw_via_1_1")
        assert first_via.GetStart()[:2] == pytest.approx([-0.25, -2.3])

    @pytest.mark.parametrize(
        ("via_diameter_mm", "length_mm", "message"),
        [(10.0, 10.0, "less than length_mm"), (3.0, 10.0, "less than ground_width")],
        ids=["longer-than-the-segment", "wider-than-the-ground"],
    )
    def test_gcpw_rejects_a_via_that_does_not_fit(
        self, generic_structure, via_diameter_mm, length_mm, message
    ):
        structure, _ = generic_structure

        with pytest.raises(ValueError, match=message):
            structure.create_gcpw(
                position=(0.0, 0.0),
                length_mm=length_mm,
                via_diameter_mm=via_diameter_mm,
                via_pitch_mm=2.0,
                via_z_bottom_mm=0.0,
                via_z_top_mm=1.6,
                **CPW,
            )

        assert structure.CSX.GetAllPrimitives() == []


# ---------------------------------------------------------------------
# Substrate, ground and plain lumped ports
# ---------------------------------------------------------------------
@pytest.mark.needs_csxcad
class TestBoardAndPorts:
    def test_overriding_the_material_recomputes_kappa(self, generic_structure):
        import numpy as np
        from openEMS.physical_constants import EPS0

        structure, _ = generic_structure

        substrate = structure.create_substrate(eps_r=3.0, tand=0.02, main_freq=1e9)

        material = substrate.GetProperty()
        expected = 0.02 * 2 * np.pi * 1e9 * EPS0 * 3.0
        assert material.GetMaterialProperty("epsilon") == pytest.approx(3.0)
        assert material.GetMaterialProperty("kappa") == pytest.approx(expected)

    def test_the_substrate_registers_its_interior_z_lines(self, generic_structure):
        import numpy as np

        structure, params = generic_structure

        structure.create_substrate()

        expected = np.linspace(0, params.substrate_thickness_mm, params.substrate_cells)
        assert structure._requested_mesh_lines[2] == pytest.approx(expected[1:-1])

    def test_ground_defaults_to_the_board_footprint(self, generic_structure):
        structure, params = generic_structure

        ground = structure.create_ground()

        bound_box = ground.GetBoundBox()
        assert bound_box[1][0] - bound_box[0][0] == pytest.approx(
            params.substrate_width_mm
        )
        assert bound_box[0][2] == pytest.approx(-params.copper_thickness_mm)

    def test_lumped_port_defaults_to_the_characteristic_impedance(
        self, generic_structure
    ):
        structure, params = generic_structure

        port = structure.create_lumped_port(1, [0, 0, 0], [1, 0, 1.6], excite=1)

        resistance = port.R
        assert resistance == pytest.approx(params.charac_imp)
        assert port.excite == 1

    def test_lumped_port_takes_an_explicit_impedance(self, generic_structure):
        structure, _ = generic_structure

        port = structure.create_lumped_port(2, [0, 0, 0], [1, 0, 1.6], impedance=75)

        resistance = port.R
        assert resistance == pytest.approx(75)
        assert port.number == 2


# ---------------------------------------------------------------------
# CPW waveguide port
# ---------------------------------------------------------------------
@pytest.fixture
def cpw_board(generic_structure):
    """A meshed board carrying a straight CPW along y."""
    structure, params = generic_structure
    structure.create_substrate()
    structure.create_ground()
    structure.create_cpw(position=(0.0, -15.0), length_mm=30.0, **CPW)
    return structure, params


class RecordingFDTD:
    """Stands in for the openEMS object, whose methods cannot be patched."""

    def __init__(self):
        self.calls = []

    def AddCPWPort(self, *args, **kwargs):  # noqa: N802 - matches openEMS
        self.calls.append((args, kwargs))


@pytest.mark.needs_csxcad
class TestCPWPort:
    START = [-0.5, -15.0, 1.6]
    STOP = [0.5, -10.0, 1.6]

    def test_builds_on_the_meshed_grid(self, cpw_board):
        from openEMS.ports import CPWPort

        structure, _ = cpw_board
        structure.create_mesh()

        port = structure.create_cpw_port(
            port_nr=1, start=self.START, stop=self.STOP, gap_mm=0.3, excite=1
        )

        assert isinstance(port, CPWPort)
        assert port.number == 1

    def test_passes_the_feed_and_measurement_shifts(self, cpw_board, monkeypatch):
        structure, _ = cpw_board
        structure.create_mesh()
        recorder = RecordingFDTD()
        monkeypatch.setattr(structure, "FDTD", recorder)

        structure.create_cpw_port(
            port_nr=1,
            start=self.START,
            stop=self.STOP,
            gap_mm=0.3,
            impedance=60,
            feed_shift_mm=1.0,
            meas_plane_shift_mm=2.0,
        )

        ((_args, kwargs),) = recorder.calls
        assert kwargs["FeedShift"] == 1.0
        assert kwargs["MeasPlaneShift"] == 2.0
        assert kwargs["Feed_R"] == 60

    def test_snaps_onto_the_grid_and_spans_the_slots(self, cpw_board, monkeypatch):
        import numpy as np

        structure, _ = cpw_board
        structure.create_mesh()
        recorder = RecordingFDTD()
        monkeypatch.setattr(structure, "FDTD", recorder)

        structure.create_cpw_port(
            port_nr=1, start=[-0.5, -14.99, 1.61], stop=[0.5, -10.0, 1.61], gap_mm=0.3
        )

        ((args, kwargs),) = recorder.calls
        _port_nr, _metal, start, stop, prop_dir, slot_dir, gap_mm = args
        grid = structure.CSX.GetGrid()
        assert start[2] == stop[2]
        assert start[2] in np.asarray(grid.GetLines(2))
        assert start[1] in np.asarray(grid.GetLines(1))
        assert (prop_dir, slot_dir, gap_mm) == ("y", "x", 0.3)
        assert "FeedShift" not in kwargs
        assert "MeasPlaneShift" not in kwargs

    def test_refuses_an_out_of_plane_direction(self, cpw_board):
        structure, _ = cpw_board

        with pytest.raises(ValueError, match="prop_dir"):
            structure.create_cpw_port(
                port_nr=1, start=self.START, stop=self.STOP, gap_mm=0.3, prop_dir="z"
            )

    def test_refuses_a_port_that_is_not_a_sheet(self, cpw_board):
        structure, _ = cpw_board

        with pytest.raises(ValueError, match="share one z"):
            structure.create_cpw_port(
                port_nr=1, start=self.START, stop=[0.5, -10.0, 1.7], gap_mm=0.3
            )

    def test_refuses_a_port_with_no_length(self, cpw_board):
        structure, _ = cpw_board

        with pytest.raises(ValueError, match="must differ along y"):
            structure.create_cpw_port(
                port_nr=1, start=self.START, stop=[0.5, -15.0, 1.6], gap_mm=0.3
            )

    def test_refuses_an_unmeshed_structure(self, cpw_board):
        structure, _ = cpw_board

        with pytest.raises(ValueError, match="call create_mesh"):
            structure.create_cpw_port(
                port_nr=1, start=self.START, stop=self.STOP, gap_mm=0.3
            )

    def test_refuses_a_slot_the_grid_does_not_resolve(self, cpw_board):
        structure, _ = cpw_board
        structure.create_mesh()

        with pytest.raises(ValueError, match="holds"):
            structure.create_cpw_port(
                port_nr=1, start=self.START, stop=self.STOP, gap_mm=0.01
            )


# ---------------------------------------------------------------------
# Lumped-fed CPW port
# ---------------------------------------------------------------------
@pytest.mark.needs_csxcad
class TestCPWLumpedPort:
    START = [-0.5, -15.0, 1.6]
    STOP = [0.5, -14.0, 1.635]

    @pytest.fixture
    def port(self, cpw_board):
        structure, _ = cpw_board
        return structure, structure.create_cpw_lumped_port(
            port_nr=1, start=self.START, stop=self.STOP, gap_mm=0.3, excite=1
        )

    def test_fills_each_slot_with_a_lumped_port(self, port):
        _structure, cpw_port = port

        assert len(cpw_port.slots) == 2
        slot_edges = sorted(
            edge for slot in cpw_port.slots for edge in (slot.start[0], slot.stop[0])
        )
        assert slot_edges == pytest.approx([-0.8, -0.5, 0.5, 0.8])

    def test_each_slot_carries_twice_the_port_impedance(self, port):
        _structure, cpw_port = port

        assert [slot.R for slot in cpw_port.slots] == pytest.approx([100, 100])
        resistance = cpw_port.R
        assert resistance == pytest.approx(100)
        assert cpw_port.Z_ref == pytest.approx(50)

    def test_collects_both_slots_probe_files_and_excitations(self, port):
        _structure, cpw_port = port

        assert len(cpw_port.U_filenames) == 2 * len(cpw_port.slots[0].U_filenames)
        assert len(cpw_port.I_filenames) == 2 * len(cpw_port.slots[0].I_filenames)
        assert len(cpw_port.port_props) == 2 * len(cpw_port.slots[0].port_props)
        assert any("_lo_" in name for name in cpw_port.U_filenames)
        assert any("_hi_" in name for name in cpw_port.U_filenames)

    def test_registers_the_slot_and_port_edges_with_the_mesher(self, port):
        structure, _cpw_port = port

        assert {-0.8, -0.5, 0.5, 0.8} <= set(structure._requested_mesh_lines[0])
        assert {-15.0, -14.0} <= set(structure._requested_mesh_lines[1])
        assert {1.6, 1.635} <= set(structure._requested_mesh_lines[2])

    def test_takes_an_explicit_impedance(self, cpw_board):
        structure, _ = cpw_board

        cpw_port = structure.create_cpw_lumped_port(
            port_nr=1, start=self.START, stop=self.STOP, gap_mm=0.3, impedance=75
        )

        assert cpw_port.Z_ref == pytest.approx(75)

    def test_reads_half_the_summed_slot_voltage(self, port, monkeypatch):
        """The two slots are in parallel, so their voltages are one voltage
        read twice."""
        import numpy as np
        from openEMS.ports import Port

        _structure, cpw_port = port

        def fake_read(self, sim_path, freq, signal_type="pulse"):
            self.uf_tot = np.array([4.0 + 2j])
            self.ut_tot = np.array([6.0])

        monkeypatch.setattr(Port, "ReadUIData", fake_read)

        cpw_port.ReadUIData("sim", np.array([1e9]))

        assert cpw_port.uf_tot == pytest.approx([2.0 + 1j])
        assert cpw_port.ut_tot == pytest.approx([3.0])

    def test_calc_port_drops_the_plane_shift(self, port, monkeypatch):
        from openEMS.ports import Port

        _structure, cpw_port = port
        received = []
        monkeypatch.setattr(Port, "CalcPort", lambda self, *args: received.append(args))

        cpw_port.CalcPort("sim", [1e9], 50)

        assert received == [("sim", [1e9], 50, None, "pulse")]

    def test_calc_port_refuses_a_plane_shift(self, port):
        _structure, cpw_port = port

        with pytest.raises(ValueError, match="no\\s+propagation constant"):
            cpw_port.CalcPort("sim", [1e9], ref_plane_shift=1.0)

    @pytest.mark.parametrize(
        ("start", "stop", "kwargs", "message"),
        [
            (START, STOP, {"prop_dir": "z"}, "prop_dir"),
            (START, [0.5, -15.0, 1.635], {}, "must differ along y"),
            (START, [-0.5, -14.0, 1.635], {}, "must differ along it"),
            (START, [0.5, -14.0, 1.6], {}, "must differ in z"),
        ],
        ids=["out-of-plane", "no-length", "no-trace-width", "no-thickness"],
    )
    def test_refuses_a_malformed_port(self, cpw_board, start, stop, kwargs, message):
        structure, _ = cpw_board

        with pytest.raises(ValueError, match=message):
            structure.create_cpw_lumped_port(
                port_nr=1, start=start, stop=stop, gap_mm=0.3, **kwargs
            )

    def test_refuses_to_be_added_after_meshing(self, cpw_board):
        structure, _ = cpw_board
        structure.create_mesh()

        with pytest.raises(ValueError, match="before create_mesh"):
            structure.create_cpw_lumped_port(
                port_nr=1, start=self.START, stop=self.STOP, gap_mm=0.3
            )
