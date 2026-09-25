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
