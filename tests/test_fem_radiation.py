"""Tests for the FEM far-field path.

``fem_radiation`` turns the near fields of a GetDP solve into a radiation
pattern and serves it through :class:`FEMNF2FF`, whose whole purpose is to
imitate the openEMS ``nf2ff`` box closely enough that the ``SimTools``
radiation plots cannot tell the backends apart.

Most of what can go wrong here is silent. A pattern file that fails to parse
falls back to scattered interpolation, a Huygens box placed outside the mesh
reads as zeros rather than as an error, and a mirrored half-model with the
wrong component parities produces a pattern that looks entirely plausible and
is wrong. So the parsers, the parity rules, and the angle conventions are
pinned individually here; the end-to-end solve at the bottom (``slow``,
``needs_getdp_bin``) only checks that the pieces still fit together.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

# ``fem_radiation`` imports gmsh, which dlopen()s libGLU at import time. On a
# host without it -- a bare CI image, say -- that surfaces as OSError rather
# than ImportError, so importorskip does not catch it and the whole module
# fails to collect. Skip on any import failure instead.
try:
    import gmsh  # noqa: F401
except Exception as error:  # pragma: no cover - depends on the host
    pytest.skip(f"gmsh is not importable: {error}", allow_module_level=True)

from simpleEMS import fem_radiation  # noqa: E402
from simpleEMS.fem_materials import C0  # noqa: E402
from simpleEMS.fem_radiation import (  # noqa: E402
    FEMFarField,
    FEMNF2FF,
    _check_farfield_margin,
    _mirror_boundary_view,
    _parse_matlab_grid,
    _read_scalar_points,
)


@pytest.fixture
def gmsh_session():
    """A gmsh session that is always torn down.

    gmsh is a process-global singleton, and ``compute_pattern`` finalizes it
    on the way out. A test that leaves it initialised corrupts whatever runs
    next, so every test that touches a view goes through here.
    """
    import gmsh

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield gmsh
    if gmsh.isInitialized():
        gmsh.finalize()


def write_matlab(path: Path, phi, theta, ff) -> Path:
    """Write a pattern file in the form the gmsh plugin emits."""

    def block(name, values):
        return f"{name} = [{' '.join(repr(float(v)) for v in np.ravel(values))}];\n"

    path.write_text(block("phi", phi) + block("theta", theta) + block("farField", ff))
    return path


# ---------------------------------------------------------------------
# FEMFarField
# ---------------------------------------------------------------------
class TestFEMFarField:
    def test_carries_every_documented_field(self):
        """``SimTools`` reads these by name off both backends' results."""
        one = np.array([1.0])
        ff = FEMFarField(
            E_norm=one, Dmax=one, Prad=one, P_rad=one, theta=one, phi=one, Ploss=one
        )

        for name in ("E_norm", "Dmax", "Prad", "P_rad", "theta", "phi", "Ploss"):
            assert hasattr(ff, name)

    @pytest.mark.needs_csxcad
    def test_ploss_is_the_fem_only_extra(self):
        """openEMS's nf2ff has no Ploss; the FEM one adds it so gain can be
        de-rated by the radiation efficiency."""
        from openEMS.nf2ff import nf2ff_results

        assert not hasattr(nf2ff_results, "Ploss")
        assert "Ploss" in FEMFarField.__dataclass_fields__


# ---------------------------------------------------------------------
# _parse_matlab_grid
# ---------------------------------------------------------------------
class TestParseMatlabGrid:
    def test_reads_the_three_grids(self, tmp_path):
        nphi, ntheta = 2, 3
        size = (nphi + 1) * (ntheta + 1)
        path = write_matlab(
            tmp_path / "p.m", range(size), range(size), np.arange(size) * 2.0
        )

        phi, theta, ff = _parse_matlab_grid(path, nphi, ntheta)

        assert phi.shape == (nphi + 1, ntheta + 1)
        assert theta.shape == (nphi + 1, ntheta + 1)
        assert ff.shape == (nphi + 1, ntheta + 1)

    def test_values_land_in_row_major_order(self, tmp_path):
        """The reshape has to agree with the plugin's write order or the
        pattern comes out transposed -- which still plots, just wrongly."""
        nphi, ntheta = 1, 1
        path = write_matlab(
            tmp_path / "p.m", [0, 1, 2, 3], [0, 1, 2, 3], [10, 20, 30, 40]
        )

        _phi, _theta, ff = _parse_matlab_grid(path, nphi, ntheta)

        assert ff.tolist() == [[10.0, 20.0], [30.0, 40.0]]

    def test_missing_file_returns_none(self, tmp_path):
        """Signals the caller to fall back to scattered interpolation."""
        assert _parse_matlab_grid(tmp_path / "absent.m", 2, 2) is None

    def test_a_directory_returns_none(self, tmp_path):
        assert _parse_matlab_grid(tmp_path, 2, 2) is None

    @pytest.mark.parametrize("missing", ["phi", "theta", "farField"])
    def test_a_missing_variable_returns_none(self, tmp_path, missing):
        size = 4
        text = ""
        for name in ("phi", "theta", "farField"):
            if name != missing:
                text += f"{name} = [{' '.join(['1.0'] * size)}];\n"
        path = tmp_path / "p.m"
        path.write_text(text)

        assert _parse_matlab_grid(path, 1, 1) is None

    def test_wrong_angle_count_returns_none(self, tmp_path):
        """The file is from a different-sized run; reshaping would misalign the
        pattern against the axes."""
        path = write_matlab(tmp_path / "p.m", range(4), range(4), range(4))

        assert _parse_matlab_grid(path, 5, 5) is None

    def test_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "p.m"
        path.write_text("")

        assert _parse_matlab_grid(path, 1, 1) is None

    def test_accepts_a_string_path(self, tmp_path):
        path = write_matlab(tmp_path / "p.m", range(4), range(4), range(4))

        assert _parse_matlab_grid(str(path), 1, 1) is not None

    def test_tolerates_surrounding_text(self, tmp_path):
        """The plugin writes comments and other assignments around the arrays."""
        path = tmp_path / "p.m"
        path.write_text(
            "% gmsh NearToFarField\n"
            "phi = [0.0 1.0 2.0 3.0];\n"
            "someOther = [9.0];\n"
            "theta = [0.0 1.0 2.0 3.0];\n"
            "farField = [1.0 2.0 3.0 4.0];\n"
        )

        assert _parse_matlab_grid(path, 1, 1) is not None


# ---------------------------------------------------------------------
# _check_farfield_margin
# ---------------------------------------------------------------------
class TestCheckFarfieldMargin:
    def quarter_wave(self, freq):
        return 0.25 * (C0 / freq)

    def test_generous_padding_passes(self):
        freq = 2.45e9
        pad = 2 * self.quarter_wave(freq)
        bbox = (0, 0, 0, 0.05, 0.05, 0.002)
        domain = (
            bbox[0] - pad,
            bbox[1] - pad,
            bbox[2] - pad,
            bbox[3] + pad,
            bbox[4] + pad,
            bbox[5] + pad,
        )

        _check_farfield_margin(bbox, domain, freq, None)

    def test_exactly_a_quarter_wavelength_passes(self):
        freq = 2.45e9
        pad = self.quarter_wave(freq)
        bbox = (0, 0, 0, 0.05, 0.05, 0.002)
        domain = tuple(
            [bbox[i] - pad for i in range(3)] + [bbox[3 + i] + pad for i in range(3)]
        )

        _check_farfield_margin(bbox, domain, freq, None)

    def test_tight_padding_raises(self):
        freq = 2.45e9
        pad = 0.1 * self.quarter_wave(freq)
        bbox = (0, 0, 0, 0.05, 0.05, 0.002)
        domain = tuple(
            [bbox[i] - pad for i in range(3)] + [bbox[3 + i] + pad for i in range(3)]
        )

        with pytest.raises(ValueError, match="Air padding too small"):
            _check_farfield_margin(bbox, domain, freq, None)

    def test_the_error_names_the_offending_face(self):
        """Which face is short is the one thing the user needs to act on."""
        freq = 2.45e9
        pad = self.quarter_wave(freq)
        bbox = (0, 0, 0, 0.05, 0.05, 0.002)
        domain = [bbox[i] - pad for i in range(3)] + [
            bbox[3 + i] + pad for i in range(3)
        ]
        domain[4] = bbox[4] + 0.001  # starve the y+ face only

        with pytest.raises(ValueError, match=r"y\+ face"):
            _check_farfield_margin(bbox, tuple(domain), freq, None)

    def test_the_error_reports_the_padding_needed(self):
        freq = 3e9
        expected_mm = self.quarter_wave(freq) * 1e3
        bbox = (0, 0, 0, 0.05, 0.05, 0.002)
        domain = tuple(
            [bbox[i] - 1e-4 for i in range(3)] + [bbox[3 + i] + 1e-4 for i in range(3)]
        )

        with pytest.raises(ValueError, match=f"{expected_mm:.2f}"):
            _check_farfield_margin(bbox, domain, freq, None)

    def test_the_symmetry_face_is_exempt(self):
        """The x- face is a mirror plane, not open air, so it has no padding to
        check -- and checking it would reject every valid half model."""
        freq = 2.45e9
        pad = self.quarter_wave(freq)
        bbox = (0, 0, 0, 0.05, 0.05, 0.002)
        domain = [bbox[i] - pad for i in range(3)] + [
            bbox[3 + i] + pad for i in range(3)
        ]
        domain[0] = bbox[0]  # symmetry plane sits flush against the structure

        _check_farfield_margin(bbox, tuple(domain), freq, symmetry_axis=0)

    def test_a_starved_face_still_raises_with_symmetry(self):
        freq = 2.45e9
        pad = self.quarter_wave(freq)
        bbox = (0, 0, 0, 0.05, 0.05, 0.002)
        domain = [bbox[i] - pad for i in range(3)] + [
            bbox[3 + i] + pad for i in range(3)
        ]
        domain[0] = bbox[0]
        domain[5] = bbox[5] + 1e-4  # z+ is genuinely too tight

        with pytest.raises(ValueError, match=r"z\+ face"):
            _check_farfield_margin(bbox, tuple(domain), freq, symmetry_axis=0)

    def test_the_requirement_scales_with_frequency(self):
        """A padding that is fine at 10 GHz is a tenth of what 1 GHz needs."""
        pad = self.quarter_wave(10e9)
        bbox = (0, 0, 0, 0.01, 0.01, 0.001)
        domain = tuple(
            [bbox[i] - pad for i in range(3)] + [bbox[3 + i] + pad for i in range(3)]
        )

        _check_farfield_margin(bbox, domain, 10e9, None)
        with pytest.raises(ValueError):
            _check_farfield_margin(bbox, domain, 1e9, None)


# ---------------------------------------------------------------------
# _read_scalar_points
# ---------------------------------------------------------------------
class TestReadScalarPoints:
    def add_view(self, gmsh, dtype, nelem, data):
        tag = gmsh.view.add("pattern")
        gmsh.view.addListData(tag, dtype, nelem, list(np.ravel(data).astype(float)))
        return tag

    def test_scalar_points_become_angles(self, gmsh_session):
        gmsh = gmsh_session
        # a point on +z, one on +x, relative to the origin
        tag = self.add_view(gmsh, "SP", 2, [0, 0, 1, 5.0, 1, 0, 0, 7.0])

        theta, phi, val = _read_scalar_points(tag, (0.0, 0.0, 0.0))

        assert val == [5.0, 7.0]
        assert theta[0] == pytest.approx(0.0)  # +z is theta = 0
        assert theta[1] == pytest.approx(math.pi / 2)  # +x is on the equator
        assert phi[1] == pytest.approx(0.0)

    def test_positions_are_measured_from_the_given_centre(self, gmsh_session):
        """The sampling box is not centred on the origin, so directions have to
        be taken relative to its centre."""
        gmsh = gmsh_session
        tag = self.add_view(gmsh, "SP", 1, [10, 10, 11, 1.0])

        theta, _phi, _val = _read_scalar_points(tag, (10.0, 10.0, 10.0))

        assert theta[0] == pytest.approx(0.0)

    def test_azimuth_is_wrapped_into_zero_to_two_pi(self, gmsh_session):
        """RegularGridInterpolator's phi axis runs 0..2pi; a negative atan2
        result would fall off the end of it."""
        gmsh = gmsh_session
        tag = self.add_view(gmsh, "SP", 1, [0, -1, 0, 1.0])

        _theta, phi, _val = _read_scalar_points(tag, (0.0, 0.0, 0.0))

        assert phi[0] == pytest.approx(1.5 * math.pi)

    def test_a_point_at_the_centre_is_dropped(self, gmsh_session):
        """Its direction is undefined; keeping it would inject a garbage angle."""
        gmsh = gmsh_session
        tag = self.add_view(gmsh, "SP", 2, [0, 0, 0, 9.0, 0, 0, 1, 1.0])

        theta, _phi, val = _read_scalar_points(tag, (0.0, 0.0, 0.0))

        assert val == [1.0]
        assert len(theta) == 1

    def test_scalar_triangles_yield_one_sample_per_node(self, gmsh_session):
        gmsh = gmsh_session
        # x-block, y-block, z-block, then one value per node: nodes are
        # (1,0,0), (0,1,0), (0,0,1)
        data = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1.0, 2.0, 3.0]
        tag = self.add_view(gmsh, "ST", 1, data)

        theta, phi, val = _read_scalar_points(tag, (0.0, 0.0, 0.0))

        assert val == [1.0, 2.0, 3.0]
        # node 3 is on +z, so its theta is 0
        assert theta[2] == pytest.approx(0.0)
        # node 2 is on +y, so its phi is 90 degrees
        assert phi[1] == pytest.approx(math.pi / 2)

    def test_coordinates_are_read_per_axis_not_per_node(self, gmsh_session):
        """gmsh stores an element as x1..xn, y1..yn, z1..zn. Reading it as
        interleaved triples transposes every sample position, which silently
        scrambles the pattern instead of failing."""
        gmsh = gmsh_session
        # block layout -> nodes (11,21,31), (12,22,32), (13,23,33)
        data = [11, 12, 13, 21, 22, 23, 31, 32, 33, 1.0, 2.0, 3.0]
        tag = self.add_view(gmsh, "ST", 1, data)

        theta, _phi, _val = _read_scalar_points(tag, (0.0, 0.0, 0.0))

        expected = math.acos(31 / math.sqrt(11**2 + 21**2 + 31**2))
        assert theta[0] == pytest.approx(expected)

    def test_scalar_quads_yield_one_sample_per_node(self, gmsh_session):
        gmsh = gmsh_session
        # nodes (1,0,0), (0,1,0), (0,0,1), (1,1,1)
        data = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1.0, 2.0, 3.0, 4.0]
        tag = self.add_view(gmsh, "SQ", 1, data)

        _theta, _phi, val = _read_scalar_points(tag, (0.0, 0.0, 0.0))

        assert val == [1.0, 2.0, 3.0, 4.0]

    def test_unknown_element_types_are_skipped(self, gmsh_session):
        """The pattern view holds only scalar elements; anything else would
        mis-stride the buffer."""
        gmsh = gmsh_session
        tag = gmsh.view.add("vectors")
        gmsh.view.addListData(tag, "VP", 1, [0.0, 0.0, 1.0, 1.0, 2.0, 3.0])

        assert _read_scalar_points(tag, (0.0, 0.0, 0.0)) == ([], [], [])

    def test_theta_stays_within_range(self, gmsh_session):
        """The acos argument is clamped; float error just outside [-1, 1] would
        otherwise produce NaN."""
        gmsh = gmsh_session
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(20, 3))
        data = np.hstack([pts, np.ones((20, 1))])
        tag = self.add_view(gmsh, "SP", 20, data)

        theta, phi, _val = _read_scalar_points(tag, (0.0, 0.0, 0.0))

        assert all(0.0 <= t <= math.pi for t in theta)
        assert all(0.0 <= p < 2 * math.pi + 1e-12 for p in phi)

    def test_an_empty_view_yields_nothing(self, gmsh_session):
        gmsh = gmsh_session
        tag = gmsh.view.add("empty")

        assert _read_scalar_points(tag, (0.0, 0.0, 0.0)) == ([], [], [])


# ---------------------------------------------------------------------
# _mirror_boundary_view
# ---------------------------------------------------------------------
class TestMirrorBoundaryView:
    def add_vector_triangle(self, gmsh, coords, values, name="huygens"):
        """``coords`` is (3, 3) as [x-block, y-block, z-block]; ``values`` is
        (3 nodes, 3 components)."""
        tag = gmsh.view.add(name)
        data = np.concatenate(
            [np.asarray(coords, float).ravel(), np.asarray(values, float).ravel()]
        )
        gmsh.view.addListData(tag, "VT", 1, list(data))
        return tag

    def read_elements(self, gmsh, tag):
        dtypes, counts, data = gmsh.view.getListData(tag)
        assert len(dtypes) == 1
        arr = np.asarray(data[0], dtype=float)
        return counts[0], arr.reshape(counts[0], -1)

    def test_the_surface_is_doubled(self, gmsh_session):
        gmsh = gmsh_session
        coords = [[1, 2, 1], [0, 0, 1], [0, 1, 0]]  # x-block, y-block, z-block
        tag = self.add_vector_triangle(gmsh, coords, np.ones((3, 3)))

        out = _mirror_boundary_view(tag, 0, 0.0, "pec", "E")

        nelem, _rows = self.read_elements(gmsh, out)
        assert nelem == 2

    def test_the_mirrored_copy_is_reflected_in_the_plane(self, gmsh_session):
        gmsh = gmsh_session
        coords = [[1, 2, 3], [0, 0, 1], [0, 1, 0]]
        tag = self.add_vector_triangle(gmsh, coords, np.ones((3, 3)))

        out = _mirror_boundary_view(tag, 0, 0.0, "pec", "E")

        _nelem, rows = self.read_elements(gmsh, out)
        mirrored_x = rows[1][:3]
        # reflected about x = 0, and node order reversed
        assert sorted(mirrored_x) == sorted([-1.0, -2.0, -3.0])

    def test_reflection_is_about_an_arbitrary_plane(self, gmsh_session):
        gmsh = gmsh_session
        coords = [[1, 2, 3], [0, 0, 1], [0, 1, 0]]
        tag = self.add_vector_triangle(gmsh, coords, np.ones((3, 3)))

        out = _mirror_boundary_view(tag, 0, 5.0, "pec", "E")

        _nelem, rows = self.read_elements(gmsh, out)
        assert sorted(rows[1][:3]) == sorted([2.0 * 5.0 - c for c in (1.0, 2.0, 3.0)])

    def test_node_order_is_reversed(self, gmsh_session):
        """A reflection flips orientation, and the far-field transform takes
        the surface normal from the first three nodes -- an unreversed copy
        would radiate inwards."""
        gmsh = gmsh_session
        coords = [[1, 2, 3], [10, 20, 30], [100, 200, 300]]
        tag = self.add_vector_triangle(gmsh, coords, np.ones((3, 3)))

        out = _mirror_boundary_view(tag, 0, 0.0, "pec", "E")

        _nelem, rows = self.read_elements(gmsh, out)
        assert rows[1][3:6].tolist() == [30.0, 20.0, 10.0]  # y-block, reversed

    @pytest.mark.parametrize(
        ("field", "kind", "normal_sign"),
        [
            ("E", "pec", 1.0),
            ("H", "pec", -1.0),
            ("E", "pmc", -1.0),
            ("H", "pmc", 1.0),
        ],
    )
    def test_component_parity_follows_the_wall_type(
        self, gmsh_session, field, kind, normal_sign
    ):
        """Getting these signs wrong yields a plausible but wrong pattern, so
        all four combinations are pinned."""
        gmsh = gmsh_session
        coords = [[1, 2, 3], [0, 0, 1], [0, 1, 0]]
        values = np.tile([1.0, 2.0, 4.0], (3, 1))  # same vector on every node
        tag = self.add_vector_triangle(gmsh, coords, values)

        out = _mirror_boundary_view(tag, 0, 0.0, kind, field)

        _nelem, rows = self.read_elements(gmsh, out)
        mirrored_values = rows[1][9:].reshape(3, 3)
        assert mirrored_values[0][0] == pytest.approx(normal_sign * 1.0)
        assert mirrored_values[0][1] == pytest.approx(-normal_sign * 2.0)
        assert mirrored_values[0][2] == pytest.approx(-normal_sign * 4.0)

    def test_elements_lying_in_the_plane_are_dropped(self, gmsh_session):
        """They become interior once the halves are joined; keeping them would
        put a radiating sheet through the middle of the structure."""
        gmsh = gmsh_session
        coords = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]  # entirely on x = 0
        tag = self.add_vector_triangle(gmsh, coords, np.ones((3, 3)))

        out = _mirror_boundary_view(tag, 0, 0.0, "pec", "E")

        dtypes, _counts, _data = gmsh.view.getListData(out)
        assert dtypes == []

    def test_mirroring_across_y_uses_the_y_components(self, gmsh_session):
        gmsh = gmsh_session
        coords = [[0, 0, 1], [1, 2, 3], [0, 1, 0]]
        values = np.tile([1.0, 2.0, 4.0], (3, 1))
        tag = self.add_vector_triangle(gmsh, coords, values)

        out = _mirror_boundary_view(tag, 1, 0.0, "pec", "E")

        _nelem, rows = self.read_elements(gmsh, out)
        mirrored_values = rows[1][9:].reshape(3, 3)
        assert mirrored_values[0][1] == pytest.approx(2.0)  # normal component kept
        assert mirrored_values[0][0] == pytest.approx(-1.0)

    def test_a_malformed_layout_is_reported(self, gmsh_session):
        """Silently mis-striding would scramble coordinates into field values.

        45 floats over 2 elements is not a whole number of strides: gmsh reads
        it back as one time step, so each element should occupy 9 coordinates
        plus 9 field components.
        """
        gmsh = gmsh_session
        tag = gmsh.view.add("bad")
        gmsh.view.addListData(tag, "VT", 2, [0.0] * 45)

        with pytest.raises(RuntimeError, match="unexpected VT list-data layout"):
            _mirror_boundary_view(tag, 0, 0.0, "pec", "E")


# ---------------------------------------------------------------------
# FEMNF2FF.CalcNF2FF
# ---------------------------------------------------------------------
class TestCalcNF2FF:
    @pytest.fixture
    def nf2ff(self, monkeypatch):
        """A calculator with a known analytic pattern stitched in.

        ``_pattern`` is what runs the solver; replacing it isolates the angle
        handling and interpolation, which is the part shared with openEMS.
        """
        theta_axis = np.linspace(0.0, math.pi, 37)
        phi_axis = np.linspace(0.0, 2 * math.pi, 73)
        # a broadside pattern: peak at theta = 0, independent of phi
        u_grid = np.tile(np.cos(theta_axis / 2) ** 2, (phi_axis.size, 1))
        pattern = (theta_axis, phi_axis, u_grid, 6.0, 0.75, 0.25)

        calc = FEMNF2FF()
        monkeypatch.setattr(
            FEMNF2FF, "_pattern", lambda self, output_path, freq: pattern
        )
        return calc

    def test_returns_a_femfarfield(self, nf2ff):
        result = nf2ff.CalcNF2FF("out", 2.45e9, np.array([0.0]), np.array([0.0]))

        assert isinstance(result, FEMFarField)

    def test_result_is_shaped_theta_by_phi(self, nf2ff):
        """``SimTools`` indexes ``E_norm[:, k]`` for the k-th phi cut."""
        theta = np.linspace(-180, 180, 91)
        phi = np.array([0.0, 90.0])

        result = nf2ff.CalcNF2FF("out", 2.45e9, theta, phi)

        assert result.E_norm.shape == (91, 2)
        assert result.P_rad.shape == (91, 2)

    def test_angles_come_back_in_radians(self, nf2ff):
        theta = np.array([0.0, 90.0, 180.0])
        phi = np.array([0.0, 180.0])

        result = nf2ff.CalcNF2FF("out", 2.45e9, theta, phi)

        assert result.theta == pytest.approx([0.0, math.pi / 2, math.pi])
        assert result.phi == pytest.approx([0.0, math.pi])

    def test_e_norm_is_the_root_of_the_intensity(self, nf2ff):
        result = nf2ff.CalcNF2FF("out", 2.45e9, np.array([0.0, 45.0]), np.array([0.0]))

        assert result.E_norm == pytest.approx(np.sqrt(result.P_rad))

    def test_the_broadside_peak_lands_at_theta_zero(self, nf2ff):
        theta = np.linspace(0, 180, 181)

        result = nf2ff.CalcNF2FF("out", 2.45e9, theta, np.array([0.0]))

        assert int(np.argmax(result.P_rad[:, 0])) == 0

    def test_negative_theta_mirrors_to_the_opposite_phi(self, nf2ff):
        """A principal-plane cut runs -180..180; the negative half is the
        phi + 180 side of the same plane, not an out-of-range angle."""
        result = nf2ff.CalcNF2FF(
            "out", 2.45e9, np.array([-45.0, 45.0]), np.array([0.0, 180.0])
        )

        # pattern is phi-independent, so -45 at phi=0 equals +45 at phi=180
        assert result.P_rad[0, 0] == pytest.approx(result.P_rad[1, 1], rel=1e-6)

    def test_a_full_cut_is_symmetric_for_a_symmetric_pattern(self, nf2ff):
        theta = np.linspace(-180, 180, 73)

        result = nf2ff.CalcNF2FF("out", 2.45e9, theta, np.array([0.0]))

        assert result.P_rad[:, 0] == pytest.approx(result.P_rad[::-1, 0], rel=1e-6)

    def test_dmax_is_the_directivity_delinearised(self, nf2ff):
        """``_pattern`` reports 6 dB."""
        result = nf2ff.CalcNF2FF("out", 2.45e9, np.array([0.0]), np.array([0.0]))

        assert result.Dmax == pytest.approx([10.0 ** (6.0 / 10.0)])

    def test_powers_are_passed_through_as_length_one_arrays(self, nf2ff):
        result = nf2ff.CalcNF2FF("out", 2.45e9, np.array([0.0]), np.array([0.0]))

        assert result.Prad == pytest.approx([0.75])
        assert result.Ploss == pytest.approx([0.25])

    def test_intensity_is_never_zero(self, nf2ff):
        """A null would become -inf dB and blow up the polar plots."""
        theta = np.linspace(-180, 180, 181)

        result = nf2ff.CalcNF2FF("out", 2.45e9, theta, np.array([0.0]))

        assert np.all(result.P_rad > 0)

    def test_scalar_angles_are_accepted(self, nf2ff):
        result = nf2ff.CalcNF2FF("out", 2.45e9, 0.0, 0.0)

        assert result.E_norm.shape == (1, 1)

    def test_openems_only_arguments_are_tolerated(self, nf2ff):
        """The signature has to match the openEMS box, whose callers pass
        these; they are documented as ignored."""
        result = nf2ff.CalcNF2FF(
            "out",
            2.45e9,
            np.array([0.0]),
            np.array([0.0]),
            read_cached=True,
            outfile="ignored.h5",
            verbose=3,
        )

        assert isinstance(result, FEMFarField)

    def test_a_path_output_path_is_accepted(self, nf2ff):
        result = nf2ff.CalcNF2FF(Path("out"), 2.45e9, np.array([0.0]), np.array([0.0]))

        assert isinstance(result, FEMFarField)


# ---------------------------------------------------------------------
# FEMNF2FF._pattern
# ---------------------------------------------------------------------
class TestPatternMetadata:
    """The bookkeeping ``_pattern`` does before and around the solve."""

    @pytest.fixture
    def staged(self, tmp_path, monkeypatch):
        """Stub the solve and the transform; the test supplies fem_mesh.json."""
        solves = []
        patterns = []

        def fake_solve(pro, msh, workdir, freq):
            solves.append((pro, msh, str(workdir), freq))
            return ("e.pos", "h.pos", 0.25, 0.75)

        def fake_compute(*args, **kwargs):
            patterns.append((args, kwargs))
            theta_axis = np.linspace(0.0, math.pi, 5)
            phi_axis = np.linspace(0.0, 2 * math.pi, 5)
            return theta_axis, phi_axis, np.ones((5, 5)), 3.0

        monkeypatch.setattr(
            fem_radiation.fem_solver, "solve_fields_and_power", fake_solve
        )
        monkeypatch.setattr(fem_radiation, "compute_pattern", fake_compute)
        return tmp_path, solves, patterns

    def write_meta(self, path, **overrides):
        pad = 0.25 * (C0 / 2.45e9) * 1.5
        bbox = [0, 0, 0, 0.05, 0.05, 0.002]
        meta = {
            "pro_path": "p.pro",
            "msh_path": "m.msh",
            "bbox": bbox,
            "domain_bbox": [bbox[i] - pad for i in range(3)]
            + [bbox[3 + i] + pad for i in range(3)],
            "symmetry_axis": None,
            "symmetry_plane": None,
            "symmetry_kind": None,
        }
        meta.update(overrides)
        (path / "fem_mesh.json").write_text(json.dumps(meta))
        return meta

    def test_solves_at_the_requested_frequency(self, staged):
        out, solves, _patterns = staged
        self.write_meta(out)

        FEMNF2FF()._pattern(str(out), 2.45e9)

        assert solves[0][3] == 2.45e9

    def test_the_pattern_is_computed_once_per_frequency(self, staged):
        """Each solve is expensive; the plots ask for the same frequency
        repeatedly (2D cut, 3D surface, gain, power)."""
        out, solves, _patterns = staged
        self.write_meta(out)

        calc = FEMNF2FF()
        calc._pattern(str(out), 2.45e9)
        calc._pattern(str(out), 2.45e9)

        assert len(solves) == 1

    def test_a_different_frequency_solves_again(self, staged):
        out, solves, _patterns = staged
        self.write_meta(out)

        calc = FEMNF2FF()
        calc._pattern(str(out), 2.45e9)
        calc._pattern(str(out), 2.60e9)

        assert len(solves) == 2

    def test_powers_are_carried_through(self, staged):
        out, _solves, _patterns = staged
        self.write_meta(out)

        *_rest, p_rad, p_loss = FEMNF2FF()._pattern(str(out), 2.45e9)

        assert p_rad == 0.75
        assert p_loss == 0.25

    def test_symmetry_metadata_reaches_the_transform(self, staged):
        out, _solves, patterns = staged
        self.write_meta(out, symmetry_axis=1, symmetry_plane=0.0, symmetry_kind="pmc")

        FEMNF2FF()._pattern(str(out), 2.45e9)

        assert patterns[0][1]["symmetry"] == (1, 0.0, "pmc")

    def test_no_symmetry_passes_none(self, staged):
        out, _solves, patterns = staged
        self.write_meta(out)

        FEMNF2FF()._pattern(str(out), 2.45e9)

        assert patterns[0][1]["symmetry"] is None

    def test_a_stale_symmetric_mesh_is_rejected(self, staged):
        """Half the Huygens surface with no way to mirror it would give a wrong
        pattern silently, so this refuses rather than guessing."""
        out, _solves, _patterns = staged
        self.write_meta(out, symmetry_axis=1, symmetry_plane=None)

        with pytest.raises(RuntimeError, match="predates the far-field mirroring"):
            FEMNF2FF()._pattern(str(out), 2.45e9)

    def test_a_mesh_without_domain_bbox_warns_but_proceeds(self, staged, capsys):
        out, _solves, patterns = staged
        meta = self.write_meta(out)
        del meta["domain_bbox"]
        (out / "fem_mesh.json").write_text(json.dumps(meta))

        FEMNF2FF()._pattern(str(out), 2.45e9)

        assert "stale mesh" in capsys.readouterr().out
        assert patterns[0][1]["domain_bbox"] is None

    def test_tight_padding_is_caught_before_solving(self, staged):
        """The check costs nothing; the solve it precedes costs minutes."""
        out, solves, _patterns = staged
        bbox = [0, 0, 0, 0.05, 0.05, 0.002]
        self.write_meta(
            out,
            domain_bbox=[bbox[i] - 1e-4 for i in range(3)]
            + [bbox[3 + i] + 1e-4 for i in range(3)],
        )

        with pytest.raises(ValueError, match="Air padding too small"):
            FEMNF2FF()._pattern(str(out), 2.45e9)

        assert solves == []

    def test_a_missing_metadata_file_raises(self, staged):
        out, _solves, _patterns = staged

        with pytest.raises(FileNotFoundError):
            FEMNF2FF()._pattern(str(out), 2.45e9)

    def test_separate_output_paths_are_cached_separately(self, staged, tmp_path):
        out, solves, _patterns = staged
        other = tmp_path / "other"
        other.mkdir()
        self.write_meta(out)
        self.write_meta(other)

        calc = FEMNF2FF()
        calc._pattern(str(out), 2.45e9)
        calc._pattern(str(other), 2.45e9)

        assert len(solves) == 2


# ---------------------------------------------------------------------
# compute_pattern, against a real solve
# ---------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.needs_getdp_bin
@pytest.mark.needs_csxcad
class TestComputePatternEndToEnd:
    """The only test that runs the whole far-field chain for real.

    Everything above stubs either the solver or the transform. This one meshes
    a patch antenna, solves it, samples the near fields on a Huygens box and
    runs the gmsh far-field transform, because the pieces can each be correct
    while the seams between them are not.

    Accuracy is not asserted -- the mesh is deliberately coarse. What is
    asserted is the failure mode the module actually has: a Huygens box that
    lands outside the meshed region reads as zeros, and a pattern of zeros
    plots as a perfectly plausible empty polar chart.
    """

    @pytest.fixture(scope="class")
    def radiated(self, tmp_path_factory):
        pytest.importorskip("CSXCAD")
        pytest.importorskip("openEMS")

        from simpleEMS.patch_antenna import ProbeFedPatchAntenna, ProbeFedPatchParams
        from simpleEMS.sim_tools import SimTools, setup_simulation

        params = ProbeFedPatchParams(
            resonant_freq=2.45e9,
            span_freq=0.4e9,
            substrate_eps_r=4.4,
            substrate_tand=0.001,
            substrate_thickness_mm=1.6,
            charac_imp=50,
            num_points=5,
            backend_engine="FEM",
            FEM_num_solve_points=4,
            FEM_elems_per_wavelength=4.0,
            FEM_min_layers=1,
        )
        sim = setup_simulation(params)
        antenna = ProbeFedPatchAntenna(params, sim)
        antenna.build_probe_fed_patch_antenna()

        out = tmp_path_factory.mktemp("fem_farfield")
        SimTools.run_simulation(sim, output_path=out)

        nf2ff = SimTools.create_nf2ff(sim)
        theta = np.linspace(-180, 180, 37)
        phi = np.array([0.0, 90.0])
        result = nf2ff.CalcNF2FF(str(out), 2.45e9, theta, phi)
        return result, out, nf2ff, theta, phi

    def test_returns_a_femfarfield(self, radiated):
        result, *_rest = radiated

        assert isinstance(result, FEMFarField)

    def test_shape_follows_the_requested_angles(self, radiated):
        result, _out, _nf, theta, phi = radiated

        assert result.E_norm.shape == (theta.size, phi.size)

    def test_the_pattern_is_not_identically_zero(self, radiated):
        """The documented silent failure: a Huygens box outside the mesh makes
        CutBox return zeros everywhere, and nothing downstream complains."""
        result, *_rest = radiated

        assert np.max(result.P_rad) > 1e-30

    def test_the_pattern_varies_with_angle(self, radiated):
        """A constant pattern means the transform ran but read nothing useful."""
        result, *_rest = radiated

        assert np.ptp(result.P_rad) > 0

    def test_everything_is_finite(self, radiated):
        result, *_rest = radiated

        assert np.all(np.isfinite(result.E_norm))
        assert np.all(np.isfinite(result.P_rad))
        assert np.isfinite(result.Dmax).all()

    def test_directivity_is_physical(self, radiated):
        """An isotropic radiator has D = 1; nothing this size reaches 100."""
        result, *_rest = radiated

        assert 1.0 <= float(result.Dmax[0]) < 100.0

    def test_powers_are_non_negative(self, radiated):
        result, *_rest = radiated

        assert float(result.Prad[0]) >= 0.0
        assert float(result.Ploss[0]) >= 0.0

    def test_some_power_is_radiated(self, radiated):
        result, *_rest = radiated

        assert float(result.Prad[0]) > 0.0

    def test_the_transform_wrote_its_intermediate_files(self, radiated):
        """``_parse_matlab_grid`` reads the .m file; if it is missing the code
        silently falls back to scattered interpolation."""
        _result, out, *_rest = radiated

        assert (out / "output" / "pattern_ntf.m").exists()
        assert (out / "output" / "pattern_ntf.pos").exists()

    def test_the_regular_grid_path_was_taken(self, radiated):
        """The written pattern file should parse at the default resolution, so
        the scattered-interpolation fallback is not being used."""
        _result, out, *_rest = radiated

        assert _parse_matlab_grid(out / "output" / "pattern_ntf.m", 72, 36) is not None

    def test_gmsh_is_left_finalized(self, radiated):
        """``compute_pattern`` initialises gmsh and must hand it back; a live
        session leaks into whatever meshes next."""
        import gmsh

        assert not gmsh.isInitialized()

    def test_a_second_call_reuses_the_cached_pattern(self, radiated):
        """Re-solving would double the cost of every extra plot."""
        _result, out, nf2ff, _theta, _phi = radiated
        marker = out / "output" / "pattern_ntf.m"
        before = marker.stat().st_mtime_ns

        nf2ff.CalcNF2FF(str(out), 2.45e9, np.array([0.0]), np.array([0.0]))

        assert marker.stat().st_mtime_ns == before

    def test_different_angles_come_from_the_same_solve(self, radiated):
        """Any angle is interpolated from the one computed grid."""
        result, out, nf2ff, theta, _phi = radiated

        again = nf2ff.CalcNF2FF(str(out), 2.45e9, theta, np.array([0.0]))

        assert again.P_rad[:, 0] == pytest.approx(result.P_rad[:, 0])


def test_a_fresh_calculator_has_an_empty_cache():
    assert FEMNF2FF()._cache == {}


@pytest.mark.needs_csxcad
def test_create_nf2ff_returns_the_fem_adapter_without_a_sim():
    """The standalone STEP-FEM workflow has no SimSetup to dispatch on."""
    from simpleEMS.sim_tools import SimTools

    assert isinstance(SimTools.create_nf2ff(None), FEMNF2FF)
