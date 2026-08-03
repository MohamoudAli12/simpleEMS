"""Tests for :class:`simpleEMS.sim_params.SimParams` and its subclasses.

``__post_init__`` derives kappa, lambda0, and the three mesh quantities that
every structure and mesh generator reads. Those derivations are asserted here
against their closed forms, together with the validation that guards the
solver-backend selection.
"""

import dataclasses

import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from openEMS.physical_constants import C0, EPS0  # noqa: E402

from simpleEMS.fem_backend import FEMOptions
from simpleEMS.sim_params import SimParams


@pytest.fixture
def params(inset_params):
    """A concrete ``SimParams`` instance (the base class is abstract)."""
    return inset_params


# ---------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------
class TestDerivedQuantities:
    def test_kappa_follows_the_loss_tangent_formula(self, params):
        expected = (
            params.substrate_tand
            * 2
            * np.pi
            * params.main_freq
            * EPS0
            * params.substrate_eps_r
        )

        assert params.substrate_kappa == pytest.approx(expected, rel=1e-12)

    def test_lossless_substrate_has_zero_kappa(self, fr4):
        from simpleEMS.patch_antenna import InsetFedPatchParams

        lossless = {**fr4, "substrate_tand": 0.0}
        result = InsetFedPatchParams(resonant_freq=2.45e9, span_freq=0.5e9, **lossless)

        assert result.substrate_kappa == 0.0

    def test_lambda0_is_the_guided_wavelength_in_drawing_units(self, params):
        expected = C0 / (
            params.main_freq * np.sqrt(params.substrate_eps_r) * params.unit
        )

        assert params.lambda0 == pytest.approx(expected, abs=10**-params.fp_precision)

    def test_lambda0_is_in_millimetres(self, params):
        """``unit`` is 1e-3, so lambda0 comes out in mm -- roughly 58 mm for a
        2.45 GHz wave in FR-4."""
        assert params.lambda0 == pytest.approx(58.3, abs=1.0)

    def test_mesh_resolution_divides_lambda0(self, params):
        expected = params.lambda0 / params.FDTD_mesh_resolution_factor

        assert params.FDTD_mesh_resolution == pytest.approx(
            expected, abs=10**-params.fp_precision
        )

    def test_metal_mesh_is_finer_than_the_global_mesh(self, params):
        """The metal factor (40) is larger than the global one (10), so metal
        edges get a finer grid. Inverting these would coarsen conductor edges
        without any visible error."""
        assert params.FDTD_metal_mesh_resolution < params.FDTD_mesh_resolution

    def test_metal_mesh_resolution_divides_lambda0(self, params):
        expected = params.lambda0 / params.FDTD_metal_mesh_resolution_factor

        assert params.FDTD_metal_mesh_resolution == pytest.approx(
            expected, abs=10**-params.fp_precision
        )

    def test_thirds_rule_has_two_opposite_signed_offsets(self, params):
        """The thirds rule places one line inside the metal and one outside."""
        assert params.FDTD_thirds_rule.shape == (2,)
        assert params.FDTD_thirds_rule[0] > 0
        assert params.FDTD_thirds_rule[1] < 0

    def test_thirds_rule_is_two_thirds_and_one_third_of_a_quarter_cell(self, params):
        quarter = params.FDTD_mesh_resolution / 4
        expected = np.array([2 * quarter / 3, -quarter / 3])

        assert params.FDTD_thirds_rule == pytest.approx(
            expected, abs=10**-params.fp_precision
        )

    def test_custom_resolution_factors_are_honoured(self, fr4):
        from simpleEMS.patch_antenna import InsetFedPatchParams

        result = InsetFedPatchParams(
            resonant_freq=2.45e9,
            span_freq=0.5e9,
            FDTD_mesh_resolution_factor=20,
            FDTD_metal_mesh_resolution_factor=80,
            **fr4,
        )

        assert result.FDTD_mesh_resolution == pytest.approx(
            result.lambda0 / 20, abs=1e-3
        )
        assert result.FDTD_metal_mesh_resolution == pytest.approx(
            result.lambda0 / 80, abs=1e-3
        )


# ---------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------
class TestBackendValidation:
    def test_fdtd_is_the_default_backend(self, params):
        assert params.backend_engine == "FDTD"

    @pytest.mark.parametrize("backend", ["FDTD", "FEM"])
    def test_supported_backends_are_accepted(self, fr4, backend):
        from simpleEMS.patch_antenna import InsetFedPatchParams

        result = InsetFedPatchParams(
            resonant_freq=2.45e9, span_freq=0.5e9, backend_engine=backend, **fr4
        )

        assert result.backend_engine == backend

    @pytest.mark.parametrize("backend", ["fdtd", "fem", "MoM", "", "FDTD "])
    def test_unknown_backend_raises(self, fr4, backend):
        """The comparison is case-sensitive, so ``"fdtd"`` must be rejected
        rather than silently falling through to the FEM branch elsewhere."""
        from simpleEMS.patch_antenna import InsetFedPatchParams

        with pytest.raises(ValueError, match="backend_engine must be"):
            InsetFedPatchParams(
                resonant_freq=2.45e9, span_freq=0.5e9, backend_engine=backend, **fr4
            )

    @pytest.mark.parametrize("num_points", [0, 1, 3, -1])
    def test_too_few_fem_solve_points_raises(self, fr4, num_points):
        """AAA needs at least 4 support points for a stable rational fit."""
        from simpleEMS.patch_antenna import InsetFedPatchParams

        with pytest.raises(ValueError, match="FEM_num_solve_points must be >= 4"):
            InsetFedPatchParams(
                resonant_freq=2.45e9,
                span_freq=0.5e9,
                FEM_num_solve_points=num_points,
                **fr4,
            )

    def test_solve_point_check_applies_even_to_the_fdtd_backend(self, fr4):
        """``_validate_backend`` checks it unconditionally; this pins that
        behaviour so the check is not accidentally moved under the FEM
        branch."""
        from simpleEMS.patch_antenna import InsetFedPatchParams

        with pytest.raises(ValueError, match="FEM_num_solve_points"):
            InsetFedPatchParams(
                resonant_freq=2.45e9,
                span_freq=0.5e9,
                backend_engine="FDTD",
                FEM_num_solve_points=2,
                **fr4,
            )

    def test_four_solve_points_is_the_boundary_and_is_allowed(self, fr4):
        from simpleEMS.patch_antenna import InsetFedPatchParams

        result = InsetFedPatchParams(
            resonant_freq=2.45e9, span_freq=0.5e9, FEM_num_solve_points=4, **fr4
        )

        assert result.FEM_num_solve_points == 4


# ---------------------------------------------------------------------
# FEM option bundling
# ---------------------------------------------------------------------
class TestFEMOptions:
    def test_defaults_are_taken_from_femoptions(self, params):
        """The flat ``FEM_*`` fields default off a single ``FEMOptions()``
        instance, so the two must not drift apart."""
        defaults = FEMOptions()

        assert params.FEM_boundary == defaults.boundary
        assert params.FEM_symmetry == defaults.symmetry
        assert params.FEM_fe_order == defaults.fe_order
        assert params.FEM_air_pad_frac == defaults.air_pad_frac
        assert params.FEM_air_pad_mm == defaults.air_pad_mm
        assert params.FEM_elems_per_wavelength == defaults.elems_per_wavelength
        assert params.FEM_mesh_fine_scale == defaults.mesh_fine_scale
        assert params.FEM_min_layers == defaults.min_layers

    def test_fem_options_property_round_trips_every_field(self, params):
        """Every ``FEMOptions`` field must be fed by a ``FEM_*`` param; a
        field added to one and not the other would silently use the default."""
        options = params.fem_options

        for field in dataclasses.fields(FEMOptions):
            flat_name = f"FEM_{field.name}"
            assert hasattr(params, flat_name), f"{flat_name} missing on SimParams"
            assert getattr(options, field.name) == getattr(params, flat_name)

    def test_custom_values_reach_the_bundled_options(self, fr4):
        from simpleEMS.patch_antenna import InsetFedPatchParams

        result = InsetFedPatchParams(
            resonant_freq=2.45e9,
            span_freq=0.5e9,
            backend_engine="FEM",
            FEM_fe_order=2,
            FEM_boundary="pml",
            FEM_air_pad_mm=5.0,
            FEM_num_solve_points=12,
            **fr4,
        )

        options = result.fem_options

        assert options.fe_order == 2
        assert options.boundary == "pml"
        assert options.air_pad_mm == 5.0
        assert options.num_solve_points == 12

    def test_property_returns_a_fresh_object_each_time(self, params):
        """It is a property, not a cached field; mutating the result must not
        leak back into the params object."""
        first = params.fem_options
        first.fe_order = 99

        assert params.fem_options.fe_order != 99


# ---------------------------------------------------------------------
# Simulation box
# ---------------------------------------------------------------------
class TestSimulationBox:
    def test_box_has_three_dimensions(self, params):
        assert params.simulation_box.shape == (3,)

    def test_box_is_rounded_to_fp_precision(self, params):
        box = params.simulation_box

        assert box == pytest.approx(np.round(box, params.fp_precision))

    def test_box_encloses_the_substrate(self, params):
        box = params.simulation_box

        assert box[0] >= params.substrate_width_mm
        assert box[1] >= params.substrate_length_mm

    def test_box_dimensions_are_positive(self, params):
        assert all(params.simulation_box > 0)

    def test_create_simulation_box_rounds_its_inputs(self, params):
        box = params._create_simulation_box(1.23456789, 2.0, 3.987654321)

        assert box == pytest.approx([1.235, 2.0, 3.988])


# ---------------------------------------------------------------------
# Abstract base behaviour
# ---------------------------------------------------------------------
class TestAbstractProperties:
    @pytest.mark.parametrize(
        "name",
        [
            "freq_range",
            "main_freq",
            "simulation_box",
            "substrate_width_mm",
            "substrate_length_mm",
        ],
    )
    def test_base_class_properties_raise_not_implemented(self, name):
        """``SimParams`` cannot be instantiated directly (``__post_init__``
        reads ``main_freq``), so the properties are probed on the class."""
        with pytest.raises(NotImplementedError):
            getattr(SimParams, name).fget(object.__new__(SimParams))

    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(NotImplementedError):
            SimParams(
                substrate_eps_r=4.4,
                substrate_tand=0.001,
                substrate_thickness_mm=1.6,
            )

    def test_params_are_keyword_only(self):
        """``@dataclass(kw_only=True)`` -- positional construction must fail,
        otherwise a caller could silently swap eps_r and tand."""
        with pytest.raises(TypeError):
            SimParams(4.4, 0.001, 1.6)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
class TestDefaults:
    def test_documented_defaults(self, params):
        assert params.unit == 1e-3
        assert params.num_points == 1000
        assert params.substrate_cells == 4
        assert params.copper_thickness_mm == 0.035
        assert params.min_trace_width_mm == 0.1
        assert params.min_trace_spacing_mm == 0.089
        assert params.fp_precision == 3
        assert params.charac_imp == 50
        assert params.FDTD_timestep == 90000000
        assert params.FDTD_end_criteria == 1e-4
        assert params.FDTD_mesh_resolution_factor == 10
        assert params.FDTD_metal_mesh_resolution_factor == 40

    def test_derived_fields_are_not_constructor_arguments(self):
        """Fields declared ``field(init=False)`` must stay out of __init__, or
        a user could pass a kappa that ``__post_init__`` then overwrites."""
        init_fields = {f.name for f in dataclasses.fields(SimParams) if f.init}

        assert "substrate_kappa" not in init_fields
        assert "lambda0" not in init_fields
        assert "FDTD_mesh_resolution" not in init_fields
        assert "FDTD_metal_mesh_resolution" not in init_fields
        assert "FDTD_thirds_rule" not in init_fields
