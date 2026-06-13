"""
Auto-generated FDTD mesh from CSXCAD primitives.

Adapted from pyems mesh.py — uses geometric series for smooth grid
transitions and the thirds rule at metal boundaries.
"""

from enum import Enum
from bisect import bisect_left, bisect_right, insort_left

import numpy as np
import scipy.optimize

from CSXCAD import ContinuousStructure
from CSXCAD.CSPrimitives import CSPrimitives

from .sim_params import SimParams

PREC = 10


class Type(Enum):
    metal = 0
    nonmetal = 1
    air = 2


class BoundedType:
    def __init__(self, prop_type: Type, lower_bound: float, upper_bound: float) -> None:
        self.prop_type = prop_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_type(self) -> Type:
        return self.prop_type

    def get_bounds(self) -> list[float]:
        return [self.lower_bound, self.upper_bound]

    def get_midpoint(self) -> float:
        return np.average([self.lower_bound, self.upper_bound])

    def size(self) -> float:
        return self.upper_bound - self.lower_bound


def fp_nearest(val_or_arr: float) -> float:
    return np.around(val_or_arr, PREC)


def fp_equalp(val1: float, val2: float) -> bool:
    return np.around(val1, PREC) == np.around(val2, PREC)


def fp_gtp(val1: float, val2: float) -> bool:
    return np.around(val1, PREC) > np.around(val2, PREC)


def fp_gep(val1: float, val2: float) -> bool:
    return np.around(val1, PREC) >= np.around(val2, PREC)


def fp_ltp(val1: float, val2: float) -> bool:
    return np.around(val1, PREC) < np.around(val2, PREC)


def fp_lep(val1: float, val2: float) -> bool:
    return np.around(val1, PREC) <= np.around(val2, PREC)


def _prim_metalp(prim: CSPrimitives) -> bool:
    type_str = prim.GetProperty().GetTypeString()
    return type_str in ("Metal", "ConductingSheet", "LumpedElement")


def _prim_materialp(prim: CSPrimitives) -> bool:
    return prim.GetProperty().GetTypeString() == "Material"


def _get_prim_bounds(prim: CSPrimitives) -> np.ndarray:
    orig_bounds = prim.GetBoundBox()
    tr = prim.GetTransform()
    orig_bounds[0] = np.array(tr.Transform(orig_bounds[0]))
    orig_bounds[1] = np.array(tr.Transform(orig_bounds[1]))
    bounds = np.array([[None, None], [None, None], [None, None]])
    for i in range(3):
        lower = np.min([orig_bounds[0][i], orig_bounds[1][i]])
        upper = np.max([orig_bounds[0][i], orig_bounds[1][i]])
        bounds[i] = np.array([lower, upper])
    return bounds


def _is_linpoly(prim: CSPrimitives) -> bool:
    cls = prim.__class__.__name__
    return cls in ("CSPrimLinPoly", "CSPrimPolygon")


def _get_linpoly_vertex_bounds(prim: CSPrimitives) -> list[list[float]]:
    bounds: list[list[float]] = [[], [], []]
    try:
        coords = prim.GetCoords()
        x_verts, y_verts = coords[0], coords[1]
        elev = float(prim.GetElevation())
        norm_dir = int(prim.GetNormDir())
        tr = prim.GetTransform()
        for x, y in zip(x_verts, y_verts, strict=True):
            pt = [0.0, 0.0, 0.0]
            if norm_dir == 0:
                pt = [elev, float(x), float(y)]
            elif norm_dir == 1:
                pt = [float(x), elev, float(y)]
            else:
                pt = [float(x), float(y), elev]
            if tr is not None:
                pt = tr.Transform(pt)
            for d in range(3):
                bounds[d].append(float(pt[d]))
    except Exception:
        pass
    return bounds


def _physical_prims(prims: list[CSPrimitives]) -> list[CSPrimitives]:
    physical = []
    for prim in prims:
        if _prim_metalp(prim) or _prim_materialp(prim):
            physical.append(prim)
    return physical


def _remove_dups(lst: list, fixed: list | None = None) -> list:
    if fixed is None:
        fixed = []
    new_lst = []
    last = None
    for elt in lst:
        if last is not None:
            if elt == last or fp_equalp(elt, last) and elt not in fixed:
                continue
            elif fp_equalp(elt, last) and elt in fixed:
                del new_lst[-1]
        last = elt
        new_lst.append(elt)
    return new_lst


def _bounds_from_prims(
    prims: list[CSPrimitives], fixed: list[list[float]]
) -> list[list[float]]:
    dim_bounds = [[], [], []]
    for prim in prims:
        prim_bounds = _get_prim_bounds(prim)
        for dim, bounds in enumerate(prim_bounds):
            dim_bounds[dim].append(bounds[0])
            dim_bounds[dim].append(bounds[1])
    for dim, bounds in enumerate(dim_bounds):
        dim_bounds[dim] = sorted(bounds)
        dim_bounds[dim] = _remove_dups(dim_bounds[dim], fixed[dim])
    return dim_bounds


def _collect_all_bounds(
    prims: list[CSPrimitives], fixed: list[list[float]]
) -> list[list[float]]:
    dim_bounds: list[list[float]] = [[], [], []]
    for prim in prims:
        prim_bounds = _get_prim_bounds(prim)
        for dim, bounds in enumerate(prim_bounds):
            dim_bounds[dim].append(float(bounds[0]))
            dim_bounds[dim].append(float(bounds[1]))
        if _is_linpoly(prim):
            vert_bounds = _get_linpoly_vertex_bounds(prim)
            for dim in range(3):
                for v in vert_bounds[dim]:
                    dim_bounds[dim].append(v)
    for dim, bounds in enumerate(dim_bounds):
        dim_bounds[dim] = sorted(bounds)
        dim_bounds[dim] = _remove_dups(dim_bounds[dim], fixed[dim])
    return dim_bounds


def _float_inside(val: float, lower: float, upper: float) -> bool:
    return lower <= val <= upper


def _pos_in_bounds(pos: float, lower: float, upper: float) -> bool:
    return fp_gep(pos, lower) and fp_lep(pos, upper)


def _type_at_pos(prims: list[CSPrimitives], dim: int, pos: float) -> Type | None:
    smallest_dim = np.inf
    current_type = None
    for prim in prims:
        prim_bounds = _get_prim_bounds(prim)
        if _float_inside(pos, prim_bounds[dim][0], prim_bounds[dim][1]):
            dim_size = prim_bounds[dim][1] - prim_bounds[dim][0]
            if np.isclose(dim_size, smallest_dim, rtol=1e-3, atol=0):
                if _prim_metalp(prim):
                    current_type = Type.metal
                    smallest_dim = dim_size
            elif dim_size < smallest_dim:
                smallest_dim = dim_size
                current_type = Type.metal if _prim_metalp(prim) else Type.nonmetal
    return current_type


def _sort_bounded_types(
    bounded_types: list[list[BoundedType]],
) -> list[list[BoundedType]]:
    new_bounded_types = [[], [], []]
    for dim, btype_list in enumerate(bounded_types):
        new_bounded_types[dim] = sorted(btype_list, key=lambda x: x.size())
    return new_bounded_types


def _factor_for_num(num: int, smaller_spacing: float, dist: float) -> float:
    lo = 1.0001
    if _geom_dist_zero(lo, num, smaller_spacing, dist) >= 0:
        return lo
    hi = max(2.0, dist / (smaller_spacing * max(1, num - 1)))
    while _geom_dist_zero(hi, num, smaller_spacing, dist) <= 0:
        hi *= 2
        if hi > 1e6:
            return hi
    try:
        result = scipy.optimize.root_scalar(
            _geom_dist_zero,
            args=(num, smaller_spacing, dist),
            bracket=(lo, hi),
            method="brentq",
        )
        return result.root
    except (ValueError, RuntimeError):
        return 1.5


def _factor_ubound(num: int, ratio: float, max_factor: float) -> float:
    return np.min([max_factor, np.power(ratio, 1 / (num - 1))])


def _geom_dist(factor: float, num: int, smaller_spacing: float) -> float:
    powers = np.arange(1, num, 1)
    return smaller_spacing * np.sum(np.power(factor, powers))


def _geom_dist_zero(
    factor: float, num: int, smaller_spacing: float, dist: float
) -> float:
    return _geom_dist(factor, num, smaller_spacing) - dist


def _num_for_factor(
    factor: float, smaller_spacing: float, dist: float
) -> tuple[float, int]:
    dist = float(np.asarray(dist).flat[0])
    num = int(
        np.ceil(
            np.log(1 - (((dist / smaller_spacing) + 1) * (1 - factor))) / np.log(factor)
            + 1
        )
    )
    factor = _factor_for_num(num, smaller_spacing, dist)
    while factor < 1:
        num -= 1
        if num == 0:
            raise RuntimeError("_num_for_factor failed. This is a bug.")
        factor = _factor_for_num(num, smaller_spacing, dist)
    return (factor, num)


def _geom_series(
    smaller_spacing: float,
    larger_spacing: float,
    dist: float,
    min_num: int,
    max_factor: float,
) -> tuple[float, int]:
    num = np.max([int(np.ceil(dist / larger_spacing)) + 1, min_num])
    factor = _factor_for_num(num, smaller_spacing, dist)
    while factor >= _factor_ubound(num, larger_spacing / smaller_spacing, max_factor):
        num += 1
        factor = _factor_for_num(num, smaller_spacing, dist)
    return (factor, num)


def _lines_const_factor_in_bounds(
    lower: float,
    upper: float,
    lower_spacing: float,
    upper_spacing: float,
    dim: int,
    min_lines: int,
    smooth: float,
) -> np.ndarray:
    if np.isclose(lower_spacing, upper_spacing, rtol=1e-3, atol=0):
        num_lines = int(np.ceil((upper - lower) / lower_spacing)) + 1
        num_lines = int(np.max([num_lines, min_lines]))
        return np.linspace(lower, upper, num_lines)

    (factor, num_lines) = _geom_series(
        smaller_spacing=np.min([lower_spacing, upper_spacing]),
        larger_spacing=np.max([lower_spacing, upper_spacing]),
        dist=upper - lower,
        min_num=min_lines,
        max_factor=smooth,
    )

    powers = np.arange(1, num_lines, 1)
    if lower_spacing < upper_spacing:
        spacings = lower_spacing * np.power(factor, powers)
        lines = np.array(lower + np.cumsum(spacings))
        lines = np.concatenate(([lower], lines))
    else:
        spacings = upper_spacing * np.power(factor, powers)
        lines = np.array(upper - np.cumsum(spacings))
        lines = np.concatenate(([upper], lines))
        lines = np.flip(lines)

    lines[-1] = upper
    return lines


def _spacing_at_dist(spacing: float, dist: float, max_factor: float) -> float:
    factor, num = _num_for_factor(max_factor, spacing, dist)
    return spacing * (factor ** (num - 1))


def _spacings_at_dist_zero(
    dist: float,
    lower_spacing: float,
    upper_spacing: float,
    total_dist: float,
    max_factor: float,
) -> float:
    spacing1 = _spacing_at_dist(lower_spacing, dist, max_factor)
    spacing2 = _spacing_at_dist(upper_spacing, total_dist - dist, max_factor)
    return spacing2 - spacing1


def _dist_for_max_spacings(
    lower_spacing: float, upper_spacing: float, dist: float, max_factor: float
) -> float:
    try:
        result = scipy.optimize.root_scalar(
            _spacings_at_dist_zero,
            args=(lower_spacing, upper_spacing, dist, max_factor),
            bracket=(dist * 1e-6, dist * (1 - 1e-6)),
            method="brentq",
        )
        return result.root
    except (ValueError, RuntimeError):
        return dist / 2


def _dim_idx_to_desc(idx: int) -> str:
    if idx == 0:
        return "xmin"
    if idx == 1:
        return "xmax"
    if idx == 2:
        return "ymin"
    if idx == 3:
        return "ymax"
    if idx == 4:
        return "zmin"
    if idx == 5:
        return "zmax"
    raise ValueError("Index must be between 0 and 5, inclusive.")


def _mesh_lines_in_box(
    mesh_lines: list[list[float]], box_lower: list[float], box_upper: list[float]
) -> list[list[float]]:
    mesh_lines_inside = []
    for dim in range(3):
        lower_pos = box_lower[dim]
        upper_pos = box_upper[dim]
        dim_lines = mesh_lines[dim]
        dim_lines_inside = dim_lines[
            bisect_left(dim_lines, lower_pos) : bisect_right(dim_lines, upper_pos)
        ]
        mesh_lines_inside.append(dim_lines_inside)
    return mesh_lines_inside


class Mesh:
    """Auto-generates an FDTD mesh from CSXCAD primitives.

    Adapts the pyems automatic mesh generation algorithm.  Scans all
    physical primitives (metal and material), classifies regions by
    type, generates mesh lines using geometric series for smooth
    transitions, and applies the thirds rule at metal boundaries.

    Parameters
    ----------
    csx : ContinuousStructure
        CSXCAD structure with primitives already added.
    params : SimParams
        Simulation parameters providing simulation_box, mesh_resolution,
        metal_mesh_resolution, unit, and main_freq.
    smooth_ratio : float
        Maximum ratio between adjacent cells (default 1.5).
    """

    def __init__(
        self,
        csx: ContinuousStructure,
        params: SimParams,
        smooth_ratio: float = 1.5,
        min_lines: int = 5,
    ) -> None:
        self._csx = csx
        self._mesh_res = float(params.mesh_resolution)
        self._metal_res = float(params.metal_mesh_resolution)
        self._smooth = (smooth_ratio, smooth_ratio, smooth_ratio)
        self._unit = float(params.unit)
        self._lambda0 = float(params.lambda0)
        self._min_lines = min_lines
        sb = params.simulation_box
        self._sim_box = tuple((float(-s / 2), float(s / 2)) for s in sb)
        self.sim_bounds = [[], [], []]
        self.ranges_meshed = [[], [], []]
        self.metal_bounds = [[], [], []]
        self.fixed_lines = [[], [], []]
        self.smallest_res = self._metal_res
        self.mesh_lines = [[], [], []]
        self.mesh = self._csx.GetGrid()
        self._generate()

    def _generate(self) -> None:
        prims = self._csx.GetAllPrimitives()
        physical_prims = _physical_prims(prims)
        self._set_fixed_lines(physical_prims)
        bounds = _collect_all_bounds(physical_prims, self.fixed_lines)
        self._set_sim_bounds_from_geometry(bounds)
        bounded_types = self._bounded_types(bounds, physical_prims)
        bounded_types = self._set_expanded_bounds(bounded_types)
        self.bounded_types = bounded_types
        self._set_metal_bounds(bounded_types)
        size_ordered = _sort_bounded_types(bounded_types)
        self._gen_mesh_for_bounded_types(size_ordered)
        self._set_mesh_from_lines()
        self.mesh.SmoothMeshLines("all", self._mesh_res, self._smooth[0])

    def _set_fixed_lines(self, prims: list[CSPrimitives]) -> None:
        for prim in prims:
            prim_bounds = _get_prim_bounds(prim)
            for dim in range(3):
                if fp_equalp(prim_bounds[dim][0], prim_bounds[dim][1]):
                    self.add_fixed_line(dim, fp_nearest(prim_bounds[dim][0]))
                self.fixed_lines[dim].sort()
                self.fixed_lines[dim] = _remove_dups(self.fixed_lines[dim])

    def add_fixed_line(self, dim: int, pos: float) -> None:
        self.fixed_lines[dim].append(pos)
        self.fixed_lines[dim].sort()

    def _bounded_types(
        self, bounds: list[list[float]], prims: list[CSPrimitives]
    ) -> list[list[BoundedType]]:
        bounded_types = [[], [], []]
        for dim, dim_bounds in enumerate(bounds):
            last_bound = None
            for bound in dim_bounds:
                if bound in self.fixed_lines[dim]:
                    if last_bound is not None:
                        mid_pos = np.average([last_bound, bound])
                        prop_type = _type_at_pos(prims, dim, mid_pos)
                        btype = BoundedType(prop_type, last_bound, bound)
                        bounded_types[dim].append(btype)
                    prop_type = _type_at_pos(prims, dim, bound)
                    btype = BoundedType(prop_type, bound, bound)
                    bounded_types[dim].append(btype)
                elif last_bound is not None:
                    mid_pos = np.average([last_bound, bound])
                    prop_type = _type_at_pos(prims, dim, mid_pos)
                    btype = BoundedType(prop_type, last_bound, bound)
                    bounded_types[dim].append(btype)
                last_bound = bound
        return bounded_types

    def _set_sim_bounds_from_geometry(self, dim_bounds: list[list[float]]) -> None:
        new_sim_box = []
        for dim in range(3):
            if not dim_bounds[dim]:
                new_sim_box.append(self._sim_box[dim])
                continue
            geo_min = dim_bounds[dim][0]
            geo_max = dim_bounds[dim][-1]
            span = geo_max - geo_min
            padding = max(self._lambda0 / 2, span * 0.15)
            new_sim_box.append((geo_min - padding, geo_max + padding))
        self._sim_box = tuple(new_sim_box)

    def _set_expanded_bounds(
        self, bounded_types: list[list[BoundedType]]
    ) -> list[list[BoundedType]]:
        for dim in range(3):
            if not bounded_types[dim]:
                btype = BoundedType(
                    Type.air, self._sim_box[dim][0], self._sim_box[dim][1]
                )
                bounded_types[dim].append(btype)
                continue
            existing_lower = bounded_types[dim][0].get_bounds()[0]
            existing_upper = bounded_types[dim][-1].get_bounds()[1]
            sim_lower = self._sim_box[dim][0]
            sim_upper = self._sim_box[dim][1]
            if fp_gtp(sim_lower, existing_lower) or fp_ltp(sim_upper, existing_upper):
                raise ValueError(
                    "Simulation box too small for structures in "
                    f"dimension {dim}. Need [{existing_lower}, {existing_upper}] "
                    f"but have [{sim_lower}, {sim_upper}]."
                )
            if not fp_equalp(sim_lower, existing_lower):
                btype = BoundedType(Type.air, sim_lower, existing_lower)
                bounded_types[dim].insert(0, btype)
            if not fp_equalp(sim_upper, existing_upper):
                btype = BoundedType(Type.air, existing_upper, sim_upper)
                bounded_types[dim].append(btype)
        for dim in range(3):
            self.sim_bounds[dim] = [
                bounded_types[dim][0].get_bounds()[0],
                bounded_types[dim][-1].get_bounds()[1],
            ]
        return bounded_types

    def _set_metal_bounds(self, bounded_types: list[list[BoundedType]]) -> None:
        for dim, btypes in enumerate(bounded_types):
            for btype in btypes:
                if btype.get_type() == Type.metal:
                    bounds = btype.get_bounds()
                    if not fp_equalp(bounds[0], bounds[1]):
                        self._add_metal_bound(dim, bounds[0])
                        self._add_metal_bound(dim, bounds[1])
            self.metal_bounds[dim] = _remove_dups(
                self.metal_bounds[dim], self.fixed_lines[dim]
            )

    def _add_metal_bound(self, dim: int, pos: float) -> None:
        insort_left(self.metal_bounds[dim], pos)

    def _is_fixed_line(self, dim: int, pos: float) -> bool:
        return any(fp_equalp(fp, pos) for fp in self.fixed_lines[dim])

    def _is_metal_bound(self, dim: int, pos: float) -> bool:
        return any(fp_equalp(mb, pos) for mb in self.metal_bounds[dim])

    def _pos_meshed(self, dim: int, pos: float) -> bool:
        for rng in self.ranges_meshed[dim]:
            if _pos_in_bounds(pos, rng[0], rng[1]):
                return True
        return False

    def _type_below(self, dim: int, upper: float) -> Type | None:
        for btype in self.bounded_types[dim]:
            if fp_equalp(btype.get_bounds()[1], upper):
                return btype.get_type()
        return None

    def _type_below_meshed(self, dim: int, lower: float) -> bool:
        for btype in self.bounded_types[dim]:
            if fp_equalp(btype.get_bounds()[1], lower) and btype.size() != 0:
                return self._pos_meshed(dim, btype.get_midpoint())
        return False

    def _type_above(self, dim: int, lower: float) -> Type | None:
        for btype in self.bounded_types[dim]:
            if fp_equalp(btype.get_bounds()[0], lower):
                return btype.get_type()
        return None

    def _type_above_meshed(self, dim: int, upper: float) -> bool:
        for btype in self.bounded_types[dim]:
            if fp_equalp(btype.get_bounds()[0], upper) and btype.size() != 0:
                return self._pos_meshed(dim, btype.get_midpoint())
        return False

    def _add_to_ranges_meshed(self, dim: int, lower: float, upper: float) -> None:
        self.ranges_meshed[dim].append([lower, upper])

    def _gen_mesh_for_bounded_types(
        self, bounded_types: list[list[BoundedType]]
    ) -> None:
        for dim, btypes in enumerate(bounded_types):
            for btype in btypes:
                lower = btype.get_bounds()[0]
                upper = btype.get_bounds()[1]
                is_metal = btype.get_type() == Type.metal
                _, line_below = self._line_below(dim, lower)
                _, line_above = self._line_above(dim, upper)
                self._gen_mesh_in_bounds(
                    dim, lower, upper, line_below, line_above, is_metal
                )
                self._add_to_ranges_meshed(dim, lower, upper)

    def _min_spacing(self, dist: float) -> float:
        return dist / (self._min_lines - 1)

    def _lower_spacing(
        self,
        dim: int,
        lower: float,
        line_below: float | None,
        dist: float,
        is_metal: bool,
    ) -> float:
        lower_spacing = self._metal_res if is_metal else self._mesh_res
        lower_spacing = np.min([lower_spacing, self._min_spacing(dist)])
        if line_below is not None and self._type_below_meshed(dim, lower):
            factor = 1.0
            if self._is_metal_bound(dim, lower) and not self._is_fixed_line(dim, lower):
                factor = 1.5 if self._type_below(dim, lower) == Type.nonmetal else 3.0
            spacing = factor * (lower - line_below)
            lower_spacing = np.min([lower_spacing, spacing])
            if dim == 2:
                lower_spacing = np.max([lower_spacing, self._mesh_res / 5])
        return lower_spacing

    def _upper_spacing(
        self,
        dim: int,
        upper: float,
        line_above: float | None,
        dist: float,
        is_metal: bool,
    ) -> float:
        upper_spacing = self._metal_res if is_metal else self._mesh_res
        upper_spacing = np.min([upper_spacing, self._min_spacing(dist)])
        if line_above is not None and self._type_above_meshed(dim, upper):
            factor = 1.0
            if self._is_metal_bound(dim, upper) and not self._is_fixed_line(dim, upper):
                factor = 1.5 if self._type_above(dim, upper) == Type.nonmetal else 3.0
            spacing = factor * (line_above - upper)
            upper_spacing = np.min([upper_spacing, spacing])
            if dim == 2:
                upper_spacing = np.max([upper_spacing, self._mesh_res / 5])
        return upper_spacing

    def _gen_mesh_in_bounds(
        self,
        dim: int,
        lower: float,
        upper: float,
        line_below: float | None,
        line_above: float | None,
        is_metal: bool,
    ) -> None:
        dist = upper - lower
        lower_spacing = self._lower_spacing(dim, lower, line_below, dist, is_metal)
        upper_spacing = self._upper_spacing(dim, upper, line_above, dist, is_metal)
        if is_metal:
            below_type = self._type_below(dim, lower)
            above_type = self._type_above(dim, upper)
            touches_air = (
                below_type is None
                or below_type == Type.air
                or above_type is None
                or above_type == Type.air
            )
            max_spacing = self._mesh_res if touches_air else self._metal_res
        else:
            max_spacing = self._mesh_res

        if fp_equalp(lower, upper):
            self._add_lines_to_mesh([lower], dim)
        elif is_metal and dim == 2 and dist < self._metal_res / 4.0:
            mid = fp_nearest((lower + upper) / 2.0)
            self._add_lines_to_mesh(np.array([mid]), dim)
        else:
            lines = self._gen_lines_in_bounds(
                lower, upper, lower_spacing, upper_spacing, max_spacing, dim
            )

            if is_metal:
                first_spacing = lines[1] - lines[0]
                last_spacing = lines[-1] - lines[-2]
                if not fp_equalp(
                    lower, self.sim_bounds[dim][0]
                ) and not self._is_fixed_line(dim, lower):
                    if (
                        self._pos_meshed(dim, lower)
                        and self._type_below(dim, lower) == Type.metal
                    ):
                        adj = 2 * first_spacing / 3
                    else:
                        adj = first_spacing / 3
                    lower += adj
                if not fp_equalp(
                    upper, self.sim_bounds[dim][1]
                ) and not self._is_fixed_line(dim, upper):
                    if (
                        self._pos_meshed(dim, upper)
                        and self._type_above(dim, upper) == Type.metal
                    ):
                        adj = 2 * last_spacing / 3
                    else:
                        adj = last_spacing / 3
                    upper -= adj
                lines = self._gen_lines_in_bounds(
                    lower, upper, lower_spacing, upper_spacing, max_spacing, dim
                )
            else:
                rebuild_lines = False
                if self._is_metal_bound(dim, lower):
                    rebuild_lines = True
                    first_spacing = lines[1] - lines[0]
                    spacing = np.min([first_spacing, self._metal_res])
                    lower += 2 * spacing / 3
                if self._is_metal_bound(dim, upper):
                    rebuild_lines = True
                    last_spacing = lines[-1] - lines[-2]
                    spacing = np.min([last_spacing, self._metal_res])
                    upper -= 2 * spacing / 3
                if rebuild_lines:
                    lines = self._gen_lines_in_bounds(
                        lower, upper, lower_spacing, upper_spacing, max_spacing, dim
                    )

            self._add_lines_to_mesh(lines, dim)

    def _gen_lines_in_bounds(
        self,
        lower: float,
        upper: float,
        lower_spacing: float,
        upper_spacing: float,
        max_spacing: float,
        dim: int,
    ) -> np.ndarray:
        dist = upper - lower
        smaller_spacing = np.min([lower_spacing, upper_spacing])
        larger_spacing = np.max([lower_spacing, upper_spacing])
        num_lower = dist / larger_spacing

        if (
            num_lower < self._min_lines
            or _spacing_at_dist(smaller_spacing, dist, self._smooth[dim])
            < larger_spacing
        ):
            return _lines_const_factor_in_bounds(
                lower,
                upper,
                lower_spacing,
                upper_spacing,
                dim,
                self._min_lines,
                self._smooth[dim],
            )

        mid_spacing_dist = _dist_for_max_spacings(
            lower_spacing, upper_spacing, dist, self._smooth[dim]
        )
        midpt = lower + mid_spacing_dist
        lower_factor, lower_num = _num_for_factor(
            self._smooth[dim], lower_spacing, midpt - lower
        )
        upper_factor, upper_num = _num_for_factor(
            self._smooth[dim], upper_spacing, upper - midpt
        )
        mid_spacing = np.min(
            [
                max_spacing,
                lower_spacing * (lower_factor**lower_num),
                upper_spacing * (upper_factor**upper_num),
            ]
        )

        while lower_num + upper_num < self._min_lines:
            lower_num += 1
            upper_num += 1

        lines_lower = _lines_const_factor_in_bounds(
            lower, midpt, lower_spacing, mid_spacing, dim, lower_num, self._smooth[dim]
        )
        lines_upper = _lines_const_factor_in_bounds(
            midpt, upper, mid_spacing, upper_spacing, dim, upper_num, self._smooth[dim]
        )
        lines = np.concatenate([lines_lower, lines_upper])
        return _remove_dups(lines, self.fixed_lines[dim])

    def _add_lines_to_mesh(self, lines: np.ndarray, dim: int) -> None:
        for line in lines:
            self._add_mesh_line(dim, line)
        self.mesh_lines[dim] = _remove_dups(self.mesh_lines[dim], self.fixed_lines[dim])

    def _add_mesh_line(self, dim: int, pos: float) -> None:
        insort_left(self.mesh_lines[dim], pos)

    def _set_mesh_from_lines(self) -> None:
        grid = self._csx.GetGrid()
        grid.SetDeltaUnit(self._unit)
        for i in range(3):
            grid.ClearLines(i)
        for dim in range(3):
            for line in self.mesh_lines[dim]:
                ch = "xyz"[dim]
                grid.AddLine(ch, fp_nearest(line))

    def nearest_mesh_line(
        self, dim: int, pos: float
    ) -> tuple[int | None, float | None]:
        lines = self.mesh_lines[dim]
        if not lines:
            return (None, None)
        bisect_pos = bisect_left(self.mesh_lines[dim], pos)
        if bisect_pos == 0:
            return (0, lines[0])
        elif bisect_pos == len(lines):
            return (bisect_pos - 1, lines[bisect_pos - 1])
        else:
            lower = lines[bisect_pos - 1]
            upper = lines[bisect_pos]
            if pos - lower < upper - pos:
                return (bisect_pos - 1, lower)
            else:
                return (bisect_pos, upper)

    def _line_below(self, dim: int, pos: float) -> tuple[int | None, float | None]:
        idx, act_pos = self.nearest_mesh_line(dim, pos)
        if act_pos is None:
            return (None, None)
        if fp_equalp(act_pos, pos):
            idx -= 1
            if self._mesh_valid_index(dim, idx):
                act_pos = self.mesh_lines[dim][idx]
        if fp_ltp(act_pos, pos):
            return (idx, act_pos)
        idx -= 1
        if self._mesh_valid_index(dim, idx):
            act_pos = self.mesh_lines[dim][idx]
            return (idx, act_pos)
        return (None, None)

    def _line_above(self, dim: int, pos: float) -> tuple[int | None, float | None]:
        idx, act_pos = self.nearest_mesh_line(dim, pos)
        if act_pos is None:
            return (None, None)
        if fp_equalp(act_pos, pos):
            idx += 1
            if self._mesh_valid_index(dim, idx):
                act_pos = self.mesh_lines[dim][idx]
        if fp_gtp(act_pos, pos):
            return (idx, act_pos)
        idx += 1
        if self._mesh_valid_index(dim, idx):
            act_pos = self.mesh_lines[dim][idx]
            return (idx, act_pos)
        return (None, None)

    def _mesh_valid_index(self, dim: int, index: int) -> bool:
        return 0 <= index < len(self.mesh_lines[dim])
