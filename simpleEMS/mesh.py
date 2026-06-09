"""
Simple FDTD mesh generator from CSXCAD primitives.

Algorithm:
  1. Discover geometry from all primitives (incl. LinPolygon vertices)
  2. Classify regions between boundaries as metal/nonmetal/air
  3. Generate mesh lines per region using np.linspace
  4. Apply thirds rule at metal boundaries (mark as fixed)
  5. Smooth between fixed lines via per-segment SmoothMeshLines
"""

from enum import Enum
from bisect import bisect_left, insort_left

import numpy as np

from CSXCAD import ContinuousStructure
from CSXCAD.CSPrimitives import CSPrimitives

from .sim_params import SimParams

PREC = 10


class Type(Enum):
    """Material type classification for mesh regions."""

    metal = 0
    nonmetal = 1
    air = 2


class BoundedType:
    """A region bounded by lower and upper bounds with a material type."""

    def __init__(
        self,
        prop_type: Type,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
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
    return type_str in (
        "Metal",
        "ConductingSheet",
        "LumpedElement",
        "Excitation",
    )


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
    """Extract individual vertex positions from a LinPolygon."""
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
            if elt == last or (fp_equalp(elt, last) and elt not in fixed):
                continue
            if fp_equalp(elt, last) and elt in fixed:
                del new_lst[-1]
        last = elt
        new_lst.append(elt)
    return new_lst


def _collect_all_bounds(
    prims: list[CSPrimitives], fixed: list[list[float]]
) -> list[list[float]]:
    """Collect boundary positions from all primitives, including LinPoly vertices."""
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
    new_bounded_types: list[list[BoundedType]] = [[], [], []]
    for dim, btype_list in enumerate(bounded_types):
        new_bounded_types[dim] = sorted(btype_list, key=lambda x: x.size())
    return new_bounded_types


class Mesh:
    """Auto-generates an FDTD mesh from CSXCAD primitives.

    Scans all physical primitives (metal + material), classifies regions
    by type, generates mesh lines using ``np.linspace``, applies the
    thirds rule at metal boundaries, and smooths between fixed lines.

    Parameters
    ----------
    csx : ContinuousStructure
        CSXCAD structure with primitives already added.
    params : SimParams
        Simulation parameters providing *simulation_box*, *mesh_resolution*,
        *metal_mesh_resolution*, *unit*, and *main_freq*.
    smooth_ratio : float
        Maximum ratio between adjacent cells (default 1.5).
    min_lines : int
        Minimum mesh lines per region (default 5).
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
        self._substrate_cells = int(params.substrate_cells)
        sb = params.simulation_box
        self._sim_box = tuple((float(-s / 2), float(s / 2)) for s in sb)
        self.sim_bounds: list[list[float]] = [[], [], []]
        self.ranges_meshed: list[list[list[float]]] = [[], [], []]
        self.metal_bounds: list[list[float]] = [[], [], []]
        self.fixed_lines: list[list[float]] = [[], [], []]
        self.smallest_res = self._metal_res
        self.mesh_lines: list[list[float]] = [[], [], []]
        self.mesh = self._csx.GetGrid()
        self._generate()

    # -----------------------------------------------------------------
    #  Main generation pipeline
    # -----------------------------------------------------------------

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
        self._smooth_non_fixed_segments()
        self._merge_close_lines()
        self._set_mesh_from_lines()

    # -----------------------------------------------------------------
    #  Fixed lines (zero-thickness primitives)
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    #  Region classification
    # -----------------------------------------------------------------

    def _bounded_types(
        self, bounds: list[list[float]], prims: list[CSPrimitives]
    ) -> list[list[BoundedType]]:
        bounded_types: list[list[BoundedType]] = [[], [], []]
        for dim, dim_bounds in enumerate(bounds):
            last_bound = None
            for bound in dim_bounds:
                if last_bound is not None:
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

    # -----------------------------------------------------------------
    #  Metal boundaries
    # -----------------------------------------------------------------

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

    def _metal_thickness_at(self, dim: int, pos: float) -> float | None:
        for bt in self.bounded_types[dim]:
            if bt.get_type() != Type.metal:
                continue
            lower, upper = bt.get_bounds()
            if fp_equalp(lower, pos) or fp_equalp(upper, pos):
                return upper - lower
        return None

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

    # -----------------------------------------------------------------
    #  Mesh generation per bounded type  (np.linspace based)
    # -----------------------------------------------------------------

    def _gen_mesh_for_bounded_types(
        self, bounded_types: list[list[BoundedType]]
    ) -> None:
        for dim, btypes in enumerate(bounded_types):
            for btype in btypes:
                lower = btype.get_bounds()[0]
                upper = btype.get_bounds()[1]
                _, line_below = self._line_below(dim, lower)
                _, line_above = self._line_above(dim, upper)
                self._gen_mesh_in_bounds(
                    dim, lower, upper, line_below, line_above, btype
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
        return upper_spacing

    def _gen_mesh_in_bounds(
        self,
        dim: int,
        lower: float,
        upper: float,
        line_below: float | None,
        line_above: float | None,
        btype: BoundedType,
    ) -> None:
        dist = upper - lower
        rtype = btype.get_type()
        is_metal = rtype == Type.metal
        is_air = rtype is None or rtype == Type.air

        if fp_equalp(lower, upper):
            self._add_lines_to_mesh([lower], dim)
            return

        # --- fixed offsets for thirds rule (based on mesh_res, not local spacing) ---
        offset_in = self._metal_res / 12.0  # 1/3 inside metal
        offset_out = self._metal_res / 6.0  # 2/3 outside metal
        thirds_cell = offset_in + offset_out  # metal_res / 4

        # --- skip thirds rule for metal regions thinner than the boundary cell ---
        skip_thirds = is_metal and dist < thirds_cell and dim == 2

        # --- number of points based on region size ---
        if dist < (2 * thirds_cell if dim == 1 else thirds_cell):
            if is_metal and dim != 2:
                ext_lower = lower - offset_out
                ext_upper = upper + offset_out
                total_span = ext_upper - ext_lower
                num = max(4, int(np.ceil(total_span / (offset_out * 0.6))) + 1)
                lines = np.linspace(ext_lower, ext_upper, num)
                lines = np.array(
                    [
                        line
                        for line in lines
                        if not (fp_equalp(line, lower) or fp_equalp(line, upper))
                    ]
                )
                self._add_lines_to_mesh(lines, dim)
                return
            else:
                num = 2
        elif is_air:
            air_spacing = max(
                self._mesh_res * 3,
                (self.sim_bounds[dim][1] - self.sim_bounds[dim][0]) / 10,
            )
            num = max(int(np.ceil(dist / air_spacing)) + 1, self._min_lines)
        elif is_metal:
            num = max(self._min_lines, int(np.ceil(dist / (self._metal_res * 5))) + 1)
        else:
            # nonmetal regions like substrate: finer initial density
            num = max(self._min_lines, int(np.ceil(dist / (self._mesh_res / 4))) + 1)
            if dim == 2:
                num = max(self._substrate_cells + 1, num)

        lines = np.linspace(lower, upper, num)
        if skip_thirds:
            mid = fp_nearest((lower + upper) / 2.0)
            self._add_lines_to_mesh(np.array([mid]), dim)
            return

        orig_lower = lower
        orig_upper = upper

        # --- thirds rule for metal regions ---
        if is_metal:
            # scale offsets for thin metals (dist < thirds_cell)
            scale = min(1.0, dist / thirds_cell)
            adj_in = offset_in * scale
            adj_out = offset_out * scale

            adj_lower = 0.0
            adj_upper = 0.0
            if not fp_equalp(
                lower, self.sim_bounds[dim][0]
            ) and not self._is_fixed_line(dim, lower):
                adj_lower = adj_in
                if (
                    self._pos_meshed(dim, lower)
                    and self._type_below(dim, lower) == Type.metal
                ):
                    adj_lower = adj_out
                lower += adj_lower
            if not fp_equalp(
                upper, self.sim_bounds[dim][1]
            ) and not self._is_fixed_line(dim, upper):
                adj_upper = adj_in
                if (
                    self._pos_meshed(dim, upper)
                    and self._type_above(dim, upper) == Type.metal
                ):
                    adj_upper = adj_out
                upper -= adj_upper

            if lower != orig_lower or upper != orig_upper:
                new_dist = upper - lower
                if new_dist > 0:
                    lines = np.linspace(lower, upper, max(num, 2))

            # add 2/3-outside lines in adjacent region
            # only for metal-air boundaries; skip for metal-metal where the
            # adjacent metal already provides mesh lines
            is_lower_metal = (
                self._pos_meshed(dim, orig_lower)
                and self._type_below(dim, orig_lower) == Type.metal
            )
            if adj_lower > 0 and not is_lower_metal:
                ol = fp_nearest(orig_lower - offset_out)
                if fp_gep(ol, self.sim_bounds[dim][0]):
                    self.add_fixed_line(dim, ol)
                    self._add_mesh_line(dim, ol)
            is_upper_metal = (
                self._pos_meshed(dim, orig_upper)
                and self._type_above(dim, orig_upper) == Type.metal
            )
            if adj_upper > 0 and not is_upper_metal:
                ol = fp_nearest(orig_upper + offset_out)
                if fp_lep(ol, self.sim_bounds[dim][1]):
                    self.add_fixed_line(dim, ol)
                    self._add_mesh_line(dim, ol)

            # mark adjusted boundary lines as fixed
            if adj_lower > 0 and len(lines) > 0:
                self.add_fixed_line(dim, fp_nearest(lines[0]))
            if adj_upper > 0 and len(lines) > 0:
                self.add_fixed_line(dim, fp_nearest(lines[-1]))

        # --- thirds rule for nonmetal regions adjacent to metal ---
        else:
            if self._is_metal_bound(dim, lower):
                metal_t = self._metal_thickness_at(dim, lower)
                if metal_t is not None and metal_t < thirds_cell:
                    offset_adj = min(offset_out, max(offset_in, metal_t * 3))
                else:
                    offset_adj = offset_out
                lower += offset_adj
            if self._is_metal_bound(dim, upper):
                metal_t = self._metal_thickness_at(dim, upper)
                if metal_t is not None and metal_t < thirds_cell:
                    offset_adj = min(offset_out, max(offset_in, metal_t * 3))
                else:
                    offset_adj = offset_out
                upper -= offset_adj
            if lower != orig_lower or upper != orig_upper:
                new_dist = upper - lower
                if new_dist > 0:
                    lines = np.linspace(lower, upper, max(num, 2))

        self._add_lines_to_mesh(lines, dim)

    # -----------------------------------------------------------------
    #  Smoothing  (preserves fixed lines)
    # -----------------------------------------------------------------

    def _smooth_non_fixed_segments(self) -> None:
        """Smooth mesh globally using CSXCAD's ``SmoothMeshLines``.

        ``SmoothMeshLines`` preserves all input points (including thirds-rule
        lines) and only adds intermediate points where adjacent cell ratios
        exceed the given limit.  Global smoothing ensures a smooth transition
        from coarse air regions to fine metal-edge regions.
        """
        from CSXCAD.SmoothMeshLines import SmoothMeshLines

        for dim in range(3):
            all_lines = sorted(set(self.mesh_lines[dim]))
            if len(all_lines) < 3:
                continue
            try:
                smoothed = SmoothMeshLines(all_lines, self._mesh_res, self._smooth[dim])
                self.mesh_lines[dim] = sorted(set(smoothed))
            except Exception:
                pass

    def _merge_close_lines(self) -> None:
        """Merge mesh lines closer than a threshold after smoothing.

        SmoothMeshLines can insert interpolation points that land extremely close
        to primitive boundary lines, creating tiny Yee cells that unnecessarily
        constrain the FDTD timestep.  This method merges any pair of lines
        closer than ``mesh_resolution / 100``, preferring to keep fixed lines
        (from zero-thickness primitives) when one of the pair is fixed.
        """
        threshold = max(self._mesh_res, self._metal_res) / 100.0
        for dim in range(3):
            lines = sorted(set(self.mesh_lines[dim]))
            if len(lines) < 2:
                continue
            cleaned: list[float] = [lines[0]]
            for i in range(1, len(lines)):
                gap = lines[i] - cleaned[-1]
                if gap >= threshold:
                    cleaned.append(lines[i])
                    continue
                prev_fixed = self._is_fixed_line(dim, cleaned[-1])
                curr_fixed = self._is_fixed_line(dim, lines[i])
                if prev_fixed and not curr_fixed:
                    pass
                elif curr_fixed and not prev_fixed:
                    cleaned[-1] = lines[i]
                else:
                    cleaned[-1] = (cleaned[-1] + lines[i]) * 0.5
            self.mesh_lines[dim] = cleaned

    # -----------------------------------------------------------------
    #  Line management
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    #  Query helpers
    # -----------------------------------------------------------------

    def nearest_mesh_line(
        self, dim: int, pos: float
    ) -> tuple[int | None, float | None]:
        lines = self.mesh_lines[dim]
        if not lines:
            return (None, None)
        bp = bisect_left(self.mesh_lines[dim], pos)
        if bp == 0:
            return (0, lines[0])
        if bp == len(lines):
            return (bp - 1, lines[bp - 1])
        lower = lines[bp - 1]
        upper = lines[bp]
        if pos - lower < upper - pos:
            return (bp - 1, lower)
        return (bp, upper)

    def _line_below(self, dim: int, pos: float) -> tuple[int | None, float | None]:
        idx, act_pos = self.nearest_mesh_line(dim, pos)
        if act_pos is None:
            return (None, None)
        if fp_equalp(act_pos, pos):
            idx = idx - 1 if idx is not None else None
            if idx is not None and self._mesh_valid_index(dim, idx):
                act_pos = self.mesh_lines[dim][idx]
        if act_pos is not None and fp_ltp(act_pos, pos):
            return (idx, act_pos)
        if idx is not None:
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
            idx = idx + 1 if idx is not None else None
            if idx is not None and self._mesh_valid_index(dim, idx):
                act_pos = self.mesh_lines[dim][idx]
        if act_pos is not None and fp_gtp(act_pos, pos):
            return (idx, act_pos)
        if idx is not None:
            idx += 1
            if self._mesh_valid_index(dim, idx):
                act_pos = self.mesh_lines[dim][idx]
                return (idx, act_pos)
        return (None, None)

    def _mesh_valid_index(self, dim: int, index: int) -> bool:
        return 0 <= index < len(self.mesh_lines[dim])
