# pyems
# Copyright (C) 2020-2026 Matt Huszabianlou
# Copyright (C) 2026 Mohamoud Ali
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
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
Auto-generated FDTD mesh from CSXCAD primitives.

Adapted from pyems mesh.py — uses geometric series for smooth grid
transitions and the thirds rule at metal boundaries.
"""

from enum import Enum
from bisect import bisect_left, insort_left

import numpy as np
import scipy.optimize

from CSXCAD import ContinuousStructure
from CSXCAD.CSPrimitives import CSPrimitives

from .sim_params import SimParams

__all__ = ["Mesh"]

PREC = 20


class Type(Enum):
    """Material classification of a meshed interval: conductor, non-metal
    (dielectric or free space treated as bulk material), or open air."""

    metal = 0
    nonmetal = 1
    air = 2


class BoundedType:
    """An interval along one mesh dimension tagged with the material
    (:class:`Type`) that occupies it."""

    def __init__(self, prop_type: Type, lower_bound: float, upper_bound: float) -> None:
        self.prop_type = prop_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_type(self) -> Type:
        """Return the :class:`Type` occupying this interval."""
        return self.prop_type

    def get_bounds(self) -> list[float]:
        """Return ``[lower_bound, upper_bound]``."""
        return [self.lower_bound, self.upper_bound]

    def get_midpoint(self) -> float:
        """Return the midpoint of the interval."""
        return np.average([self.lower_bound, self.upper_bound])

    def size(self) -> float:
        """Return the interval's length (``upper_bound - lower_bound``)."""
        return self.upper_bound - self.lower_bound


def fp_nearest(val_or_arr: float) -> float:
    """Round a value (or array) to ``PREC`` decimal places, for stable
    floating-point comparisons."""
    return np.around(val_or_arr, PREC)


def fp_equalp(val1: float, val2: float) -> bool:
    """Return True if ``val1`` and ``val2`` are equal once rounded to
    ``PREC`` decimal places."""
    return np.around(val1, PREC) == np.around(val2, PREC)


def fp_gtp(val1: float, val2: float) -> bool:
    """Return True if ``val1`` > ``val2`` once rounded to ``PREC`` decimal
    places."""
    return np.around(val1, PREC) > np.around(val2, PREC)


def fp_gep(val1: float, val2: float) -> bool:
    """Return True if ``val1`` >= ``val2`` once rounded to ``PREC`` decimal
    places."""
    return np.around(val1, PREC) >= np.around(val2, PREC)


def fp_ltp(val1: float, val2: float) -> bool:
    """Return True if ``val1`` < ``val2`` once rounded to ``PREC`` decimal
    places."""
    return np.around(val1, PREC) < np.around(val2, PREC)


def fp_lep(val1: float, val2: float) -> bool:
    """Return True if ``val1`` <= ``val2`` once rounded to ``PREC`` decimal
    places."""
    return np.around(val1, PREC) <= np.around(val2, PREC)


def _prim_metalp(prim: CSPrimitives) -> bool:
    """Return True if ``prim``'s CSXCAD property is a conductor (metal,
    conducting sheet, or lumped element/port)."""
    type_str = prim.GetProperty().GetTypeString()
    return type_str in ("Metal", "ConductingSheet", "LumpedElement")


def _prim_materialp(prim: CSPrimitives) -> bool:
    """Return True if ``prim``'s CSXCAD property is a dielectric
    (``Material``)."""
    return prim.GetProperty().GetTypeString() == "Material"


def _get_prim_bounds(prim: CSPrimitives) -> np.ndarray:
    """Return ``prim``'s axis-aligned bounding box, with its CSXCAD
    transform applied, as ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``."""
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
    """Return True if ``prim`` is a linear-extrusion polygon or flat polygon
    primitive."""
    cls = prim.__class__.__name__
    return cls in ("CSPrimLinPoly", "CSPrimPolygon")


def _get_linpoly_vertex_bounds(prim: CSPrimitives) -> list[list[float]]:
    """Return a polygon primitive's vertex coordinates in world space, as
    ``[x_coords, y_coords, z_coords]``.

    Returns three empty lists (rather than raising) if the primitive's
    coordinates cannot be read.
    """
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
    """Filter ``prims`` down to the conductor and dielectric primitives that
    should drive mesh generation (helper/non-physical primitives are
    excluded)."""
    physical = []
    for prim in prims:
        if _prim_metalp(prim) or _prim_materialp(prim):
            physical.append(prim)
    return physical


def _remove_dups(lst: list, fixed: list | None = None) -> list:
    """Collapse near-duplicate consecutive values in a sorted list.

    Two consecutive values are duplicates once rounded to ``PREC`` decimal
    places. Normally the later of a duplicate pair is dropped; if that later
    value is itself in ``fixed`` (a must-keep value, e.g. an explicit mesh
    line), the earlier one is dropped instead so the fixed value survives.

    Parameters
    ----------
    lst : list
        Sorted values to deduplicate.
    fixed : list | None
        Values that must never be dropped. Default ``None`` (treated as
        empty).

    Returns
    -------
    list
        ``lst`` with near-duplicates collapsed.
    """
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


def _collect_all_bounds(
    prims: list[CSPrimitives], fixed: list[list[float]]
) -> list[list[float]]:
    """Collect candidate mesh-line positions from ``prims``, per dimension.

    Adds both bounding-box edges of every primitive along all three
    dimensions; for polygon primitives, also adds every vertex coordinate so
    the mesh conforms to non-rectangular metal edges. Near-duplicates are
    then removed per dimension via :func:`_remove_dups`.

    Parameters
    ----------
    prims : list[CSPrimitives]
        Physical primitives to collect bounds from.
    fixed : list[list[float]]
        Per-dimension must-keep positions, forwarded to :func:`_remove_dups`.

    Returns
    -------
    list[list[float]]
        ``[x_bounds, y_bounds, z_bounds]``, each sorted and deduplicated.
    """
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
    """Return True if ``lower <= val <= upper``."""
    return lower <= val <= upper


def _point_in_poly(px: float, py: float, verts: list[tuple[float, float]]) -> bool:
    """Return True if point ``(px, py)`` lies inside the polygon defined by
    ``verts``, using the standard ray-casting (even-odd rule) test."""
    inside = False
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _pos_in_bounds(pos: float, lower: float, upper: float) -> bool:
    """Return True if ``lower <= pos <= upper``, using the floating-point-
    tolerant comparisons :func:`fp_gep`/:func:`fp_lep`."""
    return fp_gep(pos, lower) and fp_lep(pos, upper)


def _type_at_pos(prims: list[CSPrimitives], dim: int, pos: float) -> Type | None:
    """Classify the material type at a point along one dimension.

    Finds the smallest primitive (by extent along ``dim``) whose bounding
    box covers ``pos``; metal wins ties against non-metal primitives of the
    same size. A metal polygon primitive that does not actually cover the
    transverse position at ``pos`` (e.g. an inset notch cut into a patch) is
    treated as a gap rather than solid metal; if every primitive covering
    ``pos`` turns out to be such a notch, the position is reclassified as
    non-metal.

    Parameters
    ----------
    prims : list[CSPrimitives]
        Physical primitives to test.
    dim : int
        Dimension to test along: ``0`` (x), ``1`` (y), or ``2`` (z).
    pos : float
        Position along ``dim`` to classify.

    Returns
    -------
    Type | None
        The material type at ``pos``, or ``None`` if no primitive covers it.
    """

    def _covers(prim: CSPrimitives) -> bool:
        """Return True if polygon primitive ``prim`` actually covers the
        transverse position at ``pos`` along ``dim`` (tested against its
        world-space vertex outline), rather than just its bounding box."""
        try:
            coords = prim.GetCoords()
            x_verts = list(coords[0])
            y_verts = list(coords[1])
            norm_dir = int(prim.GetNormDir())
            tr = prim.GetTransform()
            elev = float(prim.GetElevation())
            bb = prim.GetBoundBox()
            var_dims = [d for d in range(3) if d != norm_dir]
            if dim == norm_dir:
                return _float_inside(pos, bb[0][dim], bb[1][dim])
            d0, d1 = var_dims
            p0 = pos if dim == d0 else (bb[0][d0] + bb[1][d0]) * 0.5
            p1 = pos if dim == d1 else (bb[0][d1] + bb[1][d1]) * 0.5
            world_verts: list[tuple[float, float]] = []
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
                world_verts.append((pt[d0], pt[d1]))
            return _point_in_poly(p0, p1, world_verts)
        except Exception:
            return True

    smallest_dim = np.inf
    current_type = None
    in_notch = False

    for prim in prims:
        prim_bounds = _get_prim_bounds(prim)
        if _float_inside(pos, prim_bounds[dim][0], prim_bounds[dim][1]):
            if _is_linpoly(prim) and _prim_metalp(prim) and not _covers(prim):
                in_notch = True
                continue
            dim_size = prim_bounds[dim][1] - prim_bounds[dim][0]
            if np.isclose(dim_size, smallest_dim, rtol=1e-3, atol=0):
                if _prim_metalp(prim):
                    current_type = Type.metal
                    smallest_dim = dim_size
            elif dim_size < smallest_dim:
                smallest_dim = dim_size
                current_type = Type.metal if _prim_metalp(prim) else Type.nonmetal

    if in_notch and current_type == Type.metal:
        for prim in prims:
            if _is_linpoly(prim) and _prim_metalp(prim) and _covers(prim):
                return Type.metal
        return Type.nonmetal

    return current_type


def _sort_bounded_types(
    bounded_types: list[list[BoundedType]],
) -> list[list[BoundedType]]:
    """Sort each dimension's list of :class:`BoundedType` intervals by
    increasing size, so meshing processes the finest features first."""
    new_bounded_types = [[], [], []]
    for dim, btype_list in enumerate(bounded_types):
        new_bounded_types[dim] = sorted(btype_list, key=lambda x: x.size())
    return new_bounded_types


def _factor_for_num(num: int, smaller_spacing: float, dist: float) -> float:
    """Solve for the geometric growth factor that spans ``dist`` in exactly
    ``num`` steps starting from ``smaller_spacing``."""
    roots = scipy.optimize.fsolve(
        func=_geom_dist_zero, x0=1.5, args=(num, smaller_spacing, dist)
    )
    return roots[0]


def _factor_ubound(num: int, ratio: float, max_factor: float) -> float:
    """Return ``min(max_factor, ratio ** (1 / (num - 1)))``: the growth
    factor that would exactly reach a total spacing ratio of ``ratio`` over
    ``num`` mesh lines, capped at ``max_factor``."""
    return np.min([max_factor, np.power(ratio, 1 / (num - 1))])


def _geom_dist(factor: float, num: int, smaller_spacing: float) -> float:
    """Return the total distance spanned by ``num - 1`` geometrically
    growing steps, starting at ``smaller_spacing`` and growing by ``factor``
    each step."""
    powers = np.arange(1, num, 1)
    return smaller_spacing * np.sum(np.power(factor, powers))


def _geom_dist_zero(
    factor: float, num: int, smaller_spacing: float, dist: float
) -> float:
    """Return ``_geom_dist(factor, num, smaller_spacing) - dist``; the
    root-finding objective used by :func:`_factor_for_num`."""
    return _geom_dist(factor, num, smaller_spacing) - dist


def _num_for_factor(
    factor: float, smaller_spacing: float, dist: float
) -> tuple[float, int]:
    """Find the number of geometric-series steps needed to span ``dist``
    without exceeding growth factor ``factor``, and the exact growth factor
    for that step count.

    Parameters
    ----------
    factor : float
        Maximum allowed growth factor between consecutive spacings.
    smaller_spacing : float
        Spacing of the first step.
    dist : float
        Total distance to span.

    Returns
    -------
    tuple[float, int]
        ``(factor, num)`` -- the growth factor that exactly spans ``dist`` in
        ``num`` steps (``num`` is reduced until this factor is at least 1).
    """
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
    """Find the number of lines and growth factor for a geometric-series
    mesh spanning ``dist``, starting at ``smaller_spacing`` and growing
    (at most ``max_factor`` per step) toward ``larger_spacing``, without
    dropping below ``min_num`` lines.

    Returns
    -------
    tuple[float, int]
        ``(factor, num)``.
    """
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
    """Generate mesh line positions across ``[lower, upper]`` as a single
    geometric series.

    Uses a uniform grid if ``lower_spacing`` and ``upper_spacing`` are
    (nearly) equal; otherwise grows geometrically from whichever end has the
    smaller spacing toward the other end, at a growth factor no larger than
    ``smooth``.

    Parameters
    ----------
    lower, upper : float
        Interval to fill with mesh lines.
    lower_spacing, upper_spacing : float
        Target spacing at each end of the interval.
    dim : int
        Dimension being meshed. Unused inside this function; kept only for a
        uniform call signature with its callers.
    min_lines : int
        Minimum number of mesh lines to generate.
    smooth : float
        Maximum ratio between adjacent cell sizes.

    Returns
    -------
    np.ndarray
        Mesh line positions from ``lower`` to ``upper`` inclusive.
    """
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
    """Return the cell spacing reached after growing (at up to ``max_factor``
    per step) from ``spacing`` across a distance ``dist``."""
    factor, num = _num_for_factor(max_factor, spacing, dist)
    return spacing * (factor ** (num - 1))


def _spacings_at_dist_zero(
    dist: float,
    lower_spacing: float,
    upper_spacing: float,
    total_dist: float,
    max_factor: float,
) -> float:
    """Return the mismatch between the two spacings that meet if a
    geometric series grows inward from each end of a ``total_dist``
    interval and meets at offset ``dist`` from the lower end.

    Root-finding objective used by :func:`_dist_for_max_spacings` to locate
    the meeting point where both series reach the same cell spacing.
    """
    spacing1 = _spacing_at_dist(lower_spacing, dist, max_factor)
    spacing2 = _spacing_at_dist(upper_spacing, total_dist - dist, max_factor)
    return spacing2 - spacing1


def _dist_for_max_spacings(
    lower_spacing: float, upper_spacing: float, dist: float, max_factor: float
) -> float:
    """Return the offset from the lower end of a ``dist``-long interval
    where two geometric series (growing inward from each end, at up to
    ``max_factor`` per step, starting at ``lower_spacing``/``upper_spacing``)
    meet at the same cell spacing."""
    roots = scipy.optimize.fsolve(
        func=_spacings_at_dist_zero,
        x0=dist / 2,
        args=(lower_spacing, upper_spacing, dist, max_factor),
    )
    return roots[0]


class Mesh:
    """Auto-generates an FDTD mesh from CSXCAD primitives.

    Adapts the pyems automatic mesh generation algorithm. Scans all
    physical primitives (metal and material), classifies regions by
    type, generates mesh lines using geometric series for smooth
    transitions, and applies the thirds rule at metal boundaries (mesh
    lines a third of a cell away from a metal-air interface, which
    improves FDTD accuracy at conductor edges).

    The mesh is generated immediately on construction and written directly
    into ``csx``'s grid (via CSXCAD's ``ContinuousStructure.GetGrid()``); the
    resulting mesh lines are also kept on ``self.mesh_lines`` for inspection.

    Parameters
    ----------
    csx : ContinuousStructure
        CSXCAD structure with primitives already added.
    params : SimParams
        Simulation parameters; reads ``simulation_box``, ``mesh_resolution``,
        ``metal_mesh_resolution``, ``unit``, and ``lambda0``.
    smooth_ratio : float
        Maximum ratio between adjacent cell sizes. Default ``1.5``.
    min_lines : int
        Minimum number of mesh lines generated across any bounded interval.
        Default ``5``.
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
        """Run the full mesh-generation pipeline and write the result into
        ``self._csx``'s grid."""
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
        self._clean_close_lines()

    def _set_fixed_lines(self, prims: list[CSPrimitives]) -> None:
        """Add a fixed mesh line for every zero-thickness primitive (a
        primitive whose bounding box has zero extent along a dimension,
        e.g. a flat metal sheet), which must be meshed exactly."""
        for prim in prims:
            prim_bounds = _get_prim_bounds(prim)
            for dim in range(3):
                if fp_equalp(prim_bounds[dim][0], prim_bounds[dim][1]):
                    self.add_fixed_line(dim, fp_nearest(prim_bounds[dim][0]))
                self.fixed_lines[dim].sort()
                self.fixed_lines[dim] = _remove_dups(self.fixed_lines[dim])

    def add_fixed_line(self, dim: int, pos: float) -> None:
        """Register a must-keep mesh line position at ``pos``.

        Fixed lines are protected from the deduplication/smoothing passes
        applied to the rest of the mesh. Must be called before the mesh is
        generated to have any effect, since ``__init__`` runs mesh
        generation immediately.

        Parameters
        ----------
        dim : int
            Dimension to add the line to: ``0`` (x), ``1`` (y), or ``2`` (z).
        pos : float
            Position of the line, in the structure's drawing units.
        """
        self.fixed_lines[dim].append(pos)
        self.fixed_lines[dim].sort()

    def _bounded_types(
        self, bounds: list[list[float]], prims: list[CSPrimitives]
    ) -> list[list[BoundedType]]:
        """Turn each dimension's sorted candidate positions (``bounds``) into
        a list of adjacent :class:`BoundedType` intervals, classified via
        :func:`_type_at_pos`. Zero-length intervals are also emitted at each
        fixed line, so a fixed position is always its own interval."""
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
        """Grow ``self._sim_box`` (in place) so every dimension pads the
        geometry's own extent by at least ``lambda0 / 2`` or 15% of the
        geometry's span, whichever is larger. Dimensions with no geometry
        keep the simulation box passed in at construction."""
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
        """Extend ``bounded_types`` with air intervals so each dimension
        spans the full simulation box, and record the result in
        ``self.sim_bounds``.

        Parameters
        ----------
        bounded_types : list[list[BoundedType]]
            Per-dimension intervals covering (at most) the geometry's own
            extent.

        Returns
        -------
        list[list[BoundedType]]
            ``bounded_types``, padded with air intervals out to the
            simulation box boundary in each dimension.

        Raises
        ------
        ValueError
            If the simulation box is smaller than the structure's extent in
            some dimension.
        """
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
        """Record the (non-degenerate) edges of every metal interval into
        ``self.metal_bounds``, per dimension."""
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
        """Insert ``pos`` into ``self.metal_bounds[dim]``, keeping it sorted."""
        insort_left(self.metal_bounds[dim], pos)

    def _is_fixed_line(self, dim: int, pos: float) -> bool:
        """Return True if ``pos`` is a fixed (must-keep) mesh line in
        dimension ``dim``."""
        return any(fp_equalp(fp, pos) for fp in self.fixed_lines[dim])

    def _is_metal_bound(self, dim: int, pos: float) -> bool:
        """Return True if ``pos`` is a metal interval edge in dimension
        ``dim``."""
        return any(fp_equalp(mb, pos) for mb in self.metal_bounds[dim])

    def _pos_meshed(self, dim: int, pos: float) -> bool:
        """Return True if ``pos`` falls inside a dimension-``dim`` interval
        that has already had mesh lines generated for it."""
        for rng in self.ranges_meshed[dim]:
            if _pos_in_bounds(pos, rng[0], rng[1]):
                return True
        return False

    def _type_below(self, dim: int, upper: float) -> Type | None:
        """Return the :class:`Type` of the bounded interval whose upper edge
        is at ``upper`` in dimension ``dim``, or ``None`` if there is none."""
        for btype in self.bounded_types[dim]:
            if fp_equalp(btype.get_bounds()[1], upper):
                return btype.get_type()
        return None

    def _type_below_meshed(self, dim: int, lower: float) -> bool:
        """Return True if the (non-degenerate) interval immediately below
        ``lower`` in dimension ``dim`` has already been meshed."""
        for btype in self.bounded_types[dim]:
            if fp_equalp(btype.get_bounds()[1], lower) and btype.size() != 0:
                return self._pos_meshed(dim, btype.get_midpoint())
        return False

    def _type_above(self, dim: int, lower: float) -> Type | None:
        """Return the :class:`Type` of the bounded interval whose lower edge
        is at ``lower`` in dimension ``dim``, or ``None`` if there is none."""
        for btype in self.bounded_types[dim]:
            if fp_equalp(btype.get_bounds()[0], lower):
                return btype.get_type()
        return None

    def _type_above_meshed(self, dim: int, upper: float) -> bool:
        """Return True if the (non-degenerate) interval immediately above
        ``upper`` in dimension ``dim`` has already been meshed."""
        for btype in self.bounded_types[dim]:
            if fp_equalp(btype.get_bounds()[0], upper) and btype.size() != 0:
                return self._pos_meshed(dim, btype.get_midpoint())
        return False

    def _add_to_ranges_meshed(self, dim: int, lower: float, upper: float) -> None:
        """Record ``[lower, upper]`` as an already-meshed range in dimension
        ``dim``."""
        self.ranges_meshed[dim].append([lower, upper])

    def _gen_mesh_for_bounded_types(
        self, bounded_types: list[list[BoundedType]]
    ) -> None:
        """Generate mesh lines for every interval in ``bounded_types``, in
        the order given.

        Intended to be called with intervals sorted smallest-first (see
        :func:`_sort_bounded_types`) so finer features are meshed before the
        coarser intervals that reference their neighbors' spacing.
        """
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
        """Return the largest cell spacing that still fits at least
        ``self._min_lines`` mesh lines across a distance of ``dist``."""
        return dist / (self._min_lines - 1)

    def _lower_spacing(
        self,
        dim: int,
        lower: float,
        line_below: float | None,
        dist: float,
        is_metal: bool,
    ) -> float:
        """Choose the target cell spacing at the lower edge of an interval.

        Starts from the metal or non-metal target resolution (capped so at
        least ``min_lines`` lines fit across ``dist``); if the interval below
        has already been meshed, further caps it by the thirds rule against
        the neighboring cell's spacing (a factor of 1.5 across a metal/
        non-metal boundary, 3.0 otherwise) so the new interval doesn't open
        with an abrupt jump in cell size.

        Parameters
        ----------
        dim : int
            Dimension being meshed.
        lower : float
            Lower edge of the interval being meshed.
        line_below : float | None
            Nearest existing mesh line below ``lower``, if any.
        dist : float
            Length of the interval being meshed.
        is_metal : bool
            Whether the interval is metal (uses ``metal_mesh_resolution``
            instead of ``mesh_resolution``).

        Returns
        -------
        float
            Target spacing at the lower edge.
        """
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
        """Mirror of :func:`_lower_spacing` for the upper edge of an
        interval."""
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
        """Generate and add mesh lines across one bounded interval.

        A degenerate (zero-length) interval gets a single line. A metal
        interval in z thinner than a quarter of ``metal_mesh_resolution``
        gets a single line at its midpoint, since it is too thin to resolve
        with a full metal-resolution grid. Otherwise, generates a
        geometric-series grid across the interval (:func:`_gen_lines_in_bounds`);
        for a metal interval, the interval is first shrunk at each edge that
        is not a fixed line or the outer simulation boundary, by a third (or
        two-thirds, if the far side is also already-meshed metal) of a cell
        width -- the "thirds rule", which keeps mesh lines off the metal
        edge itself for FDTD accuracy -- and the lines regenerated over the
        shrunk interval; a non-metal interval abutting a metal boundary is
        instead nudged inward by two-thirds of a cell at that edge and
        rebuilt.

        Parameters
        ----------
        dim : int
            Dimension being meshed: ``0`` (x), ``1`` (y), or ``2`` (z).
        lower, upper : float
            Interval to mesh.
        line_below, line_above : float | None
            Nearest existing mesh line outside the interval on each side, if
            any.
        is_metal : bool
            Whether the interval is a metal region.
        """
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
        """Generate mesh line positions across ``[lower, upper]``.

        If too few lines would fit at the coarser of the two target
        spacings, or growing from the finer spacing at the allowed
        smoothing ratio can't reach the coarser one before crossing the
        full distance, falls back to a single geometric series between the
        two spacings (:func:`_lines_const_factor_in_bounds`). Otherwise,
        splits the interval at the point where series grown inward from
        each end would meet at the same cell size (capped at
        ``max_spacing``), and generates two series that join there -- so
        both ends transition smoothly to their target spacing without
        exceeding ``max_spacing`` in the middle.

        Parameters
        ----------
        lower, upper : float
            Interval to fill with mesh lines.
        lower_spacing, upper_spacing : float
            Target cell spacing at each end of the interval.
        max_spacing : float
            Largest cell spacing allowed anywhere in the interval.
        dim : int
            Dimension being meshed (selects ``self._smooth[dim]``).

        Returns
        -------
        np.ndarray
            Mesh line positions from ``lower`` to ``upper``.
        """
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
        """Insert ``lines`` into ``self.mesh_lines[dim]`` (sorted) and remove
        any resulting near-duplicates."""
        for line in lines:
            self._add_mesh_line(dim, line)
        self.mesh_lines[dim] = _remove_dups(self.mesh_lines[dim], self.fixed_lines[dim])

    def _add_mesh_line(self, dim: int, pos: float) -> None:
        """Insert ``pos`` into ``self.mesh_lines[dim]``, keeping it sorted."""
        insort_left(self.mesh_lines[dim], pos)

    def _set_mesh_from_lines(self) -> None:
        """Write ``self.mesh_lines`` into the CSXCAD grid, replacing any
        lines already there."""
        grid = self._csx.GetGrid()
        grid.SetDeltaUnit(self._unit)
        for i in range(3):
            grid.ClearLines(i)
        for dim in range(3):
            for line in self.mesh_lines[dim]:
                ch = "xyz"[dim]
                grid.AddLine(ch, fp_nearest(line))

    def _clean_close_lines(self, min_spacing: float | None = None) -> None:
        """Merge mesh lines that ended up closer together than ``min_spacing``.

        Reads the current lines from the CSXCAD grid (after
        ``SmoothMeshLines`` has run) and, for each dimension, collapses any
        pair closer than ``min_spacing`` into one -- keeping whichever line
        of the pair is fixed, or averaging the two if neither is -- then
        writes the cleaned lines back into the grid.

        Parameters
        ----------
        min_spacing : float | None
            Minimum allowed spacing between adjacent lines. Default ``None``
            uses 1% of ``self.smallest_res``.
        """
        if min_spacing is None:
            min_spacing = self.smallest_res * 0.01
        for dim in range(3):
            ch = "xyz"[dim]
            lines = list(self.mesh.GetLines(ch))
            if len(lines) < 2:
                continue
            cleaned = []
            i = 0
            while i < len(lines):
                j = i + 1
                if j < len(lines) and (lines[j] - lines[i]) < min_spacing:
                    if self._is_fixed_line(dim, lines[i]):
                        cleaned.append(fp_nearest(lines[i]))
                        i = j + 1
                        continue
                    if self._is_fixed_line(dim, lines[j]):
                        cleaned.append(fp_nearest(lines[j]))
                        i = j + 1
                        continue
                    avg = fp_nearest((lines[i] + lines[j]) / 2.0)
                    cleaned.append(avg)
                    i = j + 1
                else:
                    cleaned.append(fp_nearest(lines[i]))
                    i += 1
            cleaned = _remove_dups(cleaned, self.fixed_lines[dim])
            self.mesh_lines[dim] = cleaned
            grid = self._csx.GetGrid()
            grid.ClearLines(dim)
            for line in cleaned:
                grid.AddLine(ch, line)

    def nearest_mesh_line(
        self, dim: int, pos: float
    ) -> tuple[int | None, float | None]:
        """Find the existing mesh line closest to ``pos`` in dimension ``dim``.

        Parameters
        ----------
        dim : int
            Dimension to search: ``0`` (x), ``1`` (y), or ``2`` (z).
        pos : float
            Position to search near.

        Returns
        -------
        tuple[int | None, float | None]
            ``(index, position)`` of the nearest line in
            ``self.mesh_lines[dim]``, or ``(None, None)`` if that dimension
            has no mesh lines yet.
        """
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
        """Return ``(index, position)`` of the nearest existing mesh line
        strictly below ``pos`` in dimension ``dim``, or ``(None, None)`` if
        there is none."""
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
        """Return ``(index, position)`` of the nearest existing mesh line
        strictly above ``pos`` in dimension ``dim``, or ``(None, None)`` if
        there is none."""
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
        """Return True if ``index`` is a valid index into
        ``self.mesh_lines[dim]``."""
        return 0 <= index < len(self.mesh_lines[dim])
