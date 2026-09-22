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

PREC = 5


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
    transform applied, as ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.

    Transforms all 8 corners of the untransformed box and takes the min/max
    of the transformed set per axis -- not just the 2 diagonal corners
    ``GetBoundBox()`` returns. Transforming only those 2 corners is exact
    for a translation or an axis-aligned (90-degree-multiple) rotation, but
    silently wrong for any other rotation: e.g. a box rotated 45 degrees
    becomes a diamond, and the two transformed diagonal corners can land at
    the same coordinate along an axis, reporting a fully degenerate
    (zero-width) bound along that axis when the rotated shape actually spans
    a wide range there. A primitive built with a non-90-degree
    ``RotateAxis`` transform (e.g. a rotated radial stub) hits this
    directly.
    """
    orig_bounds = prim.GetBoundBox()
    tr = prim.GetTransform()
    lo, hi = orig_bounds[0], orig_bounds[1]
    corners = [
        [
            lo[0] if cx == 0 else hi[0],
            lo[1] if cy == 0 else hi[1],
            lo[2] if cz == 0 else hi[2],
        ]
        for cx in (0, 1)
        for cy in (0, 1)
        for cz in (0, 1)
    ]
    if tr is not None:
        corners = [tr.Transform(c) for c in corners]
    corners = np.array(corners, dtype=float)
    bounds = np.array([[None, None], [None, None], [None, None]])
    for i in range(3):
        bounds[i] = np.array([corners[:, i].min(), corners[:, i].max()])
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


def _is_polyhedron(prim: CSPrimitives) -> bool:
    """Return True if ``prim`` is a tessellated polyhedron primitive (e.g.
    an STL/PLY import via ``CSPrimPolyhedronReader``)."""
    cls = prim.__class__.__name__
    return cls in ("CSPrimPolyhedron", "CSPrimPolyhedronReader")


def _can_be_partial(prim: CSPrimitives) -> bool:
    """Return True if ``prim``'s bounding box may not equal its actual solid
    extent (a concave or vertex-defined shape), so a query point inside its
    bounding box still needs an explicit coverage test rather than being
    assumed solid."""
    return _is_linpoly(prim) or _is_polyhedron(prim)


def _get_polyhedron_vertex_bounds(prim: CSPrimitives) -> list[list[float]]:
    """Return a polyhedron primitive's vertex coordinates in world space, as
    ``[x_coords, y_coords, z_coords]``.

    Returns three empty lists (rather than raising) if the primitive's
    vertices cannot be read.
    """
    bounds: list[list[float]] = [[], [], []]
    try:
        tr = prim.GetTransform()
        for i in range(prim.GetNumVertices()):
            pt = [float(c) for c in prim.GetVertex(i)]
            if tr is not None:
                pt = tr.Transform(pt)
            for d in range(3):
                bounds[d].append(float(pt[d]))
    except Exception:
        pass
    return bounds


def _prim_centroid(prim: CSPrimitives) -> list[float] | None:
    """Return the average of a polygon/polyhedron primitive's own vertices
    in world space, or ``None`` if ``prim`` has no vertices to average (not
    a linpoly/polyhedron, or its coordinates could not be read).

    A weaker fallback than :func:`_linpoly_cross_section_point` (see there
    for why a single fixed point is unreliable in general) -- the vertex
    average is not even reliably interior by itself for a shape whose
    vertices are unevenly distributed around its boundary (many points
    along a stub's outer arc, only two at its neck skews the average toward
    the arc, possibly past the opposite edge entirely). Used only where
    slicing isn't available: a polyhedron (a 3D vertex loop isn't a single
    cross-section to slice) or the z axis of a linpoly (uniform thickness,
    no real cross-section shape to get wrong).
    """
    if _is_linpoly(prim):
        coords = _get_linpoly_vertex_bounds(prim)
    elif _is_polyhedron(prim):
        coords = _get_polyhedron_vertex_bounds(prim)
    else:
        return None
    if not coords[0]:
        return None
    return [float(np.mean(c)) for c in coords]


def _linpoly_cross_section_point(
    prim: CSPrimitives, dim: int, pos: float
) -> list[float] | None:
    """Return a point known to lie inside ``prim``'s flat outline at
    ``pos`` along ``dim``, or ``None`` if ``prim`` isn't a linpoly, ``dim``
    isn't one of its two in-plane axes (``0``/x or ``1``/y -- its z extent
    is a uniform extrusion thickness, not a shape to slice), or no edge of
    the polygon actually straddles ``pos``.

    Slices the polygon's vertex loop at ``pos``: walks each edge, and where
    it straddles ``pos`` along ``dim``, linearly interpolates the crossing's
    position along the other in-plane axis, then returns the midpoint
    between the outermost two crossings. This is the shape's own local
    cross-section at exactly the position being classified, not one point
    meant to represent the whole shape -- which matters whenever the
    cross-section varies along the sliced axis: a radial stub's width
    changes with radius, and a right-triangle miter's bounding-box
    midpoint (or even its vertex centroid) can land outside the triangle,
    or exactly on its hypotenuse, depending where along the axis you ask.
    """
    if not _is_linpoly(prim) or dim not in (0, 1):
        return None
    coords = _get_linpoly_vertex_bounds(prim)
    if not coords[0]:
        return None
    other_dim = 1 - dim
    axis_vals = coords[dim]
    other_vals = coords[other_dim]
    elev = coords[2][0]
    n = len(axis_vals)
    crossings = []
    for i in range(n):
        j = (i + 1) % n
        a, b = axis_vals[i], axis_vals[j]
        if a == b:
            continue
        if (a <= pos <= b) or (b <= pos <= a):
            t = (pos - a) / (b - a)
            crossings.append(other_vals[i] + t * (other_vals[j] - other_vals[i]))
    if len(crossings) < 2:
        return None
    point = [0.0, 0.0, 0.0]
    point[dim] = pos
    point[other_dim] = (min(crossings) + max(crossings)) / 2.0
    point[2] = elev
    return point


def _linpoly_interior_xy(prim: CSPrimitives, bb: np.ndarray) -> list[float] | None:
    """Return an ``[x, y]`` point known to lie inside ``prim``'s flat
    outline, or ``None`` if it can't be found.

    For testing coverage along z, the outline's shape at *some* x,y is what
    matters, not a value that varies along the tested axis (z is just a
    uniform extrusion thickness for a linpoly) -- so, unlike
    :func:`_linpoly_cross_section_point`, there is no caller-given position
    to slice at. Slices along x at the shape's own bounding-box midpoint and
    takes the midpoint of the resulting y-crossings; if that particular
    slice misses the outline (a rotated shape whose vertices might all sit
    to one side of it), falls back to slicing along y at its own
    bounding-box midpoint instead.

    Neither the vertex centroid nor the bounding-box midpoint stand in for
    this reliably in general: an inset notch skews the vertex average
    toward the cut -- which can be well outside the polygon, e.g. inside
    the notch's own gap -- while a shape with a varying cross-section can
    put the bounding-box midpoint outside the polygon too (see
    :func:`_linpoly_cross_section_point`). Slicing at a fixed axis value
    still assumes the resulting cross-section's *midpoint* is solid rather
    than, say, a second notch splitting it in two -- true for every shape
    this codebase currently generates, but not guaranteed for an arbitrary
    concave outline.
    """
    x_mid = 0.5 * (bb[0][0] + bb[0][1])
    point = _linpoly_cross_section_point(prim, 0, x_mid)
    if point is not None:
        return [point[0], point[1]]
    y_mid = 0.5 * (bb[1][0] + bb[1][1])
    point = _linpoly_cross_section_point(prim, 1, y_mid)
    if point is not None:
        return [point[0], point[1]]
    return None


def _physical_prims(prims: list[CSPrimitives]) -> list[CSPrimitives]:
    """Filter ``prims`` down to the conductor and dielectric primitives that
    should drive mesh generation (helper/non-physical primitives are
    excluded)."""
    physical = []
    for prim in prims:
        if _prim_metalp(prim) or _prim_materialp(prim):
            physical.append(prim)
    return physical


def geometry_extent(csx: ContinuousStructure) -> list[list[float]]:
    """Per-dimension ``[min, max]`` of every physical primitive in ``csx``.

    Uses the same primitive filter and transform-aware bounding box as the
    mesher, so helper primitives (ports, dumps, NF2FF boxes) never widen it.

    Parameters
    ----------
    csx : ContinuousStructure
        Structure to measure.

    Returns
    -------
    list[list[float]]
        ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``, or ``[[], [], []]``
        when the structure has no physical primitives.
    """
    physical_primitives = _physical_prims(csx.GetAllPrimitives())
    if not physical_primitives:
        return [[], [], []]
    primitive_bounds = np.array(
        [_get_prim_bounds(primitive) for primitive in physical_primitives],
        dtype=float,
    )
    return [
        [
            float(primitive_bounds[:, dimension, 0].min()),
            float(primitive_bounds[:, dimension, 1].max()),
        ]
        for dimension in range(3)
    ]


def auto_simulation_bounds(
    geometry_bounds: list[list[float]], lambda0: float
) -> tuple[tuple[float, float], ...]:
    """Simulation box derived from the geometry when the user defined none.

    Each dimension pads the geometry's extent by ``max(lambda0, 15% of
    span)``; a dimension with no geometry spans ``[-lambda0, lambda0]``.

    Parameters
    ----------
    geometry_bounds : list[list[float]]
        Per-dimension positions; only the minimum and maximum are used.
    lambda0 : float
        Wavelength in the substrate, in drawing units.

    Returns
    -------
    tuple[tuple[float, float], ...]
        ``((xmin, xmax), (ymin, ymax), (zmin, zmax))``.
    """
    simulation_bounds = []
    for dimension_bounds in geometry_bounds:
        if not dimension_bounds:
            simulation_bounds.append((-lambda0, lambda0))
            continue
        geometry_min = min(dimension_bounds)
        geometry_max = max(dimension_bounds)
        padding = max(lambda0, (geometry_max - geometry_min) * 0.15)
        simulation_bounds.append((geometry_min - padding, geometry_max + padding))
    return tuple(simulation_bounds)


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


def _decimate_curve_coords(coords: list[float], min_feature: float) -> list[float]:
    """Collapse a single primitive's own vertex coordinates down to points
    that are actually distinguishable at the mesh's resolution.

    Works on the sorted values: keeps the smallest, then each value at least
    ``min_feature`` beyond the last one kept, and always the largest, so the
    shape's full extent survives. Sorting is what makes that a spacing
    guarantee -- walking the vertex sequence instead only compares
    neighbours in a list, and an outline that doubles back over the same
    range (a curved bend's polygon walks its inner arc out and its outer arc
    back) restarts the walk at an arbitrary phase, so a keeper from each
    pass can land a hundredth of a resolution apart. Where the largest value
    is itself within ``min_feature`` of the previous keeper, that keeper is
    dropped rather than the extent: the bounding box contributes the extent
    anyway, so keeping both would only open the sliver interval this
    function exists to prevent.

    A finely-faceted curve (e.g. a round pad's arc, built from many straight
    polygon segments) otherwise contributes one near-duplicate mesh-boundary
    candidate per facet vertex -- tens of them, each a tiny fraction of the
    target resolution apart. :func:`_gen_mesh_in_bounds`'s thin-interval
    collapse keeps any *one* of those gaps from ballooning into several
    lines, but does nothing about there being dozens of separate gaps in the
    first place, each still getting its own line barely a facet-width from
    the next -- exactly the runaway that forces an FDTD timestep far smaller
    than the geometry warrants. Decimating here, before the vertices are
    merged into the global candidate list, fixes that at the source. A
    genuine polygon corner (a miter, a taper, an inset notch) sits many
    resolution-widths from its neighbors, so this never touches it.
    """
    if not coords:
        return []
    ordered = sorted(coords)
    kept = [ordered[0]]
    for c in ordered[1:]:
        if c - kept[-1] >= min_feature:
            kept.append(c)
    if not fp_equalp(kept[-1], ordered[-1]):
        if len(kept) > 1:
            kept.pop()
        kept.append(ordered[-1])
    return kept


def _collect_all_bounds(
    prims: list[CSPrimitives], fixed: list[list[float]], min_feature: float
) -> list[list[float]]:
    """Collect candidate mesh-line positions from ``prims``, per dimension.

    Adds both bounding-box edges of every primitive along all three
    dimensions; for polygon and polyhedron primitives, also adds every
    vertex coordinate (decimated per :func:`_decimate_curve_coords`) so the
    mesh conforms to non-rectangular metal edges (including tessellated
    STL/PLY imports) without flooding it with a faceted curve's redundant
    near-duplicate vertices. A vertex strictly inside its own primitive's
    extent counts as an edge when the outline has a second vertex at the same
    coordinate (an inset notch, a step) and otherwise as one facet of a
    discretised curve, which is kept only where it stays ``min_feature`` clear
    of every edge collected here and of the facets already kept -- so a curve's
    facet cannot open a sliver interval against a neighbouring primitive's
    edge. Near-duplicates are then removed per dimension via
    :func:`_remove_dups`.

    Parameters
    ----------
    prims : list[CSPrimitives]
        Physical primitives to collect bounds from.
    fixed : list[list[float]]
        Per-dimension must-keep positions, forwarded to :func:`_remove_dups`.
    min_feature : float
        Minimum spacing, along a single dimension, between two vertices of
        the same polygon/polyhedron primitive for both to be kept as
        separate mesh-boundary candidates; see :func:`_decimate_curve_coords`.

    Returns
    -------
    list[list[float]]
        ``[x_bounds, y_bounds, z_bounds]``, each sorted and deduplicated.
    """
    dim_bounds: list[list[float]] = [[], [], []]
    interior: list[list[float]] = [[], [], []]
    for prim in prims:
        prim_bounds = _get_prim_bounds(prim)
        for dim, bounds in enumerate(prim_bounds):
            dim_bounds[dim].append(float(bounds[0]))
            dim_bounds[dim].append(float(bounds[1]))
        if _is_linpoly(prim):
            vert_bounds = _get_linpoly_vertex_bounds(prim)
        elif _is_polyhedron(prim):
            vert_bounds = _get_polyhedron_vertex_bounds(prim)
        else:
            continue
        for dim in range(3):
            lower, upper = float(prim_bounds[dim][0]), float(prim_bounds[dim][1])
            for v in _decimate_curve_coords(vert_bounds[dim], min_feature):
                if fp_equalp(v, lower) or fp_equalp(v, upper):
                    continue
                repeats = sum(1 for c in vert_bounds[dim] if fp_equalp(c, v))
                if repeats > 1:
                    # Two vertices share the coordinate, so an edge of the
                    # outline runs along it: a real corner (an inset notch, a
                    # step in a stub), meshed like any other edge.
                    dim_bounds[dim].append(v)
                else:
                    interior[dim].append(v)
    for dim, bounds in enumerate(dim_bounds):
        # What is left is a lone vertex on a slanted or curved run -- one facet
        # of a discretised arc, whose position says more about how finely the
        # curve was drawn than about the geometry. Take it only where it is at
        # least min_feature clear of every edge (and of the facets already
        # taken): a facet landing a fraction of a resolution from the trace the
        # curve meets would otherwise split off a sliver interval, the same
        # runaway _decimate_curve_coords guards against within one primitive.
        kept = sorted(bounds)
        for v in sorted(interior[dim]):
            index = bisect_left(kept, v)
            below = v - kept[index - 1] if index else np.inf
            above = kept[index] - v if index < len(kept) else np.inf
            if min(below, above) >= min_feature:
                insort_left(kept, v)
        dim_bounds[dim] = _remove_dups(kept, fixed[dim])
    return dim_bounds


def _float_inside(val: float, lower: float, upper: float) -> bool:
    """Return True if ``lower <= val <= upper``."""
    return lower <= val <= upper


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

    def _covers(prim: CSPrimitives, bb: np.ndarray) -> bool:
        """Return True if ``prim`` actually covers the transverse position
        at ``pos`` along ``dim``, tested via CSXCAD's own solid-membership
        test (``IsInside``) rather than just its bounding box ``bb`` --
        e.g. an inset notch cut into a patch, or a cutout in a tessellated
        polyhedron import.

        The other two dimensions are sampled via, in order of preference:
        :func:`_linpoly_cross_section_point` (a linpoly's own local
        cross-section at ``pos``, when ``dim`` is one of its two in-plane
        axes -- correct regardless of how its shape varies along ``dim``);
        :func:`_linpoly_interior_xy` (a linpoly's z axis, where the in-plane
        shape doesn't vary with ``pos`` but still needs a reliably-interior
        x,y -- an inset notch, say, skews the vertex centroid outside the
        polygon); :func:`_prim_centroid` (a polyhedron); or, failing all of
        those, the bounding-box midpoint. Never the bounding-box midpoint
        for a linpoly's in-plane axes: for a shape whose cross-section
        changes along ``dim``, one fixed point can be outside the polygon
        at some positions and, for a diagonally-cut shape like a
        full-miter triangle, exactly on an edge at every position along a
        slice through the bbox center -- where ``IsInside()`` gives an
        implementation-defined answer right where it matters most."""
        try:
            sample = _linpoly_cross_section_point(prim, dim, pos)
            if sample is None and _is_linpoly(prim):
                xy = _linpoly_interior_xy(prim, bb)
                if xy is not None:
                    sample = [xy[0], xy[1], 0.5 * (bb[2][0] + bb[2][1])]
            if sample is None:
                centroid = _prim_centroid(prim)
                sample = [
                    centroid[d] if centroid is not None else 0.5 * (bb[d][0] + bb[d][1])
                    for d in range(3)
                ]
            point = [pos if d == dim else sample[d] for d in range(3)]
            return bool(prim.IsInside(point))
        except Exception:
            return True

    smallest_dim = np.inf
    current_type = None
    in_notch = False

    for prim in prims:
        prim_bounds = _get_prim_bounds(prim)
        if _float_inside(pos, prim_bounds[dim][0], prim_bounds[dim][1]):
            if (
                _can_be_partial(prim)
                and _prim_metalp(prim)
                and not _covers(prim, prim_bounds)
            ):
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
            if (
                _can_be_partial(prim)
                and _prim_metalp(prim)
                and _covers(prim, _get_prim_bounds(prim))
            ):
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
        Simulation parameters; reads ``simulation_box``, ``FDTD_mesh_resolution``,
        ``FDTD_metal_mesh_resolution``, ``unit``, ``lambda0``,
        ``substrate_thickness_mm``, and ``substrate_cells``. A defined
        ``simulation_box`` is meshed exactly as given; when it is ``None`` the
        box is derived from the geometry (see :func:`auto_simulation_bounds`).
        Every dielectric layer ``substrate_thickness_mm`` thick gets exactly
        ``substrate_cells`` evenly spaced lines through it in z, counting both
        faces (see :meth:`_gen_substrate_lines`).
    smooth_ratio : float
        Maximum ratio between adjacent cell sizes. Default ``1.5``.
    min_lines : int
        Minimum number of mesh lines generated across any bounded interval.
        Default ``5``.
    requested_lines : list[list[float]] | None
        Per-dimension ``[x, y, z]`` positions the caller needs a mesh line on,
        registered as fixed lines (see :func:`add_fixed_line`) and added to the
        candidate bounds so each one becomes an interval boundary in its own
        right. A structure whose feature the automatic passes cannot see --
        the two slots of a CPW, narrow enough that the thin-interval collapse
        in :func:`_gen_mesh_in_bounds` leaves a single line across one -- asks
        for its edges this way. Default ``None``, which registers nothing and
        leaves the generated mesh bit-for-bit what it would otherwise be.
    """

    def __init__(
        self,
        csx: ContinuousStructure,
        params: SimParams,
        smooth_ratio: float = 1.5,
        min_lines: int = 5,
        requested_lines: list[list[float]] | None = None,
    ) -> None:
        self._csx = csx
        self._mesh_res = float(params.FDTD_mesh_resolution)
        self._metal_res = float(params.FDTD_metal_mesh_resolution)
        self._smooth = (smooth_ratio, smooth_ratio, smooth_ratio)
        self._unit = float(params.unit)
        self._lambda0 = float(params.lambda0)
        self._min_lines = min_lines
        self._substrate_thickness = float(params.substrate_thickness_mm)
        self._substrate_cells = int(params.substrate_cells)
        self.substrate_spans: list[tuple[float, float]] = []
        # Kept apart from self.fixed_lines, which _set_fixed_lines also fills
        # with the zero-thickness primitives' own positions: those are already
        # candidate bounds (a sheet contributes both of its equal bbox edges),
        # but as fp_nearest-rounded copies, so merging them into the bounds
        # below would swap a rounded value in for the raw one and shift an
        # existing line by up to a thousandth of a unit. Only what the caller
        # asked for is merged.
        self._requested_lines = (
            [[], [], []]
            if requested_lines is None
            else [[float(pos) for pos in dim_lines] for dim_lines in requested_lines]
        )
        user_simulation_bounds = params.simulation_bounds
        self._user_sim_box = user_simulation_bounds is not None
        self._sim_box = (
            None
            if user_simulation_bounds is None
            else tuple(
                (float(lower), float(upper)) for lower, upper in user_simulation_bounds
            )
        )
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
        # IsInside() (used below for notch/coverage testing) reads each
        # primitive's cached bounding box/winding data, which is only
        # populated by Update() -- never called automatically for a
        # freshly-built polygon/polyhedron, so every IsInside() call would
        # otherwise spuriously return False.
        for prim in physical_prims:
            prim.Update()
        self._set_fixed_lines(physical_prims)
        self._set_substrate_spans(physical_prims)
        bounds = _collect_all_bounds(
            physical_prims, self.fixed_lines, min_feature=self._metal_res
        )
        bounds = self._merge_requested_lines(bounds)
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
        e.g. a flat metal sheet) and for every position the caller asked for
        via ``requested_lines``, all of which must be meshed exactly."""
        for prim in prims:
            prim_bounds = _get_prim_bounds(prim)
            for dim in range(3):
                if fp_equalp(prim_bounds[dim][0], prim_bounds[dim][1]):
                    self.add_fixed_line(dim, fp_nearest(prim_bounds[dim][0]))
                self.fixed_lines[dim].sort()
                self.fixed_lines[dim] = _remove_dups(self.fixed_lines[dim])
        for dim, dim_lines in enumerate(self._requested_lines):
            if not dim_lines:
                continue
            for pos in dim_lines:
                self.add_fixed_line(dim, fp_nearest(pos))
            self.fixed_lines[dim] = _remove_dups(self.fixed_lines[dim])

    def _set_substrate_spans(self, primitives: list[CSPrimitives]) -> None:
        """Record the z-extent of every dielectric primitive that is exactly
        ``substrate_thickness_mm`` thick into ``self.substrate_spans``.

        The substrate is found from its own primitive rather than from a
        single bounded interval: requested lines or other primitives' edges
        inside the substrate split its thickness into several intervals, none
        of which would then match the substrate thickness on its own.
        """
        for primitive in primitives:
            if not _prim_materialp(primitive):
                continue
            z_bounds = _get_prim_bounds(primitive)[2]
            z_lower, z_upper = float(z_bounds[0]), float(z_bounds[1])
            if not fp_equalp(z_upper - z_lower, self._substrate_thickness):
                continue
            already_recorded = any(
                fp_equalp(z_lower, span_lower) and fp_equalp(z_upper, span_upper)
                for span_lower, span_upper in self.substrate_spans
            )
            if not already_recorded:
                self.substrate_spans.append((z_lower, z_upper))

    def _substrate_span_containing(
        self, dimension: int, bounded_type: BoundedType
    ) -> tuple[float, float] | None:
        """Return the substrate span a z-interval lies inside, or ``None``.

        Only a non-degenerate, non-metal interval in z qualifies; a
        zero-length interval (a fixed line) is left to
        :meth:`_gen_mesh_in_bounds` so the line it stands for is still placed.
        """
        if dimension != 2 or bounded_type.get_type() != Type.nonmetal:
            return None
        interval_lower, interval_upper = bounded_type.get_bounds()
        if fp_equalp(interval_lower, interval_upper):
            return None
        for span_lower, span_upper in self.substrate_spans:
            starts_inside = fp_gep(interval_lower, span_lower)
            ends_inside = fp_lep(interval_upper, span_upper)
            if starts_inside and ends_inside:
                return span_lower, span_upper
        return None

    def _gen_substrate_lines(
        self,
        span_lower: float,
        span_upper: float,
        interval_lower: float,
        interval_upper: float,
    ) -> None:
        """Mesh one interval of a substrate with evenly spaced z-lines.

        The substrate gets ``substrate_cells`` lines through its thickness,
        counting both faces: ``linspace(span_lower, span_upper,
        substrate_cells)``. Only the interior lines are added here. The faces
        are metal or air boundaries whose lines come from the neighbouring
        intervals -- a thin copper layer collapses to its own midline, and
        adding the face as well would open a sliver cell that shrinks the FDTD
        timestep. Every interval of a split substrate adds the same positions,
        which :meth:`_add_lines_to_mesh` de-duplicates, so the result does not
        depend on how many intervals the substrate was split into. An interval
        edge inside the substrate that is a fixed line (a requested line, say)
        is kept as well.
        """
        substrate_lines = list(
            np.linspace(span_lower, span_upper, self._substrate_cells)[1:-1]
        )
        for interval_bound in (interval_lower, interval_upper):
            inside_faces = not fp_equalp(interval_bound, span_lower) and not fp_equalp(
                interval_bound, span_upper
            )
            if inside_faces and self._is_fixed_line(2, interval_bound):
                substrate_lines.append(interval_bound)
        self._add_lines_to_mesh(np.array(sorted(substrate_lines)), 2)

    def _merge_requested_lines(self, bounds: list[list[float]]) -> list[list[float]]:
        """Fold ``requested_lines`` into the per-dimension candidate bounds.

        Registering a position as a fixed line is not on its own enough to put
        a mesh line there: :func:`_bounded_types` builds its intervals by
        walking the candidate bounds and only consults ``self.fixed_lines`` to
        decide whether a bound it is *already* visiting gets an interval of its
        own. A requested position that is not also a primitive edge -- an
        interior line across a substrate, say -- would never be visited, and so
        would be silently dropped. Every dimension the caller left empty is
        returned untouched, including the list objects themselves.
        """
        for dim, dim_lines in enumerate(self._requested_lines):
            if not dim_lines:
                continue
            merged = sorted(set(bounds[dim]) | {fp_nearest(pos) for pos in dim_lines})
            bounds[dim] = _remove_dups(merged, self.fixed_lines[dim])
        return bounds

    def add_fixed_line(self, dim: int, pos: float) -> None:
        """Register a must-keep mesh line position at ``pos``.

        Fixed lines are protected from the deduplication/smoothing passes
        applied to the rest of the mesh, and a line lands *exactly* on one: the
        thirds-rule shrink in :func:`_gen_mesh_in_bounds` is suppressed at an
        interval edge that is a fixed line, rather than pulling the line a
        third of a cell inside the metal. That is what makes this the right
        mechanism for a CPW's slot edges, where openEMS's own port probes, its
        excitation and its termination are all boxes spanning trace edge to
        ground edge and need a line on each.

        Must be called before the mesh is generated to have any effect, since
        ``__init__`` runs mesh generation immediately -- so outside the class
        this means passing ``requested_lines`` to the constructor. Note that a
        position which is not also a primitive's own bound has to reach the
        candidate bounds too (:func:`_merge_requested_lines`) before it is
        meshed.

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
        """Set ``self._sim_box`` from the geometry unless the user defined one.

        A user-defined ``simulation_box`` is used exactly as given;
        :meth:`_set_expanded_bounds` raises if the geometry does not fit it.
        Otherwise see :func:`auto_simulation_bounds`.
        """
        if self._user_sim_box:
            return
        self._sim_box = auto_simulation_bounds(dim_bounds, self._lambda0)

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
        coarser intervals that reference their neighbors' spacing. A z-interval
        inside a substrate is meshed by :meth:`_gen_substrate_lines` instead of
        :meth:`_gen_mesh_in_bounds`.
        """
        for dim, btypes in enumerate(bounded_types):
            for btype in btypes:
                lower = btype.get_bounds()[0]
                upper = btype.get_bounds()[1]
                substrate_span = self._substrate_span_containing(dim, btype)
                if substrate_span is not None:
                    self._gen_substrate_lines(*substrate_span, lower, upper)
                else:
                    is_metal = btype.get_type() == Type.metal
                    _, line_below = self._line_below(dim, lower)
                    _, line_above = self._line_above(dim, upper)
                    self._gen_mesh_in_bounds(
                        dim, lower, upper, line_below, line_above, is_metal
                    )
                self._add_to_ranges_meshed(dim, lower, upper)

    def _scaled_min_lines(self, dist: float, is_metal: bool, dim: int) -> int:
        """Return the minimum mesh-line count to force across an interval of
        size ``dist``, scaled down for a feature much smaller than its own
        target resolution -- in the x/y dimensions only.

        ``self._min_lines`` (the configured default, e.g. 5) is a sensible
        minimum for an interval that actually spans close to a full
        resolution cell -- but forcing that same count onto a feature far
        smaller than the resolution (a via a third the size of
        ``FDTD_metal_mesh_resolution``, say) packs unnecessarily fine lines
        into it for no accuracy benefit. Scales linearly (rounded down, so a
        feature has to actually earn the next line rather than being rounded
        up to it) from a floor of 3 lines (2 edges plus one interior line --
        fewer would give no interior resolution at all) as ``dist`` goes from
        0 up to its target resolution (``FDTD_metal_mesh_resolution`` for
        metal, ``FDTD_mesh_resolution`` otherwise), reaching the full
        ``self._min_lines`` once the interval is at least one resolution
        cell wide. An interval already much larger than its resolution
        (the common case -- a trace, a patch) saturates at
        ``self._min_lines`` unchanged, so this only affects features
        smaller than their own target resolution.

        ``dim == 2`` (z) is exempted and always returns ``self._min_lines``
        unscaled. In every structure this codebase generates, z is the
        layer stackup -- ground, substrate, conductor, air -- not a
        collection of independent lateral features. A via or a stub being
        physically small relative to the mesh resolution is a reasonable
        excuse for fewer lines across *it specifically*; the substrate
        being electrically thin at a high operating frequency (so its
        thickness ends up small relative to ``FDTD_mesh_resolution``,
        itself derived from the wavelength) is not the same kind of
        "small feature" -- it is the primary region of field variation
        between patch and ground, and under-resolving it for that reason
        measurably changes the simulation, not just its cell count.
        """
        if dim == 2:
            return self._min_lines
        res = self._metal_res if is_metal else self._mesh_res
        floor = min(3, self._min_lines)
        scaled = floor + (self._min_lines - floor) * (dist / res)
        return int(np.clip(np.floor(scaled), floor, self._min_lines))

    def _min_spacing(self, dist: float, is_metal: bool, dim: int) -> float:
        """Return the largest cell spacing that still fits at least
        :func:`_scaled_min_lines` mesh lines across a distance of ``dist``."""
        return dist / (self._scaled_min_lines(dist, is_metal, dim) - 1)

    def _min_cell(self, dim: int) -> float:
        """Return the smallest cell size to aim for in dimension ``dim``.

        A gap narrower than ``FDTD_metal_mesh_resolution`` still has to be
        meshed, but there is no reason to spend cells a quarter of that
        resolution on it. The FDTD timestep follows the smallest cell in the
        entire grid, so one sub-millimetre slot otherwise shrinks the timestep
        of every cell in the model -- the 0.77 mm inset slot of a patch
        antenna takes four lines at 0.198 mm and halves the timestep of the
        whole simulation.

        This floors the *target* spacing, not the generated cells:
        :func:`_gen_lines_in_bounds` still rounds its line count up, so an
        interval that does not divide evenly lands a little under the floor
        (0.255 mm against a 0.365 mm floor, on that patch). That is the
        intent -- the floor exists to stop a runaway, not to override the
        grid's own arithmetic.

        ``dim == 2`` is exempt for the same reason :func:`_scaled_min_lines`
        exempts it: z is the layer stackup, and a thin substrate is not a
        "small feature" that deserves fewer cells.
        """
        return 0.0 if dim == 2 else self._metal_res / 4.0

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
        with an abrupt jump in cell size. The result is floored at
        :func:`_min_cell`, which is where both caps are kept from running away
        on a sub-resolution feature.

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
            Whether the interval is metal (uses ``FDTD_metal_mesh_resolution``
            instead of ``FDTD_mesh_resolution``).

        Returns
        -------
        float
            Target spacing at the lower edge.
        """
        lower_spacing = self._metal_res if is_metal else self._mesh_res
        lower_spacing = np.min([lower_spacing, self._min_spacing(dist, is_metal, dim)])
        if line_below is not None and self._type_below_meshed(dim, lower):
            factor = 1.0
            if self._is_metal_bound(dim, lower) and not self._is_fixed_line(dim, lower):
                factor = 1.5 if self._type_below(dim, lower) == Type.nonmetal else 3.0
            spacing = factor * (lower - line_below)
            lower_spacing = np.min([lower_spacing, spacing])
            if dim == 2:
                lower_spacing = np.max([lower_spacing, self._mesh_res / 5])
        return float(np.max([lower_spacing, self._min_cell(dim)]))

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
        upper_spacing = np.min([upper_spacing, self._min_spacing(dist, is_metal, dim)])
        if line_above is not None and self._type_above_meshed(dim, upper):
            factor = 1.0
            if self._is_metal_bound(dim, upper) and not self._is_fixed_line(dim, upper):
                factor = 1.5 if self._type_above(dim, upper) == Type.nonmetal else 3.0
            spacing = factor * (line_above - upper)
            upper_spacing = np.min([upper_spacing, spacing])
            if dim == 2:
                upper_spacing = np.max([upper_spacing, self._mesh_res / 5])
        return float(np.max([upper_spacing, self._min_cell(dim)]))

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

        A z-interval inside a substrate never reaches this method; it gets
        ``substrate_cells`` evenly spaced lines from :meth:`_gen_substrate_lines`
        instead, so a thin substrate is not caught by the thin-interval
        collapse below.

        A degenerate (zero-length) interval gets a single line. An interval
        thinner than a quarter of its target resolution (``FDTD_metal_mesh_resolution``
        for metal, ``FDTD_mesh_resolution`` otherwise) gets a single line at
        its midpoint instead of a full geometric-series grid, since it is too
        thin to resolve with that grid -- without this, ``_gen_lines_in_bounds``
        would still be forced to place at least ``min_lines`` lines across it
        (via ``_min_spacing``), collapsing a sub-resolution sliver into a
        cluster of needlessly dense lines. This most commonly matters in z,
        for copper layers far thinner than the metal resolution, but applies
        in any dimension -- e.g. the many near-duplicate x/y vertices a
        finely-faceted curved polygon (a round pad's arc, say) contributes to
        the candidate bounds in :func:`_collect_all_bounds`, each pair of
        which would otherwise open its own over-resolved micro-interval.
        Otherwise, generates a
        geometric-series grid across the interval (:func:`_gen_lines_in_bounds`);
        for a metal interval, the interval is first shrunk at each edge that
        is not a fixed line or the outer simulation boundary, by a third (or
        two-thirds, if the far side is also already-meshed metal) of a cell
        width -- the "thirds rule", which keeps mesh lines off the metal
        edge itself for FDTD accuracy -- and the lines regenerated over the
        shrunk interval; a non-metal interval abutting a metal boundary is
        instead nudged inward by two-thirds of a cell at that edge and
        rebuilt.

        A metal interval in z collapses on the full metal resolution rather
        than a quarter of it. A z metal interval is a copper layer of the
        stackup, and one line through its thickness is all it ever needs --
        the layer carries a surface current, not a field varying across the
        foil. ``FDTD_metal_mesh_resolution`` scales with the wavelength while
        ``copper_thickness_mm`` does not, so the quarter-resolution cutoff
        collapses the copper at low frequencies and, once the metal
        resolution falls under four foil thicknesses -- 0.14 mm, reached
        around 25-30 GHz on a typical board -- stops: the layer suddenly takes
        ``min_lines`` lines across 35 um, a spacing several times finer than
        anything else in the grid, which costs cells and (since the FDTD
        timestep follows the smallest cell) timesteps -- 2.3x the total work
        in ``examples/InsetFedPatch_60GHz.py``. ``_scaled_min_lines`` does not
        soften this, since it deliberately exempts z. Non-metal z keeps the
        quarter-resolution cutoff: an air or dielectric interval that thin is
        a genuine gap between features, and the field really does vary across
        it.

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
            # A conductor far wider than its own edge resolution needs that
            # resolution only at its edges; its interior carries no more field
            # detail than the space around it, so let the geometric series
            # grade up to the global resolution in the middle. Without this the
            # test above almost never fires on a real board: Type.air is only
            # ever assigned outside the geometry's own extent (see
            # _set_expanded_bounds), so a full ground plane makes every metal
            # interval report metal on both sides, and a patch is filled edge
            # to edge at the metal resolution -- most of its lateral lines.
            wide = dim in (0, 1) and (upper - lower) > 2.0 * self._metal_res
            max_spacing = self._mesh_res if (touches_air or wide) else self._metal_res
        else:
            max_spacing = self._mesh_res

        if is_metal and dim == 2:
            # A copper layer of the stackup: one line through the foil is
            # enough at any frequency, so collapse on the full metal
            # resolution rather than a quarter of it. See the docstring.
            thin_threshold = self._metal_res
        else:
            thin_threshold = (self._metal_res if is_metal else self._mesh_res) / 4.0
        # Deliberately much smaller than thin_threshold above, and not
        # scaled off either resolution: this one only guards the
        # post-thirds-rule-shrink regeneration below against the shrink
        # degenerating the interval to near-zero or negative width, which
        # _gen_lines_in_bounds cannot handle. It is not a second "is this
        # interval thin enough to deserve only one line" judgment -- a
        # regular, moderately-sized interval (a via's own diameter, a
        # substrate's post-shrink thickness) can easily fall below a
        # resolution-scaled cutoff (e.g. metal_res / 4) purely from the
        # shrink's roughly one-cell pullback, without being anywhere near
        # actually degenerate; using that looser cutoff here once collapsed
        # both a legitimate ~1mm substrate sub-interval and a via's own
        # ~0.36mm remaining span down to a single line apiece.
        regen_threshold = self._metal_res * 0.01
        if fp_equalp(lower, upper):
            self._add_lines_to_mesh([lower], dim)
        elif dist < thin_threshold:
            mid = fp_nearest((lower + upper) / 2.0)
            self._add_lines_to_mesh(np.array([mid]), dim)
        else:
            lines = self._gen_lines_in_bounds(
                lower, upper, lower_spacing, upper_spacing, max_spacing, dim, is_metal
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
                lines = self._regen_lines_or_collapse(
                    lower,
                    upper,
                    lower_spacing,
                    upper_spacing,
                    max_spacing,
                    regen_threshold,
                    dim,
                    is_metal,
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
                    lines = self._regen_lines_or_collapse(
                        lower,
                        upper,
                        lower_spacing,
                        upper_spacing,
                        max_spacing,
                        regen_threshold,
                        dim,
                        is_metal,
                    )

            self._add_lines_to_mesh(lines, dim)

    def _regen_lines_or_collapse(
        self,
        lower: float,
        upper: float,
        lower_spacing: float,
        upper_spacing: float,
        max_spacing: float,
        regen_threshold: float,
        dim: int,
        is_metal: bool,
    ) -> np.ndarray:
        """Regenerate lines across a thirds-rule-shrunk interval, or collapse
        it to a single midpoint line if the shrink left it too thin to
        resolve.

        The thirds-rule shrink in :func:`_gen_mesh_in_bounds` moves ``lower``
        and ``upper`` inward by roughly a third of a cell each, *after* that
        function's own thin-interval check already passed on the pre-shrink
        bounds -- so a shrunk interval can end up thinner than
        ``regen_threshold`` (or even inverted, if the two shrinks together
        exceed the original span) without ever being caught by that check.
        Calling :func:`_gen_lines_in_bounds` unconditionally on such a sliver
        would still force :func:`_scaled_min_lines` lines across it via
        ``_min_spacing``, reproducing the exact runaway the thin-interval
        check exists to prevent.

        ``regen_threshold`` is deliberately an absolute floor
        (``FDTD_metal_mesh_resolution * 0.01``) rather than the type-scaled
        ``thin_threshold`` used for the pre-shrink check in
        :func:`_gen_mesh_in_bounds`: the question here is whether the shrink
        degenerated the interval into a sliver too thin for *any* resolution
        to usefully resolve, not whether it is thinner than its own
        (possibly much coarser, ``FDTD_mesh_resolution``-scaled) target. A
        substrate's post-shrink thickness sub-interval, for instance, is
        routinely well above this floor even though it can be well below the
        coarser, type-scaled threshold.
        """
        if upper - lower < regen_threshold:
            mid = fp_nearest((lower + upper) / 2.0)
            return np.array([mid])
        return self._gen_lines_in_bounds(
            lower, upper, lower_spacing, upper_spacing, max_spacing, dim, is_metal
        )

    def _gen_lines_in_bounds(
        self,
        lower: float,
        upper: float,
        lower_spacing: float,
        upper_spacing: float,
        max_spacing: float,
        dim: int,
        is_metal: bool,
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
        is_metal : bool
            Whether the interval is metal; passed to :func:`_scaled_min_lines`
            so the minimum line count scales off the right target resolution.

        Returns
        -------
        np.ndarray
            Mesh line positions from ``lower`` to ``upper``.
        """
        dist = upper - lower
        min_lines = self._scaled_min_lines(dist, is_metal, dim)
        smaller_spacing = np.min([lower_spacing, upper_spacing])
        larger_spacing = np.max([lower_spacing, upper_spacing])
        num_lower = dist / larger_spacing

        if (
            num_lower < min_lines
            or _spacing_at_dist(smaller_spacing, dist, self._smooth[dim])
            < larger_spacing
        ):
            return _lines_const_factor_in_bounds(
                lower,
                upper,
                lower_spacing,
                upper_spacing,
                dim,
                min_lines,
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

        while lower_num + upper_num < min_lines:
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
                grid.AddLine(ch, line)

    def _clean_close_lines(self, min_spacing: float | None = None) -> None:
        """Merge mesh lines that ended up closer together than ``min_spacing``.

        Reads the current lines from the CSXCAD grid (after
        ``SmoothMeshLines`` has run) and, for each dimension, collapses any
        run of lines closer than ``min_spacing`` into one -- keeping
        whichever line is fixed, or averaging if neither is -- then writes
        the cleaned lines back into the grid.

        Walks left to right comparing each line against the last *accepted*
        (possibly already-merged) line, rather than pairing lines two at a
        time: pairing would merge a close pair and then move on to compare
        the next line against whatever followed the pair, never re-checking
        the pair's own average against its neighbor -- so a chain of three
        or more mutually-close lines could still leave a sub-``min_spacing``
        gap next to the merged result. Comparing against the last accepted
        line instead folds an entire close run into one line, however long.

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
            for line in lines:
                if cleaned and (line - cleaned[-1]) < min_spacing:
                    if self._is_fixed_line(dim, cleaned[-1]):
                        continue
                    if self._is_fixed_line(dim, line):
                        cleaned[-1] = line
                    else:
                        cleaned[-1] = fp_nearest((cleaned[-1] + line) / 2.0)
                else:
                    cleaned.append(line)
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
