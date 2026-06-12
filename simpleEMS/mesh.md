# `mesh.py` — Algorithm Documentation

A thorough explanation of the FDTD auto-meshing algorithm implemented in `mesh.py`. Adapted from [pyems](https://github.com/edward-kay/pyems). All dimensions are in the project's geometry unit (default `unit = 1e-3`, i.e. **millimetres**).

---

## 0. Goal

`Mesh` takes a CSXCAD `ContinuousStructure` (populated with primitives) and generates a Cartesian FDTD grid. The pipeline:

```
1. collect bounding-box boundaries from all physical primitives
2. classify each inter-boundary region as metal / nonmetal / air
3. generate mesh lines per region using geometric series
4. apply the "thirds rule" at metal boundaries
5. globally smooth with CSXCAD's SmoothMeshLines
```

The constructor (`__init__`, line 337) stores parameters and calls `_generate()`. All logic is in `_generate()`.

---

## 1. Floating-point helpers (lines 19, 47–68)

`PREC = 10` rounds every comparison to 10 decimal places. This prevents floating-point drift between CSXCAD's geometry engine and the mesh generator.

| helper | meaning |
|--------|---------|
| `fp_nearest` | round to `PREC` |
| `fp_equalp` | equal up to `PREC` |
| `fp_gtp` / `fp_gep` | strictly greater / greater-or-equal |
| `fp_ltp` / `fp_lep` | strictly less / less-or-equal |

---

## 2. Primitive classification (lines 22–44, 71–98)

Two predicates on a CSXCAD primitive:

- `_prim_metalp` → type string in `{"Metal", "ConductingSheet", "LumpedElement"}`
- `_prim_materialp` → `"Material"`

Only these are considered **physical** primitives — ports, excitation, etc. are ignored.

`Type` (line 22) is the region classification enum: `metal=0`, `nonmetal=1`, `air=2` (None returned from `_type_at_pos` is also treated as air).

`BoundedType` (line 28) wraps a `[lower, upper]` interval with a `Type` and exposes `get_midpoint()` and `size()`.

---

## 3. Geometry discovery

### 3.1 Bounding boxes (`_get_prim_bounds`, line 80)
Returns the AABB of a primitive in **world coordinates**:
1. asks CSXCAD for the local AABB (`prim.GetBoundBox()`),
2. transforms both corners through `prim.GetTransform()`,
3. takes per-axis min/max.

### 3.2 Boundary pool (`_bounds_from_prims`, line 117)
For every physical primitive, appends both `lower` and `upper` bounds (6 values per primitive: 2 per axis). Then sorts and deduplicates per axis, preferring to keep `fixed` lines (see §4). Only AABB corners are used — no per-vertex extraction.

### 3.3 Region classification (`_bounded_types`, line 390; `_type_at_pos`, line 140)
With sorted boundaries `[b₀, b₁, b₂, …]` per axis, walks consecutive pairs `(bᵢ, bᵢ₊₁)` and classifies via `_type_at_pos()` at the midpoint. 

There is special handling for **fixed-line boundaries** (line 397): if a bound is also a fixed line, a zero-width `BoundedType` is created at that exact position (so zero-thickness metals are represented as single-point regions). The previous region is closed at the fixed line before the zero-width region, ensuring correct adjacency.

`_type_at_pos` resolves overlapping primitives with a **smallest-extent wins** rule (line 140):
- checks every primitive whose AABB contains the test point,
- the smallest-extent primitive (along the test dimension) wins,
- ties (within `rtol=1e-3`) prefer **metal**.

### 3.4 Adjacency queries (lines 487–509)
- `_type_below(dim, upper)` → type of the region whose upper bound equals `upper`
- `_type_above(dim, lower)` → type whose lower bound equals `lower`
- `_type_below_meshed` / `_type_above_meshed` → whether that region has already been meshed

These are used during spacing computation and the thirds rule to detect metal–metal vs metal–nonmetal adjacencies.

---

## 4. Fixed lines (`_set_fixed_lines`, line 377)

A **fixed line** is a mesh position locked in place — smoothing will not move it. Added by `_set_fixed_lines`:
- for every physical primitive, if its AABB is degenerate on any axis (`lower == upper`), that coordinate is added as a fixed line.

This pins the planes of **2D primitives** (zero-thickness sheets, ports, LumpedElements) into the mesh.

`add_fixed_line` (line 386) is also available as a public method for external callers.

---

## 5. Simulation box (`_set_sim_bounds_from_geometry`, line 414)

The user-supplied `simulation_box` is a starting point. It gets **overridden** by a box grown from the actual geometry:

```python
span = geo_max - geo_min
padding = max(lambda0 / 2, span * 0.15)
new_box = (geo_min - padding, geo_max + padding)
```

This guarantees at least λ₀/2 of air padding around the structure (standard for PML absorption).

### 5.1 Expansion to fill the box (`_set_expanded_bounds`, line 427)

Prepends/appends `Type.air` regions to fill any gap between the sim box and the outermost geometry. Raises `ValueError` if the final sim box is smaller than the geometry (with a message naming the offending dimension and required interval). Caches the final bounds as `self.sim_bounds[dim]`.

After this, every position from `sim_box.lower` to `sim_box.upper` on every axis is covered by a `BoundedType`.

### 5.2 Metal bounds (`_set_metal_bounds`, line 460)
Records every metal region's `[lower, upper]` endpoints for use in the thirds rule. Zero-width metal bounded types (from fixed lines) are excluded.

---

## 6. Per-region meshing

### 6.1 Processing order

Regions are sorted **by size, smallest first** (`_sort_bounded_types`, line 157). Small features (thin metals, narrow gaps) are meshed first so adjacent larger regions can see their lines and maintain smooth transitions.

`_gen_mesh_for_bounded_types` (line 514) iterates each region:
```python
for each (dim, btype):
    lower, upper = bounds
    line_below = nearest existing mesh line strictly below lower
    line_above = nearest existing mesh line strictly above  upper
    _gen_mesh_in_bounds(dim, lower, upper, line_below, line_above, is_metal)
    mark range as meshed
```

### 6.2 Spacing computation (`_lower_spacing` / `_upper_spacing`, lines 532–566)

Each end of the region gets a target cell spacing:

1. **Base resolution**: `metal_res` if metal, else `mesh_res`.
2. **Min-spacing clamp**: `min(base, dist / (min_lines - 1))` prevents fewer cells than `min_lines=5`.
3. **Adjacent-region constraint**: if the neighbour has already been meshed, the spacing is capped at `factor * (distance to neighbour line)` where:
   - `factor = 1.0` — default (no special interface)
   - `factor = 1.5` — at a **non-fixed** metal–nonmetal boundary (gentle growth)
   - `factor = 3.0` — at a **non-fixed** metal–air or metal–metal boundary (faster growth)

These spacings are **consumed**: they are passed directly into `_gen_lines_in_bounds`.

### 6.3 Thin z-metal special case (lines 584–586)

```python
elif is_metal and dim == 2 and dist < self._metal_res / 4.0:
    mid = fp_nearest((lower + upper) / 2.0)
    self._add_lines_to_mesh(np.array([mid]), dim)
```

If a metal region in the vertical dimension is thinner than one boundary cell (`metal_res / 4`), the full geometric series and thirds rule are skipped. A **single mesh line** is placed at the midpoint. This prevents sub-cell copper foils from generating many unnecessary lines.

### 6.4 Line generation (`_gen_lines_in_bounds`, line 639)

The core mesh-line generator for a region `[lower, upper]` with target spacings at each end.

**If spacings are equal** (within 1e-3 rel tolerance):
```python
num_lines = max(ceil(dist / spacing) + 1, min_lines)
return linspace(lower, upper, num_lines)
```

**If spacings differ**, a bidirectional geometric series is constructed:

1. Find the optimal split point `midpt` where the two series meet.
2. Grow from `lower_spacing → mid_spacing` on the left, and `mid_spacing → upper_spacing` on the right.
3. The split point is found numerically by solving `spacing_at_dist(lower, d) = spacing_at_dist(upper, dist - d)` (i.e. where both series reach the same cell size).
4. `mid_spacing` is capped at `max_spacing` (metal_res for metal, mesh_res otherwise).

Each half is generated by `_lines_const_factor_in_bounds` (line 222): starting from `lower`, each subsequent cell is `spacing * factor^k` until the target spacing is reached, with the growth factor bound by `smooth_ratio`.

### 6.5 Geometric series mathematics

Given `num - 1` cells growing geometrically from `small_spacing` by factor `factor`:

```
dist = small_spacing · Σ(factor^k),  k = 1 … num-1
```

`_factor_for_num` (line 166) solves for `factor` numerically via `scipy.optimize.fsolve`.

`_geom_series` (line 207) determines the minimal `num` such that the growth factor does not exceed the `smooth_ratio` limit and does not overshoot the target `larger_spacing`.

`_num_for_factor` (line 188) inverts the geometric sum to predict how many cells are needed given a known factor, spacing, and distance.

`_dist_for_max_spacings` (line 276) finds the optimal meeting point for two opposing series by solving `spacing1 - spacing2 = 0`.

### 6.6 Thirds rule for metals (lines 592–619)

After initial line generation, the metal boundaries are shifted to place mesh lines at 1/3 and 2/3 offsets from the metal edge:

```python
if lower not at sim-box edge and not fixed:
    if neighbour below is also metal (already meshed):
        lower += 2 * first_spacing / 3    # further in — neighbour has lines
    else:
        lower += first_spacing / 3         # 1/3 inside the metal
```

Same logic for the upper boundary. Lines are then **regenerated** with the shifted bounds.

The resulting cell at the metal boundary is `first_spacing / 3 + 2 * last_spacing / 3` (or vice versa), giving the canonical 1/3–2/3 split.

### 6.7 Thirds rule for nonmetals (lines 620–635)

If a nonmetal region abuts a metal boundary, the boundary is shifted **inward** by `2 * spacing / 3`:

```python
if metal_bound_at(lower):
    spacing = min(first_spacing, metal_res)
    lower += 2 * spacing / 3
if metal_bound_at(upper):
    spacing = min(last_spacing, metal_res)
    upper -= 2 * spacing / 3
```

Together with the metal-side shift (§6.6), this creates the complementary cell: the metal occupies 1/3 of the boundary cell, the nonmetal occupies 2/3 — placing a mesh line at the metal surface.

---

## 7. Smoothing (line 375)

After all regions are meshed, a single global smoothing pass is applied:

```python
self.mesh.SmoothMeshLines("all", self._mesh_res, self._smooth[0])
```

- `"all"` — applies to all dimensions
- `self._mesh_res` — the global target cell size
- `self._smooth[0]` — the maximum ratio between adjacent cells (default 1.5)

`SmoothMeshLines` **preserves all existing points** (fixed lines, thirds-rule anchors, region boundaries) and adds intermediate points wherever adjacent cell sizes exceed the ratio limit. This stitches together regions of different densities (coarse air → fine metal-edge) into a smooth progression.

---

## 8. Grid assembly (`_set_mesh_from_lines`, line 707)

```python
grid = self._csx.GetGrid()
grid.SetDeltaUnit(self._unit)
for i in range(3):
    grid.ClearLines(i)
for dim in range(3):
    for line in self.mesh_lines[dim]:
        grid.AddLine('xyz'[dim], fp_nearest(line))
```

All values are pre-rounded to `PREC` (10 decimal places). The grid unit is set explicitly so CSXCAD interprets coordinates correctly.

---

## 9. State accumulated on `self`

| Attribute | Type | Purpose |
|-----------|------|---------|
| `sim_bounds[dim]` | `[float, float]` | Final `[lower, upper]` per axis after geometry expansion |
| `ranges_meshed[dim]` | `list[[float, float]]` | Intervals already populated; used by `_pos_meshed` |
| `metal_bounds[dim]` | `list[float]` | Every metal region's boundary positions |
| `fixed_lines[dim]` | `list[float]` | Locked positions: zero-thickness primitives, thirds-rule anchors |
| `mesh_lines[dim]` | `list[float]` | The final per-axis mesh lines (sorted, deduplicated) |
| `bounded_types[dim]` | `list[BoundedType]` | Region list with types (cached for adjacency queries) |
| `smallest_res` | `float` | Set to `_metal_res`; currently unused in the main flow |
| `mesh` | `CSXCAD grid` | Alias for `csx.GetGrid()` |

---

## 10. Key design decisions

1. **Smallest-first processing order** (`_sort_bounded_types`) ensures thin features are meshed before bulk regions, so neighbour-aware spacing constraints in `_lower_spacing`/`_upper_spacing` see the correct adjacent lines.

2. **Smallest-AABB-wins classification with metal tiebreak** (`_type_at_pos`) handles overlapping primitives robustly — a thin metal patch inside a thick substrate is correctly classified as metal.

3. **Geometric series for grid transitions** — rather than uniform spacing, cells grow geometrically from a fine boundary spacing to a coarser interior spacing, with the growth factor bounded by `smooth_ratio=1.5`. This provides natural mesh grading.

4. **Bidirectional geometric series** — when lower and upper spacings differ, the two series meet at an interior point where both reach the same cell size, ensuring symmetry when adjacent metal features demand fine resolution at both ends.

5. **Fixed lines as anchors** — zero-thickness primitives (sheets, ports, LumpedElements) are pinned as fixed lines, and the thirds rule marks its offset lines as fixed. Everything else is fair game for smoothing to adjust.

6. **Thin z-metal special case** — copper foils thinner than `metal_res / 4` in the vertical dimension get a single line at midpoint rather than the full geometric series + thirds rule treatment.

7. **Global SmoothMeshLines** — a single call on all axes simultaneously ensures cross-axis consistency. It only adds points, never removes or moves the input lines.

8. **Sim box is grown from geometry** — the user-supplied `simulation_box` is a minimum; it expands by `max(λ₀/2, 15% × span)` around the geometry. If the geometry exceeds the given box, a `ValueError` is raised.

9. **`_lower_spacing`/`_upper_spacing` are consumed** — unlike earlier versions of this code, the spacing values computed by these methods are passed directly to `_gen_lines_in_bounds` and actively control the mesh density.

10. **`PREC=10` round-trip on every float** prevents floating-point misalignment between geometry and mesh. "Two positions are the same" is defined as "they agree to 10 decimal places".

---

## 11. Query helpers (`nearest_mesh_line`, `_line_below`, `_line_above`, lines 717–765)

`nearest_mesh_line` bisects the sorted `mesh_lines[dim]` and returns the closest line (by absolute distance). `_line_below` and `_line_above` return the nearest line **strictly** below or above the given position, walking past any exact match. These are used by `_gen_mesh_for_bounded_types` to find existing neighbour lines when constraining spacing.

---

## 12. Data flow

```
csx.GetAllPrimitives()
        |
        v
_physical_prims()            — Metal/ConductingSheet/LumpedElement/Material only
        |
        +--> _set_fixed_lines()          — zero-thickness planes → fixed lines
        |
        +--> _bounds_from_prims()        — sorted AABB boundaries per axis
        |
        +--> _set_sim_bounds_from_geometry() — grow box by max(lambda0/2, 15%)
        |
        +--> _bounded_types()            — classify regions via _type_at_pos
        |
        +--> _set_expanded_bounds()      — prepend/append air regions
        |
        +--> _set_metal_bounds()         — record metal boundaries
        |
        +--> _sort_bounded_types()       — smallest regions first
        |
        +--> _gen_mesh_for_bounded_types()
        |       +-- _gen_mesh_in_bounds()
        |              +-- _lower_spacing / _upper_spacing (target cell sizes)
        |              +-- _gen_lines_in_bounds (geometric series)
        |              +-- thirds rule adjustments
        |
        +--> SmoothMeshLines("all", ...) — global smoothing
        |
        +--> _set_mesh_from_lines()      — write to csx.GetGrid()
```
