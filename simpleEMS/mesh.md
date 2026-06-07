# `mesh.py` — Deep Walkthrough

A line-by-line explanation of the FDTD meshing algorithm. All numbers are in the project's geometry unit (default `unit = 1e-3`, i.e. **millimetres**).

---

## 0. Goal (from the module docstring, lines 1-10)

`Mesh` is an **auto-mesh generator** that turns a CSXCAD `ContinuousStructure` (already populated with primitives) into a Cartesian FDTD grid. The high-level pipeline is:

```
1. discover geometry from primitives (+ LinPolygon vertices)
2. classify each inter-boundary region as metal / nonmetal / air
3. seed each region with np.linspace lines
4. apply the "thirds rule" near metal edges (mark lines as fixed)
5. smooth between fixed lines using CSXCAD's SmoothMeshLines
```

The constructor (`Mesh.__init__`, lines 267-291) is small — it only copies parameters and calls `self._generate()`. All logic lives in `_generate()`.

---

## 1. Floating-point helpers (lines 22, 55-76)

A constant `PREC = 10` rounds every comparison/sort coordinate to 10 decimal places. This is critical: CSXCAD returns floats, and `0.1 + 0.2 != 0.3`-style errors would otherwise produce duplicated mesh lines or missed boundary alignments. Every comparison goes through:

| helper | meaning |
|---|---|
| `fp_nearest` | round to PREC |
| `fp_equalp` | equal up to PREC |
| `fp_gtp` / `fp_gep` | strictly greater / greater-or-equal |
| `fp_ltp` / `fp_lep` | strictly less / less-or-equal |

This means "two boundaries are the same" is defined as "they agree to 10 decimal places", which matches the geometry generation precision used elsewhere (`fp_precision` in `SimParams` is 3, but mesh comparisons round at 10 to be safer).

---

## 2. Primitive classification (lines 79-90)

Two predicates:

- `_prim_metalp` → type string in `{"Metal", "ConductingSheet", "LumpedElement", "Excitation"}`
- `_prim_materialp` → `"Material"`

Only these two categories are considered *physical*. Excitation ports, lumped elements, and conducting sheets are treated as metals for meshing (they need sub-cell resolution at their edges).

`Type` (lines 25-28) is the enum used per region: `metal=0`, `nonmetal=1`, `air=2`.

`BoundedType` (lines 31-52) is a region descriptor: a `[lower, upper]` interval tagged with a `Type`. It also exposes `get_midpoint()` and `size()`.

---

## 3. Geometry discovery

### 3.1 Bounding boxes (lines 93-103)
`_get_prim_bounds` returns the AABB in **world coordinates** by:
1. asking CSXCAD for the local AABB (`prim.GetBoundBox()`),
2. transforming the two corners through `prim.GetTransform()`,
3. taking the min/max along each axis.

Note: an axis-aligned primitive returns a degenerate box (lower==upper) on axes it doesn't extend along. This is how the algorithm later detects 2D primitives (see §4).

### 3.2 LinPolygon vertex handling (lines 106-134)
Standard `GetBoundBox` for a `CSPrimLinPoly`/`CSPrimPolygon` only gives the polygon's overall AABB, which loses individual vertex positions. `_get_linpoly_vertex_bounds`:
- reads `(x_verts, y_verts)` from `prim.GetCoords()`,
- folds the polygon's `Elevation` and `NormDir` into 3D points,
- applies the primitive's transform,
- returns per-axis lists of vertex coordinates.

These per-vertex points are added to the boundary pool (see §3.3) so that mesh lines land **on every polygon vertex** — important for conformal trace edges that have slanted sides.

### 3.3 Boundary pool (`_collect_all_bounds`, lines 161-179)
For every physical primitive:
- append `lower` and `upper` of the transformed AABB on each axis,
- if it's a LinPolygon, also append every vertex coordinate.

Then sort each axis's list and de-duplicate (preferring `fixed` lines — see §4).

### 3.4 Region classification (`_bounded_types`, lines 333-346; `_type_at_pos`, lines 190-204)
With sorted axis-wise boundaries `[b0, b1, b2, …]`, the algorithm walks consecutive pairs `(b_i, b_{i+1})` and asks `_type_at_pos(prims, dim, midpoint)`.

`_type_at_pos` resolves overlapping primitives using a **smallest-volume wins** rule (lines 191-204):
- for every primitive whose AABB contains the test point, compute its extent along `dim`,
- the smallest-extent primitive wins,
- ties (within `rtol=1e-3`) prefer **metal** — so a thin metal sheet inside a thick substrate is correctly classified as metal.

This produces a `list[list[BoundedType]]` indexed by `[dim][region]`.

### 3.5 Type at adjacent regions (lines 434-456)
`_type_below(dim, upper)` and `_type_above(dim, lower)` find the region whose upper/lower bound equals the given boundary. They're used later to detect metal↔nonmetal vs metal↔air vs metal↔metal adjacencies.

---

## 4. Fixed lines (lines 316-327)

A **fixed line** is a mesh position that is locked in place and will not be moved by smoothing. They are added by `_set_fixed_lines`:
- for every physical primitive, if the AABB is degenerate on an axis (`lower == upper` on that axis), that coordinate is added as a fixed line.

This handles:
- **2D primitives** (`CSPrimBox` with zero thickness) → their plane becomes a hard mesh line,
- **LinPolygons** → their vertex x/y/z positions are pinned,
- **ports / sheets** → their plane is pinned.

`add_fixed_line` (lines 325-327) is also called during mesh generation to lock the thirds-rule offset lines (see §6.4).

---

## 5. Simulation box & expansion (lines 348-392)

The user supplies a `simulation_box`; the geometry is then allowed to push it outward:

```python
span = geo_max - geo_min
padding = max(lambda0/2, span * 0.15)
new_box = (geo_min - padding, geo_max + padding)
```

So the sim box is the geometric bounding box plus **at least λ₀/2 air on every side** (or 15% of the span — whichever is bigger). The factor of λ₀/2 is standard for absorbing boundaries (PML needs room to ramp the field down).

`_set_expanded_bounds` then:
- prepends/appends `Type.air` regions to fill the gap between the sim box and the geometry,
- **raises `ValueError`** if the user's sim box is *smaller* than the geometry (with a useful message naming the offending dimension and interval),
- caches the result as `self.sim_bounds[dim]`.

After this step, every region from `sim_box.lower` to `sim_box.upper` along every axis is covered by a `BoundedType`.

---

## 6. Mesh generation per region (`_gen_mesh_for_bounded_types` → `_gen_mesh_in_bounds`)

This is the heart of the file. It iterates regions sorted **by size, smallest first** (`_sort_bounded_types`, lines 207-213) — the idea is that small/thin regions are seeded first, then larger regions can be told "there's already a line nearby, don't go coarser than that".

For each region it calls `_gen_mesh_in_bounds(dim, lower, upper, line_below, line_above, btype)` with the nearest already-placed line on each side (queried via `_line_below` / `_line_above` at lines 739-771 — they bisect `self.mesh_lines` and step past any line that coincides with the boundary).

### 6.1 Min-spacing guard (line 479-480)
`_min_spacing(dist) = dist / (min_lines - 1)` enforces at least `min_lines=5` cells across any region, no matter how big.

### 6.2 Lower / upper spacing caps (lines 482-516)
For each side, the per-region spacing is:
```python
spacing = min(target_res, min_spacing(dist))
if neighbour_already_meshed:
    spacing = min(spacing, factor * neighbour_cell_size)
```
- `target_res` is `_metal_res` for metal regions, `_mesh_res` otherwise.
- The neighbour cap uses a **factor**:
  - `1.0` by default,
  - `1.5` if the boundary is a metal↔nonmetal interface and the line is **not** fixed (so the cell growth at dielectric edges is gentle),
  - `3.0` if the boundary is a metal↔air or metal↔metal interface and the line is not fixed (air interfaces can grow much faster because there's no field singularity to resolve).

This factor logic is computed in `_lower_spacing`/`_upper_spacing` but — as far as I can see in the file — the returned `lower_spacing` / `upper_spacing` values are *not actually consumed* in `_gen_mesh_in_bounds`. The seeding uses `np.linspace(lower, upper, num)` where `num` is decided heuristically (§6.3). This is likely a **latent helper** that is currently dormant; the neighbour-cap idea is implemented differently inside the per-region logic. Worth flagging — see §10.

### 6.3 Number-of-points heuristic (lines 545-576)
The branch on region size (`dist` = upper − lower):

| Condition | Region type | Action |
|---|---|---|
| `dist < thirds_cell` (2D dim=1) | metal & not dim=2 | **extended seeding** (lines 547-560): expand `lower` and `upper` by `offset_out = metal_res/6`, linspace across the expanded span with `~offset_out * 0.6` cell size, then **drop the two original boundary lines** so the result contains only the new interior points. Used for ultra-thin metal lines. |
| `dist < thirds_cell` (1D dim=1) | any other | `num = 2` → just the two endpoints. |
| `dist < 2*thirds_cell` (dim=1) | metal | same as the first row (treats thin metals on the y-dim specially). |
| else | **air** | `air_spacing = max(mesh_res*3, sim_box/10)`; `num = max(ceil(dist/air_spacing)+1, min_lines)` |
| else | **metal** | `num = max(min_lines, ceil(dist/(metal_res*5))+1)` |
| else | **nonmetal** (substrate) | `num = max(min_lines, ceil(dist/(mesh_res/4))+1)`, **and for z-axis**: `num = max(substrate_cells+1, num)` — this forces ≥ 5 cells through the substrate thickness by default. |

Constants used:
- `offset_in = metal_res / 12`
- `offset_out = metal_res / 6`
- `thirds_cell = offset_in + offset_out = metal_res / 4`

For dim=2 (z), thin-metal regions are special-cased with `skip_thirds` (line 542) — when the metal is thinner than the thirds cell, the algorithm **does not** apply the offset logic, and instead emits a single midpoint line (line 580) so extremely thin foils (e.g. copper cladding) don't generate unnecessary inner lines.

### 6.4 The "thirds rule" (lines 587-647)

The thirds rule is the textbook openEMS trick: at a metal boundary, place mesh lines at
- `+1/3 * cell_size` *inside* the metal, and
- `+2/3 * cell_size` *outside* the metal,

so the cell edges are **never on the conductor surface** and the field can be interpolated. Here it's implemented as **offsets from the metal boundary** rather than from a target cell size — the cell size becomes `offset_in + offset_out = metal_res/4`, which is the desired sub-cell size at the metal edge.

For each metal region:

1. **Scale offsets for thin metals** (lines 588-591):
   ```python
   scale = min(1.0, dist / thirds_cell)
   adj_in  = offset_in  * scale
   adj_out = offset_out * scale
   ```
   so for metals thinner than `metal_res/4`, the inner and outer thirds are scaled down proportionally.

2. **Move the seeding bounds inward** (lines 595-614) for any non-fixed, non-sim-box boundary:
   - default: `lower += adj_in` (1/3 inside),
   - **but** if the region *below* this metal is also metal (back-to-back metals share a face, e.g. two substrates around a shared ground plane), use `adj_out` instead — there is no field to resolve on the other side, so the line can be placed farther out.

3. **Re-linspace between the new bounds** if they moved (lines 616-619).

4. **Drop a 2/3-outside fixed line into the adjacent region** (lines 622-641), unless the neighbour is also metal:
   ```python
   ol = orig_lower - offset_out   # 2/3 outside the metal, into the dielectric
   add_fixed_line(dim, ol); add_mesh_line(dim, ol)
   ```
   This is how the **outer** thirds line gets placed when the metal's own region doesn't seed it. The check `is_lower_metal` prevents duplicating the line in the metal-on-both-sides case (where the neighbour metal has already provided the corresponding line on its own side).

5. **Mark the adjusted `lines[0]` and `lines[-1]` as fixed** (lines 644-647) so smoothing won't move them.

### 6.5 Thirds rule for nonmetal regions (lines 649-668)
If the *nonmetal* region is adjacent to a metal boundary, push its seeding bounds *outward* by the outer offset:
```python
if is_metal_bound(lower): lower += offset_adj
if is_metal_bound(upper): upper -= offset_adj
```
where `offset_adj`:
- equals `offset_out` for normal metals,
- for **thin metals** (`metal_t < thirds_cell`): `min(offset_out, max(offset_in, metal_t*3))` — so the offset never exceeds the half cell on the air side, and never undershoots the inner offset.

Then re-linspace between the shrunk bounds and emit the new lines.

### 6.6 Why both metal-side and nonmetal-side adjustments?
After metal processing, the metal region's linspace sits *inside* `[lower+adj_in, upper-adj_in]` (or further out for metal-on-metal). The nonmetal region's linspace sits *outside* `[lower+offset_out, upper-offset_out]`. Together with the 2/3-outside fixed line dropped by the metal side (step 4), this yields the canonical four-line neighbourhood at a metal interface (1/3 in, 2/3 in, 2/3 out, …). The fixed-line tags guarantee smoothing keeps this structure.

---

## 7. Smoothing (`_smooth_non_fixed_segments`, lines 676-694)

After every region is seeded, the code calls CSXCAD's `SmoothMeshLines`:

```python
smoothed = SmoothMeshLines(all_lines, mesh_res, smooth_ratio)
```

- `mesh_res` is the **target cell size**,
- `smooth_ratio=1.5` is the **max ratio** between adjacent cells,
- `SmoothMeshLines` only *adds* new lines, never moves or removes the input points. So all fixed lines (thirds-rule pins, zero-thickness primitive planes, LinPolygon vertex coordinates) are preserved.
- It's applied per-dimension with the same `1.5` ratio.

The role of smoothing is to **stitch together** the regions of different densities produced in step 6 — the coarse air region and the fine metal-edge region must transition smoothly so FDTD dispersion is bounded. Because fixed lines are inputs, smoothing can only add cells *between* them.

A `try/except` wraps the call so a CSXCAD-version mismatch doesn't kill the pipeline; on failure the unsmoothed lines stand.

---

## 8. Writing the grid back to CSXCAD (`_set_mesh_from_lines`, lines 708-716)

```python
grid = csx.GetGrid()
grid.SetDeltaUnit(unit)
for i in range(3):
    grid.ClearLines(i)
for dim, line in enumerate(self.mesh_lines[dim]):
    grid.AddLine('xyz'[dim], fp_nearest(line))
```

`SetDeltaUnit` sets the *unit* used by the grid (default `1e-3` mm). Lines are added one at a time so CSXCAD can sort/deduplicate internally; we pre-round to 10 decimals to match the helper precision.

---

## 9. State the algorithm accumulates on `self`

| attribute | meaning |
|---|---|
| `sim_bounds[dim]` | final `[lower, upper]` per axis (sim box after geometry expansion) |
| `ranges_meshed[dim]` | list of `[lo, hi]` intervals already populated, used by `_pos_meshed` |
| `metal_bounds[dim]` | every metal region's `[lo, hi]` endpoints, used by `_is_metal_bound` |
| `fixed_lines[dim]` | locked positions: zero-thickness planes, LinPoly vertices, thirds-rule anchors |
| `mesh_lines[dim]` | the final per-axis mesh (sorted, deduplicated) |
| `bounded_types[dim]` | the region list with types (cached for queries) |
| `smallest_res` | currently set to `_metal_res`; no further use observed in this file |
| `self.mesh` | alias for `csx.GetGrid()` |

---

## 10. Key design decisions & things worth knowing

1. **Sort regions by size first** (`_sort_bounded_types`) so small thin features are meshed before the surrounding bulk; this lets `_line_below`/`_line_above` see a neighbour before the larger region is seeded, so the spacing cap kicks in correctly.

2. **Smallest-AABB-wins classification with metal tiebreak** (`_type_at_pos`) — robust to overlapping primitives (e.g. a thin substrate inside a thicker air box, or a metal patch on top of substrate that is *also* inside the air box).

3. **Two-sided thirds rule with thin-metal scaling** — the algorithm correctly handles metals thinner than the natural thirds cell, metals sitting on the sim-box boundary (no thirds line emitted outside the box), and metal-on-metal interfaces (no duplicate outer line).

4. **Substrate override** — for the z-axis (`dim==2`), the nonmetal region is forced to have at least `substrate_cells + 1` lines, guaranteeing a minimum number of vertical mesh cells through the dielectric regardless of the thickness-based heuristic.

5. **Air is treated very coarsely** — `air_spacing = max(mesh_res*3, sim_box/10)` means the air region is sampled at ≥ 3× the substrate resolution, with a lower bound of one tenth of the sim box (so a tiny patch in a huge sim box still has at least 10 cells across). Combined with smoothing, the air gradually densifies near the structure.

6. **PREC=10 round-trip on every input** — every float that enters a comparison, an `insort`, or the CSXCAD grid is rounded to 10 decimals. This is what makes "the vertex at `x=3.0`" align with "the box's edge at `x=3.0`" even though they were generated by different code paths.

7. **LinPolygon special path** — without vertex extraction, a triangular patch with vertices at `(0,0), (10,0), (0,10)` would only contribute AABB corners, so the diagonal would have no mesh alignment. `_get_linpoly_vertex_bounds` pins all vertices as fixed lines, which is what makes slanted edges resolvable.

8. **Sim box is grown automatically** — the user-supplied `simulation_box` is a *minimum*; if geometry sticks out, it expands by `max(λ₀/2, 15% × span)`. The check is one-way: the sim box can only grow, not shrink. If the geometry exceeds the sim box the user gave, a `ValueError` is raised (lines 375-380).

9. **Smoothing is permissive-fail** — `SmoothMeshLines` is wrapped in `try/except`, so a bad line set won't crash the mesher; it'll just skip smoothing for that axis.

10. **Latent helper** — `_lower_spacing` / `_upper_spacing` compute neighbour-aware spacing caps but their results are never used in `_gen_mesh_in_bounds`. The neighbouring-cell cap is currently implemented implicitly via the "drop a 2/3-outside fixed line" branch and via smoothing. If you wanted to fix the apparent dead code, those helpers could be the basis for tightening `num` directly inside the linspace.

11. **`_line_below` / `_line_above` query helpers** (lines 739-771) — they bisect `self.mesh_lines` to find the nearest line *strictly below* or *strictly above* the given position. They walk past coincident lines so the "below" of position `3.0` is the line at `2.9`, not the `3.0` itself. This is what makes the seeding aware of lines that have already been placed at a boundary.

12. **`_metal_thickness_at` is the bridge to thin-metal handling on the dielectric side** — the nonmetal branch in §6.5 needs to know how thick the adjacent metal is to decide whether to scale down its `offset_adj`. This is the only place where the metal's *thickness* (not just its boundary) is used.

---

## 11. Quick reference: the data flow

```
csx.GetAllPrimitives()
        |
        v
 _physical_prims()  -- filters to Metal/Material only
        |
        +--> _set_fixed_lines()         -- zero-thickness planes, LinPoly vertices
        |
        +--> _collect_all_bounds()      -- AABBs + LinPoly vertex coords
        |
        +--> _set_sim_bounds_from_geometry() -- grow box by max(lambda0/2, 15%)
        |
        +--> _bounded_types()           -- regions + types via _type_at_pos
        |
        +--> _set_expanded_bounds()     -- prepend/append Type.air regions
        |
        +--> _set_metal_bounds()        -- record metal boundaries
        |
        +--> _sort_bounded_types()      -- smallest regions first
        |
        +--> _gen_mesh_for_bounded_types()
        |       +-- _gen_mesh_in_bounds()   -- linspace + thirds rule per region
        |
        +--> _smooth_non_fixed_segments() -- CSXCAD SmoothMeshLines
        |
        +--> _set_mesh_from_lines()     -- write back to csx.GetGrid()
```
