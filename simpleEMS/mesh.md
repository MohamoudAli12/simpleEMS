# Mesh Generation Algorithm

## Overview

The mesh generator discovers geometry from CSXCAD primitives, classifies regions
between boundaries as metal/nonmetal/air, generates mesh lines per region, applies
the thirds rule at metal boundaries, and globally smooths between fixed lines.

```
_prims_  →  collect bounds  →  classify regions  →  mesh per region  →  smooth  →  CSXCAD grid
```

---

## Constants

**`PREC = 10`** — floating-point precision for all comparisons. All positions are
rounded to 10 decimal places (`fp_nearest`, `fp_equalp`, etc.). This prevents
micro-errors from repeated arithmetic from creating near-duplicate lines or
misidentifying boundaries.

---

## Pipeline (`_generate`, line 287)

```
1. GetAllPrimitives() from CSXCAD
2. Filter to physical primitives (metal + material)
3. Set fixed lines from zero-thickness primitives (sheets, 2D ports)
4. Collect all unique boundary positions (box edges + LinPoly vertices)
5. Set simulation bounds from geometry (geometry + λ₀/2 padding)
6. Classify intervals between adjacent boundaries → BoundedType
7. Expand bounds with air padding to fill the simulation box
8. Record metal boundaries (for _is_metal_bound queries)
9. Sort bounded types by size (smallest first)
10. For each bounded type: generate mesh in bounds
11. Smooth non-fixed segments globally via CSXCAD SmoothMeshLines
12. Write mesh lines to CSXCAD grid
```

---

## Key Data Structures

### `Type` (line 25)
```python
metal = 0     # PEC / conductive
nonmetal = 1  # dielectric (substrate)
air = 2       # free space / unoccupied (also `None`)
```

### `BoundedType` (line 31)
A region `[lower, upper]` in one dimension, classified by probing the
midpoint with `_type_at_pos`. Each interval between adjacent primitive
boundaries produces one `BoundedType`.

### `ranges_meshed[dim]` (line 275)
Tracks which intervals have been meshed already. Used by the thirds rule
to detect whether an adjacent region already exists (metal-metal boundaries
need different treatment than metal-air boundaries).

### `metal_bounds[dim]` (line 276)
All positions that are edges of metal regions (populated before any mesh
lines are generated). Used by `_is_metal_bound` to determine if a given
coordinate is a metal/nonmetal interface.

### `fixed_lines[dim]` (line 277)
Lines that must NOT be moved or removed during smoothing. Includes
thirds-rule boundary lines and lines from zero-thickness primitives
(sheets, ports). Fixed lines take priority in `_remove_dups`.

---

## Boundary Collection (`_collect_all_bounds`, line 151)

For each physical primitive:
- Box primitives → add both box edges in each dimension
- LinPoly primitives → add both extrusion bounds + every vertex coordinate

All positions are merged, sorted, and deduplicated (`_remove_dups` with
`fp_equalp` precision). Fixed lines are passed so that a fixed line and a
geometric boundary at the same position are kept as one entry.

---

## Region Classification (`_type_at_pos`, line 180)

For a given position `pos` in dimension `dim`:

```
For each primitive:
  if pos falls inside the primitive's bounding box in dim:
    dim_size = bounding box span in dim
    if dim_size < smallest_dim_found:
      → replace winner with this primitive's type
    elif dim_size ≈ smallest_dim_found (within 1e-3 rtol):
      → metal wins over nonmetal (tie-breaker)
```

The **smallest primitive in the given dimension** wins. This correctly
handles overlapping structures: a thin copper trace on a thick substrate
has a smaller z-extent than the substrate, so `_type_at_pos` returns
`Type.metal` for z-positions inside the copper even though the substrate
also contains that position.

**Important limitation**: `_type_at_pos` projects all primitives onto a
single dimension. A stub at x=`[a,b]` with y-extent `[0, c]` will cause
ALL y-positions in `[0, c]` to be classified as metal, even at x-positions
far from the stub. This is a 1D projection — it cannot distinguish
(x,y) locations. The mesh in each dimension is generated independently.

---

## Simulation Bounds (`_set_sim_bounds_from_geometry`, line 338)

```
padding = max(λ₀ / 2, geometry_span * 0.15)
sim_box = [geometry_min - padding, geometry_max + padding]
```

Pads the geometry by at least λ₀/2 on each side, ensuring the PML
absorbs outgoing waves. The `λ₀/2` minimum guarantees the simulation
box is electrically large enough.

---

## Metal Boundaries (`_set_metal_bounds`, line 390)

Iterates all bounded types and records every interval endpoint where
`type == metal`. The resulting `metal_bounds[dim]` list is used by
`_is_metal_bound` (line 408) and `_metal_thickness_at` (line 411)
which are called during the thirds rule.

This is a PURELY GEOMETRIC step — it runs before any mesh lines are
generated, so it doesn't depend on mesh content.

---

## Mesh Generation Order

Bounded types are sorted BY SIZE ascending (smallest interval first).
This ensures:

1. **Thin metal regions** (copper traces, ground plane) are meshed first
2. **Thicker metals** are meshed next
3. **Nonmetal regions** (substrate, air) are meshed last

The thirds-rule for metal regions uses `_pos_meshed` to detect whether an
adjacent region has already been meshed. When it has, the boundary is
treated as a metal-metal interface (shift outward by `offset_out`).

---

## Core: `_gen_mesh_in_bounds` (line 510)

This is the heart of the mesh generator. Every bounded type in every
dimension is processed once.

### Parameters

- `lower`, `upper`: region boundaries
- `line_below`, `line_above`: nearest already-meshed lines outside the region
  (currently unused — vestigial from pyems)
- `btype`: the `BoundedType` (metal/nonmetal/air)

### Step 1: Thirds Rule Constants (line 528)

```
mesh_res = λ₀ / mesh_resolution_factor    (default 20)
offset_in  = mesh_res / 12    # 1/3 of a boundary cell, inside metal
offset_out = mesh_res / 6     # 2/3 of a boundary cell, outside metal
thirds_cell = offset_in + offset_out = mesh_res / 4
```

The **thirds rule** shifts the first/last mesh line of a region away from
a metal boundary so the FDTD cell at the interface is 1/3 inside the metal
and 2/3 outside. This improves field accuracy at PEC boundaries.

### Step 2: skip_thirds (line 534)

```python
skip_thirds = is_metal and dist < thirds_cell and dim == 2
```

For **z-dimension metal regions** thinner than `thirds_cell` (copper
traces, ground planes — typically 0.035mm), the full thirds-rule
machinery is skipped. Instead, a **single line at the midpoint** is
added (line 564–566). This prevents spending mesh lines on features
far below mesh resolution.

### Step 3: Determine Number of Points (lines 536–561)

Five cases:

| Condition | Density |
|-----------|---------|
| `dist < threshold` (thin region) | `num = 2` (just boundaries) |
| — extended linspace (metal, x/y) | see below |
| Air | `air_spacing = max(mesh_res * 3, sim_box_span / 10)` |
| Metal | `spacing = mesh_res * 5` (very coarse — thirds rule and SmoothMeshLines add detail) |
| Nonmetal (substrate) | `spacing = mesh_res / 4`, forced to ≥ `substrate_cells + 1` in z |

#### Extended Linspace for Thin Metals (x/y dimension, line 538)

For metal regions narrower than the threshold that are NOT in the z-dimension,
the standard metal-density + thirds-rule approach produces poor results
(the thirds-rule shifts can overshoot or leave only 2 boundary lines).
Instead:

```python
ext_lower = lower - offset_out
ext_upper = upper + offset_out
num = max(4, ceil(total_span / (offset_out * 0.6)) + 1)
lines = np.linspace(ext_lower, ext_upper, num)
# Remove any line that lands exactly on the metal boundary
lines = filter(lambda l: not fp_equalp(l, lower) and not fp_equalp(l, upper), lines)
```

This spans from `offset_out` outside one side to `offset_out` outside the
other, producing a smooth transition from exterior → metal → exterior in a
single linspace. Lines that would land exactly on the metal boundary are
dropped (preserving the thirds rule constraint that no line is at the
edge). The density `offset_out * 0.6` provides ~5-7 points across the span
for typical thin metals.

Threshold for entry:

| Dimension | Threshold | Rationale |
|-----------|-----------|-----------|
| x (dim=0) | `thirds_cell` | Narrow transverse features |
| y (dim=1) | `2 × thirds_cell` | Wider threshold for stub-end transitions |
| z (dim=2) | (excluded) | Uses `skip_thirds` instead |

### Step 4: Initial Linspace (line 563)

```python
lines = np.linspace(lower, upper, num)
```

Uniform point distribution. All subsequent steps shift boundaries and
regenerate the linspace.

### Step 5: Metal Thirds Rule (lines 572–621)

For metal regions that are NOT handled by extended linspace or skip_thirds:

1. **Scale offsets** for thin metals: `scale = min(1.0, dist / thirds_cell)`
   Reduces `adj_in` and `adj_out` proportionally for metals thinner than
   `thirds_cell`.

2. **Shift lower boundary**:
   - Default: `adj_lower = adj_in` (1/3 cell inside)
   - If region below is meshed metal: `adj_lower = adj_out` (2/3 cell inward
     — the adjacent metal's external line already provides the 2/3 offset)
   - Skip if at simulation box edge or already a fixed line

3. **Shift upper boundary**: Same logic.

4. **Regenerate linspace** if boundaries were shifted, with at least 2 points.

5. **Add external lines** (`lower - offset_out`, `upper + offset_out`) in the
   adjacent air/nonmetal region. These are fixed lines. Skip if the adjacent
   region is also metal (its own thirds rule provides the external line).

6. **Mark adjusted boundary lines as fixed** to prevent smoothing from
   moving them.

### Step 6: Nonmetal Thirds Rule (lines 623–642)

For nonmetal and air regions adjacent to metal boundaries:

```
if lower is a metal bound:
    metal_t = thickness of that metal region
    if metal_t < thirds_cell:
        offset_adj = min(offset_out, max(offset_in, metal_t * 3))
    else:
        offset_adj = offset_out
    lower += offset_adj

if upper is a metal bound:
    same logic for upper
    upper -= offset_adj

if either shift happened:
    regenerate linspace with at least 2 points
```

The `metal_t * 3` cap prevents the shift from pushing too far past a thin
adjacent metal trace (e.g., a 0.035mm copper trace caps the shift at
0.105mm). The `max(offset_in, ...)` guard ensures the gap doesn't shrink
below 1/3 of a mesh cell, which would unnecessarily limit the CFL time
step.

**For very thick metals** (metal_t ≥ thirds_cell): the full `offset_out`
(= 2/3 of a cell) is used, giving the standard 1/3-inside + 2/3-outside
FDTD cell at the boundary.

### Step 7: Add Lines to Mesh (line 644)

All lines for the region are inserted, deduplicated with `_remove_dups`,
and the region is recorded in `ranges_meshed` for subsequent regions.

---

## Smoothing (`_smooth_non_fixed_segments`, line 650)

Calls CSXCAD's `SmoothMeshLines(all_lines, mesh_res, smooth_ratio=1.5)`.

**SmoothMeshLines preserves all input points** — it only adds intermediate
points where adjacent cell ratios exceed 1.5. This ensures:

- Fixed lines (thirds-rule boundaries, zero-thickness primitives) stay exact
- The transition from coarse air mesh to fine substrate/metal mesh is gradual
- The global mesh resolves field variations without abrupt cell-size changes

---

## Line Management

### `_remove_dups` (line 135)

Deduplicates with a priority rule:
- If two lines are equal (within `fp_equalp`) and neither is fixed:
  keep the first occurrence.
- If two lines are equal and one is fixed: **keep the fixed one**,
  delete the non-fixed one.

This ensures thirds-rule boundary lines take precedence over linspace lines
that happen to land at the same position.

### `_add_lines_to_mesh` (line 676)

Inserts each line via `insort_left` (maintains sorted order), then
deduplicates with `_remove_dups`.

### `_set_mesh_from_lines` (line 684)

Writes all mesh lines to the CSXCAD grid, rounding each to PREC via
`fp_nearest`.

---

## Key Tuning Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `mesh_resolution_factor` | `sim_params.py:125` | 20 | Coarser = larger `mesh_res` → fewer xy cells, less accuracy |
| `metal_mesh_resolution_factor` | `sim_params.py:126` | 60 | Coarser = larger metal spacing |
| `substrate_cells` | `sim_params.py:114` | 4 | Minimum FDTD cells through substrate thickness |
| `min_lines` | `mesh.py:270` | 5 | Minimum lines per bounded region |
| `smooth_ratio` | `mesh.py:267` | 1.5 | Max adjacent cell ratio for SmoothMeshLines |

---

## Common Pitfalls

### 1. No mesh line at the midpoint of copper thickness

The copper trace (and ground plane) in z are handled by `skip_thirds`,
which adds exactly one line at the midpoint. If you see no midpoint line,
check that `dim == 2` and `dist < thirds_cell`. If the extended linspace
path is taken instead (`is_metal and dim != 2`), the z-dimension metal
gets 6+ extended lines instead of 1 midpoint line.

### 2. Very dense lines inside the substrate

Three possible causes:
1. `substrate_cells` is too high → reduces to 4 or lower
2. Extended linspace from copper/ground spills lines into the substrate
   → check that dim==2 is excluded from extended linspace
3. SmoothMeshLines adds transition points between coarse air and fine
   substrate → normal behavior, controlled by `smooth_ratio`

### 3. Lines at metal boundaries (thirds rule violation)

The thirds rule requires no mesh line exactly at a metal edge. The code
ensures this in two ways:
- Metal thirds rule: shifts boundaries inward by `offset_in`/`offset_out`
- Extended linspace: filters out any line that `fp_equalp` considers on
  the boundary
- Nonmetal thirds rule: shifts boundaries outward by `offset_adj`

If boundary lines appear, check `fp_equalp` precision — a line at
`0.0000000001` and a boundary at `0.0` are NOT equal at PREC=10, but
they are effectively at the boundary for FDTD.

### 4. Slow simulation due to very fine cells

The CFL time step is limited by the **minimum cell size in any dimension**.
The most common culprit is the gap between the substrate's first/last mesh
line and an adjacent metal `skip_thirds` line. This gap is controlled by
the nonmetal thirds rule `offset_adj` formula. Increasing
`max(offset_in, metal_t * 3)` ensures a larger minimum gap.

---

## Algorithm History (for maintainers)

- **Extended linspace for thin metals** (dim 0 and 1): Replaced the previous
  `adj_in` + 3-interior-lines + external-lines approach. A single linspace
  from `lower-offset_out` to `upper+offset_out` with boundary-line filtering
  produces smoother transitions and avoids boundary hits by construction.

- **dim==1 threshold widened to 2×thirds_cell**: The y-dimension stub-end
  region (~1.37mm) falls between `thirds_cell` and `2×thirds_cell` at the
  default mesh resolution. The widened threshold catches these near-threshold
  regions that otherwise get a poor 2-line mesh.

- **dim==2 excluded from extended linspace**: The z-dimension has its own
  correct handler (`skip_thirds`, single midpoint line for metals thinner
  than `thirds_cell`). The extended linspace would spill extra lines into
  adjacent substrate/air regions.

- **Nonmetal thirds rule thin-metal clamp**: `min(offset_out, max(offset_in,
  metal_t * 3))` replaces the earlier `min(offset_out, metal_t * 3)`. The
  `max(offset_in, ...)` ensures a minimum boundary gap of 1/3 cell for CFL
  performance, even when the adjacent metal is extremely thin.

- **substrate_cells reduced to 4**: The default of 8 forced unnecessarily
  many z-cells through the substrate. 4 is sufficient for most geometries.
