"""
Auto-generated FDTD mesh from CSXCAD primitives and lumped ports.

The Mesh class scans all geometry and port elements, places mesh
lines at every boundary, applies sub-cell refinement near metal
edges and inside port regions, then calls SmoothMeshLines for a
stable grid transition.
"""

from collections.abc import Iterable

import numpy as np
from CSXCAD import ContinuousStructure

from .sim_params import SimParams


class Mesh:
    """Auto-generates an FDTD mesh from CSXCAD primitives and lumped ports.

    Scans all geometry and ports, places mesh lines at every boundary,
    applies sub-cell refinement at metal edges and inside port regions,
    then calls SmoothMeshLines to fill between fixed lines.

    Parameters
    ----------
    csx : ContinuousStructure
        CSXCAD structure with primitives already added.
    params : SimParams
        Simulation parameters providing simulation_box, mesh_resolution,
        metal_mesh_resolution, unit, and thirds_rule.
    smooth_ratio : float
        Maximum ratio between adjacent cells (default 1.5).
    """

    def __init__(
        self, csx: ContinuousStructure, params: SimParams, smooth_ratio: float = 1.5
    ) -> None:
        """Initialise the Mesh generator and automatically generate the grid."""
        self._csx = csx
        self._mesh_res = float(params.mesh_resolution)
        self._metal_res = float(params.metal_mesh_resolution)
        self._smooth_ratio = float(smooth_ratio)
        self._unit = float(params.unit)
        self._thirds_rule = np.asarray(params.thirds_rule, dtype=float)
        sb = params.simulation_box
        self._sim_box = tuple((float(-s / 2), float(s / 2)) for s in sb)
        self._generate()

    @staticmethod
    def _dedup(
        coords: Iterable[float], min_spacing: float | None = None
    ) -> list[float]:
        """Remove duplicate coordinates within a minimum spacing threshold."""
        s = sorted(coords)
        if not s:
            return s
        out = [s[0]]
        for v in s[1:]:
            if v - out[-1] >= (min_spacing if min_spacing is not None else 1e-12):
                out.append(v)
        return out

    def _generate(self) -> None:
        """Generate the FDTD mesh by scanning all primitives and adding grid lines."""
        grid = self._csx.GetGrid()
        grid.SetDeltaUnit(self._unit)

        for dim in range(3):
            ch = "xyz"[dim]
            coords = set()

            coords.add(self._sim_box[dim][0])
            coords.add(self._sim_box[dim][1])

            for prim in self._csx.GetAllPrimitives():
                ptype = prim.GetProperty().GetTypeString()
                if ptype in ("ProbeBox", "Excitation"):
                    continue
                bb = prim.GetBoundBox()
                tr = prim.GetTransform()
                p0 = tr.Transform(bb[0])
                p1 = tr.Transform(bb[1])
                lo = min(p0[dim], p1[dim])
                hi = max(p0[dim], p1[dim])
                coords.add(lo)
                coords.add(hi)

                if ptype == "Metal":
                    t = self._thirds_rule
                    coords.add(lo - t[0])
                    coords.add(lo - t[1])
                    coords.add(hi + t[1])
                    coords.add(hi + t[0])

                elif ptype == "LumpedElement":
                    if hi - lo > 1e-12 * (abs(hi) + 1e-12):
                        n = max(2, int(np.ceil((hi - lo) / self._metal_res)) + 1)
                        for v in np.linspace(lo, hi, n):
                            coords.add(v)

            for v in self._dedup(coords, min_spacing=self._metal_res / 10):
                grid.AddLine(ch, v)

        grid.SmoothMeshLines("all", self._mesh_res, self._smooth_ratio)
