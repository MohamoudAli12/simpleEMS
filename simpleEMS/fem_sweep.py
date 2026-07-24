# simpleEMS
# Copyright (C) 2026 Mohamoud Ali
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
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
Fast FEM frequency sweep by adaptive rational interpolation.

S-parameters of linear microwave structures are smooth rational functions of
frequency, so a handful of full FEM solves suffice to reconstruct the whole
curve. Each S_ij is fitted with a barycentric rational approximant using the
AAA algorithm (Nakatsukasa, Sete & Trefethen 2018) via
:class:`scipy.interpolate.AAA`, and new solve points are added greedily where
successive models disagree most. Typically 8-15 solves reproduce a 200+ point
dense sweep.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import AAA

from .console import console


def rational_sweep(
    freqs_out: NDArray,
    port_numbers: list[int],
    solve_at: Callable[[float], NDArray],
    num_solves: int,
    tol: float = 0.0,
    verbose: bool = True,
) -> NDArray:
    """
    Adaptively solve and rational-interpolate S(f) over the output grid.

    Performs ``num_solves`` full FEM solves (a few uniform seeds, the rest
    placed adaptively where the model varies fastest), then evaluates the
    barycentric rational model on ``freqs_out``.

    Parameters
    ----------
    freqs_out : NDArray
        Dense output frequency grid to interpolate onto.
    port_numbers : list[int]
        Sorted port numbers; the returned matrix is indexed in this order.
    solve_at : Callable[[float], NDArray]
        Callback returning the ``[nports, nports]`` S-matrix at one frequency.
    num_solves : int
        Number of full FEM solves to perform.
    tol : float
        Optional early-stop guard on model change between solves. ``0`` (the
        default) disables it, so exactly ``num_solves`` solves are performed.
    verbose : bool
        Print progress through the shared console. Default ``True``.

    Returns
    -------
    NDArray
        Interpolated ``S[len(freqs_out), nports, nports]`` (complex).
    """
    fgrid = np.asarray(freqs_out, dtype=float)
    fmin, fmax = float(fgrid[0]), float(fgrid[-1])
    n_grid = len(fgrid)
    npt = len(port_numbers)
    span = (fmax - fmin) or 1.0

    # AAA is much better conditioned on data scaled to [-1, 1] than on raw Hz
    # (frequencies ~1e10), so fit in this normalised coordinate throughout.
    def zof(f: NDArray) -> NDArray:
        return (2 * np.asarray(f) - (fmin + fmax)) / span  # scale to [-1, 1]

    zg = zof(fgrid)

    # Seed with a few uniform full solves to give AAA something to fit.
    solved: dict[float, NDArray] = {}
    n_init = min(5, num_solves)
    for f in np.linspace(fmin, fmax, n_init):
        solved[float(f)] = solve_at(float(f))
        if verbose:
            console.print(f"[info]FEM solved {f / 1e9:.4f} GHz ({len(solved)})[/info]")

    # Fit one rational approximant per S_ij over the solved points and evaluate
    # it on the dense output grid -> the current best model of S(f).
    def build_model() -> NDArray:
        fs = np.array(sorted(solved))
        s_arr = np.array([solved[f] for f in fs])  # [n, npt, npt]
        model = np.zeros((n_grid, npt, npt), dtype=complex)
        for i in range(npt):
            for j in range(npt):
                model[:, i, j] = AAA(zof(fs), s_arr[:, i, j])(zg)
        return model

    # Choose the next frequency to solve: where the model changes fastest
    # (var, likely a resonance) AND far from existing solves (dist, avoids
    # clustering) -- the product balances "interesting" against "unexplored".
    def pick_next(model: NDArray) -> float:
        fmid = 0.5 * (fgrid[:-1] + fgrid[1:])
        var = np.max(np.abs(np.diff(model, axis=0)), axis=(1, 2))
        sf = np.array(sorted(solved))
        dist = np.min(np.abs(fmid[:, None] - sf[None, :]), axis=1)
        return float(fmid[int(np.argmax(var * dist))])

    # Greedily add solves up to the budget, stopping early only if the model has
    # essentially stopped changing (tol > 0). Each iteration = one FEM solve.
    prev = build_model()
    while len(solved) < num_solves:
        fnext = pick_next(prev)
        if fnext in solved:
            break
        solved[fnext] = solve_at(fnext)
        cur = build_model()
        change = float(np.max(np.abs(cur - prev)))
        if verbose:
            console.print(
                f"[info]FEM added {fnext / 1e9:.4f} GHz -> model change "
                f"{change:.2e} ({len(solved)} solves)[/info]"
            )
        prev = cur
        if tol > 0 and change < tol:
            break
    return prev
