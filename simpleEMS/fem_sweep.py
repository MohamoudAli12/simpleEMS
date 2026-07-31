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

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import AAA

from .console import console

# A passive structure can never have a scattering matrix with a singular value
# above 1. Left to itself AAA fits each S_ij to near machine precision, which
# on a handful of solve points routinely produces Froissart doublets: spurious
# pole/zero pairs sitting almost on the real frequency axis. They show up as
# towering fake resonances mid-band, and AAA's own `clean_up` does not catch
# them. On a 10-solve bandstop sweep whose every solve point was passive, the
# exact-interpolating fit reached |S11| = 5.7 (+15 dB) from a pole 0.01 off the
# real axis, with sigma_max = 9.6 over 40% of the band.
#
# Passivity is the right discriminator: a genuine high-Q resonance keeps
# sigma_max <= 1 no matter how sharp it is, a spurious pole does not. Guessing
# at pole positions instead would reject real narrowband physics.
#
# Dropping the fit order kills the blow-up (that same sweep goes to |S11| = 0.99
# at one order lower), but it does NOT guarantee passivity: too few solve points
# simply cannot represent the response, and some overshoot survives at every
# order. So the goal here is to pick the best order available and be honest when
# the result is still not passive -- not to silently clamp it into looking fine.
_PASSIVITY_TOL = 1e-3

# How much worse than the best achievable passivity a fit may be and still be
# preferred for its higher order. Among models that break passivity by about
# the same amount, the richer one tracks the solved points better.
_PASSIVITY_SLACK = 1.5


def _passivity_excess(model: NDArray) -> float:
    """
    Amount by which ``model`` violates passivity.

    Parameters
    ----------
    model : NDArray
        Scattering matrices ``[n_freq, nports, nports]``.

    Returns
    -------
    float
        ``max(sigma_max) - 1`` over the grid, clamped at ``0.0`` for a passive
        model.
    """
    return float(max(np.linalg.svd(model, compute_uv=False).max() - 1.0, 0.0))


def _fit_matrix(zs: NDArray, s_arr: NDArray, zg: NDArray, max_terms: int) -> NDArray:
    """
    Fit one AAA approximant per S_ij and evaluate them on ``zg``.

    Parameters
    ----------
    zs : NDArray
        Normalised frequencies of the solved points.
    s_arr : NDArray
        Solved S-matrices ``[n_solved, nports, nports]``.
    zg : NDArray
        Normalised output grid.
    max_terms : int
        Cap on the number of barycentric terms (the fit order knob).

    Returns
    -------
    NDArray
        Interpolated ``[len(zg), nports, nports]``.
    """
    npt = s_arr.shape[1]
    model = np.empty((len(zg), npt, npt), dtype=complex)
    with warnings.catch_warnings():
        # Capping max_terms below what AAA needs to reach its tolerance is the
        # whole point here, so its "failed to converge" notice is expected.
        warnings.filterwarnings("ignore", "AAA failed to converge", RuntimeWarning)
        for i in range(npt):
            for j in range(npt):
                model[:, i, j] = AAA(zs, s_arr[:, i, j], max_terms=max_terms)(zg)
    return model


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
    #
    # Fit at every order and score each by how far it breaks passivity, then
    # take the richest fit that is within _PASSIVITY_SLACK of the best score.
    # A lower-order model no longer interpolates the solve points exactly,
    # which is the right trade: a slightly loose curve beats an
    # exact-interpolating one carrying a fake 15 dB resonance. Orders at or
    # above the one AAA would pick unaided all collapse to the same fit, and
    # each fit costs microseconds against a full FEM solve, so this is free.
    def build_model() -> NDArray:
        fs = np.array(sorted(solved))
        zs = zof(fs)
        s_arr = np.array([solved[f] for f in fs])  # [n, npt, npt]
        # highest order first, so the `next(...)` below picks the richest fit
        scored = [
            (model, _passivity_excess(model))
            for model in (
                _fit_matrix(zs, s_arr, zg, terms) for terms in range(len(fs), 0, -1)
            )
        ]
        limit = max(_PASSIVITY_TOL, min(e for _m, e in scored) * _PASSIVITY_SLACK)
        return next(model for model, excess in scored if excess <= limit)

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

    # Final guard on the model actually handed back: if no fit order came out
    # passive, say so rather than silently returning a curve with a fake
    # resonance in it. Non-passive solve *points* would mean the FEM solves
    # themselves are wrong -- a different problem -- so name which one it is.
    excess = _passivity_excess(prev)
    if excess > _PASSIVITY_TOL and verbose:
        solved_arr = np.array([solved[f] for f in sorted(solved)])
        cause = (
            "the FEM solves themselves are non-passive"
            if _passivity_excess(solved_arr) > _PASSIVITY_TOL
            else f"rational interpolation from {len(solved)} solve points"
        )
        console.print(
            f"[warning]FEM sweep: |S| exceeds passivity by {excess:.3f} "
            f"({cause}); the curve may contain spurious resonances. Try "
            f"raising num_solve_points.[/warning]"
        )
    return prev
