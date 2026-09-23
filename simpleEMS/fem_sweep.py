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
Sweep the FEM solver over frequency without solving at every point.

S-parameters vary smoothly with frequency, so a handful of full solves is
enough to reconstruct the whole curve by interpolation. Solve frequencies are
chosen as the sweep goes, concentrating them where the response changes
fastest; typically 8-15 of them reproduce a 200-point sweep.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import AAA

from .console import console
from .fem_solver import describe_solve, format_duration, last_solve

# Each S_ij is fitted with a barycentric rational approximant by the AAA
# algorithm (Nakatsukasa, Sete & Trefethen 2018), via scipy.interpolate.AAA.
#
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
# order. So a non-passive fit drives where the next solve goes -- the violation
# names the interval the spurious pole lives in -- and if refinement still ends
# up non-passive, the sweep says so rather than clamping it into looking fine.
_PASSIVITY_TOL = 1e-3

# How far a fit may miss the solve points and still count as faithful, so that
# the leanest passive fit can win on order without losing accuracy: every extra
# term is one more chance at a Froissart doublet. Kept near machine precision
# deliberately -- the rule is "drop order only where the extra terms were
# redundant anyway". At 1e-3 it cost three digits of accuracy on a three-pole
# target, and it bought nothing: on noisy solve points no lean fit is faithful
# at any threshold, so the fallback below runs regardless.
_FIT_TOL = 1e-9

# The most a returned curve may miss the solve points by, in linear S. When no
# fit is both passive and faithful the sweep trades accuracy for passivity, and
# this is the floor on that trade: past it the curve has stopped describing the
# simulation at all. Without it, "least non-passive" picks a near-constant fit
# -- perfectly passive, and wrong by more than the response itself varies.
#
# 0.1 measured best on both axes over a scattered near-unity two-port: against
# 0.02 it cut the worst error from 0.70 to 0.073 while leaving the worst excess
# unchanged at 0.036. The curve either side of it is flat, so the exact value
# is not load-bearing -- having one is.
_MAX_RESIDUAL = 0.1

# Refinement rounds allowed to show no improvement before the sweep concludes
# that the solve points simply disagree and more of them will not help. AAA
# interpolates exactly, so scatter between solve points has nowhere to go but
# into poles: on data carrying 5e-3 of per-solve scatter, doubling the budget
# left more runs non-passive, not fewer.
_STALL_ROUNDS = 3


def _passivity_excess(model: NDArray) -> float:
    """
    How far ``model`` exceeds what a passive structure can do.

    Parameters
    ----------
    model : NDArray
        S-matrices over frequency, shaped ``[n_freq, nports, nports]``.

    Returns
    -------
    float
        The worst excess over the frequency range, or ``0.0`` if the model is
        passive throughout.
    """
    return float(max(np.linalg.svd(model, compute_uv=False).max() - 1.0, 0.0))


def _fit_matrix(zs: NDArray, s_arr: NDArray, zg: NDArray, max_terms: int) -> NDArray:
    """
    Fit each S-parameter across the solved points and evaluate it on ``zg``.

    Parameters
    ----------
    zs : NDArray
        Scaled frequencies that were solved at.
    s_arr : NDArray
        The S-matrices solved there, shaped ``[n_solved, nports, nports]``.
    zg : NDArray
        Scaled frequencies to evaluate the fit at.
    max_terms : int
        Highest order the fit may use.

    Returns
    -------
    NDArray
        The fitted S-matrices, shaped ``[len(zg), nports, nports]``.
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


def _bisect_violation(
    model: NDArray, fgrid: NDArray, solved_freqs: NDArray
) -> float | None:
    """
    Bisect the solved interval bracketing the worst passivity violation.

    A spurious pole lives *between* two solve points: it honours both and
    misbehaves in the gap, so a solve in that gap is what pins the curve back
    down. Placing the next solve by curvature instead can send it to an
    unrelated part of the band and leave the violation standing.

    Parameters
    ----------
    model : NDArray
        S-matrices over ``fgrid``, shaped ``[n_freq, nports, nports]``.
    fgrid : NDArray
        The frequencies (Hz) ``model`` is sampled on.
    solved_freqs : NDArray
        The frequencies (Hz) already solved at, in ascending order.

    Returns
    -------
    float or None
        The frequency (Hz) to solve at next, or ``None`` when the violation
        sits outside every solved interval and so cannot be bracketed.
    """
    sigma = np.linalg.svd(model, compute_uv=False).max(axis=1)
    worst = float(fgrid[int(np.argmax(sigma))])
    below = solved_freqs[solved_freqs < worst]
    above = solved_freqs[solved_freqs > worst]
    if not below.size or not above.size:
        return None
    return float(np.sqrt(below[-1] * above[0]))  # geometric midpoint


def rational_sweep(
    freqs_out: NDArray,
    port_numbers: list[int],
    solve_at: Callable[[float], NDArray],
    num_solves: int,
    max_solves: int | None = None,
    tol: float = 0.0,
    verbose: bool = True,
) -> NDArray:
    """
    Sweep the S-parameters over a frequency range.

    Solves at ``num_solves`` frequencies -- a few spread evenly, the rest
    placed where the response changes fastest -- then interpolates the results
    onto ``freqs_out``.

    If the interpolated curve breaks passivity, the sweep keeps solving past
    ``num_solves``, aiming each extra solve at the frequency interval the
    violation sits in, up to ``max_solves``. It gives up early once the extra
    solves stop improving matters, since solve points that disagree with each
    other cannot be interpolated into a passive curve however many there are.

    Parameters
    ----------
    freqs_out : NDArray
        Frequency points (Hz) to report results at.
    port_numbers : list[int]
        Sorted port numbers. The returned matrix is indexed in this order.
    solve_at : Callable[[float], NDArray]
        Function returning the ``[nports, nports]`` S-matrix at one frequency.
    num_solves : int
        Number of frequencies to solve at.
    max_solves : int, optional
        Ceiling on the solve count when chasing passivity. Default ``None``,
        which allows twice ``num_solves``. The extra solves are spent only
        while the model is non-passive and still improving, so a well-behaved
        structure costs exactly ``num_solves``.
    tol : float
        Stop early once the interpolated curve changes by less than this
        between solves. Default ``0``, which disables the check so that
        exactly ``num_solves`` solves are performed.
    verbose : bool
        Print progress. Default ``True``.

    Returns
    -------
    NDArray
        The complex S-matrices over ``freqs_out``, shaped
        ``[len(freqs_out), nports, nports]``.
    """
    fgrid = np.asarray(freqs_out, dtype=float)
    fmin, fmax = float(fgrid[0]), float(fgrid[-1])
    span = (fmax - fmin) or 1.0
    # The budget the caller asked for is what a passive response costs; the
    # ceiling is only ever reached chasing a passivity violation.
    ceiling = int(max_solves) if max_solves is not None else 2 * num_solves
    solve_cap = max(ceiling, num_solves)

    # AAA is much better conditioned on data scaled to [-1, 1] than on raw Hz
    # (frequencies ~1e10), so fit in this normalised coordinate throughout.
    def zof(f: NDArray) -> NDArray:
        return (2 * np.asarray(f) - (fmin + fmax)) / span  # scale to [-1, 1]

    zg = zof(fgrid)

    # Seed with a few uniform full solves to give AAA something to fit.
    # A single-point grid has nothing to spread across: np.linspace(f, f, 5)
    # would solve the same frequency five times and discard four of them.
    solved: dict[float, NDArray] = {}
    started = time.perf_counter()

    # One line per solve, columns lined up: how far along, which frequency,
    # what that solve cost, and how long the whole sweep still has to run.
    # Everything constant across the sweep -- problem size, peak memory --
    # waits for the closing summary rather than repeating on every line.
    def report(freq: float, seconds: float, change: float | None = None) -> None:
        """Print one solve's line."""
        elapsed = time.perf_counter() - started
        # Count against the requested budget until refinement overruns it, then
        # against the ceiling -- so the column never reads "13/10".
        target = num_solves if len(solved) <= num_solves else solve_cap
        left = max(target - len(solved), 0) * elapsed / len(solved)
        # Fixed-width fields, so the columns hold still as the durations grow
        # and the last line's fit figure does not jump left when the estimate
        # of what is left drops out.
        elapsed_text = f"elapsed {format_duration(elapsed)}"
        left_text = f"~{format_duration(left)} left" if left > 0 else ""
        line = (
            f"{len(solved):>3}/{target}  {freq / 1e9:8.4f} GHz  "
            f"{format_duration(seconds):>7}  {elapsed_text:<18}{left_text:<17}"
        )
        if change is not None:
            line += f"Δfit {change:.0e}"
        console.print(f"[info]{line.rstrip()}[/info]")

    def timed(freq: float) -> tuple[NDArray, float]:
        """Solve at ``freq``, and say how long it took."""
        start = time.perf_counter()
        return solve_at(freq), time.perf_counter() - start

    if verbose:
        # The ceiling only comes into play if the curve breaks passivity, so
        # name it as the exception it is rather than as the headline budget.
        budget = f"up to {num_solves} solves"
        if solve_cap > num_solves:
            budget += f" ({solve_cap} if non-passive)"
        console.rule(
            f"[info]FEM sweep · {fmin / 1e9:.3f}-{fmax / 1e9:.3f} GHz · {budget}[/info]"
        )

    n_init = min(5, num_solves) if fgrid.size >= 2 else 1
    for f in np.linspace(fmin, fmax, n_init):
        solved[float(f)], seconds = timed(float(f))
        if verbose:
            report(float(f), seconds)

    # Fit one rational approximant per S_ij over the solved points and evaluate
    # it on the dense output grid -> the current best model of S(f).
    #
    # Fit at every order and score each twice: how far it breaks passivity, and
    # how far it misses the solve points it was fitted from. The leanest fit
    # that passes both wins. A lower-order model no longer interpolates the
    # solve points exactly, which is the right trade: a slightly loose curve
    # beats an exact-interpolating one carrying a fake 15 dB resonance. Orders
    # at or above the one AAA would pick unaided all collapse to the same fit,
    # and each fit costs microseconds against a full FEM solve, so this is free.
    def build_model() -> tuple[NDArray, float, float]:
        """The best fit available: (curve on ``fgrid``, passivity excess, residual)."""
        fs = np.array(sorted(solved))
        zs = zof(fs)
        s_arr = np.array([solved[f] for f in fs])  # [n, npt, npt]
        scored = []
        for terms in range(len(fs), 0, -1):  # richest order first
            model = _fit_matrix(zs, s_arr, zg, terms)
            residual = float(np.abs(_fit_matrix(zs, s_arr, zs, terms) - s_arr).max())
            scored.append((model, _passivity_excess(model), residual))
        passing = [
            fit for fit in scored if fit[1] <= _PASSIVITY_TOL and fit[2] <= _FIT_TOL
        ]
        if passing:
            return passing[-1]  # scored runs richest-first, so the last is leanest
        # Nothing qualifies, so trade accuracy for passivity -- but only down to
        # a curve that still represents the solve points. Minimising excess on
        # its own hands the prize to a near-constant fit, which is trivially
        # passive and describes nothing: it misses the data by more than the
        # data varies. Stay inside _MAX_RESIDUAL, then be as passive as possible.
        usable = [fit for fit in scored if fit[2] <= _MAX_RESIDUAL] or scored
        return min(usable, key=lambda fit: (fit[1], fit[2]))

    # Choose the next frequency to solve: where the model changes fastest
    # (var, likely a resonance) AND far from existing solves (dist, avoids
    # clustering) -- the product balances "interesting" against "unexplored".
    def pick_next(model: NDArray) -> float:
        fmid = 0.5 * (fgrid[:-1] + fgrid[1:])
        var = np.max(np.abs(np.diff(model, axis=0)), axis=(1, 2))
        sf = np.array(sorted(solved))
        dist = np.min(np.abs(fmid[:, None] - sf[None, :]), axis=1)
        return float(fmid[int(np.argmax(var * dist))])

    def pick_violation(model: NDArray) -> float:
        """Aim at the violation, falling back to curvature if it is not bracketed."""
        target = _bisect_violation(model, fgrid, np.array(sorted(solved)))
        return pick_next(model) if target is None else target

    # Greedily add solves up to the budget, stopping early only if the model has
    # essentially stopped changing (tol > 0). Each iteration = one FEM solve.
    #
    # Past the budget the loop keeps going for one reason only: the curve is
    # not passive and each extra solve is still pulling it back towards passive.
    # Once that stops paying, more solves are wasted FEM time -- solve points
    # that disagree with each other admit no passive interpolant at any count.
    #
    # pick_next bisects intervals between output points, so it needs at least
    # two of them; with a single-point grid the seeded model is already final.
    prev, excess, residual = build_model()
    best_excess, stalled, overrunning = excess, 0, False
    while len(solved) < solve_cap and fgrid.size >= 2:
        if len(solved) >= num_solves:
            if excess <= _PASSIVITY_TOL or stalled >= _STALL_ROUNDS:
                break
            if not overrunning:
                # Entering the overrun: the stall count so far described solves
                # spent covering the band, not solves aimed at the violation.
                # Carrying it over would condemn the refinement before it has
                # taken a single step.
                overrunning, stalled = True, 0
        fnext = pick_violation(prev) if excess > _PASSIVITY_TOL else pick_next(prev)
        if fnext in solved:
            break
        solved[fnext], seconds = timed(fnext)
        cur, excess, residual = build_model()
        change = float(np.max(np.abs(cur - prev)))
        if verbose:
            report(fnext, seconds, change)
        prev = cur
        # A passive curve resets the count too: `excess < best_excess - tol`
        # cannot fire once both are pinned at zero.
        if excess <= _PASSIVITY_TOL or excess < best_excess - _PASSIVITY_TOL:
            best_excess, stalled = excess, 0
        else:
            stalled += 1
        # A settled curve is only done if it is also passive; otherwise the
        # refinement above still has work to do.
        if tol > 0 and change < tol and excess <= _PASSIVITY_TOL:
            break

    if verbose:
        # The problem the solves shared, reported once instead of per line.
        size = describe_solve(last_solve())
        console.rule(
            f"[info]{len(solved)} solves in "
            f"{format_duration(time.perf_counter() - started)}"
            f"{f' · {size}' if size else ''}[/info]"
        )

    # Final guard on the model actually handed back: if refinement could not
    # reach a passive curve, say so rather than silently returning one with a
    # fake resonance in it. The three causes want three different fixes, so
    # name which one this is instead of always asking for more solve points.
    if excess > _PASSIVITY_TOL and verbose:
        solved_arr = np.array([solved[f] for f in sorted(solved)])
        if _passivity_excess(solved_arr) > _PASSIVITY_TOL:
            # A single solve came back active: the FEM solve itself is wrong,
            # and no amount of interpolation work touches that.
            cause = "the FEM solves themselves are non-passive"
            advice = "check the port de-embedding and the mesh resolution"
        elif stalled >= _STALL_ROUNDS:
            # Every solve point is passive, yet no smooth rational curve runs
            # through all of them. That is solve quality, not sampling: the
            # points scatter by more than a rational model can absorb, and
            # adding points measurably does not close the gap.
            cause = (
                f"{len(solved)} solve points, each passive, admit no passive "
                f"model -- the best fit still misses them by {residual:.1e}"
            )
            advice = "raise FEM_elems_per_wavelength or FEM_fe_order"
        else:
            cause = f"rational interpolation from {len(solved)} solve points"
            advice = "raise FEM_max_solve_points"
        console.print(
            f"[warning]FEM sweep: |S| exceeds passivity by {excess:.3f} "
            f"({cause}); the curve may contain spurious resonances. "
            f"Try to {advice}.[/warning]"
        )
    return prev
