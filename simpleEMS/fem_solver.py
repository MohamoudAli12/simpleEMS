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
Drive the GetDP finite-element solver as a subprocess.

Locates the ``getdp`` binary on ``PATH``, runs one solve of a generated
``.pro`` file, and reads back the results: per-port ``xS_<n>.txt``
S-parameter tables (one row per driven port), or the field-view files and
dielectric-loss/radiated-power totals from a combined fields-and-power solve.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def find_getdp() -> str:
    """
    Resolve the ``getdp`` binary path from ``PATH``.

    Returns
    -------
    str
        Path to the ``getdp`` executable.

    Raises
    ------
    RuntimeError
        If no ``getdp`` binary can be found.
    """
    found = shutil.which("getdp")
    if found:
        return found
    raise RuntimeError(
        "getdp binary not found. Run `simpleems install getdp`, or install "
        "GetDP yourself (https://getdp.info) and put it on your PATH."
    )


def read_complex(path: str | Path) -> complex:
    """
    Read a single complex value from a GetDP Table output file.

    Parameters
    ----------
    path : str | Path
        Path to a Table file with ``[tag, Re, Im]`` columns. Rows accumulate
        across a sweep (GetDP appends, see ``write_problem``), so the most
        recently written value is always the last row.

    Returns
    -------
    complex
        The last row's complex value, or ``0j`` if the file is missing.
    """
    if not Path(path).exists():
        return 0j
    d = np.loadtxt(path)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    return complex(d[-1, 1], d[-1, 2])


def read_complex_rows(path: str | Path, count: int) -> list[complex]:
    """
    Read the last ``count`` complex values from a GetDP Table output file.

    One ``Analysis`` launch drives every port in turn and appends a row per
    port, in the Resolution's port order, so the last ``count`` rows of
    ``xS_<n>.txt`` are that frequency's ``S[n, :]`` -- one entry per driven
    port.

    Parameters
    ----------
    path : str | Path
        Path to a Table file with ``[tag, Re, Im]`` columns.
    count : int
        Number of trailing rows to return (the port count).

    Returns
    -------
    list[complex]
        The last ``count`` values, oldest first (i.e. in driven-port order).

    Raises
    ------
    RuntimeError
        If the file is missing or holds fewer than ``count`` rows. Unlike
        :func:`read_complex`, a missing value cannot be defaulted here: rows are
        identified by *position*, so anything short would misalign the whole
        S-matrix rather than zero one entry of it.
    """
    if not Path(path).exists():
        raise RuntimeError(
            f"{path} not found. The Analysis resolution writes one row per "
            f"driven port to this file; a .pro generated before the "
            f"single-launch port loop instead writes xS_<observed><driven>.txt. "
            f"Re-run build_mesh to regenerate the problem file."
        )
    d = np.loadtxt(path)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if d.shape[0] < count:
        raise RuntimeError(
            f"{path} has {d.shape[0]} row(s) but {count} were expected (one per "
            f"driven port); the GetDP solve did not complete every port."
        )
    return [complex(row[1], row[2]) for row in d[-count:]]


def run_getdp(
    pro_path: str | Path,
    msh_path: str | Path,
    workdir: str | Path,
    setnumbers: dict,
    postop: str | list[str] | None,
    resolution: str = "Analysis",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run GetDP once for a given set of numeric parameters.

    Parameters
    ----------
    pro_path : str | Path
        Path to the ``.pro`` problem file.
    msh_path : str | Path
        Path to the ``.msh`` mesh file.
    workdir : str | Path
        Working directory for the run (outputs go under ``workdir/output``).
    setnumbers : dict
        Numeric ``-setnumber`` overrides (e.g. ``{"FREQ": f, "ACTIVE_PORT": k}``).
    postop : str | list[str] | None
        Name(s) of the ``-pos`` post-operation(s) to run, or ``None``. GetDP's
        ``-pos`` accepts multiple post-operation ids in one invocation, run
        against whatever the ``-solve``/``-cal`` step already assembled and
        solved -- passing a list here runs them all off a single
        ``Generate``/``Solve`` instead of one GetDP process per post-operation.
    resolution : str
        Name of the GetDP resolution to solve. Default ``"Analysis"``.
    extra_args : list[str] | None
        Extra getdp/PETSc flags (e.g. iterative solver options).

    Returns
    -------
    subprocess.CompletedProcess
        The completed process.

    Raises
    ------
    RuntimeError
        If getdp exits with a non-zero return code.
    """
    # Build the getdp command line:
    #   getdp <name>.pro -msh <name>.msh -setnumber FREQ <f>
    #         -solve <Resolution> [-pos <PostOperation(s)>] -v2
    # The default "Analysis" resolution drives every port at that frequency and
    # runs Get_SParameters itself (so no -pos, and no ACTIVE_PORT), appending a
    # row per port to output/xS_<n>.txt. "AnalysisSinglePort" instead solves the
    # single ACTIVE_PORT and keeps the solution for a -pos field extraction.
    args = [find_getdp(), str(pro_path), "-msh", str(msh_path)]
    for k, v in setnumbers.items():
        args += ["-setnumber", k, repr(float(v))]
    args += ["-solve", resolution]
    if postop:  # omit for the internal sweep (its Resolution calls PostOperation)
        postops = [postop] if isinstance(postop, str) else list(postop)
        args += ["-pos", *postops]
    args += ["-v2"]  # verbosity level 2 (progress but not per-iteration spam)
    if extra_args:  # passthrough getdp/PETSc flags
        args += list(extra_args)
    res = subprocess.run(args, cwd=workdir, capture_output=False, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"getdp failed ({res.returncode}):\n"
            f"{res.stdout[-2000:]}\n{res.stderr[-2000:]}"
        )
    return res


def _read_power_value(outdir: str | Path, fname: str) -> float:
    # rows accumulate across calls (GetDP appends); take the latest one.
    path = Path(outdir) / fname
    if not path.exists():
        return 0.0
    return float(np.atleast_2d(np.loadtxt(path))[-1, -2])


def solve_fields_and_power(
    pro_path: str | Path,
    msh_path: str | Path,
    workdir: str | Path,
    freq: float,
    active: int = 1,
) -> tuple[str, str, float, float]:
    """
    Solve at one frequency and return both the field views and the powers.

    Drives ``Get_Fields`` and ``Get_Power`` together in a single GetDP
    invocation, so the linear system is assembled and solved only once (both
    post-operations read off the same solved system) instead of once per
    post-operation as two separate GetDP runs would require.

    Parameters
    ----------
    pro_path, msh_path : str | Path
        Paths to the ``.pro`` and ``.msh`` files.
    workdir : str | Path
        Working directory (outputs go under ``workdir/output``).
    freq : float
        Frequency in Hz.
    active : int
        Driven port number. Default ``1``.

    Returns
    -------
    tuple[str, str, float, float]
        ``(e_pos, h_pos, p_loss, p_rad)`` -- paths to the written ``e.pos``
        and ``h.pos`` field views, and the dielectric-loss and radiated
        powers. These are computed directly from the unnormalised field
        solution, so only their ratio (e.g. radiation efficiency) is
        physically meaningful, not their absolute magnitude.
    """
    run_getdp(
        pro_path,
        msh_path,
        workdir,
        {"FREQ": freq, "ACTIVE_PORT": active},
        ["Get_Fields", "Get_Power"],
        resolution="AnalysisSinglePort",
    )
    outdir = Path(workdir).absolute() / "output"
    return (
        str(outdir / "e.pos"),
        str(outdir / "h.pos"),
        _read_power_value(outdir, "Ploss.txt"),
        _read_power_value(outdir, "Prad.txt"),
    )
