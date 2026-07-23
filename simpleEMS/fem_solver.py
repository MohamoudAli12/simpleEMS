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

Locates the ``getdp`` binary (environment variable, then ``PATH``), runs a
single frequency/port solve of a generated ``.pro`` file, and reads the
resulting per-port ``xS_<n><k>.txt`` S-parameter tables.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

__all__ = [
    "find_getdp",
    "run_getdp",
    "read_complex",
    "solve_fields",
    "solve_power",
    "solve_fields_and_power",
]

_GETDP_ENV = "SIMPLEEMS_GETDP_BIN"


def find_getdp() -> str:
    """
    Resolve the ``getdp`` binary path.

    Looks first at the ``SIMPLEEMS_GETDP_BIN`` environment variable, then on
    ``PATH``.

    Returns
    -------
    str
        Path to the ``getdp`` executable.

    Raises
    ------
    RuntimeError
        If no ``getdp`` binary can be found.
    """
    env = os.environ.get(_GETDP_ENV)
    if env:
        if not os.path.exists(env):
            raise RuntimeError(
                f"{_GETDP_ENV} is set to {env!r} but that file does not exist"
            )
        return env
    found = shutil.which("getdp")
    if found:
        return found
    raise RuntimeError(
        "getdp binary not found. Install GetDP (https://getdp.info) and either "
        f"put it on your PATH or set the {_GETDP_ENV} environment variable to "
        "its full path."
    )


def read_complex(path: str) -> complex:
    """
    Read a single complex value from a GetDP Table output file.

    Parameters
    ----------
    path : str
        Path to an ``xS_<n><k>.txt`` file with ``[tag, Re, Im]`` columns. Rows
        accumulate across a sweep (GetDP appends, see ``write_problem``), so
        the most recently solved point is always the last row.

    Returns
    -------
    complex
        The last row's complex value, or ``0j`` if the file is missing.
    """
    if not os.path.exists(path):
        return 0j
    d = np.loadtxt(path)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    return complex(d[-1, 1], d[-1, 2])


def run_getdp(
    pro_path: str,
    msh_path: str,
    workdir: str,
    setnumbers: dict,
    postop: str | list[str] | None,
    resolution: str = "Analysis",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run GetDP once for a given set of numeric parameters.

    Parameters
    ----------
    pro_path : str
        Path to the ``.pro`` problem file.
    msh_path : str
        Path to the ``.msh`` mesh file.
    workdir : str
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
    #   getdp <name>.pro -msh <name>.msh -setnumber FREQ <f> -setnumber ACTIVE_PORT <k>
    #         -solve <Resolution> -pos <PostOperation(s)> -v2
    # i.e. one solve at one frequency driving one port; results are written by
    # the .pro's PostOperation(s) into output/xS_<n><k>.txt.
    args = [find_getdp(), pro_path, "-msh", msh_path]
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


def _read_power_value(outdir: str, fname: str) -> float:
    # rows accumulate across calls (GetDP appends); take the latest one.
    path = os.path.join(outdir, fname)
    if not os.path.exists(path):
        return 0.0
    return float(np.atleast_2d(np.loadtxt(path))[-1, -2])


def solve_fields(
    pro_path: str, msh_path: str, workdir: str, freq: float, active: int = 1
) -> tuple[str, str]:
    """
    Solve at one frequency and write the E/H field views for the radiation step.

    Drives the ``Get_Fields`` post-operation the problem file already emits.

    Parameters
    ----------
    pro_path, msh_path : str
        Paths to the ``.pro`` and ``.msh`` files.
    workdir : str
        Working directory (outputs go under ``workdir/output``).
    freq : float
        Frequency in Hz.
    active : int
        Driven port number. Default ``1``.

    Returns
    -------
    tuple[str, str]
        Paths to the written ``e.pos`` and ``h.pos`` field views.
    """
    run_getdp(
        pro_path,
        msh_path,
        workdir,
        {"FREQ": freq, "ACTIVE_PORT": active},
        "Get_Fields",
    )
    outdir = os.path.join(os.path.abspath(workdir), "output")
    return os.path.join(outdir, "e.pos"), os.path.join(outdir, "h.pos")


def solve_power(
    pro_path: str, msh_path: str, workdir: str, freq: float, active: int = 1
) -> tuple[float, float]:
    """
    Solve at one frequency and return the dielectric-loss and radiated powers.

    Drives the ``Get_Power`` post-operation the problem file already emits.

    Parameters
    ----------
    pro_path, msh_path : str
        Paths to the ``.pro`` and ``.msh`` files.
    workdir : str
        Working directory (outputs go under ``workdir/output``).
    freq : float
        Frequency in Hz.
    active : int
        Driven port number. Default ``1``.

    Returns
    -------
    tuple[float, float]
        ``(p_loss, p_rad)`` -- dielectric loss and radiated power (field units).
    """
    run_getdp(
        pro_path,
        msh_path,
        workdir,
        {"FREQ": freq, "ACTIVE_PORT": active},
        "Get_Power",
    )
    outdir = os.path.join(os.path.abspath(workdir), "output")
    return (
        _read_power_value(outdir, "Ploss.txt"),
        _read_power_value(outdir, "Prad.txt"),
    )


def solve_fields_and_power(
    pro_path: str, msh_path: str, workdir: str, freq: float, active: int = 1
) -> tuple[str, str, float, float]:
    """
    Solve at one frequency and return both the field views and the powers.

    Drives ``Get_Fields`` and ``Get_Power`` together in a single GetDP
    invocation, so the linear system is assembled and solved only once (both
    post-operations read off the same solved system) instead of once per
    post-operation as separate calls to :func:`solve_fields` and
    :func:`solve_power` would do.

    Parameters
    ----------
    pro_path, msh_path : str
        Paths to the ``.pro`` and ``.msh`` files.
    workdir : str
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
        powers (field units).
    """
    run_getdp(
        pro_path,
        msh_path,
        workdir,
        {"FREQ": freq, "ACTIVE_PORT": active},
        ["Get_Fields", "Get_Power"],
    )
    outdir = os.path.join(os.path.abspath(workdir), "output")
    return (
        os.path.join(outdir, "e.pos"),
        os.path.join(outdir, "h.pos"),
        _read_power_value(outdir, "Ploss.txt"),
        _read_power_value(outdir, "Prad.txt"),
    )
