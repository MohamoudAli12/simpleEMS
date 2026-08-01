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
Run the GetDP finite-element solver and read back what it writes.

Locates the ``getdp`` binary, runs one solve of a generated problem file, and
reads the results out of its output files: the S-parameters of each port, or
the near fields and the power radiated and lost.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def find_getdp() -> str:
    """
    Locate the ``getdp`` binary.

    Returns
    -------
    str
        Path to the ``getdp`` executable.

    Raises
    ------
    RuntimeError
        If no ``getdp`` binary is installed, with instructions for installing
        one.
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
    Read the most recent complex value out of a solver output file.

    Parameters
    ----------
    path : str | Path
        Path to an output file with ``[tag, Re, Im]`` columns. Rows are
        appended as the sweep runs, so the last one is the most recent.

    Returns
    -------
    complex
        The value in the last row, or ``0j`` if the file does not exist.
    """
    if not Path(path).exists():
        return 0j
    d = np.loadtxt(path)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    return complex(d[-1, 1], d[-1, 2])


def read_complex_rows(path: str | Path, count: int) -> list[complex]:
    """
    Read the most recent ``count`` complex values out of a solver output file.

    One solve appends a row per driven port, so the last ``count`` rows are
    that frequency's results for every port.

    Parameters
    ----------
    path : str | Path
        Path to an output file with ``[tag, Re, Im]`` columns.
    count : int
        Number of rows to return, i.e. the number of ports.

    Returns
    -------
    list[complex]
        The last ``count`` values, oldest first, so in driven-port order.

    Raises
    ------
    RuntimeError
        If the file is missing or holds fewer than ``count`` rows, which would
        misalign the results.
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
    Run the solver once.

    Parameters
    ----------
    pro_path : str | Path
        Path to the ``.pro`` problem file to solve.
    msh_path : str | Path
        Path to the ``.msh`` mesh file to solve it on.
    workdir : str | Path
        Working directory for the run. Results are written to an ``output``
        subdirectory of it.
    setnumbers : dict
        Values to set in the problem file, e.g.
        ``{"FREQ": f, "ACTIVE_PORT": k}``.
    postop : str | list[str] | None
        Name or names of the results to extract afterwards, or ``None`` for
        none. Passing several extracts them all from one solve.
    resolution : str
        Which of the problem file's solve steps to run. Default
        ``"Analysis"``, which drives every port in turn.
    extra_args : list[str] | None
        Extra flags to pass to the solver, e.g. iterative solver options.

    Returns
    -------
    subprocess.CompletedProcess
        The finished solver process.

    Raises
    ------
    RuntimeError
        If the solver exits with an error.
    """
    # "Analysis" drives every port and runs Get_SParameters itself (so no -pos
    # and no ACTIVE_PORT); "AnalysisSinglePort" solves the single ACTIVE_PORT
    # and keeps the solution for a -pos field extraction.
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
    # Output is streamed rather than captured, so getdp's progress is visible;
    # that leaves stdout/stderr as None here, hence the guard (indexing them
    # raised TypeError and hid the actual failure).
    res = subprocess.run(args, cwd=workdir, capture_output=False, text=True)
    if res.returncode != 0:
        tail = "\n".join(
            stream[-2000:] for stream in (res.stdout, res.stderr) if stream
        )
        raise RuntimeError(
            f"getdp failed ({res.returncode}) running {' '.join(args)}"
            + (f":\n{tail}" if tail else "; see the output above.")
        )
    return res


def _read_power_value(outdir: str | Path, fname: str) -> float:
    # rows accumulate across calls (GetDP appends); take the latest one.
    path = Path(outdir) / fname
    if not path.exists():
        return 0.0
    return float(np.atleast_2d(np.loadtxt(path))[-1, -2])


def _conductor_loss(outdir: Path) -> float:
    """Total power lost in the lossy conductors, in watts, or ``0`` if none."""
    return sum(_read_power_value(outdir, p.name) for p in outdir.glob("Pcond_*.txt"))


def solve_fields_and_power(
    pro_path: str | Path,
    msh_path: str | Path,
    workdir: str | Path,
    freq: float,
    active: int = 1,
) -> tuple[str, str, float, float]:
    """
    Solve at one frequency and return the near fields and the powers together.

    Both come out of a single solve, rather than one each.

    Parameters
    ----------
    pro_path, msh_path : str | Path
        Paths to the problem and mesh files to solve.
    workdir : str | Path
        Working directory for the run. Results are written to an ``output``
        subdirectory of it.
    freq : float
        Frequency to solve at, in Hz.
    active : int
        Number of the port to drive. Default ``1``.

    Returns
    -------
    tuple[str, str, float, float]
        ``(e_pos, h_pos, p_loss, p_rad)`` -- paths to the written electric and
        magnetic near-field files, the power lost in the materials, and the
        power radiated. The two powers sum to the power accepted by the port.
        Only their ratio, e.g. the radiation efficiency, is meaningful; their
        absolute magnitudes are not.
    """
    # p_rad is what the port accepted less what the materials dissipated, not a
    # flux through the outer boundary -- that reads ~0 behind a PML.
    run_getdp(
        pro_path,
        msh_path,
        workdir,
        {"FREQ": freq, "ACTIVE_PORT": active},
        ["Get_Fields", "Get_Power"],
        resolution="AnalysisSinglePort",
    )
    outdir = Path(workdir).absolute() / "output"

    # Dielectric plus conductor loss, so p_rad + p_loss is the accepted power.
    p_loss = _read_power_value(outdir, "Ploss.txt") + _conductor_loss(outdir)
    v = read_complex(outdir / f"Vdrv_{active}.txt")
    i = read_complex(outdir / f"Idrv_{active}.txt")
    p_acc = 0.5 * float(np.real(v * np.conj(i)))
    # Clamped at 0: a small negative value is numerical noise on a
    # non-radiating structure.
    p_rad = max(p_acc - p_loss, 0.0)
    return (str(outdir / "e.pos"), str(outdir / "h.pos"), p_loss, p_rad)
