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

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .console import console

# What getdp prints, verbatim:
#   "Info    : System 1/1: 763 Dofs"
#   "Info    : Stopped (..., Wall = 0.2s, CPU = 1.07s, Mem = 70.8633Mb)"
#   " 20%    : Pre-processing"      <- carriage-return separated, many per line
# so the stage pattern takes the percentage lines too, and the resource scrapes
# stay best-effort: a build that prints neither still reports its duration.
_STAGE = re.compile(r"^(?:Info|\d+%)\s*:\s*(\S.*?)$")

# Per-iteration residuals, eigenvalue dumps and resource/timestamp lines say
# nothing about where the solve has got to, and a half-truncated timestamp on
# the progress line reads as a glitch.
_STAGE_NOISE = re.compile(r"^(?:\(|Started|Stopped|\d+ KSP|Eigenvalue|[wf] =)", re.I)
_MEMORY = re.compile(r"Mem\s*=?\s*([\d.]+)\s*([kMG])b", re.IGNORECASE)
_DOFS = re.compile(r"(\d+)\s+Dofs?\b", re.IGNORECASE)


class SolveInfo(NamedTuple):
    """
    What one getdp run cost.

    Attributes
    ----------
    elapsed : float
        Wall seconds the run took.
    dofs : int | None
        Unknowns in the solved system, or ``None`` if getdp did not say.
    peak_mb : float | None
        Peak memory in megabytes, or ``None`` if getdp did not say.
    """

    elapsed: float
    dofs: int | None
    peak_mb: float | None


# The last run's cost, so a sweep can summarise the problem it just solved
# without threading the solver's output back through every caller.
_last: SolveInfo | None = None


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


def last_solve() -> SolveInfo | None:
    """
    What the most recent getdp run cost.

    Returns
    -------
    SolveInfo | None
        The duration, problem size and peak memory of the last run, or
        ``None`` if none has finished in this process yet.
    """
    return _last


def format_duration(seconds: float) -> str:
    """
    Render a duration the way a person reads one.

    Parameters
    ----------
    seconds : float
        A duration in seconds.

    Returns
    -------
    str
        ``"4.2 s"``, ``"3m 05s"`` or ``"1h 12m 30s"``.
    """
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, whole_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {whole_seconds:02d}s"
    return f"{minutes}m {whole_seconds:02d}s"


def format_memory(megabytes: float) -> str:
    """
    Render a memory figure at a readable scale.

    Parameters
    ----------
    megabytes : float
        An amount of memory, in megabytes.

    Returns
    -------
    str
        ``"71 MB"`` below a gigabyte, ``"2.10 GB"`` above it.
    """
    if megabytes >= 1e3:
        return f"{megabytes / 1e3:.2f} GB"
    return f"{megabytes:.0f} MB"


def _parse_resources(output: str) -> tuple[int | None, float | None]:
    """
    Problem size and peak memory, scraped out of what getdp printed.

    Parameters
    ----------
    output : str
        Everything the solver wrote during one run.

    Returns
    -------
    tuple[int | None, float | None]
        ``(dofs, peak_mb)``, each ``None`` where getdp reported nothing.
        getdp reports its resources at every stage, so the peak is the
        interesting one and it is not the last figure printed.
    """
    dofs = [int(match.group(1)) for match in _DOFS.finditer(output)]
    scale = {"k": 1e-3, "m": 1.0, "g": 1e3}  # to megabytes
    memories = [
        float(match.group(1)) * scale[match.group(2).lower()]
        for match in _MEMORY.finditer(output)
    ]
    return (max(dofs) if dofs else None, max(memories) if memories else None)


def describe_solve(info: SolveInfo | None) -> str:
    """
    Name the size of a solve, for a progress or summary line.

    Parameters
    ----------
    info : SolveInfo | None
        What a run cost, as :func:`last_solve` reports it.

    Returns
    -------
    str
        ``"9,539 DOF, peak 1.15 GB"``, whichever half getdp reported, or an
        empty string when it reported neither.
    """
    if info is None:
        return ""
    parts = []
    if info.dofs is not None:
        parts.append(f"{info.dofs:,} DOF")
    if info.peak_mb is not None:
        parts.append(f"peak {format_memory(info.peak_mb)}")
    return ", ".join(parts)


def _stages(line: str) -> list[str]:
    """
    The stages getdp announced on one line of output.

    getdp overwrites its progress indicator with carriage returns, so one
    newline-delimited line can carry a dozen of them.

    Parameters
    ----------
    line : str
        One line as read from the solver.

    Returns
    -------
    list[str]
        The stage names in the order printed, without the per-iteration
        residuals and resource lines. The last one is where the solve has
        got to.
    """
    return [
        match.group(1)
        for chunk in line.replace("\r", "\n").splitlines()
        if (match := _STAGE.match(chunk.strip()))
        and not _STAGE_NOISE.match(match.group(1))
    ]


def _stream(
    args: list[str], workdir: str | Path, description: str | None
) -> tuple[int, str]:
    """
    Run ``args``, showing getdp's current stage on one self-updating line.

    Parameters
    ----------
    args : list[str]
        The command line to run.
    workdir : str | Path
        Working directory for the run.
    description : str | None
        What to call this run in the progress line, or ``None`` to stay quiet.

    Returns
    -------
    tuple[int, str]
        The exit status, and everything the process wrote with stdout and
        stderr interleaved.
    """
    # stdin is closed, not inherited: getdp's eigenvalue solver prompts for
    # ARPACK settings when it has no eigen.par to read, and an inherited stdin
    # that never delivers a line hangs the whole sweep with no output.
    process = subprocess.Popen(
        args,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    # The live line only makes sense on a terminal; a log file or a CI run
    # would collect one copy of it per refresh, so fall back to a plain
    # start line there.
    live = description is not None and console.is_terminal
    if description is not None and not live:
        console.print(f"[info]getdp {description}: solving...[/info]")
    lines: list[str] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[info]{task.description}[/info]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=not live,
    ) as progress:
        task = progress.add_task(f"getdp {description}", total=None)
        for line in process.stdout or []:
            lines.append(line)
            if not live:
                continue
            stages = _stages(line)
            if stages:
                progress.update(
                    task, description=f"getdp {description} · {stages[-1][:60]}"
                )
    return process.wait(), "".join(lines)


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
    label: str | None = None,
    verbose: bool = True,
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
    label : str | None
        What to call this run when reporting progress, e.g.
        ``"S-params @ 2.4500 GHz"``. Defaults to the resolution and the
        problem file's name.
    verbose : bool
        Report the solver's progress and how long it took. Default ``True``.

    Returns
    -------
    subprocess.CompletedProcess
        The finished solver process. Its ``stdout`` holds everything the
        solver wrote, stderr interleaved.

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
    # Not a verbosity flag, despite the name: getdp -help lists -v2 under
    # output options as "create mesh-based Gmsh output files when possible".
    # Verbosity is -v <num>. Inert while the .pro files pin Format GmshParsed.
    args += ["-v2"]
    if extra_args:  # passthrough getdp/PETSc flags
        args += list(extra_args)

    label = label or f"{resolution} ({Path(str(pro_path)).stem})"
    start = time.perf_counter()
    returncode, output = _stream(args, workdir, label if verbose else None)
    elapsed = time.perf_counter() - start
    global _last
    _last = SolveInfo(elapsed, *_parse_resources(output))

    if returncode != 0:
        raise RuntimeError(
            f"getdp failed ({returncode}) after {format_duration(elapsed)} "
            f"running {' '.join(args)}"
            + (f":\n{output[-2000:]}" if output.strip() else "; see the output above.")
        )
    if verbose:
        size = describe_solve(_last)
        console.print(
            f"[info]getdp {label}: {format_duration(elapsed)}"
            f"{f' · {size}' if size else ''}[/info]"
        )
    return subprocess.CompletedProcess(args, returncode, output, "")


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
        label=f"fields @ {freq / 1e9:.4f} GHz",
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
