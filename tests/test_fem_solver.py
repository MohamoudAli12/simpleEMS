"""Tests for the GetDP solver driver.

``fem_solver`` is the thin layer between simpleEMS and the ``getdp`` binary:
it builds the command line, runs it, and parses the plain-text files GetDP
leaves behind. None of that needs a solver to test -- the subprocess is stubbed
and the output files are written by hand -- so this whole module runs anywhere.

The parsing rules are worth pinning precisely because they are silent when
wrong: GetDP *appends* to its output files across a sweep, so every reader here
takes the last row(s), and an off-by-one lands as a plausible-looking wrong
number rather than an error.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from simpleEMS import fem_solver
from simpleEMS.fem_solver import (
    _conductor_loss,
    _parse_resources,
    _read_power_value,
    _stages,
    SolveInfo,
    describe_solve,
    find_getdp,
    format_duration,
    format_memory,
    last_solve,
    read_complex,
    read_complex_rows,
    run_getdp,
    solve_fields_and_power,
)


def write_rows(path: Path, rows) -> Path:
    """Write ``[tag, Re, Im]`` rows in GetDP's output format."""
    path.write_text("\n".join(" ".join(str(v) for v in row) for row in rows) + "\n")
    return path


@pytest.fixture
def fake_getdp(monkeypatch):
    """Pretend a ``getdp`` binary exists, without needing one."""
    monkeypatch.setattr(fem_solver, "find_getdp", lambda: "/usr/bin/getdp")
    return "/usr/bin/getdp"


class FakeProcess:
    """A finished ``getdp`` whose output ``run_getdp`` reads line by line."""

    def __init__(self, args, returncode=0, output=""):
        self.args = args
        self.returncode = returncode
        self.stdout = iter(output.splitlines(keepends=True))

    def wait(self):
        return self.returncode


@pytest.fixture
def fake_popen(monkeypatch):
    """Stub the solver launch; the test says what it "printed" and returned."""

    def install(returncode=0, output=""):
        calls = []

        def popen(args, **kwargs):
            calls.append((args, kwargs))
            return FakeProcess(args, returncode, output)

        monkeypatch.setattr(fem_solver.subprocess, "Popen", popen)
        return calls

    return install


@pytest.fixture
def spy_run(fake_popen):
    """Capture the command line ``run_getdp`` builds, and fake a clean exit."""
    return fake_popen()


# ---------------------------------------------------------------------
# find_getdp
# ---------------------------------------------------------------------
class TestFindGetdp:
    def test_returns_the_path_from_path_lookup(self, monkeypatch):
        monkeypatch.setattr(fem_solver.shutil, "which", lambda name: f"/opt/{name}")

        assert find_getdp() == "/opt/getdp"

    def test_looks_for_the_binary_by_name(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            fem_solver.shutil, "which", lambda name: seen.append(name) or "/opt/getdp"
        )

        find_getdp()

        assert seen == ["getdp"]

    def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.setattr(fem_solver.shutil, "which", lambda name: None)

        with pytest.raises(RuntimeError, match="getdp binary not found"):
            find_getdp()

    def test_the_error_says_how_to_install_one(self, monkeypatch):
        """A user hitting this has no idea what GetDP is; the message is the
        only place that explains it."""
        monkeypatch.setattr(fem_solver.shutil, "which", lambda name: None)

        with pytest.raises(RuntimeError, match="simpleems install getdp"):
            find_getdp()


# ---------------------------------------------------------------------
# read_complex
# ---------------------------------------------------------------------
class TestReadComplex:
    def test_reads_the_real_and_imaginary_columns(self, tmp_path):
        path = write_rows(tmp_path / "v.txt", [[0, 1.5, -2.5]])

        assert read_complex(path) == complex(1.5, -2.5)

    def test_a_single_row_file_is_not_mistaken_for_a_single_column(self, tmp_path):
        """``np.loadtxt`` returns a 1-D array for a one-row file; without the
        reshape the column indexing would read the wrong values."""
        path = write_rows(tmp_path / "v.txt", [[0, 3.0, 4.0]])

        assert read_complex(path) == complex(3.0, 4.0)

    def test_the_last_row_wins(self, tmp_path):
        """GetDP appends a row per solve; the newest is the one wanted."""
        path = write_rows(
            tmp_path / "v.txt", [[0, 1.0, 1.0], [0, 2.0, 2.0], [0, 9.0, -9.0]]
        )

        assert read_complex(path) == complex(9.0, -9.0)

    def test_missing_file_reads_as_zero(self, tmp_path):
        """A quantity the problem file never wrote is absent, not an error."""
        assert read_complex(tmp_path / "absent.txt") == 0j

    def test_accepts_a_string_path(self, tmp_path):
        path = write_rows(tmp_path / "v.txt", [[0, 1.0, 2.0]])

        assert read_complex(str(path)) == complex(1.0, 2.0)


# ---------------------------------------------------------------------
# read_complex_rows
# ---------------------------------------------------------------------
class TestReadComplexRows:
    def test_returns_one_value_per_port(self, tmp_path):
        path = write_rows(tmp_path / "s.txt", [[0, 1.0, 0.0], [0, 0.0, 1.0]])

        assert read_complex_rows(path, 2) == [complex(1, 0), complex(0, 1)]

    def test_returns_the_last_block_oldest_first(self, tmp_path):
        """Two solves have run; the second one's rows are the live ones, and
        they must come back in driven-port order, not reversed."""
        path = write_rows(
            tmp_path / "s.txt",
            [[0, 1.0, 0.0], [0, 2.0, 0.0], [0, 3.0, 0.0], [0, 4.0, 0.0]],
        )

        assert read_complex_rows(path, 2) == [complex(3, 0), complex(4, 0)]

    def test_single_row_single_port(self, tmp_path):
        path = write_rows(tmp_path / "s.txt", [[0, 0.5, 0.25]])

        assert read_complex_rows(path, 1) == [complex(0.5, 0.25)]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found"):
            read_complex_rows(tmp_path / "absent.txt", 1)

    def test_the_missing_file_error_names_the_fix(self, tmp_path):
        """This fires when a stale .pro from before the single-launch port loop
        is reused, which is not something the user could guess."""
        with pytest.raises(RuntimeError, match="Re-run build_mesh"):
            read_complex_rows(tmp_path / "absent.txt", 1)

    def test_too_few_rows_raises_rather_than_misaligning(self, tmp_path):
        path = write_rows(tmp_path / "s.txt", [[0, 1.0, 0.0]])

        with pytest.raises(RuntimeError, match="1 row"):
            read_complex_rows(path, 2)

    def test_the_short_file_error_reports_both_counts(self, tmp_path):
        path = write_rows(tmp_path / "s.txt", [[0, 1.0, 0.0], [0, 2.0, 0.0]])

        with pytest.raises(RuntimeError, match="2 row.*3 were expected"):
            read_complex_rows(path, 3)


# ---------------------------------------------------------------------
# run_getdp
# ---------------------------------------------------------------------
class TestRunGetdp:
    def call(self, calls):
        """The argv of the single recorded invocation."""
        assert len(calls) == 1
        return calls[0][0]

    def test_invokes_the_located_binary_on_the_problem_file(
        self, tmp_path, fake_getdp, spy_run
    ):
        run_getdp("p.pro", "m.msh", tmp_path, {}, None)

        args = self.call(spy_run)
        assert args[0] == fake_getdp
        assert args[1] == "p.pro"

    def test_passes_the_mesh(self, tmp_path, fake_getdp, spy_run):
        run_getdp("p.pro", "m.msh", tmp_path, {}, None)

        args = self.call(spy_run)
        assert args[args.index("-msh") + 1] == "m.msh"

    def test_runs_in_the_given_working_directory(self, tmp_path, fake_getdp, spy_run):
        run_getdp("p.pro", "m.msh", tmp_path, {}, None)

        assert spy_run[0][1]["cwd"] == tmp_path

    def test_setnumbers_become_flag_triples(self, tmp_path, fake_getdp, spy_run):
        run_getdp("p.pro", "m.msh", tmp_path, {"FREQ": 2.45e9, "ACTIVE_PORT": 1}, None)

        args = self.call(spy_run)
        i = args.index("FREQ")
        assert args[i - 1] == "-setnumber"
        assert float(args[i + 1]) == pytest.approx(2.45e9)

    def test_integer_setnumbers_are_passed_as_floats(
        self, tmp_path, fake_getdp, spy_run
    ):
        """GetDP's -setnumber takes a number; an int repr like ``1`` is fine
        but the code normalises everything through float for consistency."""
        run_getdp("p.pro", "m.msh", tmp_path, {"ACTIVE_PORT": 2}, None)

        args = self.call(spy_run)
        assert args[args.index("ACTIVE_PORT") + 1] == "2.0"

    def test_resolution_defaults_to_analysis(self, tmp_path, fake_getdp, spy_run):
        run_getdp("p.pro", "m.msh", tmp_path, {}, None)

        args = self.call(spy_run)
        assert args[args.index("-solve") + 1] == "Analysis"

    def test_resolution_is_honoured(self, tmp_path, fake_getdp, spy_run):
        run_getdp("p.pro", "m.msh", tmp_path, {}, None, resolution="AnalysisSinglePort")

        args = self.call(spy_run)
        assert args[args.index("-solve") + 1] == "AnalysisSinglePort"

    def test_no_postop_means_no_pos_flag(self, tmp_path, fake_getdp, spy_run):
        """The internal sweep's Resolution calls PostOperation itself; passing
        -pos as well makes GetDP run it twice."""
        run_getdp("p.pro", "m.msh", tmp_path, {}, None)

        assert "-pos" not in self.call(spy_run)

    def test_a_single_postop_is_passed(self, tmp_path, fake_getdp, spy_run):
        run_getdp("p.pro", "m.msh", tmp_path, {}, "Get_Fields")

        args = self.call(spy_run)
        assert args[args.index("-pos") + 1] == "Get_Fields"

    def test_several_postops_are_extracted_from_one_solve(
        self, tmp_path, fake_getdp, spy_run
    ):
        run_getdp("p.pro", "m.msh", tmp_path, {}, ["Get_Fields", "Get_Power"])

        args = self.call(spy_run)
        i = args.index("-pos")
        assert args[i + 1 : i + 3] == ["Get_Fields", "Get_Power"]

    def test_the_gmsh_output_flag_is_pinned(self, tmp_path, fake_getdp, spy_run):
        """-v2 is an output-format flag, not a verbosity one: it asks for
        mesh-based Gmsh output wherever the .pro does not pin a format."""
        run_getdp("p.pro", "m.msh", tmp_path, {}, None)

        assert "-v2" in self.call(spy_run)

    def test_extra_args_are_appended(self, tmp_path, fake_getdp, spy_run):
        run_getdp(
            "p.pro", "m.msh", tmp_path, {}, None, extra_args=["-ksp_type", "gmres"]
        )

        args = self.call(spy_run)
        assert args[-2:] == ["-ksp_type", "gmres"]

    def test_paths_are_stringified(self, tmp_path, fake_getdp, spy_run):
        """subprocess would accept Path, but the argv is also interpolated into
        the failure message, where a Path repr would be noise."""
        run_getdp(Path("p.pro"), Path("m.msh"), tmp_path, {}, None)

        assert all(isinstance(a, str) for a in self.call(spy_run))

    def test_returns_the_completed_process(self, tmp_path, fake_getdp, spy_run):
        result = run_getdp("p.pro", "m.msh", tmp_path, {}, None)

        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 0

    def test_a_failing_solve_raises(self, tmp_path, fake_getdp, fake_popen):
        fake_popen(returncode=1)

        with pytest.raises(RuntimeError, match=r"getdp failed \(1\)"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)

    def test_the_failure_message_includes_the_command(
        self, tmp_path, fake_getdp, fake_popen
    ):
        fake_popen(returncode=1)

        with pytest.raises(RuntimeError, match="-solve Analysis"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)

    def test_the_failure_message_says_how_long_it_ran(
        self, tmp_path, fake_getdp, fake_popen
    ):
        """A solve that dies after an hour and one that dies on startup want
        very different debugging, so the duration goes in the message."""
        fake_popen(returncode=1)

        with pytest.raises(RuntimeError, match=r"after [\d.]+ s running"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)

    def test_a_silent_failure_does_not_mask_itself(
        self, tmp_path, fake_getdp, fake_popen
    ):
        """A solver that printed nothing leaves an empty tail; appending it
        would bury the exit status under a blank line."""
        fake_popen(returncode=2, output="")

        with pytest.raises(RuntimeError, match="see the output above"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)

    def test_the_output_is_tailed_into_the_message(
        self, tmp_path, fake_getdp, fake_popen
    ):
        fake_popen(returncode=1, output="Error : something went wrong\n")

        with pytest.raises(RuntimeError, match="something went wrong"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)

    def test_the_output_comes_back_on_the_result(
        self, tmp_path, fake_getdp, fake_popen
    ):
        """Streaming used to discard it; callers and the error tail both read
        it off the returned process."""
        fake_popen(output="Info    : Solving system {A}\n")

        result = run_getdp("p.pro", "m.msh", tmp_path, {}, None)

        assert "Solving system" in result.stdout


# ---------------------------------------------------------------------
# progress reporting
# ---------------------------------------------------------------------
class TestProgressReporting:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.42, "0.4 s"),
            (4.25, "4.2 s"),
            (59.9, "59.9 s"),
            (60.0, "1m 00s"),
            (185.0, "3m 05s"),
            (4350.0, "1h 12m 30s"),
        ],
        ids=[
            "sub-second",
            "seconds",
            "just-under-a-minute",
            "a-minute",
            "minutes",
            "hours",
        ],
    )
    def test_durations_read_the_way_a_person_says_them(self, seconds, expected):
        assert format_duration(seconds) == expected

    def test_the_problem_size_is_scraped_out_of_the_output(self):
        """Verbatim from getdp 4.0.0."""
        assert _parse_resources("Info    : System 1/1: 412033 Dofs\n")[0] == 412033

    def test_the_peak_memory_is_scraped_out_of_the_output(self):
        """Verbatim from getdp 4.0.0 -- note the ``=``, which an earlier
        pattern missed, leaving the memory silently unreported."""
        output = (
            "Info    : Stopped (Sun Sep 20 22:55:39 2026, Wall = 0.204s, "
            "CPU = 1.07s, Mem = 2100.5Mb)\n"
        )

        assert _parse_resources(output)[1] == pytest.approx(2100.5)

    def test_the_largest_report_wins(self):
        """getdp reports its resources at every stage; the peak is the useful
        one, and it is not the last line printed."""
        output = "Info : Mem = 100Mb\nInfo : Mem = 2100Mb\nInfo : Mem = 900Mb\n"

        assert _parse_resources(output)[1] == pytest.approx(2100)

    @pytest.mark.parametrize(
        ("megabytes", "expected"),
        [(70.8359, "71 MB"), (999.4, "999 MB"), (1150.0, "1.15 GB")],
        ids=["a-port-mode-solve", "just-under-a-gigabyte", "a-3d-solve"],
    )
    def test_memory_is_reported_at_a_readable_scale(self, megabytes, expected):
        """A port-mode solve peaks around 70 Mb; "0.07 GB" reads as noise."""
        assert format_memory(megabytes) == expected

    def test_a_solve_is_described_by_size_and_memory(self):
        info = SolveInfo(elapsed=1.0, dofs=9539, peak_mb=1150.0)

        assert describe_solve(info) == "9,539 DOF, peak 1.15 GB"

    def test_half_a_description_is_better_than_none(self):
        """getdp builds differ in what they report; whichever half arrived is
        still worth saying."""
        assert describe_solve(SolveInfo(1.0, 9539, None)) == "9,539 DOF"

    def test_the_stage_is_read_off_an_info_line(self):
        assert _stages("Info    : Post-processing (Compute)\n") == [
            "Post-processing (Compute)"
        ]

    def test_carriage_returns_carry_several_stages(self):
        """getdp overwrites its progress indicator in place, so a whole
        pre-processing run arrives as one newline-delimited line."""
        line = " 20%    : Pre-processing\r 90%    : Pre-processing\rInfo    : Solving\n"

        assert _stages(line)[-1] == "Solving"

    @pytest.mark.parametrize(
        "line",
        [
            "Info    :   1 KSP Residual norm 8.760107442077e-16",
            "Info    : Stopped (Sun Sep 20 22:55:30 2026, Wall = 0.26s, Mem = 70Mb)",
            "Info    : (Wall = 0.259939s, CPU = 1.01271s, Mem = 70.8984Mb)",
            "Info    : Eigenvalue 001: w^2 = 2.503984033690e+04",
        ],
        ids=["residual", "timestamp", "resources", "eigenvalue"],
    )
    def test_per_iteration_chatter_is_not_a_stage(self, line):
        """None of it says where the solve has got to, and a half-truncated
        timestamp on the progress line reads as a glitch."""
        assert _stages(line + "\n") == []

    def test_output_without_either_gets_no_detail(self):
        """A getdp build that reports neither still gets its duration; a
        separator with nothing after it would just be noise."""
        dofs, peak_mb = _parse_resources("Info    : Started\nInfo    : Stopped\n")

        assert (dofs, peak_mb) == (None, None)
        assert describe_solve(SolveInfo(1.0, None, None)) == ""

    def test_a_finished_solve_reports_its_duration(
        self, tmp_path, fake_getdp, spy_run, capsys
    ):
        run_getdp("p.pro", "m.msh", tmp_path, {}, None, label="S-params @ 2.45 GHz")

        out = capsys.readouterr().out
        assert "S-params @ 2.45 GHz: 0.0 s" in out

    def test_an_unlabelled_solve_names_the_resolution(
        self, tmp_path, fake_getdp, spy_run, capsys
    ):
        run_getdp("p.pro", "m.msh", tmp_path, {}, None, resolution="ModeAnalysis")

        assert "ModeAnalysis (p)" in capsys.readouterr().out

    def test_quiet_solves_print_nothing(self, tmp_path, fake_getdp, spy_run, capsys):
        """optimise/sweep callers thread their own verbose flag through."""
        run_getdp("p.pro", "m.msh", tmp_path, {}, None, verbose=False)

        assert capsys.readouterr().out == ""

    def test_the_last_solve_is_remembered(self, tmp_path, fake_getdp, fake_popen):
        """The sweep summarises the problem it solved without the solver's
        output having to travel back through every caller."""
        fake_popen(output="Info : System 1/1: 9539 Dofs\nInfo : Mem = 1150Mb\n")

        run_getdp("p.pro", "m.msh", tmp_path, {}, None, verbose=False)

        info = last_solve()
        assert (info.dofs, info.peak_mb) == (9539, pytest.approx(1150))
        assert info.elapsed > 0


# ---------------------------------------------------------------------
# power readers
# ---------------------------------------------------------------------
class TestPowerReaders:
    def test_reads_the_value_column(self, tmp_path):
        """GetDP writes ``[tag, value, 0]``; the wanted number is second last."""
        write_rows(tmp_path / "Ploss.txt", [[0, 0.25, 0.0]])

        assert _read_power_value(tmp_path, "Ploss.txt") == pytest.approx(0.25)

    def test_the_last_row_wins(self, tmp_path):
        write_rows(tmp_path / "Ploss.txt", [[0, 1.0, 0.0], [0, 0.5, 0.0]])

        assert _read_power_value(tmp_path, "Ploss.txt") == pytest.approx(0.5)

    def test_missing_file_reads_as_zero_power(self, tmp_path):
        """A lossless problem never writes the file at all."""
        assert _read_power_value(tmp_path, "Ploss.txt") == 0.0

    def test_conductor_loss_sums_every_conductor(self, tmp_path):
        write_rows(tmp_path / "Pcond_1.txt", [[0, 0.1, 0.0]])
        write_rows(tmp_path / "Pcond_2.txt", [[0, 0.2, 0.0]])

        assert _conductor_loss(tmp_path) == pytest.approx(0.3)

    def test_conductor_loss_is_zero_with_no_lossy_conductors(self, tmp_path):
        assert _conductor_loss(tmp_path) == 0.0

    def test_conductor_loss_ignores_other_files(self, tmp_path):
        write_rows(tmp_path / "Pcond_1.txt", [[0, 0.1, 0.0]])
        write_rows(tmp_path / "Ploss.txt", [[0, 99.0, 0.0]])

        assert _conductor_loss(tmp_path) == pytest.approx(0.1)


# ---------------------------------------------------------------------
# solve_fields_and_power
# ---------------------------------------------------------------------
class TestSolveFieldsAndPower:
    @pytest.fixture
    def staged(self, tmp_path, monkeypatch):
        """Stub the solve and let the test lay out the files it "wrote"."""
        calls = []
        outdir = tmp_path / "output"
        outdir.mkdir()

        def fake_run_getdp(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr(fem_solver, "run_getdp", fake_run_getdp)
        return tmp_path, outdir, calls

    def test_asks_for_both_results_from_one_solve(self, staged):
        """Two separate solves would double the cost of every pattern point."""
        workdir, _outdir, calls = staged

        solve_fields_and_power("p.pro", "m.msh", workdir, 2.45e9)

        _args, _kw = calls[0]
        assert _args[4] == ["Get_Fields", "Get_Power"]

    def test_uses_the_single_port_resolution(self, staged):
        workdir, _outdir, calls = staged

        solve_fields_and_power("p.pro", "m.msh", workdir, 2.45e9)

        assert calls[0][1]["resolution"] == "AnalysisSinglePort"

    def test_passes_the_frequency_and_active_port(self, staged):
        workdir, _outdir, calls = staged

        solve_fields_and_power("p.pro", "m.msh", workdir, 2.45e9, active=2)

        assert calls[0][0][3] == {"FREQ": 2.45e9, "ACTIVE_PORT": 2}

    def test_active_port_defaults_to_one(self, staged):
        workdir, _outdir, calls = staged

        solve_fields_and_power("p.pro", "m.msh", workdir, 2.45e9)

        assert calls[0][0][3]["ACTIVE_PORT"] == 1

    def test_returns_the_field_file_paths(self, staged):
        workdir, outdir, _calls = staged

        e_pos, h_pos, _p_loss, _p_rad = solve_fields_and_power(
            "p.pro", "m.msh", workdir, 2.45e9
        )

        assert Path(e_pos) == outdir / "e.pos"
        assert Path(h_pos) == outdir / "h.pos"

    def test_loss_sums_dielectric_and_conductor(self, staged):
        workdir, outdir, _calls = staged
        write_rows(outdir / "Ploss.txt", [[0, 0.2, 0.0]])
        write_rows(outdir / "Pcond_1.txt", [[0, 0.05, 0.0]])

        _e, _h, p_loss, _p_rad = solve_fields_and_power(
            "p.pro", "m.msh", workdir, 2.45e9
        )

        assert p_loss == pytest.approx(0.25)

    def test_radiated_power_is_accepted_minus_lost(self, staged):
        """p_acc = 0.5 Re(V I*); with V = 2, I = 1 that is 1 W accepted, so a
        0.25 W loss leaves 0.75 W radiated."""
        workdir, outdir, _calls = staged
        write_rows(outdir / "Vdrv_1.txt", [[0, 2.0, 0.0]])
        write_rows(outdir / "Idrv_1.txt", [[0, 1.0, 0.0]])
        write_rows(outdir / "Ploss.txt", [[0, 0.25, 0.0]])

        _e, _h, p_loss, p_rad = solve_fields_and_power(
            "p.pro", "m.msh", workdir, 2.45e9
        )

        assert p_loss == pytest.approx(0.25)
        assert p_rad == pytest.approx(0.75)

    def test_reads_the_driven_ports_own_voltage_and_current(self, staged):
        workdir, outdir, _calls = staged
        write_rows(outdir / "Vdrv_1.txt", [[0, 100.0, 0.0]])
        write_rows(outdir / "Idrv_1.txt", [[0, 100.0, 0.0]])
        write_rows(outdir / "Vdrv_2.txt", [[0, 2.0, 0.0]])
        write_rows(outdir / "Idrv_2.txt", [[0, 1.0, 0.0]])

        _e, _h, _p_loss, p_rad = solve_fields_and_power(
            "p.pro", "m.msh", workdir, 2.45e9, active=2
        )

        assert p_rad == pytest.approx(1.0)

    def test_reactive_port_power_does_not_count_as_radiated(self, staged):
        """A purely reactive port accepts no real power."""
        workdir, outdir, _calls = staged
        write_rows(outdir / "Vdrv_1.txt", [[0, 0.0, 2.0]])
        write_rows(outdir / "Idrv_1.txt", [[0, 1.0, 0.0]])

        _e, _h, _p_loss, p_rad = solve_fields_and_power(
            "p.pro", "m.msh", workdir, 2.45e9
        )

        assert p_rad == pytest.approx(0.0)

    def test_negative_radiated_power_is_clamped_to_zero(self, staged):
        """On a non-radiating structure the difference is numerical noise and
        can come out slightly negative; a negative radiated power would make
        the efficiency nonsensical downstream."""
        workdir, outdir, _calls = staged
        write_rows(outdir / "Vdrv_1.txt", [[0, 0.1, 0.0]])
        write_rows(outdir / "Idrv_1.txt", [[0, 0.1, 0.0]])
        write_rows(outdir / "Ploss.txt", [[0, 5.0, 0.0]])

        _e, _h, _p_loss, p_rad = solve_fields_and_power(
            "p.pro", "m.msh", workdir, 2.45e9
        )

        assert p_rad == 0.0

    def test_missing_output_files_give_zero_powers(self, staged):
        """Nothing was written, so nothing is claimed -- not a crash."""
        workdir, _outdir, _calls = staged

        _e, _h, p_loss, p_rad = solve_fields_and_power(
            "p.pro", "m.msh", workdir, 2.45e9
        )

        assert p_loss == 0.0
        assert p_rad == 0.0


# ---------------------------------------------------------------------
# against the real binary
# ---------------------------------------------------------------------
@pytest.mark.needs_getdp_bin
def test_find_getdp_locates_the_installed_binary():
    """The stubbed tests above all assume this resolves; check it really does
    when a binary is present."""
    assert Path(find_getdp()).exists()


def test_read_complex_rows_matches_read_complex_on_the_last_row(tmp_path):
    """The two readers disagreeing would silently desync the sweep from the
    field extraction, which both read from the same files."""
    path = write_rows(tmp_path / "s.txt", [[0, 1.0, 2.0], [0, 3.0, 4.0], [0, 5.0, 6.0]])

    assert read_complex_rows(path, 1) == [read_complex(path)]


def test_power_reader_tolerates_extra_columns(tmp_path):
    """GetDP's column count varies with the quantity written; the reader keys
    off the end of the row, not a fixed index."""
    write_rows(tmp_path / "Ploss.txt", [[0, 1, 2, 0.75, 0.0]])

    assert _read_power_value(tmp_path, "Ploss.txt") == pytest.approx(0.75)


def test_readers_handle_scientific_notation(tmp_path):
    """GetDP writes small powers in exponent form."""
    path = write_rows(tmp_path / "v.txt", [[0, "1.5e-12", "-2.5e-13"]])

    value = read_complex(path)

    assert value.real == pytest.approx(1.5e-12)
    assert value.imag == pytest.approx(-2.5e-13)


def test_loadtxt_column_convention_is_what_the_readers_assume(tmp_path):
    """Guards the shared assumption: column 0 is a tag, 1 is Re, 2 is Im."""
    path = write_rows(tmp_path / "v.txt", [[7, 1.0, 2.0]])

    raw = np.atleast_2d(np.loadtxt(path))

    assert raw[-1, 0] == 7
    assert complex(raw[-1, 1], raw[-1, 2]) == read_complex(path)
