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
    _read_power_value,
    find_getdp,
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


@pytest.fixture
def spy_run(monkeypatch):
    """Capture the command line ``run_getdp`` builds, and fake a clean exit."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, returncode=0, stdout=None, stderr=None)

    monkeypatch.setattr(fem_solver.subprocess, "run", fake_run)
    return calls


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

    def test_verbosity_is_pinned(self, tmp_path, fake_getdp, spy_run):
        """-v2 shows progress without per-iteration spam; the sweep prints a
        lot of solves and the default level buries it."""
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

    def test_a_failing_solve_raises(self, tmp_path, fake_getdp, monkeypatch):
        monkeypatch.setattr(
            fem_solver.subprocess,
            "run",
            lambda args, **kw: subprocess.CompletedProcess(args, 1, None, None),
        )

        with pytest.raises(RuntimeError, match=r"getdp failed \(1\)"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)

    def test_the_failure_message_includes_the_command(
        self, tmp_path, fake_getdp, monkeypatch
    ):
        monkeypatch.setattr(
            fem_solver.subprocess,
            "run",
            lambda args, **kw: subprocess.CompletedProcess(args, 1, None, None),
        )

        with pytest.raises(RuntimeError, match="-solve Analysis"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)

    def test_uncaptured_streams_do_not_mask_the_failure(
        self, tmp_path, fake_getdp, monkeypatch
    ):
        """Output is streamed, so stdout/stderr come back None. Indexing them
        used to raise TypeError and hide the real error."""
        monkeypatch.setattr(
            fem_solver.subprocess,
            "run",
            lambda args, **kw: subprocess.CompletedProcess(args, 2, None, None),
        )

        with pytest.raises(RuntimeError, match="see the output above"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)

    def test_captured_streams_are_tailed_into_the_message(
        self, tmp_path, fake_getdp, monkeypatch
    ):
        monkeypatch.setattr(
            fem_solver.subprocess,
            "run",
            lambda args, **kw: subprocess.CompletedProcess(
                args, 1, "stdout detail", "stderr detail"
            ),
        )

        with pytest.raises(RuntimeError, match="stderr detail"):
            run_getdp("p.pro", "m.msh", tmp_path, {}, None)


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
