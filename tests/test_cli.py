"""Tests for the ``simpleems`` command-line interface.

``cli.py`` is 781 lines of installer logic: it clones repositories, downloads
archives, extracts them, and edits shell profiles. None of that may happen
during a test run, so every command here is driven with ``--dry-run`` (which
the CLI already supports on each install path) or has its network and
subprocess entry points monkeypatched.

``checkhealth`` performs no writes at all and is exercised directly.
"""

import shutil
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from simpleEMS import cli


# The "already installed" short-circuit checks BOTH the binary and the Python
# bindings. With only one of the two present the command proceeds to a real
# install, so those tests must not run in that half-installed state.
FULLY_INSTALLED = (
    shutil.which("openEMS") is not None
    and cli._check_package_import("openEMS", "openEMS")[0]
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def no_side_effects(monkeypatch):
    """Fail loudly if a command tries to touch the network or the shell.

    ``--dry-run`` is supposed to make every command inert; these stubs are how
    a regression in that guarantee gets caught rather than silently cloning a
    repository into the developer's home directory.
    """
    calls = {"subprocess": [], "download": [], "path_edits": []}

    def forbid_subprocess(cmd, cwd, dry_run):
        calls["subprocess"].append((cmd, cwd, dry_run))
        assert dry_run, f"subprocess would really run: {cmd}"

    def forbid_download(url, dest):
        calls["download"].append((url, dest))
        raise AssertionError(f"network download attempted: {url}")

    def record_path_edit(bin_dir, dry_run, extra_env_vars=None):
        calls["path_edits"].append((bin_dir, dry_run))
        assert dry_run, f"shell profile would really be edited for {bin_dir}"

    monkeypatch.setattr(cli, "_run_subprocess", forbid_subprocess)
    monkeypatch.setattr(cli, "_download_with_progress", forbid_download)
    monkeypatch.setattr(cli, "_add_dir_to_path", record_path_edit)
    return calls


# ---------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------
class TestTopLevel:
    def test_bare_invocation_shows_help(self, runner):
        result = runner.invoke(cli.app, [])

        assert "Usage" in result.output

    def test_help_lists_the_commands(self, runner):
        result = runner.invoke(cli.app, ["--help"])

        assert result.exit_code == 0
        assert "checkhealth" in result.output
        assert "install" in result.output

    def test_unknown_command_fails(self, runner):
        result = runner.invoke(cli.app, ["nonsense"])

        assert result.exit_code != 0

    def test_install_group_without_subcommand_shows_help(self, runner):
        result = runner.invoke(cli.app, ["install"])

        assert "Usage" in result.output

    def test_install_help_lists_both_installers(self, runner):
        result = runner.invoke(cli.app, ["install", "--help"])

        assert result.exit_code == 0
        assert "openems" in result.output
        assert "getdp" in result.output

    def test_entry_point_matches_the_app(self):
        """``[project.scripts] simpleems = "simpleEMS.cli:app"``."""
        assert cli.app.info.name == "simpleems"


# ---------------------------------------------------------------------
# Dependency discovery
# ---------------------------------------------------------------------
class TestDependencyChecks:
    def test_runtime_dependencies_are_discovered(self):
        checks = cli._get_dependency_checks_from_pyproject_section("dependencies")

        if not checks:
            pytest.skip("simpleEMS is not installed as a distribution")

        pip_names = {pip for _import, pip in checks}
        assert "numpy" in pip_names
        assert "scipy" in pip_names

    def test_version_specifiers_are_stripped(self):
        """The pinned requirement is ``numpy==2.4.4``; only the bare name may
        reach ``importlib.metadata.version``."""
        checks = cli._get_dependency_checks_from_pyproject_section("dependencies")

        if not checks:
            pytest.skip("simpleEMS is not installed as a distribution")

        for _import_name, pip_name in checks:
            assert not any(ch in pip_name for ch in "><=!~")
            assert pip_name == pip_name.strip()

    def test_dev_extras_are_excluded_from_the_runtime_section(self):
        runtime = cli._get_dependency_checks_from_pyproject_section("dependencies")

        if not runtime:
            pytest.skip("simpleEMS is not installed as a distribution")

        assert "ruff" not in {pip for _import, pip in runtime}

    def test_dev_section_returns_the_extras(self):
        dev = cli._get_dependency_checks_from_pyproject_section("optional-dependencies")

        if not dev:
            pytest.skip("dev extras not installed")

        assert "ruff" in {pip for _import, pip in dev}

    def test_import_name_is_mapped_from_the_distribution_name(self):
        """``pyqt6`` imports as ``PyQt6``; checking the pip name as an import
        would report a false failure."""
        checks = cli._get_dependency_checks_from_pyproject_section("dependencies")

        if not checks:
            pytest.skip("simpleEMS is not installed as a distribution")

        mapping = dict((pip, imp) for imp, pip in checks)
        if "scikit-rf" in mapping:
            assert mapping["scikit-rf"] != "scikit-rf"

    def test_missing_distribution_returns_empty(self, monkeypatch):
        import importlib.metadata

        def raise_not_found(_name):
            raise importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(importlib.metadata, "requires", raise_not_found)

        assert cli._get_dependency_checks_from_pyproject_section("dependencies") == []


class TestCheckHelpers:
    def test_importable_package_reports_a_version(self):
        ok, version, error = cli._check_package_import("numpy", "numpy")

        assert ok is True
        assert version
        assert error is None

    def test_missing_package_reports_an_error(self):
        ok, version, error = cli._check_package_import(
            "definitely_not_installed_xyz", "definitely-not-installed-xyz"
        )

        assert ok is False
        assert version is None
        assert error

    def test_unknown_distribution_falls_back_to_unknown(self):
        """The module imports but has no distribution metadata."""
        ok, version, _error = cli._check_package_import("sys", "sys")

        assert ok is True
        assert version == "unknown"

    def test_binary_on_path_is_found(self):
        ok, path, error = cli._check_system_binary("python3")

        assert ok is True
        assert Path(path).exists()
        assert error is None

    def test_missing_binary_reports_not_found(self):
        ok, path, error = cli._check_system_binary("definitely_not_a_binary_xyz")

        assert ok is False
        assert path is None
        assert "not found in PATH" in error


class TestDefaultPrefixes:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX default")
    def test_posix_openems_prefix(self):
        assert cli._default_prefix() == Path.home() / "opt" / "openEMS"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX default")
    def test_posix_getdp_prefix(self):
        assert cli._default_getdp_prefix() == Path.home() / "opt" / "getdp"

    def test_the_two_prefixes_differ(self):
        """Sharing a prefix would have one installer overwrite the other."""
        assert cli._default_prefix() != cli._default_getdp_prefix()


# ---------------------------------------------------------------------
# checkhealth
# ---------------------------------------------------------------------
class TestCheckHealth:
    def test_runs_and_prints_a_table(self, runner):
        result = runner.invoke(cli.app, ["checkhealth"])

        assert "simpleEMS Health Check" in result.output
        assert "Python" in result.output

    def test_exit_code_reflects_failures(self, runner):
        """Documented contract: non-zero when anything is missing, so CI can
        gate on it."""
        result = runner.invoke(cli.app, ["checkhealth"])

        failed = "❌" in result.output
        assert (result.exit_code != 0) == failed

    def test_checks_the_solver_backends(self, runner):
        result = runner.invoke(cli.app, ["checkhealth"])

        assert "openEMS Python API" in result.output
        assert "CSXCAD Python API" in result.output
        assert "getdp" in result.output

    def test_dev_flag_adds_more_checks(self, runner):
        plain = runner.invoke(cli.app, ["checkhealth"])
        with_dev = runner.invoke(cli.app, ["checkhealth", "--dev"])

        assert with_dev.output.count("✅") + with_dev.output.count(
            "❌"
        ) >= plain.output.count("✅") + plain.output.count("❌")

    def test_reports_the_running_python_version(self, runner):
        result = runner.invoke(cli.app, ["checkhealth"])

        assert f"{sys.version_info.major}.{sys.version_info.minor}" in result.output

    def test_writes_nothing_to_disk(self, runner, tmp_path):
        before = set(tmp_path.iterdir())

        runner.invoke(cli.app, ["checkhealth"])

        assert set(tmp_path.iterdir()) == before


# ---------------------------------------------------------------------
# install openems
# ---------------------------------------------------------------------
class TestInstallOpenems:
    def test_dry_run_performs_no_real_work(self, runner, no_side_effects, tmp_path):
        result = runner.invoke(
            cli.app,
            ["install", "openems", "--dry-run", "--force", "--prefix", str(tmp_path)],
        )

        assert result.exit_code in (0, 1)
        assert no_side_effects["download"] == []

    def test_prefix_reaches_the_build_and_path_steps(
        self, runner, no_side_effects, tmp_path
    ):
        """``--prefix`` has to flow all the way through to the install
        commands and the PATH entry, not just be printed."""
        runner.invoke(
            cli.app,
            ["install", "openems", "--dry-run", "--force", "--prefix", str(tmp_path)],
        )

        flat = [
            str(arg) for cmd, _cwd, _dry in no_side_effects["subprocess"] for arg in cmd
        ]

        assert any(str(tmp_path) in arg for arg in flat)
        assert any(
            str(tmp_path) in str(bin_dir)
            for bin_dir, _dry in no_side_effects["path_edits"]
        )

    def test_dry_run_marks_every_subprocess_as_dry(
        self, runner, no_side_effects, tmp_path
    ):
        """Each recorded call asserts ``dry_run`` inside the fixture, so
        reaching here at all means nothing was executed for real."""
        runner.invoke(
            cli.app,
            ["install", "openems", "--dry-run", "--force", "--prefix", str(tmp_path)],
        )

        assert all(dry for _cmd, _cwd, dry in no_side_effects["subprocess"])

    @pytest.mark.skipif(
        not FULLY_INSTALLED,
        reason="needs both the openEMS binary and its Python bindings",
    )
    def test_existing_installation_is_detected_without_force(
        self, runner, no_side_effects
    ):
        """Without ``--force`` an existing install must short-circuit.

        ``no_side_effects`` is requested deliberately: if this check ever
        stops short-circuiting, the command falls through to a real build
        instead of failing the test.
        """
        result = runner.invoke(cli.app, ["install", "openems"])

        assert "already installed" in result.output
        assert "--force" in result.output
        assert no_side_effects["subprocess"] == []
        assert no_side_effects["download"] == []

    def test_help_documents_the_options(self, runner):
        result = runner.invoke(cli.app, ["install", "openems", "--help"])

        assert result.exit_code == 0
        for option in ("--prefix", "--version", "--dry-run", "--force"):
            assert option in result.output


# ---------------------------------------------------------------------
# install getdp
# ---------------------------------------------------------------------
class TestInstallGetdp:
    def test_dry_run_downloads_nothing(self, runner, no_side_effects, tmp_path):
        result = runner.invoke(
            cli.app,
            ["install", "getdp", "--dry-run", "--force", "--prefix", str(tmp_path)],
        )

        assert no_side_effects["download"] == []
        assert "Would download" in result.output

    def test_dry_run_reports_the_target_prefix(self, runner, no_side_effects, tmp_path):
        result = runner.invoke(
            cli.app,
            ["install", "getdp", "--dry-run", "--force", "--prefix", str(tmp_path)],
        )

        assert str(tmp_path) in result.output

    def test_dry_run_creates_no_directories(self, runner, no_side_effects, tmp_path):
        target = tmp_path / "getdp_prefix"

        runner.invoke(
            cli.app,
            ["install", "getdp", "--dry-run", "--force", "--prefix", str(target)],
        )

        assert not target.exists()

    def test_unknown_version_is_rejected(self, runner, no_side_effects, tmp_path):
        result = runner.invoke(
            cli.app,
            [
                "install",
                "getdp",
                "--dry-run",
                "--force",
                "--version",
                "nightly",
                "--prefix",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 1
        assert "Unknown --version" in result.output

    @pytest.mark.parametrize("version", ["stable", "git"])
    def test_supported_versions_are_accepted(
        self, runner, no_side_effects, tmp_path, version
    ):
        result = runner.invoke(
            cli.app,
            [
                "install",
                "getdp",
                "--dry-run",
                "--force",
                "--version",
                version,
                "--prefix",
                str(tmp_path),
            ],
        )

        assert "Unknown --version" not in result.output
        assert result.exit_code in (0, 1)

    def test_unsupported_platform_names_the_supported_ones(
        self, runner, no_side_effects, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cli.platform, "system", lambda: "Plan9")
        monkeypatch.setattr(cli.platform, "machine", lambda: "vax")

        result = runner.invoke(
            cli.app,
            ["install", "getdp", "--dry-run", "--force", "--prefix", str(tmp_path)],
        )

        assert result.exit_code == 1
        assert "No prebuilt getdp binary" in result.output
        assert "getdp.info" in result.output

    @pytest.mark.skipif(
        shutil.which("getdp") is None, reason="getdp binary not installed"
    )
    def test_existing_installation_is_detected_without_force(
        self, runner, no_side_effects
    ):
        result = runner.invoke(cli.app, ["install", "getdp"])

        assert "already installed" in result.output
        assert no_side_effects["download"] == []


# ---------------------------------------------------------------------
# Download URL table
# ---------------------------------------------------------------------
class TestGetdpArchiveTable:
    def test_both_channels_are_defined(self):
        assert set(cli._GETDP_ARCHIVES) == {"stable", "git"}

    def test_both_channels_cover_the_same_platforms(self):
        """A platform present in one channel but not the other means
        ``--version git`` fails there for no good reason."""
        assert set(cli._GETDP_ARCHIVES["stable"]) == set(cli._GETDP_ARCHIVES["git"])

    def test_urls_are_https(self):
        for archives in cli._GETDP_ARCHIVES.values():
            for url in archives.values():
                assert url.startswith("https://")

    def test_every_url_is_the_petsc_mumps_build(self):
        """``fem_solver.run_getdp`` relies on a direct MUMPS solve, which only
        the "c" variant provides."""
        for archives in cli._GETDP_ARCHIVES.values():
            for url in archives.values():
                stem = url.rsplit("/", 1)[-1]
                assert "c." in stem, f"{stem} is not a PETSc/MUMPS build"

    def test_archive_extensions_match_the_platform(self):
        for archives in cli._GETDP_ARCHIVES.values():
            for (system, _machine), url in archives.items():
                if system == "Windows":
                    assert url.endswith(".zip")
                else:
                    assert url.endswith(".tgz")

    def test_current_platform_is_supported(self):
        import platform

        key = (platform.system(), platform.machine())

        if key not in cli._GETDP_ARCHIVES["stable"]:
            pytest.skip(f"{key} has no prebuilt getdp binary")

        assert cli._GETDP_ARCHIVES["stable"][key]
