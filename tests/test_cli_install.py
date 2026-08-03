"""Tests for the ``simpleems install`` machinery.

``test_cli.py`` drives the commands through ``--dry-run``, which proves the
argument handling and the inertness guarantee but never runs the code that
does the work. This file tests that code directly: archive extraction, binary
discovery, shell-profile editing, wheel selection and the subprocess wrapper.

These are the functions that touch a developer's actual machine, so every one
of them is exercised against a sandboxed ``HOME`` and stubbed network. The
``home`` fixture is not optional -- without it ``_add_dir_to_path`` appends to
the real ``~/.bashrc``.
"""

import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import typer

from simpleEMS import cli


def fake_release_response():
    """What the GitHub releases API returns, trimmed to the fields read."""
    payload = json.dumps(
        {
            "assets": [
                {
                    "name": "openEMS_x64_msvc.zip",
                    "browser_download_url": "https://example.invalid/openEMS.zip",
                }
            ]
        }
    ).encode()
    return io.BytesIO(payload)


def windows_zip_bytes(py_tag):
    """A release archive laid out the way the real Windows zips are."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("openEMS/AppCSXCAD.exe", "")
        zf.writestr(f"openEMS/python/CSXCAD-0.6.3-{py_tag}-win_amd64.whl", "")
        zf.writestr(f"openEMS/python/openEMS-0.0.36-{py_tag}-win_amd64.whl", "")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _restore_environment(monkeypatch):
    """Undo the environment edits the installers make.

    They assign to ``os.environ`` directly, which nothing rolls back on its
    own. Claiming the variables through monkeypatch first makes it restore
    them at teardown. Without this the fake ``getdp`` extracted below stays on
    ``PATH`` for the rest of the session, and every later test that runs a
    real solve picks it up and fails with an unhelpful missing-results error.
    """
    for name in ("PATH", "OPENEMS_INSTALL_PATH", "CSXCAD_INSTALL_PATH"):
        monkeypatch.setenv(name, os.environ.get(name, ""))


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point ``Path.home()`` at a throwaway directory.

    ``Path.home()`` resolves through ``$HOME`` on POSIX, so setting it here
    redirects every rc-file edit and cache write the installers perform.
    """
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    assert Path.home() == fake
    return fake


@pytest.fixture
def spy_subprocess(monkeypatch):
    """Record ``_run_subprocess`` calls instead of spawning anything."""
    calls = []
    monkeypatch.setattr(
        cli,
        "_run_subprocess",
        lambda cmd, cwd, dry_run: calls.append((cmd, cwd, dry_run)),
    )
    return calls


@pytest.fixture
def spy_path_edit(monkeypatch):
    """Record ``_add_dir_to_path`` calls instead of editing a profile."""
    calls = []
    monkeypatch.setattr(
        cli,
        "_add_dir_to_path",
        lambda bin_dir, dry_run, extra_env_vars=None: calls.append(
            (bin_dir, dry_run, extra_env_vars)
        ),
    )
    return calls


# ---------------------------------------------------------------------
# _find_extracted_binary
# ---------------------------------------------------------------------
class TestFindExtractedBinary:
    def make(self, root, relative):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        return path

    def test_finds_a_binary_anywhere_under_the_root(self, tmp_path):
        expected = self.make(tmp_path, "getdp-3.5.0-Linux64c/getdp")

        assert cli._find_extracted_binary(tmp_path) == expected

    def test_prefers_the_one_in_a_bin_directory(self, tmp_path):
        """Archives ship a launcher next to the docs and the real binary under
        bin/; picking the wrong one puts a non-executable on PATH."""
        self.make(tmp_path, "getdp-3.5.0/getdp")
        expected = self.make(tmp_path, "getdp-3.5.0/bin/getdp")

        assert cli._find_extracted_binary(tmp_path) == expected

    def test_falls_back_to_the_only_candidate(self, tmp_path):
        expected = self.make(tmp_path, "deep/nested/getdp")

        assert cli._find_extracted_binary(tmp_path) == expected

    def test_returns_none_when_there_is_nothing(self, tmp_path):
        (tmp_path / "docs").mkdir()

        assert cli._find_extracted_binary(tmp_path) is None

    def test_a_directory_named_getdp_is_not_a_binary(self, tmp_path):
        (tmp_path / "getdp").mkdir()

        assert cli._find_extracted_binary(tmp_path) is None

    def test_an_empty_tree_returns_none(self, tmp_path):
        assert cli._find_extracted_binary(tmp_path) is None


# ---------------------------------------------------------------------
# _add_dir_to_path
# ---------------------------------------------------------------------
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell-profile branch")
class TestAddDirToPath:
    def test_appends_the_export_line(self, home):
        rc = home / ".bashrc"
        rc.write_text("# existing config\n")

        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)

        assert 'export PATH="/opt/getdp/bin:$PATH"' in rc.read_text()

    def test_keeps_the_existing_contents(self, home):
        rc = home / ".bashrc"
        rc.write_text("# existing config\nalias ll='ls -l'\n")

        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)

        assert "alias ll='ls -l'" in rc.read_text()

    def test_writes_to_only_the_first_rc_file_found(self, home):
        """The rc files are tried in order and the first hit wins; writing to
        all of them would duplicate the entry for anyone sourcing both."""
        bashrc = home / ".bashrc"
        zshrc = home / ".zshrc"
        bashrc.write_text("")
        zshrc.write_text("")

        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)

        assert "export PATH" in bashrc.read_text()
        assert zshrc.read_text() == ""

    def test_uses_zshrc_when_there_is_no_bashrc(self, home):
        zshrc = home / ".zshrc"
        zshrc.write_text("")

        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)

        assert "export PATH" in zshrc.read_text()

    def test_a_repeat_install_does_not_duplicate_the_line(self, home):
        rc = home / ".bashrc"
        rc.write_text("")

        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)
        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)

        assert rc.read_text().count("export PATH") == 1

    def test_dry_run_writes_nothing(self, home):
        rc = home / ".bashrc"
        rc.write_text("# untouched\n")

        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=True)

        assert rc.read_text() == "# untouched\n"

    def test_no_rc_file_leaves_the_home_directory_alone(self, home, capsys):
        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)

        assert list(home.iterdir()) == []
        assert "manually" in capsys.readouterr().out

    def test_the_manual_instructions_name_the_directory(self, home, capsys):
        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)

        assert "/opt/getdp/bin" in capsys.readouterr().out

    def test_the_line_ends_with_a_newline(self, home):
        """A profile whose last line has no newline breaks the next append."""
        rc = home / ".bashrc"
        rc.write_text("# no trailing newline")

        cli._add_dir_to_path(Path("/opt/getdp/bin"), dry_run=False)

        assert rc.read_text().endswith("\n")


# ---------------------------------------------------------------------
# _run_subprocess
# ---------------------------------------------------------------------
class TestRunSubprocess:
    def test_dry_run_spawns_nothing(self, capsys):
        cli._run_subprocess(["false"], None, dry_run=True)

        assert "Would run" in capsys.readouterr().out

    def test_runs_the_command(self, tmp_path):
        marker = tmp_path / "ran"

        cli._run_subprocess(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
            None,
            dry_run=False,
        )

        assert marker.exists()

    def test_runs_in_the_given_directory(self, tmp_path, capsys):
        cli._run_subprocess(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            tmp_path,
            dry_run=False,
        )

        assert str(tmp_path) in capsys.readouterr().out

    def test_a_failing_command_raises(self):
        with pytest.raises(subprocess.CalledProcessError):
            cli._run_subprocess(
                [sys.executable, "-c", "raise SystemExit(3)"], None, False
            )

    def test_the_error_carries_the_return_code(self):
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            cli._run_subprocess(
                [sys.executable, "-c", "raise SystemExit(7)"], None, False
            )

        assert excinfo.value.returncode == 7

    def test_output_is_streamed(self, capsys):
        cli._run_subprocess([sys.executable, "-c", "print('hello build')"], None, False)

        assert "hello build" in capsys.readouterr().out


# ---------------------------------------------------------------------
# _pip_install
# ---------------------------------------------------------------------
class TestPipInstall:
    def test_prefers_uv_when_available(self, monkeypatch, spy_subprocess):
        monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv")

        cli._pip_install("/tmp/pkg", dry_run=False)

        cmd, _cwd, _dry = spy_subprocess[0]
        assert cmd[:3] == ["uv", "pip", "install"]

    def test_uv_installs_into_the_running_interpreter(
        self, monkeypatch, spy_subprocess
    ):
        """Without --python, uv would install into whatever venv it finds."""
        monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv")

        cli._pip_install("/tmp/pkg", dry_run=False)

        cmd, _cwd, _dry = spy_subprocess[0]
        assert cmd[cmd.index("--python") + 1] == sys.executable

    def test_falls_back_to_pip(self, monkeypatch, spy_subprocess):
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)

        cli._pip_install("/tmp/pkg", dry_run=False)

        cmd, _cwd, _dry = spy_subprocess[0]
        assert cmd[:4] == [sys.executable, "-m", "pip", "install"]

    def test_the_package_path_is_last(self, monkeypatch, spy_subprocess):
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)

        cli._pip_install(Path("/tmp/pkg"), dry_run=False)

        cmd, _cwd, _dry = spy_subprocess[0]
        assert cmd[-1] == "/tmp/pkg"

    def test_dry_run_is_passed_through(self, monkeypatch, spy_subprocess):
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)

        cli._pip_install("/tmp/pkg", dry_run=True)

        assert spy_subprocess[0][2] is True


# ---------------------------------------------------------------------
# _ensure_repo
# ---------------------------------------------------------------------
class TestEnsureRepo:
    def test_clones_when_missing(self, tmp_path, spy_subprocess):
        cli._ensure_repo(tmp_path / "cache", dry_run=False)

        cmd, _cwd, _dry = spy_subprocess[0]
        assert cmd[:2] == ["git", "clone"]

    def test_the_clone_is_shallow_and_recursive(self, tmp_path, spy_subprocess):
        """openEMS-Project is submodule-heavy; a full clone is slow and a
        non-recursive one does not build."""
        cli._ensure_repo(tmp_path / "cache", dry_run=False)

        cmd, _cwd, _dry = spy_subprocess[0]
        assert "--depth" in cmd
        assert "--recursive" in cmd

    def test_creates_the_cache_directory(self, tmp_path, spy_subprocess):
        cli._ensure_repo(tmp_path / "cache", dry_run=False)

        assert (tmp_path / "cache").is_dir()

    def test_pulls_when_already_cloned(self, tmp_path, spy_subprocess):
        cache = tmp_path / "cache"
        (cache / "openEMS-Project").mkdir(parents=True)

        cli._ensure_repo(cache, dry_run=False)

        cmd, cwd, _dry = spy_subprocess[0]
        assert cmd[:2] == ["git", "pull"]
        assert cwd == cache / "openEMS-Project"

    def test_returns_the_repository_path(self, tmp_path, spy_subprocess):
        result = cli._ensure_repo(tmp_path / "cache", dry_run=False)

        assert result == tmp_path / "cache" / "openEMS-Project"


# ---------------------------------------------------------------------
# _download_with_progress
# ---------------------------------------------------------------------
class TestDownload:
    def fake_response(self, payload, length=True):
        import io

        class Response(io.BytesIO):
            headers = {"Content-Length": str(len(payload))} if length else {}

        return Response(payload)

    def test_writes_the_payload(self, tmp_path, monkeypatch):
        payload = b"archive-bytes" * 1000
        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda url: self.fake_response(payload)
        )
        dest = tmp_path / "out.tgz"

        cli._download_with_progress("https://example.invalid/a.tgz", dest)

        assert dest.read_bytes() == payload

    def test_works_without_a_content_length(self, tmp_path, monkeypatch):
        """Some mirrors omit it; the progress bar just has no total."""
        payload = b"x" * 100
        monkeypatch.setattr(
            cli.urllib.request,
            "urlopen",
            lambda url: self.fake_response(payload, length=False),
        )
        dest = tmp_path / "out.tgz"

        cli._download_with_progress("https://example.invalid/a.tgz", dest)

        assert dest.read_bytes() == payload

    def test_handles_a_payload_larger_than_one_chunk(self, tmp_path, monkeypatch):
        payload = os.urandom(8192 * 3 + 17)
        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda url: self.fake_response(payload)
        )
        dest = tmp_path / "out.tgz"

        cli._download_with_progress("https://example.invalid/a.tgz", dest)

        assert dest.read_bytes() == payload


# ---------------------------------------------------------------------
# _install_getdp_archive
# ---------------------------------------------------------------------
class TestInstallGetdpArchive:
    @pytest.fixture
    def tarball(self, tmp_path):
        """A .tgz laid out the way the real getdp archives are."""
        staging = tmp_path / "staging" / "getdp-3.5.0-Linux64c" / "bin"
        staging.mkdir(parents=True)
        (staging / "getdp").write_text("#!/bin/sh\necho getdp\n")
        archive = tmp_path / "getdp.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(tmp_path / "staging" / "getdp-3.5.0-Linux64c", arcname="getdp-3.5.0")
        return archive

    @pytest.fixture
    def offline(self, monkeypatch, tarball):
        """Serve the local tarball instead of downloading one."""

        def fake_download(url, dest):
            dest.write_bytes(tarball.read_bytes())

        monkeypatch.setattr(cli, "_download_with_progress", fake_download)

    def test_unknown_version_exits_with_an_error(self, tmp_path):
        with pytest.raises(typer.Exit) as excinfo:
            cli._install_getdp_archive(tmp_path, "nightly", dry_run=True)

        assert excinfo.value.exit_code == 1

    def test_the_unknown_version_message_names_the_valid_ones(self, tmp_path, capsys):
        with pytest.raises(typer.Exit):
            cli._install_getdp_archive(tmp_path, "nightly", dry_run=True)

        out = capsys.readouterr().out
        assert "stable" in out and "git" in out

    def test_an_unsupported_platform_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Haiku")
        monkeypatch.setattr(cli.platform, "machine", lambda: "m68k")

        with pytest.raises(typer.Exit) as excinfo:
            cli._install_getdp_archive(tmp_path, "stable", dry_run=True)

        assert excinfo.value.exit_code == 1

    def test_the_unsupported_platform_message_lists_what_works(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(cli.platform, "system", lambda: "Haiku")
        monkeypatch.setattr(cli.platform, "machine", lambda: "m68k")

        with pytest.raises(typer.Exit):
            cli._install_getdp_archive(tmp_path, "stable", dry_run=True)

        assert "Linux/x86_64" in capsys.readouterr().out

    def test_dry_run_downloads_and_extracts_nothing(self, tmp_path, monkeypatch, home):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(
            cli,
            "_download_with_progress",
            lambda url, dest: pytest.fail(f"downloaded {url}"),
        )
        prefix = tmp_path / "prefix"

        assert cli._install_getdp_archive(prefix, "stable", dry_run=True) is None
        assert not prefix.exists()

    def test_extracts_and_finds_the_binary(
        self, tmp_path, monkeypatch, home, offline, spy_path_edit
    ):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
        prefix = tmp_path / "prefix"

        binary = cli._install_getdp_archive(prefix, "stable", dry_run=False)

        assert binary is not None
        assert binary.is_file()
        assert binary.name == "getdp"

    def test_the_binary_is_made_executable(
        self, tmp_path, monkeypatch, home, offline, spy_path_edit
    ):
        """Archive members do not always carry the execute bit, and a
        non-executable getdp fails much later with a confusing error."""
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")

        binary = cli._install_getdp_archive(tmp_path / "prefix", "stable", False)

        assert os.access(binary, os.X_OK)

    def test_the_binary_directory_is_added_to_the_shell_profile(
        self, tmp_path, monkeypatch, home, offline, spy_path_edit
    ):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")

        binary = cli._install_getdp_archive(tmp_path / "prefix", "stable", False)

        assert spy_path_edit[0][0] == binary.parent

    def test_the_current_process_path_is_updated(
        self, tmp_path, monkeypatch, home, offline, spy_path_edit
    ):
        """So a checkhealth run straight after the install finds the binary
        without the user opening a new shell."""
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
        monkeypatch.setenv("PATH", "/usr/bin")

        binary = cli._install_getdp_archive(tmp_path / "prefix", "stable", False)

        assert str(binary.parent) in os.environ["PATH"].split(os.pathsep)

    def test_the_path_is_not_duplicated(
        self, tmp_path, monkeypatch, home, offline, spy_path_edit
    ):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")

        binary = cli._install_getdp_archive(tmp_path / "prefix", "stable", False)
        monkeypatch.setenv("PATH", str(binary.parent))
        cli._install_getdp_archive(tmp_path / "prefix2", "stable", False)

        assert os.environ["PATH"].split(os.pathsep).count(str(binary.parent)) == 1

    def test_an_archive_without_a_binary_exits(
        self, tmp_path, monkeypatch, home, spy_path_edit
    ):
        empty = tmp_path / "empty.tgz"
        docs = tmp_path / "staging" / "docs"
        docs.mkdir(parents=True)
        (docs / "README").write_text("no binary here")
        with tarfile.open(empty, "w:gz") as tf:
            tf.add(docs, arcname="docs")

        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(
            cli,
            "_download_with_progress",
            lambda url, dest: dest.write_bytes(empty.read_bytes()),
        )

        with pytest.raises(typer.Exit) as excinfo:
            cli._install_getdp_archive(tmp_path / "prefix", "stable", False)

        assert excinfo.value.exit_code == 1

    def test_a_zip_archive_is_extracted(
        self, tmp_path, monkeypatch, home, spy_path_edit
    ):
        """The Windows builds ship as .zip rather than .tgz."""
        archive = tmp_path / "getdp.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("getdp-3.5.0/bin/getdp", "#!/bin/sh\n")

        monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
        monkeypatch.setattr(cli.platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(
            cli,
            "_download_with_progress",
            lambda url, dest: dest.write_bytes(archive.read_bytes()),
        )
        # _find_extracted_binary looks for getdp.exe only on win32
        monkeypatch.setattr(cli.sys, "platform", "linux")

        binary = cli._install_getdp_archive(tmp_path / "prefix", "stable", False)

        assert binary.is_file()

    def test_the_git_channel_is_usable(
        self, tmp_path, monkeypatch, home, offline, spy_path_edit
    ):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")

        assert cli._install_getdp_archive(tmp_path / "prefix", "git", False) is not None


# ---------------------------------------------------------------------
# _install_unix
# ---------------------------------------------------------------------
class TestInstallUnix:
    def test_clones_builds_and_edits_the_path(
        self, tmp_path, home, spy_subprocess, spy_path_edit
    ):
        cli._install_unix(tmp_path / "prefix", dry_run=False)

        commands = [cmd[0] for cmd, _cwd, _dry in spy_subprocess]
        assert commands == ["git", "sudo", "./update_openEMS.sh"]
        assert spy_path_edit[0][0] == tmp_path / "prefix" / "bin"

    def test_builds_into_the_requested_prefix(
        self, tmp_path, home, spy_subprocess, spy_path_edit
    ):
        prefix = tmp_path / "prefix"

        cli._install_unix(prefix, dry_run=False)

        build_cmd = spy_subprocess[-1][0]
        assert str(prefix) in build_cmd

    def test_the_build_includes_the_python_bindings(
        self, tmp_path, home, spy_subprocess, spy_path_edit
    ):
        """Without --python the solver installs but the package cannot import
        CSXCAD, which is the failure the health check reports."""
        cli._install_unix(tmp_path / "prefix", dry_run=False)

        assert "--python" in spy_subprocess[-1][0]

    def test_creates_the_prefix(self, tmp_path, home, spy_subprocess, spy_path_edit):
        prefix = tmp_path / "prefix"

        cli._install_unix(prefix, dry_run=False)

        assert prefix.is_dir()

    def test_dry_run_is_threaded_through(
        self, tmp_path, home, spy_subprocess, spy_path_edit
    ):
        cli._install_unix(tmp_path / "prefix", dry_run=True)

        assert all(dry for _cmd, _cwd, dry in spy_subprocess)
        assert spy_path_edit[0][1] is True


# ---------------------------------------------------------------------
# _install_python_bindings_only
# ---------------------------------------------------------------------
class TestInstallBindingsOnly:
    def test_installs_both_bindings(self, tmp_path, home, monkeypatch, spy_subprocess):
        installed = []
        monkeypatch.setattr(
            cli,
            "_pip_install",
            lambda path, dry, force=False: installed.append(Path(path).parts[-2]),
        )

        cli._install_python_bindings_only(tmp_path / "prefix", dry_run=False)

        assert installed == ["CSXCAD", "openEMS"]

    def test_dry_run_installs_nothing(
        self, tmp_path, home, monkeypatch, spy_subprocess
    ):
        monkeypatch.setattr(
            cli,
            "_pip_install",
            lambda path, dry, force=False: pytest.fail("pip ran in dry-run"),
        )

        cli._install_python_bindings_only(tmp_path / "prefix", dry_run=True)

    def test_dry_run_reports_what_it_would_install(
        self, tmp_path, home, spy_subprocess, capsys
    ):
        cli._install_python_bindings_only(tmp_path / "prefix", dry_run=True)

        out = capsys.readouterr().out
        assert "Would install" in out


# ---------------------------------------------------------------------
# _install_windows wheel selection
# ---------------------------------------------------------------------
class TestWindowsWheelSelection:
    """The wheel-matching logic is plain Python and worth testing off Windows.

    Only the registry edit in ``_add_dir_to_path`` is Windows-specific, and it
    is stubbed here, so the interesting part -- picking a wheel that matches
    the running interpreter -- runs anywhere.
    """

    def prepared(self, tmp_path, wheel_tags):
        python_dir = tmp_path / "prefix" / "openEMS" / "python"
        python_dir.mkdir(parents=True)
        (tmp_path / "prefix" / "openEMS" / "AppCSXCAD.exe").write_text("")
        for tag in wheel_tags:
            (python_dir / f"CSXCAD-0.6.3-{tag}-win_amd64.whl").write_text("")
            (python_dir / f"openEMS-0.0.36-{tag}-win_amd64.whl").write_text("")
        return tmp_path / "prefix"

    def test_picks_the_wheel_for_the_running_python(
        self, tmp_path, home, monkeypatch, spy_path_edit
    ):
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        prefix = self.prepared(tmp_path, [tag, "cp39"])
        installed = []
        monkeypatch.setattr(
            cli, "_pip_install", lambda path, dry, force=False: installed.append(path)
        )

        cli._install_windows(prefix, "latest", dry_run=False)

        assert all(tag in name for name in installed)

    def test_no_matching_wheel_exits_with_the_available_tags(
        self, tmp_path, home, monkeypatch, spy_path_edit, capsys
    ):
        prefix = self.prepared(tmp_path, ["cp39"])
        monkeypatch.setattr(cli, "_pip_install", lambda path, dry, force=False: None)

        with pytest.raises(typer.Exit) as excinfo:
            cli._install_windows(prefix, "latest", dry_run=False)

        assert excinfo.value.exit_code == 1
        assert "cp39" in capsys.readouterr().out

    def test_missing_wheels_exit(self, tmp_path, home, monkeypatch, spy_path_edit):
        prefix = self.prepared(tmp_path, [])

        with pytest.raises(typer.Exit) as excinfo:
            cli._install_windows(prefix, "latest", dry_run=False)

        assert excinfo.value.exit_code == 1

    def test_an_existing_install_is_not_re_downloaded(
        self, tmp_path, home, monkeypatch, spy_path_edit, capsys
    ):
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        prefix = self.prepared(tmp_path, [tag])
        monkeypatch.setattr(cli, "_pip_install", lambda path, dry, force=False: None)
        monkeypatch.setattr(
            cli,
            "_download_with_progress",
            lambda url, dest: pytest.fail("re-downloaded an existing install"),
        )

        cli._install_windows(prefix, "latest", dry_run=False)

        assert "already present" in capsys.readouterr().out

    def test_force_replaces_an_existing_install(
        self, tmp_path, home, monkeypatch, spy_path_edit
    ):
        """--force has to clear the tree first; extraction only merges into it."""
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        prefix = self.prepared(tmp_path, [tag])
        stale = prefix / "openEMS" / "stale-from-an-old-release.dll"
        stale.write_text("")
        monkeypatch.setattr(cli, "_pip_install", lambda path, dry, force=False: None)
        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda req: fake_release_response()
        )
        monkeypatch.setattr(
            cli,
            "_download_with_progress",
            lambda url, dest: dest.write_bytes(windows_zip_bytes(tag)),
        )

        cli._install_windows(prefix, "latest", dry_run=False, force=True)

        assert not stale.exists()
        assert (prefix / "openEMS" / "AppCSXCAD.exe").exists()

    def test_force_reinstalls_the_python_bindings(
        self, tmp_path, home, monkeypatch, spy_path_edit
    ):
        """Same wheel version as the installed one -- pip skips it otherwise."""
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        prefix = self.prepared(tmp_path, [tag])
        forced = []
        monkeypatch.setattr(
            cli, "_pip_install", lambda path, dry, force=False: forced.append(force)
        )
        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda req: fake_release_response()
        )
        monkeypatch.setattr(
            cli,
            "_download_with_progress",
            lambda url, dest: dest.write_bytes(windows_zip_bytes(tag)),
        )

        cli._install_windows(prefix, "latest", dry_run=False, force=True)

        assert forced == [True, True]

    def test_force_keeps_the_install_in_a_dry_run(
        self, tmp_path, home, monkeypatch, spy_path_edit, capsys
    ):
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        prefix = self.prepared(tmp_path, [tag])
        monkeypatch.setattr(cli, "_pip_install", lambda path, dry, force=False: None)
        monkeypatch.setattr(
            cli,
            "_download_with_progress",
            lambda url, dest: pytest.fail("downloaded during a dry run"),
        )

        cli._install_windows(prefix, "latest", dry_run=True, force=True)

        assert (prefix / "openEMS" / "AppCSXCAD.exe").exists()
        assert "Would delete" in capsys.readouterr().out

    def test_force_leaves_the_prefix_itself_alone(
        self, tmp_path, home, monkeypatch, spy_path_edit
    ):
        """--prefix may be a shared directory; only the openEMS tree is ours."""
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        prefix = self.prepared(tmp_path, [tag])
        neighbour = prefix / "unrelated.txt"
        neighbour.write_text("not ours to delete")
        monkeypatch.setattr(cli, "_pip_install", lambda path, dry, force=False: None)
        monkeypatch.setattr(
            cli.urllib.request, "urlopen", lambda req: fake_release_response()
        )
        monkeypatch.setattr(
            cli,
            "_download_with_progress",
            lambda url, dest: dest.write_bytes(windows_zip_bytes(tag)),
        )

        cli._install_windows(prefix, "latest", dry_run=False, force=True)

        assert neighbour.read_text() == "not ours to delete"

    def test_the_install_directory_is_exported(
        self, tmp_path, home, monkeypatch, spy_path_edit
    ):
        """openEMS and CSXCAD locate their data through these variables."""
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        prefix = self.prepared(tmp_path, [tag])
        monkeypatch.setattr(cli, "_pip_install", lambda path, dry, force=False: None)
        monkeypatch.setenv("OPENEMS_INSTALL_PATH", "")
        monkeypatch.setenv("CSXCAD_INSTALL_PATH", "")

        cli._install_windows(prefix, "latest", dry_run=False)

        assert os.environ["OPENEMS_INSTALL_PATH"] == str(prefix / "openEMS")
        assert os.environ["CSXCAD_INSTALL_PATH"] == str(prefix / "openEMS")


# ---------------------------------------------------------------------
# summary and verification
# ---------------------------------------------------------------------
class TestSummaryAndVerification:
    def test_success_exits_zero(self, capsys):
        with pytest.raises(typer.Exit) as excinfo:
            cli._print_summary(True, Path("/opt/openEMS"))

        assert excinfo.value.exit_code == 0
        assert "successfully" in capsys.readouterr().out

    def test_failure_exits_one(self, capsys):
        with pytest.raises(typer.Exit) as excinfo:
            cli._print_summary(False, Path("/opt/openEMS"))

        assert excinfo.value.exit_code == 1
        assert "issues" in capsys.readouterr().out

    def test_the_label_is_used(self, capsys):
        with pytest.raises(typer.Exit):
            cli._print_summary(True, Path("/opt/getdp"), label="getdp")

        assert "getdp" in capsys.readouterr().out

    def test_verification_passes_when_everything_is_present(self, monkeypatch):
        monkeypatch.setattr(
            cli, "_check_package_import", lambda imp, pip: (True, "1.0", None)
        )
        monkeypatch.setattr(
            cli, "_check_system_binary", lambda name: (True, f"/usr/bin/{name}", None)
        )

        assert cli._verify_installation() is True

    def test_verification_fails_on_a_missing_binary(self, monkeypatch):
        monkeypatch.setattr(
            cli, "_check_package_import", lambda imp, pip: (True, "1.0", None)
        )
        monkeypatch.setattr(
            cli, "_check_system_binary", lambda name: (False, None, "not found")
        )

        assert cli._verify_installation() is False

    def test_verification_fails_on_a_missing_import(self, monkeypatch):
        monkeypatch.setattr(
            cli, "_check_package_import", lambda imp, pip: (False, None, "no module")
        )
        monkeypatch.setattr(
            cli, "_check_system_binary", lambda name: (True, "/usr/bin/x", None)
        )

        assert cli._verify_installation() is False

    def test_verification_checks_both_apis_and_both_binaries(self, monkeypatch):
        checked = []
        monkeypatch.setattr(
            cli,
            "_check_package_import",
            lambda imp, pip: (checked.append(imp), (True, "1.0", None))[1],
        )
        monkeypatch.setattr(
            cli,
            "_check_system_binary",
            lambda name: (checked.append(name), (True, "/usr/bin/x", None))[1],
        )

        cli._verify_installation()

        assert checked == ["openEMS", "CSXCAD", "AppCSXCAD", "openEMS"]
