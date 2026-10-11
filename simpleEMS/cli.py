"""simpleEMS CLI tool."""

import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

app = typer.Typer(
    name="simpleems",
    help="simpleEMS - Antenna and RF structure design, simulation and visualization",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """simpleEMS - Antenna and RF structure design, simulation and visualization"""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


def _get_dependency_checks_from_pyproject_section(section):
    try:
        requires = importlib.metadata.requires("simpleEMS")
    except importlib.metadata.PackageNotFoundError:
        return []
    if requires is None:
        return []

    distribution_to_import = {}
    try:
        for (
            import_name,
            distribution_names,
        ) in importlib.metadata.packages_distributions().items():
            if not import_name.isidentifier():
                continue
            for distribution_name in distribution_names:
                distribution_to_import.setdefault(
                    distribution_name.lower(), import_name
                )
    except Exception:
        pass

    dependency_checks = []
    for requirement in requires:
        parts = requirement.split(";")
        dep = parts[0].strip()
        marker = parts[1].strip() if len(parts) > 1 else ""

        if section == "dependencies":
            if "extra ==" in marker:
                continue
        else:
            if 'extra == "dev"' not in marker and "extra == 'dev'" not in marker:
                continue

        for separator in (">", "<", "=", "!", "~"):
            dep = dep.split(separator)[0]
        pip_name = dep.strip()
        import_name = distribution_to_import.get(pip_name.lower(), pip_name)
        dependency_checks.append((import_name, pip_name))

    return dependency_checks


def _check_package_import(import_name, pip_name):
    try:
        importlib.import_module(import_name)
        try:
            version = importlib.metadata.version(pip_name)
        except importlib.metadata.PackageNotFoundError:
            try:
                version = importlib.metadata.version(import_name)
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
        return True, version, None
    except Exception as error:
        # Not just ImportError: a compiled extension that is installed but
        # cannot dlopen its shared library raises OSError, and reporting that
        # as "not installed" is more useful than a traceback out of the CLI.
        # Drop only a trailing "(unknown location)"-style suffix; cutting at the
        # first parenthesis would also cut the useful half of a dlopen error.
        return False, None, re.sub(r"\s*\([^()]*\)$", "", str(error)).strip()


def _check_system_binary(binary_name):
    binary_path = shutil.which(binary_name)
    if binary_path is not None:
        return True, binary_path, None
    return False, None, "not found in PATH"


def _default_prefix():
    if sys.platform == "win32":
        return Path("C:\\openEMS")
    return Path.home() / "opt" / "openEMS"


def _default_getdp_prefix():
    if sys.platform == "win32":
        return Path("C:\\getdp")
    return Path.home() / "opt" / "getdp"


# Pinned per-OS/arch download URLs -- getdp.info is a plain Apache directory
# listing (no releases API like GitHub's), so "latest" can't be resolved at
# runtime; pin a known-good "stable" build per platform and offer the rolling
# "git" dev build as the alternative. The "c" variant is the PETSc+MUMPS build,
# which the FEM solve relies on (direct MUMPS solve in fem_solver.run_getdp).
_GETDP_ARCHIVES = {
    "stable": {
        ("Linux", "x86_64"): "https://getdp.info/bin/Linux/getdp-3.5.0-Linux64c.tgz",
        ("Darwin", "x86_64"): "https://getdp.info/bin/macOS/getdp-3.5.0-MacOSXc.tgz",
        ("Darwin", "arm64"): "https://getdp.info/bin/macOS/getdp-4.0.0-MacOSARMc.tgz",
        (
            "Windows",
            "AMD64",
        ): "https://getdp.info/bin/Windows/getdp-3.5.0-Windows64c.zip",
    },
    "git": {
        ("Linux", "x86_64"): "https://getdp.info/bin/Linux/getdp-git-Linux64c.tgz",
        ("Darwin", "x86_64"): "https://getdp.info/bin/macOS/getdp-git-MacOSXc.tgz",
        ("Darwin", "arm64"): "https://getdp.info/bin/macOS/getdp-git-MacOSARMc.tgz",
        ("Windows", "AMD64"): "https://getdp.info/bin/Windows/getdp-git-Windows64c.zip",
    },
}


def _ensure_repo(cache_dir, dry_run, update=True):
    """Clone openEMS-Project into ``cache_dir``, or pull it when ``update``.

    Pass ``update=False`` to build against the sources already in the cache --
    the ones an earlier ``simpleems install openems`` built the C++ side from.
    """
    repo_dir = cache_dir / "openEMS-Project"
    if repo_dir.exists():
        if update:
            _run_subprocess(["git", "pull", "--recurse-submodules"], repo_dir, dry_run)
    else:
        if not dry_run:
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_subprocess(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--recursive",
                "https://github.com/thliebig/openEMS-Project.git",
                str(repo_dir),
            ],
            None,
            dry_run,
        )
    return repo_dir


def _run_subprocess(cmd, cwd, dry_run):
    if dry_run:
        console.print(f"  [yellow]Would run:[/] {' '.join(cmd)}")
        return
    kwargs = {}
    if cwd is not None:
        kwargs["cwd"] = cwd
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs
    )
    for line in process.stdout:
        console.print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def _add_dir_to_path(bin_dir, dry_run, extra_env_vars=None):
    # The persistent edit below -- rc file or registry -- only reaches future
    # shells. Put the directory on this process's PATH as well, so the health
    # check that runs straight after an install sees what was just installed.
    if not dry_run:
        _prepend_to_process_path(bin_dir)

    if sys.platform == "win32":
        import ctypes
        import winreg

        path_value = str(bin_dir)

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS
        )

        try:
            user_path, reg_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            user_path = ""
            reg_type = winreg.REG_EXPAND_SZ

        dirs = [d.strip() for d in user_path.split(";")] if user_path else []
        if path_value not in dirs:
            new_path = f"{user_path};{path_value}" if user_path else path_value
            if not dry_run:
                winreg.SetValueEx(key, "Path", 0, reg_type, new_path)
            console.print(f"  [green]Added {path_value} to user PATH[/]")
        else:
            console.print(f"  [green]{path_value} already in user PATH[/]")

        for var, value in (extra_env_vars or {}).items():
            if not dry_run:
                winreg.SetValueEx(key, var, 0, winreg.REG_SZ, value)
            console.print(f"  [green]Set {var} to {value}[/]")

        winreg.CloseKey(key)

        if not dry_run:
            HWND_BROADCAST = 0xFFFF
            WM_SETTINGCHANGE = 0x001A
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment", 0, 5000, None
            )
        return

    line = f'export PATH="{bin_dir}:$PATH"'
    for rc in _shell_rc_files():
        if rc.exists():
            content = rc.read_text()
            if line in content:
                console.print(f"  [green]Already in PATH in {rc}[/]")
            else:
                if dry_run:
                    console.print(f"  [yellow]Would add to {rc}:[/] {line}")
                else:
                    rc.write_text(content.rstrip() + "\n" + line + "\n")
                    console.print(f"  [green]Added to {rc}[/]")
            return
    console.print("  [yellow]No shell rc file found. Add to your PATH manually:[/]")
    console.print(f'    export PATH="{bin_dir}:$PATH"')


def _shell_rc_files():
    """The rc files to try, in order, for the user's login shell.

    Only the first one that exists is edited: ``.bash_profile`` commonly
    sources ``.bashrc``, so writing to both would duplicate the entry. The
    candidates follow ``$SHELL`` -- a zsh user with a leftover ``.bashrc``
    would otherwise get the line in a file zsh never reads.
    """
    home = Path.home()
    shell_name = Path(os.environ.get("SHELL", "")).name
    if shell_name == "zsh":
        return [home / ".zshrc"]
    if shell_name == "bash":
        return [home / ".bashrc", home / ".bash_profile"]
    if shell_name == "fish":
        # fish reads none of these and has no `export`; leave it to the user.
        return []
    return [home / ".bashrc", home / ".zshrc", home / ".bash_profile"]


def _prepend_to_process_path(bin_dir):
    """Put ``bin_dir`` on this process's PATH if it is not already there."""
    bin_dir = str(bin_dir)
    current = os.environ.get("PATH", "")
    if bin_dir not in current.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + current if current else bin_dir


def _in_virtualenv():
    """True when the running interpreter is inside a venv/virtualenv."""
    return sys.prefix != sys.base_prefix


def _externally_managed():
    """True when the running interpreter is a PEP 668 system Python.

    pip and uv both refuse to install into such an interpreter (Arch, Debian
    12+, Fedora 38+ ...), so the openEMS bindings cannot go there.
    """
    if _in_virtualenv():
        return False
    stdlib = sysconfig.get_path("stdlib")
    return stdlib is not None and (Path(stdlib) / "EXTERNALLY-MANAGED").exists()


_NETWORK_TIMEOUT = 60  # seconds without data before a request gives up


def _download_with_progress(url, dest):
    # Download next to the destination and rename at the end, so a failed or
    # interrupted download never leaves a truncated archive under the real name.
    dest = Path(dest)
    partial = dest.with_name(dest.name + ".part")
    response = urllib.request.urlopen(url, timeout=_NETWORK_TIMEOUT)
    total = int(response.headers.get("Content-Length", 0))
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=total)
            with open(partial, "wb") as f:
                for chunk in iter(lambda: response.read(8192), b""):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def _pip_install(package_path, dry_run, force: bool = False):
    if shutil.which("uv") is not None:
        cmd = ["uv", "pip", "install", "--python", sys.executable]
    else:
        # A venv made by `uv venv` has no pip; bootstrap it rather than fail.
        if importlib.util.find_spec("pip") is None:
            _run_subprocess([sys.executable, "-m", "ensurepip"], None, dry_run)
        cmd = [sys.executable, "-m", "pip", "install"]
    if force:
        cmd.append("--force-reinstall")
    cmd.append(str(package_path))
    _run_subprocess(cmd, None, dry_run)


def _install_windows(prefix, version, dry_run, force: bool = False):
    cache_dir = Path.home() / ".cache" / "simpleems"
    if not dry_run:
        cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(Panel.fit("[bold]Installing openEMS...[/]", border_style="cyan"))
    console.print()

    install_dir = prefix / "openEMS"
    existing = (install_dir / "AppCSXCAD.exe").exists()

    if existing and force:
        # Extraction merges into the target, so files dropped between releases
        # would survive a reinstall. Clear the tree the archive owns -- never
        # the user's --prefix, which may hold unrelated things.
        console.print(f"Removing the existing install at {install_dir}...")
        if dry_run:
            console.print(f"  [yellow]Would delete:[/] {install_dir}")
        else:
            shutil.rmtree(install_dir)
        console.print("  ✅ Removed")
        console.print()
        existing = False

    if not existing:
        console.print("Querying GitHub releases...")
        api_url = "https://api.github.com/repos/thliebig/openEMS-Project/releases"
        if version == "latest":
            api_url = api_url + "/latest"
        else:
            api_url = api_url + f"/tags/{version}"
        if dry_run:
            console.print(f"  [yellow]Would fetch:[/] {api_url}")
        else:
            try:
                req = urllib.request.Request(
                    api_url, headers={"Accept": "application/vnd.github.v3+json"}
                )
                response = urllib.request.urlopen(req, timeout=_NETWORK_TIMEOUT)
                release_data = json.loads(response.read())
            except (urllib.error.URLError, TimeoutError) as e:
                console.print(f"  ❌ Failed to fetch release data: {e}")
                raise typer.Exit(code=1)
            assets = release_data.get("assets", [])
            asset = None
            for a in assets:
                name = a["name"]
                if name.endswith(".zip") and "msvc" in name:
                    asset = a
                    break
            if asset is None:
                for a in assets:
                    if a["name"].endswith(".zip"):
                        asset = a
                        break
            if asset is None:
                console.print("  ❌ No suitable release asset found")
                raise typer.Exit(code=1)
            download_url = asset["browser_download_url"]
            zip_path = cache_dir / "openEMS_win.zip"
            console.print(f"  Downloading {asset['name']}...")
            _download_with_progress(download_url, zip_path)
            console.print("  ✅ Download complete")
            console.print()
            console.print("Extracting...")
            prefix.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(prefix)
            console.print("  ✅ Extraction complete")
            console.print()
    else:
        console.print(f"  ✅ openEMS already present at {install_dir}")
        console.print()

    python_dir = install_dir / "python"
    if python_dir.exists():
        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

        csxcad_wheels = sorted(python_dir.glob("CSXCAD-*.whl"))
        openems_wheels = sorted(python_dir.glob("openEMS-*.whl"))

        if not csxcad_wheels or not openems_wheels:
            console.print("  ❌ Could not find required .whl files in python directory")
            raise typer.Exit(code=1)

        matching_csxcad = [w for w in csxcad_wheels if py_tag in w.name]
        matching_openems = [w for w in openems_wheels if py_tag in w.name]

        if not matching_csxcad or not matching_openems:
            available_tags = set()
            for w in csxcad_wheels + openems_wheels:
                m = re.search(r"-cp\d+-", w.name)
                if m:
                    available_tags.add(m.group(0).strip("-"))
            console.print(
                f"  ❌ No wheel for Python {sys.version_info.major}.{sys.version_info.minor}"
            )
            console.print(f"  Available wheels: {', '.join(sorted(available_tags))}")
            console.print("  Install a compatible Python version or build from source.")
            raise typer.Exit(code=1)

        console.print("Installing Python bindings...")
        _pip_install(str(matching_csxcad[0]), dry_run, force=force)
        _pip_install(str(matching_openems[0]), dry_run, force=force)
        console.print("  ✅ Python bindings installed")
        console.print()
    _add_dir_to_path(
        install_dir,
        dry_run,
        extra_env_vars={
            "OPENEMS_INSTALL_PATH": str(install_dir),
            "CSXCAD_INSTALL_PATH": str(install_dir),
        },
    )

    # _add_dir_to_path already put install_dir on this process's PATH.
    if not dry_run:
        os.environ["OPENEMS_INSTALL_PATH"] = str(install_dir)
        os.environ["CSXCAD_INSTALL_PATH"] = str(install_dir)


def _install_unix(prefix, dry_run, force: bool = False):
    cache_dir = Path.home() / ".cache" / "simpleems"
    console.print(Panel.fit("[bold]Installing openEMS...[/]", border_style="cyan"))
    console.print()
    console.print("Cloning openEMS-Project...")
    repo_dir = _ensure_repo(cache_dir, dry_run)
    console.print("  ✅ Repository ready")
    console.print()
    console.print("Installing system dependencies (requires sudo)...")
    # --python still matters: it pulls in the Python headers (python3-dev on
    # Debian) that the extension build compiles against.
    _run_subprocess(
        ["sudo", "./scripts/install_deps.sh", "--auto", "--python"], repo_dir, dry_run
    )
    console.print("  ✅ Dependencies installed")
    console.print()
    console.print("Building openEMS (this may take 10-30 minutes)...")
    if not dry_run:
        prefix.mkdir(parents=True, exist_ok=True)
    # Build only the C++ side. openEMS's own build_python.sh installs the
    # bindings with whichever python3/pip3 comes first on PATH, not the
    # interpreter running simpleEMS: on Arch that is the externally managed
    # system Python, and pip refuses (PEP 668); elsewhere it quietly lands in a
    # venv of its own that simpleEMS never sees.
    _run_subprocess(["./update_openEMS.sh", str(prefix)], repo_dir, dry_run)
    console.print("  ✅ Build complete")
    console.print()
    _add_dir_to_path(prefix / "bin", dry_run)
    console.print()
    _install_python_bindings(prefix, repo_dir, dry_run, force=force)


def _install_python_bindings(prefix, repo_dir, dry_run, force: bool = False):
    """Build the CSXCAD and openEMS bindings into the running interpreter."""
    csxcad_path = repo_dir / "CSXCAD" / "python"
    openems_path = repo_dir / "openEMS" / "python"
    # Both setup.py files locate the headers and libraries of the existing C++
    # install through these variables; without them the extension build guesses
    # from VIRTUAL_ENV and fails when the C++ side lives somewhere else.
    os.environ["CSXCAD_INSTALL_PATH"] = str(prefix)
    os.environ["OPENEMS_INSTALL_PATH"] = str(prefix)
    if dry_run:
        console.print(f"  [yellow]Would install:[/] {csxcad_path}")
        console.print(f"  [yellow]Would install:[/] {openems_path}")
    else:
        console.print("Installing CSXCAD Python bindings...")
        _pip_install(csxcad_path, dry_run, force=force)
        console.print("  ✅ CSXCAD Python bindings installed")
        console.print()
        console.print("Installing openEMS Python bindings...")
        _pip_install(openems_path, dry_run, force=force)
        console.print("  ✅ openEMS Python bindings installed")
    console.print()


def _install_python_bindings_only(prefix, dry_run):
    cache_dir = Path.home() / ".cache" / "simpleems"
    console.print("openEMS binary found but Python API is missing.")
    console.print("Installing Python bindings only...")
    console.print()
    # Reuse the cached sources as they are: they are what an earlier install
    # built the C++ side from, and bindings from a newer checkout can be
    # ABI-incompatible with it.
    if not (cache_dir / "openEMS-Project").exists():
        console.print(
            "  [yellow]No cached openEMS sources; cloning the latest. If the "
            "bindings fail to build or import, reinstall with --force.[/]"
        )
    repo_dir = _ensure_repo(cache_dir, dry_run, update=False)
    _install_python_bindings(prefix, repo_dir, dry_run)


def _find_extracted_binary(root):
    name = "getdp.exe" if sys.platform == "win32" else "getdp"
    candidates = [p for p in root.rglob(name) if p.is_file()]
    for candidate in candidates:
        if candidate.parent.name == "bin":
            return candidate
    return candidates[0] if candidates else None


def _install_getdp_archive(prefix, version, dry_run):
    archives = _GETDP_ARCHIVES.get(version)
    if archives is None:
        console.print(f"  ❌ Unknown --version {version!r}. Use 'stable' or 'git'.")
        raise typer.Exit(code=1)

    key = (platform.system(), platform.machine())
    url = archives.get(key)
    if url is None:
        supported = ", ".join(f"{system}/{machine}" for system, machine in archives)
        console.print(
            f"  ❌ No prebuilt getdp binary for {key[0]}/{key[1]}. "
            f"Supported: {supported}. Install manually from https://getdp.info "
            "or build from source (https://gitlab.onelab.info/getdp/getdp)."
        )
        raise typer.Exit(code=1)

    console.print(Panel.fit("[bold]Installing getdp...[/]", border_style="cyan"))
    console.print()

    cache_dir = Path.home() / ".cache" / "simpleems"
    archive_path = cache_dir / url.rsplit("/", 1)[-1]

    console.print(f"Downloading {url}...")
    if dry_run:
        console.print(f"  [yellow]Would download to:[/] {archive_path}")
        console.print(f"  [yellow]Would extract to:[/] {prefix}")
        console.print("  [yellow]Would add the extracted getdp directory to PATH[/]")
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    _download_with_progress(url, archive_path)
    console.print("  ✅ Download complete")
    console.print()

    console.print("Extracting...")
    prefix.mkdir(parents=True, exist_ok=True)
    if url.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(prefix)
    else:
        with tarfile.open(archive_path, mode="r:gz") as tf:
            # The "data" filter rejects absolute paths, ".." escapes and device
            # files. It exists from 3.11.4 / 3.12 on.
            if hasattr(tarfile, "data_filter"):
                tf.extractall(prefix, filter="data")
            else:
                tf.extractall(prefix)
    console.print("  ✅ Extraction complete")
    console.print()

    binary = _find_extracted_binary(prefix)
    if binary is None:
        console.print(f"  ❌ Could not find a getdp binary under {prefix}")
        raise typer.Exit(code=1)
    binary.chmod(binary.stat().st_mode | 0o111)

    bin_dir = binary.parent
    _add_dir_to_path(bin_dir, dry_run)

    return binary


def _print_summary(ok, prefix, label="openEMS"):
    if ok:
        style = "green"
        message = f"✅ {label} installed successfully"
    else:
        style = "red"
        message = f"❌ {label} installation completed with issues"

    lines = [message]

    console.print()
    console.print(Panel.fit("\n".join(lines), border_style=style))
    raise typer.Exit(code=0 if ok else 1)


def _refresh_import_paths():
    """Make packages installed during this run importable by this process.

    The extensions are pip-installed after the interpreter started. When they
    land in the user site directory and that directory did not exist at
    startup, it is not on ``sys.path`` at all, so the health check below would
    report a perfectly good install as missing.
    """
    user_site = site.getusersitepackages()
    if os.path.isdir(user_site) and user_site not in sys.path:
        site.addsitedir(user_site)
    importlib.invalidate_caches()


def _verify_installation():
    console.print("Running health check...")
    _refresh_import_paths()
    all_ok = True
    for import_name, label in [
        ("openEMS", "openEMS Python API"),
        ("CSXCAD", "CSXCAD Python API"),
    ]:
        ok, version, error = _check_package_import(import_name, import_name)
        if ok:
            console.print(f"  ✅ {label}: {version}")
        else:
            console.print(f"  ❌ {label}: {error}")
            all_ok = False
    for binary, label, required in [
        # AppCSXCAD is the Qt geometry viewer. It is not built when the Qt
        # development packages are missing, which is the normal state of a
        # headless machine -- and an install without it still simulates, so it
        # is reported but does not make the install a failure.
        ("AppCSXCAD", "AppCSXCAD binary", False),
        ("openEMS", "openEMS solver", True),
    ]:
        ok, path, error = _check_system_binary(binary)
        if ok:
            console.print(f"  ✅ {label}: {path}")
        elif required:
            console.print(f"  ❌ {label}: {error}")
            all_ok = False
        else:
            console.print(f"  ⚠️  {label}: {error} (optional, GUI viewer)")
    console.print()
    return all_ok


@app.command(name="checkhealth")
def checkhealth(
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Also check optional development dependencies",
    ),
):
    """Check that all dependencies and external tools are available."""
    console.print(
        Panel.fit("[bold cyan]simpleEMS Health Check[/]", border_style="cyan")
    )
    console.print()

    dependency_checks = _get_dependency_checks_from_pyproject_section("dependencies")
    if dev:
        dev_dependency_checks = _get_dependency_checks_from_pyproject_section(
            "optional-dependencies"
        )
        dependency_checks = dependency_checks + dev_dependency_checks

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    python_ok = sys.version_info >= (3, 11)

    total_checks = 1 + len(dependency_checks) + 5

    results = []

    if sys.platform == "win32":
        prefix = _default_prefix()
        bin_dir = str(prefix / "openEMS")
        if os.path.isdir(bin_dir):
            os.environ.setdefault("OPENEMS_INSTALL_PATH", bin_dir)
            os.environ.setdefault("CSXCAD_INSTALL_PATH", bin_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running health checks...", total=total_checks)

        progress.update(task, description="Checking Python version")
        results.append(("Python", python_ok, python_version))
        progress.advance(task)

        for import_name, pip_name in dependency_checks:
            progress.update(task, description=f"Checking {pip_name}")
            ok, version, error = _check_package_import(import_name, pip_name)
            if ok:
                results.append((pip_name, True, version))
            else:
                results.append((pip_name, False, error))
            progress.advance(task)

        progress.update(task, description="Checking openEMS Python API")
        ok, version, error = _check_package_import("openEMS", "openEMS")
        if ok:
            results.append(("openEMS Python API", True, version))
        else:
            results.append(("openEMS Python API", False, error))
        progress.advance(task)

        progress.update(task, description="Checking CSXCAD Python API")
        ok, version, error = _check_package_import("CSXCAD", "CSXCAD")
        if ok:
            results.append(("CSXCAD Python API", True, version))
        else:
            results.append(("CSXCAD Python API", False, error))
        progress.advance(task)

        progress.update(task, description="Checking AppCSXCAD binary")
        ok, path, error = _check_system_binary("AppCSXCAD")
        if ok:
            results.append(("AppCSXCAD binary", True, path))
        else:
            results.append(("AppCSXCAD binary", False, error))
        progress.advance(task)

        progress.update(task, description="Checking openEMS binary")
        ok, path, error = _check_system_binary("openEMS")
        if ok:
            results.append(("openEMS solver", True, path))
        else:
            results.append(("openEMS solver", False, error))
        progress.advance(task)

        progress.update(task, description="Checking getdp binary (FEM backend)")
        ok, path, error = _check_system_binary("getdp")
        if ok:
            results.append(("getdp solver (FEM)", True, path))
        else:
            results.append(("getdp solver (FEM)", False, error))
        progress.advance(task)

        progress.update(task, description="Done")

    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Dependency", style="cyan", no_wrap=True)
    table.add_column("Status", width=8)
    table.add_column("Version / Path")

    passed = 0
    failed = 0

    for name, ok, value in results:
        if ok:
            table.add_row(name, "✅", value)
            passed = passed + 1
        else:
            table.add_row(name, "❌", f"[red]{value}[/]")
            failed = failed + 1

    console.print(table)
    console.print()

    summary_parts = []
    if passed:
        summary_parts.append(f"[green]✅ {passed} passed[/]")
    if failed:
        summary_parts.append(f"[red]❌ {failed} failed[/]")

    style = "green" if failed == 0 else "red"
    console.print(
        Panel.fit(
            "  ".join(summary_parts)
            if summary_parts
            else "[green]All checks passed[/]",
            border_style=style,
        )
    )
    console.print()

    raise typer.Exit(code=1 if failed > 0 else 0)


install_typer = typer.Typer(
    name="install",
    help="Install openEMS and its dependencies",
)
app.add_typer(install_typer, name="install")


@install_typer.callback(invoke_without_command=True)
def install_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@install_typer.command(name="openems")
def install_openems(
    prefix: Path | None = typer.Option(None, "--prefix", help="Install directory"),
    version: str = typer.Option(
        "latest", "--version", help="Release tag (Windows only)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print actions without executing"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Reinstall even if already installed "
            "(on Windows, deletes <prefix>/openEMS first)"
        ),
    ),
):
    """Install openEMS and CSXCAD."""
    prefix_given = prefix is not None
    if prefix is None:
        prefix = _default_prefix()

    binary_path = shutil.which("openEMS")
    python_ok, _, _ = _check_package_import("openEMS", "openEMS")
    if not force and binary_path is not None and python_ok:
        console.print("openEMS is already installed. Use --force to reinstall.")
        raise typer.Exit()

    # Every remaining path pip-installs the bindings into this interpreter, so
    # stop before a 10-30 minute build that is bound to fail at the end.
    if _externally_managed():
        console.print(
            f"  ❌ {sys.executable} is an externally managed system Python "
            "(PEP 668), so the openEMS bindings cannot be installed into it.\n"
            "  Install simpleEMS into a virtual environment and rerun from there:\n"
            "    python -m venv ~/.venvs/simpleems\n"
            "    ~/.venvs/simpleems/bin/pip install simpleEMS\n"
            "    ~/.venvs/simpleems/bin/simpleems install openems"
        )
        raise typer.Exit(code=1)

    if not force and binary_path is not None and not python_ok:
        if sys.platform != "win32" and not prefix_given:
            # Build against the install that is actually on PATH, which
            # need not be the default prefix (a distro package, say).
            prefix = Path(binary_path).resolve().parent.parent
        if sys.platform == "win32":
            _install_windows(prefix, version, dry_run)
        else:
            _install_python_bindings_only(prefix, dry_run)
        if not dry_run:
            ok = _verify_installation()
        else:
            ok = True
        _print_summary(ok, prefix)
        return

    if sys.platform == "win32":
        _install_windows(prefix, version, dry_run, force=force)
    else:
        _install_unix(prefix, dry_run, force=force)

    if not dry_run:
        ok = _verify_installation()
    else:
        ok = True

    _print_summary(ok, prefix)


@install_typer.command(name="getdp")
def install_getdp(
    prefix: Path | None = typer.Option(None, "--prefix", help="Install directory"),
    version: str = typer.Option(
        "stable",
        "--version",
        help="'stable' (pinned release) or 'git' (rolling dev build)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print actions without executing"
    ),
    force: bool = typer.Option(
        False, "--force", help="Reinstall even if already installed"
    ),
):
    """Install the getdp binary (FEM backend solver)."""
    if prefix is None:
        prefix = _default_getdp_prefix()

    if not force and shutil.which("getdp") is not None:
        console.print("getdp is already installed. Use --force to reinstall.")
        raise typer.Exit()

    _install_getdp_archive(prefix, version, dry_run)

    if dry_run:
        _print_summary(True, prefix, label="getdp")
        return

    ok, path, error = _check_system_binary("getdp")
    if ok:
        console.print(f"  ✅ getdp solver: {path}")
    else:
        console.print(f"  ❌ getdp solver: {error}")
    console.print()

    _print_summary(ok, prefix, label="getdp")


if __name__ == "__main__":
    app()
