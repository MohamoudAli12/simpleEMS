"""simpleEMS CLI tool."""

import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
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
            for distribution_name in distribution_names:
                distribution_to_import[distribution_name.lower()] = import_name
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

        for separator in (">", "<", "=", "!", "~", "["):
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
    except ImportError as error:
        return False, None, str(error).split("(")[0].strip()


def _check_system_binary(binary_name):
    binary_path = shutil.which(binary_name)
    if binary_path is not None:
        return True, binary_path, None
    return False, None, "not found in PATH"


def _default_prefix():
    if sys.platform == "win32":
        return Path("C:\\openEMS")
    return Path.home() / "opt" / "openEMS"


def _ensure_repo(cache_dir, dry_run):
    repo_dir = cache_dir / "openEMS-Project"
    if repo_dir.exists():
        _run_subprocess(["git", "pull", "--recurse-submodules"], repo_dir, dry_run)
    else:
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
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs
        )
    except FileNotFoundError:
        console.print(f"  ❌ Command not found: {cmd[0]}")
        raise typer.Exit(code=1)
    for line in process.stdout:
        console.print(line, end="", markup=False, highlight=False)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def _add_to_path(prefix, dry_run):
    if sys.platform == "win32":
        import ctypes
        import winreg

        path_value = str(prefix / "openEMS")

        access = winreg.KEY_READ if dry_run else winreg.KEY_ALL_ACCESS
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, access)

        try:
            user_path, reg_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            user_path = ""
            reg_type = winreg.REG_EXPAND_SZ

        dirs = [d.strip() for d in user_path.split(";")] if user_path else []
        if path_value.lower() not in (d.lower() for d in dirs):
            new_path = f"{user_path};{path_value}" if user_path else path_value
            if dry_run:
                console.print(f"  [yellow]Would add {path_value} to user PATH[/]")
            else:
                winreg.SetValueEx(key, "Path", 0, reg_type, new_path)
                console.print(f"  [green]Added {path_value} to user PATH[/]")
        else:
            console.print(f"  [green]{path_value} already in user PATH[/]")

        for var in ("OPENEMS_INSTALL_PATH", "CSXCAD_INSTALL_PATH"):
            if dry_run:
                console.print(f"  [yellow]Would set {var} to {path_value}[/]")
            else:
                winreg.SetValueEx(key, var, 0, winreg.REG_SZ, path_value)
                console.print(f"  [green]Set {var} to {path_value}[/]")

        winreg.CloseKey(key)

        if not dry_run:
            HWND_BROADCAST = 0xFFFF
            WM_SETTINGCHANGE = 0x001A
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment", 0, 5000, None
            )
        return

    bin_dir = prefix / "bin"
    line = f'export PATH="{bin_dir}:$PATH"'
    if "zsh" in os.environ.get("SHELL", ""):
        rc_files = [
            Path.home() / ".zshrc",
            Path.home() / ".bashrc",
            Path.home() / ".bash_profile",
        ]
    else:
        rc_files = [
            Path.home() / ".bashrc",
            Path.home() / ".zshrc",
            Path.home() / ".bash_profile",
        ]
    for rc in rc_files:
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


def _download_with_progress(url, dest):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading...", total=total)
                with open(dest, "wb") as f:
                    for chunk in iter(lambda: response.read(8192), b""):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
    except OSError as e:
        dest.unlink(missing_ok=True)
        console.print(f"  ❌ Download failed: {e}")
        raise typer.Exit(code=1)


def _wheel_version_key(wheel):
    match = re.match(r"^[^-]+-([^-]+)-", wheel.name)
    if match is None:
        return []
    return [int(part) if part.isdigit() else 0 for part in match.group(1).split(".")]


def _pip_install(package_path, dry_run):
    if shutil.which("uv") is not None:
        cmd = ["uv", "pip", "install", "--python", sys.executable, str(package_path)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", str(package_path)]
    _run_subprocess(cmd, None, dry_run)


def _install_windows(prefix, version, dry_run, force=False):
    cache_dir = Path.home() / ".cache" / "simpleems"
    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(Panel.fit("[bold]Installing openEMS...[/]", border_style="cyan"))
    console.print()

    existing = (prefix / "openEMS" / "AppCSXCAD.exe").exists() and not force

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
                with urllib.request.urlopen(req, timeout=30) as response:
                    release_data = json.loads(response.read())
            except OSError as e:
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
            zip_path.unlink(missing_ok=True)
            console.print("  ✅ Extraction complete")
            console.print()
    else:
        console.print(f"  ✅ openEMS already present at {prefix}")
        console.print()

    python_dir = prefix / "openEMS" / "python"
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
        _pip_install(str(max(matching_csxcad, key=_wheel_version_key)), dry_run)
        _pip_install(str(max(matching_openems, key=_wheel_version_key)), dry_run)
        console.print("  ✅ Python bindings installed")
        console.print()
    _add_to_path(prefix, dry_run)

    bin_dir = str(prefix / "openEMS")
    current_path = os.environ.get("PATH", "")
    dirs = current_path.split(os.pathsep)
    if bin_dir not in dirs:
        os.environ["PATH"] = bin_dir + os.pathsep + current_path

    os.environ["OPENEMS_INSTALL_PATH"] = bin_dir
    os.environ["CSXCAD_INSTALL_PATH"] = bin_dir


def _install_unix(prefix, dry_run):
    cache_dir = Path.home() / ".cache" / "simpleems"
    console.print(Panel.fit("[bold]Installing openEMS...[/]", border_style="cyan"))
    console.print()
    console.print("Cloning openEMS-Project...")
    repo_dir = _ensure_repo(cache_dir, dry_run)
    console.print("  ✅ Repository ready")
    console.print()
    console.print("Installing system dependencies (requires sudo)...")
    _run_subprocess(
        ["sudo", "./scripts/install_deps.sh", "--auto", "--python"], repo_dir, dry_run
    )
    console.print("  ✅ Dependencies installed")
    console.print()
    console.print("Building openEMS (this may take 10-30 minutes)...")
    prefix.mkdir(parents=True, exist_ok=True)
    _run_subprocess(["./update_openEMS.sh", str(prefix), "--python"], repo_dir, dry_run)
    console.print("  ✅ Build complete")
    console.print()
    _add_to_path(prefix, dry_run)

    bin_dir = str(prefix / "bin")
    current_path = os.environ.get("PATH", "")
    if bin_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + current_path
    console.print()


def _install_python_bindings_only(prefix, dry_run):
    cache_dir = Path.home() / ".cache" / "simpleems"
    console.print("openEMS binary found but Python API is missing.")
    console.print("Installing Python bindings only...")
    console.print()
    repo_dir = _ensure_repo(cache_dir, dry_run)
    csxcad_path = repo_dir / "CSXCAD" / "python"
    openems_path = repo_dir / "openEMS" / "python"
    if dry_run:
        console.print(f"  [yellow]Would install:[/] {csxcad_path}")
        console.print(f"  [yellow]Would install:[/] {openems_path}")
    else:
        console.print("Installing CSXCAD Python bindings...")
        _pip_install(csxcad_path, dry_run)
        console.print("  ✅ CSXCAD Python bindings installed")
        console.print()
        console.print("Installing openEMS Python bindings...")
        _pip_install(openems_path, dry_run)
        console.print("  ✅ openEMS Python bindings installed")
    console.print()


def _print_summary(ok):
    if ok:
        style = "green"
        message = "✅ openEMS installed successfully"
    else:
        style = "red"
        message = "❌ openEMS installation completed with issues"

    console.print()
    console.print(Panel.fit(message, border_style=style))
    raise typer.Exit(code=0 if ok else 1)


def _verify_installation():
    console.print("Running health check...")
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
    for binary, label in [
        ("AppCSXCAD", "AppCSXCAD binary"),
        ("openEMS", "openEMS solver"),
    ]:
        ok, path, error = _check_system_binary(binary)
        if ok:
            console.print(f"  ✅ {label}: {path}")
        else:
            console.print(f"  ❌ {label}: {error}")
            all_ok = False
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

    total_checks = 1 + len(dependency_checks) + 4

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
        False, "--force", help="Reinstall even if already installed"
    ),
):
    """Install openEMS and CSXCAD."""
    if prefix is None:
        prefix = _default_prefix()

    if not force:
        binary_exists = shutil.which("openEMS") is not None
        python_ok, _, _ = _check_package_import("openEMS", "openEMS")

        if binary_exists and python_ok:
            console.print("openEMS is already installed. Use --force to reinstall.")
            raise typer.Exit()

        if binary_exists and not python_ok:
            if sys.platform == "win32":
                _install_windows(prefix, version, dry_run)
            else:
                _install_python_bindings_only(prefix, dry_run)
            if not dry_run:
                ok = _verify_installation()
            else:
                ok = True
            _print_summary(ok)

    if sys.platform == "win32":
        _install_windows(prefix, version, dry_run, force=force)
    else:
        _install_unix(prefix, dry_run)

    if not dry_run:
        ok = _verify_installation()
    else:
        ok = True

    _print_summary(ok)


if __name__ == "__main__":
    app()
