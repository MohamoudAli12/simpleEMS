# Installation

## Prerequisites

- Python ≥ 3.11
- pip
- git, C/C++ compiler, cmake, make (for building openEMS from source)

## 1. Install simpleEMS

The quickest way is from PyPI — this will also install the `simpleems` CLI:

```bash
pip install simpleEMS
```

## 2. Install openEMS & CSXCAD

simpleEMS needs openEMS and CSXCAD to run FDTD simulations, and GetDP to run FEM simulation. Since these are not
available on PyPI, you need to install them separately.

### Option A: Using the CLI (recommended)

After installing simpleEMS, run:

```bash
simpleems install openems
```

**What it does per platform:**

- **Linux / macOS:** Clones the openEMS-Project repository, runs
  `./scripts/install_deps.sh --auto --python` (with sudo) to install system
  dependencies, then builds from source with
  `./update_openEMS.sh ~/opt/openEMS --python`. Adds `~/opt/openEMS/bin` to
  PATH and sets `OPENEMS_INSTALL_PATH` / `CSXCAD_INSTALL_PATH` in your shell
  rc file.

- **Windows:** Downloads the latest pre-built release from GitHub, extracts to
  `C:\openEMS`, installs the matching Python wheels, and sets PATH and
  environment variables.

### Option B: Manual — Linux / macOS

Start by cloning the openEMS repository with all submodules:

```bash
git clone --depth 1 --recursive https://github.com/thliebig/openEMS-Project.git
cd openEMS-Project
```

Install the system build dependencies required to compile openEMS. This
installs compilers, HDF5, VTK, and other libraries:

```bash
./scripts/install_deps.sh --auto --python
```

Build openEMS and the Python bindings, installing to `~/opt/openEMS`:

```bash
mkdir -p ~/opt/openEMS
./update_openEMS.sh ~/opt/openEMS --python
```

Finally, add the openEMS binaries to your PATH and set the required environment
variables so simpleEMS can find them:

```bash
echo 'export PATH="$HOME/opt/openEMS/bin:$PATH"' >> ~/.bashrc
echo 'export OPENEMS_INSTALL_PATH="$HOME/opt/openEMS"' >> ~/.bashrc
echo 'export CSXCAD_INSTALL_PATH="$HOME/opt/openEMS"' >> ~/.bashrc
source ~/.bashrc
```

### Option C: Manual — Windows

1. Download the latest pre-built `.zip` from
   [github.com/thliebig/openEMS-Project/releases](https://github.com/thliebig/openEMS-Project/releases)
2. Extract to `C:\openEMS`
3. Install matching Python wheels:
   ```
   pip install C:\openEMS\python\CSXCAD-*.whl C:\openEMS\python\openEMS-*.whl
   ```
4. Add `C:\openEMS` to your system PATH
5. Set environment variables:
   - `OPENEMS_INSTALL_PATH=C:\openEMS`
   - `CSXCAD_INSTALL_PATH=C:\openEMS`

## 3. Install GetDP

simpleEMS needs the `getdp` binary to run FEM simulations (the FDTD path via
openEMS/CSXCAD above works without it). GetDP only needs to be on your
`PATH` — no extra environment variables are required.

### Option A: Using the CLI (recommended)

```bash
simpleems install getdp
```

**What it does:** downloads the pinned "c" build (PETSc+MUMPS, required by
simpleEMS's direct MUMPS solve) for your OS/architecture from
[getdp.info](https://getdp.info), extracts it to `~/opt/getdp`
(`C:\getdp` on Windows), and adds its `bin` directory to `PATH` (persisted to
your shell rc file, or the user registry on Windows).

Useful options:

```bash
simpleems install getdp --version git     # rolling dev build instead of the pinned stable release
simpleems install getdp --prefix ~/tools/getdp  # custom install directory
simpleems install getdp --dry-run         # preview actions without downloading/extracting
simpleems install getdp --force           # reinstall even if getdp is already on PATH
```

Prebuilt archives are only available for Linux x86_64, macOS (x86_64/arm64),
and Windows x86_64. On any other platform, use the manual option below.

### Option B: Manual

1. Download the archive for your platform from
   [getdp.info](https://getdp.info/bin) — for the FEM backend's direct MUMPS
   solve, pick a `...c.tgz`/`...c.zip` build (the PETSc+MUMPS variant), not a
   plain build. If no prebuilt archive exists for your platform, build from
   source: [gitlab.onelab.info/getdp/getdp](https://gitlab.onelab.info/getdp/getdp).
2. Extract it, e.g. to `~/opt/getdp` (or `C:\getdp` on Windows).
3. Add the extracted `bin` directory (containing the `getdp` / `getdp.exe`
   binary) to your `PATH`:

   ```bash
   echo 'export PATH="$HOME/opt/getdp/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

   On Windows, add the folder containing `getdp.exe` to your system or user
   PATH via System Properties → Environment Variables.
4. Verify it resolves:

   ```bash
   getdp --version
   ```

## 4. Verify Installation

```bash
simpleems checkhealth
```

This checks: Python version, all pip dependencies, openEMS and CSXCAD Python
APIs, AppCSXCAD binary, the openEMS solver binary, and the getdp solver
binary (FEM backend).

## 5. Optional: Development Dependencies

If you plan to contribute to simpleEMS or build the documentation locally,
install the optional dev dependencies:

```bash
pip install simpleEMS[dev]
```

This installs:
- **ruff** — linter and formatter
- **prek** — pre-commit hook runner (configured in `prek.toml`)
- **sphinx**, **myst-parser**, **pydata-sphinx-theme**, **sphinx-automodapi**,
  **sphinx-copybutton** — documentation toolchain

Build the documentation with:

```bash
cd docs && make html
```

The built docs will be in `docs/build/html/`.

