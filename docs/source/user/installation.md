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

simpleEMS needs openEMS and CSXCAD to run simulations. Since these are not
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

## 3. Verify Installation

```bash
simpleems checkhealth
```

This checks: Python version, all pip dependencies, openEMS and CSXCAD Python
APIs, AppCSXCAD binary, and the openEMS solver binary.

## 4. Optional: Development Dependencies

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

