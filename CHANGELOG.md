# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/) and follows [Semantic Versioning](https://semver.org/).

---

## 0.2.0 - 25 Jul 2026

### Added
- New GetDP-based FEM (finite-element) solver backend as an alternative to the
  FDTD (openEMS) backend, with shared FDTD/FEM modules unified for dual-backend
  simulation
- `simpleems install getdp` CLI command to install the GetDP solver binary
- Documentation: GetDP installation guide, `fem_backend` API reference, and a
  "Standalone FEM STEP Model" tutorial

### Fixed
- Updated examples for the `add_field_dump`/`compute_sim_data` signature changes
- Fixed README installation, PyPI, and demo links
- Fixed Sphinx docs CI build (mocked heavy/native dependencies for a
  docs-only build) and added GitHub Pages deployment

---

## 0.1.0 - 11 Jul 2026

- Initial public release
---

