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
FEM far-field radiation, exposed through an openEMS-``nf2ff``-compatible adapter.

The GetDP solve gives the near fields; a Gmsh CutBox samples E/H on a box
enclosing the antenna and the NearToFarField plugin transforms them to the
far field. :class:`FemNF2FF` wraps that computation behind the same
``CalcNF2FF(output_path, freq, theta, phi, ...)`` interface that
:class:`openEMS.nf2ff.nf2ff` exposes, returning a :class:`FemFarField` with the
attributes (``E_norm``, ``Dmax``, ``Prad``, ``P_rad``, ``theta``, ``phi``) that
the existing :class:`~simpleEMS.sim_tools.SimTools` 2D/3D radiation plots read.
That way the FDTD and FEM backends share the same plotting code unchanged.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator, griddata

from . import fem_solver
from .console import console
from .fem_materials import C0

__all__ = ["FemFarField", "FemNF2FF"]

_MESH_META = "fem_mesh.json"


@dataclass
class FemFarField:
    """Far-field result mirroring the openEMS ``nf2ff`` result surface.

    Attributes are named to match what the ``SimTools`` radiation plots read:
    ``E_norm`` is the far-field amplitude on the requested ``[theta, phi]`` grid,
    ``Dmax`` the peak directivity (linear, length-1 array), ``Prad`` the radiated
    power (length-1 array), ``P_rad`` the radiation-intensity pattern, and
    ``theta``/``phi`` the grid axes in **radians**. ``Ploss`` (dielectric loss,
    length-1 array) is FEM-specific: the accepted power is ``Prad + Ploss``,
    which is what should be passed as ``input_power`` to ``plot_3d_gain`` so gain
    is de-rated by the radiation efficiency.
    """

    E_norm: NDArray
    Dmax: NDArray
    Prad: NDArray
    P_rad: NDArray
    theta: NDArray
    phi: NDArray
    Ploss: NDArray


def compute_pattern(
    e_pos: str,
    h_pos: str,
    freq: float,
    bbox: tuple,
    workdir: str,
    nphi: int = 72,
    ntheta: int = 90,
    npts: tuple[int, int, int] = (30, 30, 30),
    margin_frac: float = 0.15,
    verbose: bool = False,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """
    Transform near fields to a far-field intensity pattern on a regular grid.

    Parameters
    ----------
    e_pos, h_pos : str
        Paths to the GetDP ``e.pos``/``h.pos`` field views.
    freq : float
        Frequency in Hz.
    bbox : tuple
        Structure extents ``(xmin, ymin, zmin, xmax, ymax, zmax)`` in metres.
    workdir : str
        Directory for the intermediate NearToFarField output.
    nphi, ntheta : int
        Far-field angular sampling (azimuth, elevation). Cost ~ ``nphi * ntheta``.
    npts : tuple[int, int, int]
        CutBox sampling density on each axis of the Huygens box.
    margin_frac : float
        Huygens-box margin as a fraction of the largest structure dimension.
    verbose : bool
        Print Gmsh progress. Default ``False``.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, float]
        ``(theta_axis, phi_axis, u_grid, directivity_db)`` -- the elevation and
        azimuth axes (radians, increasing), the radiation-intensity grid
        ``u_grid[n_phi, n_theta]``, and the peak directivity in dB.
    """
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

    gmsh.merge(e_pos)  # view index 0 -> E
    gmsh.merge(h_pos)  # view index 1 -> H

    # Huygens box just outside the structure, inside the air region.
    dx, dy, dz = bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
    m = margin_frac * max(dx, dy, dz)
    x0, y0, z0 = bbox[0] - m, bbox[1] - m, bbox[2] - m
    x1, y1, z1 = bbox[3] + m, bbox[4] + m, bbox[5] + m

    def cb(name: str, val: float) -> None:
        gmsh.plugin.setNumber("CutBox", name, val)

    cb("NumPointsU", npts[0])
    cb("NumPointsV", npts[1])
    cb("NumPointsW", npts[2])
    cb("X0", x0)
    cb("Y0", y0)
    cb("Z0", z0)
    cb("X1", x1)
    cb("Y1", y0)
    cb("Z1", z0)
    cb("X2", x0)
    cb("Y2", y1)
    cb("Z2", z0)
    cb("X3", x0)
    cb("Y3", y0)
    cb("Z3", z1)
    cb("ConnectPoints", 1)
    cb("Boundary", 1)

    cb("View", 0)
    e_box = gmsh.plugin.run("CutBox")  # new view tag for E on the box
    cb("View", 1)
    h_box = gmsh.plugin.run("CutBox")  # new view tag for H on the box

    def view_index(tag: int) -> int:
        return list(gmsh.view.getTags()).index(tag)

    k0 = 2 * math.pi * freq / C0
    gmsh.plugin.setNumber("NearToFarField", "Wavenumber", k0)
    gmsh.plugin.setNumber("NearToFarField", "RFar", 1.0)
    gmsh.plugin.setNumber("NearToFarField", "NumPointsPhi", nphi)
    gmsh.plugin.setNumber("NearToFarField", "NumPointsTheta", ntheta)
    gmsh.plugin.setNumber("NearToFarField", "EView", view_index(e_box))
    gmsh.plugin.setNumber("NearToFarField", "HView", view_index(h_box))
    gmsh.plugin.setNumber("NearToFarField", "Normalize", 0)
    gmsh.plugin.setNumber("NearToFarField", "dB", 0)

    outdir = os.path.join(os.path.abspath(workdir), "output")
    os.makedirs(outdir, exist_ok=True)
    # The plugin writes an exact regular (phi, theta, farField) grid to this
    # MATLAB file; we read the pattern from there rather than the deformed .pos.
    mfile = os.path.join(outdir, "pattern_ntf.m")
    gmsh.plugin.setString("NearToFarField", "MatlabOutputFile", mfile)
    ntf = gmsh.plugin.run("NearToFarField")
    gmsh.view.write(ntf, os.path.join(outdir, "pattern_ntf.pos"))

    grid = _parse_matlab_grid(mfile, nphi, ntheta)
    if grid is not None:
        phi_grid, theta_grid, u_grid = grid  # each (nphi+1, ntheta+1)
        theta_axis = theta_grid[0, :]
        phi_axis = phi_grid[:, 0]
        val = u_grid.ravel()
    else:
        # Fallback: recover scattered samples from the deformed .pos view and
        # resample them onto a regular grid (see _read_scalar_points).
        cx, cy, cz = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
        theta_s, phi_s, val_s = _read_scalar_points(ntf, (cx, cy, cz))
        theta_axis = np.linspace(0.0, math.pi, ntheta + 1)
        phi_axis = np.linspace(0.0, 2 * math.pi, nphi + 1)
        pg, tg = np.meshgrid(phi_axis, theta_axis, indexing="ij")
        u_grid = griddata(
            (np.asarray(phi_s), np.asarray(theta_s)),
            np.asarray(val_s),
            (pg, tg),
            method="linear",
            fill_value=0.0,
        )
        val = np.asarray(val_s)
    gmsh.finalize()

    # keep axes strictly increasing so RegularGridInterpolator is happy
    if theta_axis[0] > theta_axis[-1]:
        theta_axis = theta_axis[::-1]
        u_grid = u_grid[:, ::-1]
    if phi_axis[0] > phi_axis[-1]:
        phi_axis = phi_axis[::-1]
        u_grid = u_grid[::-1, :]

    # directivity D = U_max / U_avg, with U_avg the sin(theta)-weighted mean.
    umax = float(val.max()) if val.size else 1.0
    weights = np.abs(np.sin(theta_axis))[None, :] * np.ones_like(u_grid)
    denom = float(np.sum(weights))
    u_avg = float(np.sum(u_grid * weights) / denom) if denom > 0 else umax
    directivity_db = 10.0 * math.log10(max(umax, 1e-30) / max(u_avg, 1e-30))

    return theta_axis, phi_axis, u_grid, directivity_db


def _parse_matlab_grid(path: str, nphi: int, ntheta: int) -> tuple | None:
    """Parse the NearToFarField MATLAB dump into ``(phi, theta, val)`` grids.

    The plugin writes ``phi``/``theta``/``farField`` as flat row-major matrices
    with phi as the outer index and theta the inner one. Returns ``None`` if the
    file is missing or cannot be reshaped to ``(nphi + 1, ntheta + 1)``.
    """
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None

    def vec(name: str) -> NDArray | None:
        match = re.search(name + r"\s*=\s*\[([^\]]*)\]", text)
        return np.fromstring(match.group(1), sep=" ") if match else None

    phi, theta, ff = vec("phi"), vec("theta"), vec("farField")
    if phi is None or theta is None or ff is None:
        return None
    shape = (nphi + 1, ntheta + 1)
    if phi.size != shape[0] * shape[1]:
        return None
    try:
        return phi.reshape(shape), theta.reshape(shape), ff.reshape(shape)
    except ValueError:
        return None


def _read_scalar_points(
    view_tag: int, center: tuple[float, float, float]
) -> tuple[list, list, list]:
    """Fallback: extract ``(theta, phi, value)`` from a NearToFarField .pos view.

    The plugin places each vertex at a radius proportional to the field
    magnitude, offset by the sampling-box ``center``; angles are only meaningful
    once that centre is subtracted. Each node carries its own value, so corner
    nodes are emitted directly. Handles points ('SP'), triangles ('ST') and
    quads ('SQ').
    """
    dtypes, tags, data = gmsh.view.getListData(view_tag)
    cx, cy, cz = center
    theta, phi, val = [], [], []

    def add(x: float, y: float, z: float, v: float) -> None:
        x, y, z = x - cx, y - cy, z - cz
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-30:  # collapsed null point -> direction undefined; skip it
            return
        theta.append(math.acos(max(-1.0, min(1.0, z / r))))
        phi.append(math.atan2(y, x) % (2 * math.pi))
        val.append(v)

    for dt, _tag, arr in zip(dtypes, tags, data, strict=True):
        arr = np.asarray(arr, dtype=float)
        if dt == "SP":  # 1 node: x,y,z then v
            for row in arr.reshape(-1, 4):
                add(row[0], row[1], row[2], row[3])
        elif dt == "ST":  # 3 nodes: 9 coords then v1,v2,v3
            for row in arr.reshape(-1, 12):
                for n in range(3):
                    add(row[3 * n], row[3 * n + 1], row[3 * n + 2], row[9 + n])
        elif dt == "SQ":  # 4 nodes: 12 coords then v1..v4
            for row in arr.reshape(-1, 16):
                for n in range(4):
                    add(row[3 * n], row[3 * n + 1], row[3 * n + 2], row[12 + n])
    return theta, phi, val


class FemNF2FF:
    """
    openEMS-``nf2ff``-compatible far-field calculator for the FEM backend.

    Like the openEMS box it is created without an ``output_path``; each
    :meth:`CalcNF2FF` call reads ``fem_mesh.json`` from the ``output_path`` it is
    given (the ``.pro``/``.msh`` paths and structure bbox) and mirrors
    ``openEMS.nf2ff.nf2ff.CalcNF2FF`` so the ``SimTools`` radiation plots work
    unchanged.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple] = {}  # (workdir, freq) -> pattern data

    def _pattern(self, output_path: str, freq: float) -> tuple:
        """Solve fields/power at ``freq`` and build the pattern (cached)."""
        key = (output_path, freq)
        if key not in self._cache:
            meta = json.loads((Path(output_path) / _MESH_META).read_text())
            pro, msh, bbox = meta["pro_path"], meta["msh_path"], tuple(meta["bbox"])
            console.print(
                f"[info]FEM far-field: solving fields at {freq / 1e9:.4f} GHz[/info]"
            )
            e_pos, h_pos = fem_solver.solve_fields(pro, msh, output_path, freq)
            p_loss, p_rad = fem_solver.solve_power(pro, msh, output_path, freq)
            theta_axis, phi_axis, u_grid, dir_db = compute_pattern(
                e_pos, h_pos, freq, bbox, output_path
            )
            self._cache[key] = (theta_axis, phi_axis, u_grid, dir_db, p_rad, p_loss)
        return self._cache[key]

    def CalcNF2FF(  # noqa: N802  (matches the openEMS nf2ff method name)
        self,
        output_path: str | Path,
        freq: float,
        theta: NDArray,
        phi: NDArray,
        read_cached: bool = False,
        outfile: str | None = None,
        verbose: int = 0,
    ) -> FemFarField:
        """
        Evaluate the far field on a ``theta``/``phi`` grid (degrees in).

        Matches ``openEMS.nf2ff.nf2ff.CalcNF2FF``. ``theta``/``phi`` are in
        degrees (as ``SimTools`` passes them); negative ``theta`` is mapped to
        ``(|theta|, phi + 180)`` so principal-plane cuts spanning -180..180 work.
        """
        theta_axis, phi_axis, u_grid, dir_db, p_rad, p_loss = self._pattern(
            str(output_path), float(freq)
        )
        interp = RegularGridInterpolator(
            (phi_axis, theta_axis), u_grid, bounds_error=False, fill_value=None
        )

        theta_deg = np.atleast_1d(np.asarray(theta, dtype=float))
        phi_deg = np.atleast_1d(np.asarray(phi, dtype=float))
        th_mesh, ph_mesh = np.meshgrid(theta_deg, phi_deg, indexing="ij")

        # negative theta -> mirror to the phi+180 half-plane
        neg = th_mesh < 0
        th_q = np.where(neg, -th_mesh, th_mesh)
        ph_q = np.where(neg, ph_mesh + 180.0, ph_mesh)
        th_r = np.clip(np.deg2rad(th_q), 0.0, math.pi)
        ph_r = np.deg2rad(ph_q) % (2 * math.pi)

        u_vals = np.maximum(interp((ph_r, th_r)), 1e-30)  # [n_theta, n_phi]
        return FemFarField(
            E_norm=np.sqrt(u_vals),
            Dmax=np.array([10.0 ** (dir_db / 10.0)]),
            Prad=np.array([float(p_rad)]),
            P_rad=u_vals,
            theta=np.deg2rad(theta_deg),
            phi=np.deg2rad(phi_deg),
            Ploss=np.array([float(p_loss)]),
        )
