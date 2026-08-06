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
Far-field radiation patterns for the FEM backend.

Computes the far field from the near fields of a solve, and exposes it through
:class:`FEMNF2FF`, which offers the same ``CalcNF2FF`` interface as the openEMS
near-field-to-far-field box so both backends share the same radiation plots.
"""

from __future__ import annotations

import json
import math
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

__all__ = ["FEMNF2FF"]

_MESH_META = "fem_mesh.json"

# CutBox places nodes exactly on the box faces, so this only absorbs float noise.
_PLANE_TOL = 1e-12


@dataclass
class FEMFarField:
    """
    A far-field result, matching what the openEMS ``nf2ff`` box returns.

    Attributes
    ----------
    E_norm : NDArray
        Far-field amplitude over the requested ``[theta, phi]`` grid.
    Dmax : NDArray
        Peak directivity, linear, as a length-1 array.
    Prad : NDArray
        Radiated power in watts, as a length-1 array.
    P_rad : NDArray
        Radiation intensity over the requested grid.
    theta : NDArray
        Elevation angles of the grid, in radians.
    phi : NDArray
        Azimuth angles of the grid, in radians.
    Ploss : NDArray
        Power lost in the materials, in watts, as a length-1 array. Only the
        FEM backend reports this; ``Prad + Ploss`` is the accepted power, which
        is what to pass as ``input_power`` to ``plot_3d_gain`` so that gain is
        de-rated by the radiation efficiency.
    """

    E_norm: NDArray
    Dmax: NDArray
    Prad: NDArray
    P_rad: NDArray
    theta: NDArray
    phi: NDArray
    Ploss: NDArray


def compute_pattern(
    e_pos: str | Path,
    h_pos: str | Path,
    freq: float,
    bbox: tuple,
    workdir: str | Path,
    domain_bbox: tuple | None = None,
    nphi: int = 72,
    ntheta: int = 36,
    npts: tuple[int, int, int] = (14, 14, 14),
    margin_frac: float = 0.15,
    safety_frac: float = 0.6,
    symmetry: tuple[int, float, str] | None = None,
    verbose: bool = False,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """
    Turn the near fields of a solve into a far-field pattern on a regular grid.

    Parameters
    ----------
    e_pos, h_pos : str | Path
        Paths to the electric and magnetic near-field files written by the
        solver.
    freq : float
        Frequency the fields were solved at, in Hz.
    bbox : tuple
        Extents of the structure as ``(xmin, ymin, zmin, xmax, ymax, zmax)``,
        in metres.
    workdir : str | Path
        Directory to write the intermediate pattern files into.
    domain_bbox : tuple | None
        Extents of the meshed region, in metres. The near fields are sampled
        on a box placed ``safety_frac`` of the way from ``bbox`` out to this,
        so the samples stay inside the mesh. Default ``None``, which falls
        back to ``margin_frac`` instead.
    nphi, ntheta : int
        Number of azimuth and elevation angles to compute the pattern at.
        Defaults ``72`` and ``36``. Every angle later requested is
        interpolated from this grid.
    npts : tuple[int, int, int]
        Number of near-field sample points along each axis of the sampling
        box. Default ``(14, 14, 14)``.
    margin_frac : float
        Gap between the structure and the sampling box, as a fraction of the
        largest structure dimension. Used only when ``domain_bbox`` is
        ``None``. Default ``0.15``.
    safety_frac : float
        Gap between the structure and the sampling box, as a fraction of the
        distance from the structure to the edge of the mesh. Default ``0.6``.
    symmetry : tuple[int, float, str] | None
        ``(axis, plane, kind)`` when only half the structure was meshed, so
        the missing half of the pattern can be filled in by reflection.
        Default ``None``.
    verbose : bool
        Print progress. Default ``False``.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, float]
        ``(theta_axis, phi_axis, u_grid, directivity_db)`` -- the elevation
        and azimuth angles in radians, the radiation intensity over them as
        ``u_grid[n_phi, n_theta]``, and the peak directivity in dB.
    """
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

    gmsh.merge(str(e_pos))  # view index 0 -> E
    gmsh.merge(str(h_pos))  # view index 1 -> H

    # Huygens box just outside the structure, inside the meshed air region.
    if domain_bbox is not None:
        lo = [bbox[i] - safety_frac * (bbox[i] - domain_bbox[i]) for i in range(3)]
        hi = [
            bbox[3 + i] + safety_frac * (domain_bbox[3 + i] - bbox[3 + i])
            for i in range(3)
        ]
        if symmetry is not None:
            # `domain_bbox` predates the cut, so without this the box reaches
            # across the plane into unmeshed space, where CutBox reads zeros.
            axis_i, sym_plane, _kind = symmetry
            lo[axis_i] = sym_plane
        x0, y0, z0 = lo
        x1, y1, z1 = hi
    else:
        # This can land the box outside the mesh if the air padding is smaller,
        # silently zeroing the far field: CutBox returns 0 for points outside
        # every element rather than failing.
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

    if symmetry is not None:
        axis_i, sym_plane, sym_kind = symmetry
        e_box = _mirror_boundary_view(e_box, axis_i, sym_plane, sym_kind, "E")
        h_box = _mirror_boundary_view(h_box, axis_i, sym_plane, sym_kind, "H")

    def view_index(tag: int) -> int:
        return list(gmsh.view.getTags()).index(tag)

    k0 = 2 * math.pi * freq / C0
    gmsh.plugin.setNumber("NearToFarField", "Wavenumber", k0)
    # The .pro solves in exp(+i w t) (GetDP's native convention, see
    # fem_formulation), which is the plugin's default, so the transform is
    # left as-is. That branch returns |E_inf|^2, scaled arbitrarily, so
    # callers normalise by the peak and take the level from the directivity.
    gmsh.plugin.setNumber("NearToFarField", "NegativeTime", 0)
    gmsh.plugin.setNumber("NearToFarField", "NumPointsPhi", nphi)
    gmsh.plugin.setNumber("NearToFarField", "NumPointsTheta", ntheta)
    gmsh.plugin.setNumber("NearToFarField", "EView", view_index(e_box))
    gmsh.plugin.setNumber("NearToFarField", "HView", view_index(h_box))
    gmsh.plugin.setNumber("NearToFarField", "Normalize", 0)
    gmsh.plugin.setNumber("NearToFarField", "dB", 0)

    outdir = Path(workdir).absolute() / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    # The plugin writes an exact regular (phi, theta, farField) grid to this
    # MATLAB file; we read the pattern from there rather than the deformed .pos.
    mfile = outdir / "pattern_ntf.m"
    gmsh.plugin.setString("NearToFarField", "MatlabOutputFile", str(mfile))
    ntf = gmsh.plugin.run("NearToFarField")
    gmsh.view.write(ntf, str(outdir / "pattern_ntf.pos"))

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


def _mirror_boundary_view(
    tag: int, axis: int, plane: float, kind: str, field: str
) -> int:
    """Complete a half-model sampling surface by reflecting it in the symmetry plane.

    Parameters
    ----------
    tag : int
        The half surface to complete.
    axis : int
        Axis the structure is mirrored across: ``0``, ``1``, or ``2`` for x,
        y, or z.
    plane : float
        Position of the symmetry plane along ``axis``, in metres.
    kind : str
        Symmetry plane type: ``"pec"`` or ``"pmc"``.
    field : str
        Which field is being reflected: ``"E"`` or ``"H"``. Together with
        ``kind`` this sets which components change sign.

    Returns
    -------
    int
        The completed surface: the original samples, minus those lying in the
        plane, plus their mirror images.
    """
    # Three things have to hold together or the pattern comes out plausible but
    # wrong: samples in the plane are dropped (they end up interior), node
    # order is reversed (a reflection flips orientation, and the far-field
    # transform reads normals from the first three nodes), and each component
    # follows the parity of the wall.
    dtypes, counts, data = gmsh.view.getListData(tag)
    nsteps = int(gmsh.view.option.getNumber(tag, "NbTimeStep"))

    # Even component keeps its sign under reflection, odd flips. The normal
    # component is the one along `axis`.
    electric = (field == "E") == (kind == "pec")
    normal_sign = 1.0 if electric else -1.0
    tangential_sign = -normal_sign

    out = gmsh.view.add(f"huygens_{field}")
    for dtype, nelem, arr in zip(dtypes, counts, data, strict=True):
        arr = np.asarray(arr, dtype=float)
        nnodes = {"T": 3, "Q": 4}[dtype[1]]  # VT / VQ
        ncoord = 3 * nnodes
        stride = ncoord + nsteps * nnodes * 3
        if nelem * stride != arr.size:
            raise RuntimeError(
                f"unexpected {dtype} list-data layout: {arr.size} floats for "
                f"{nelem} elements ({nnodes} nodes, {nsteps} steps)"
            )

        kept, mirrored = [], []
        for e in range(nelem):
            el = arr[e * stride : (e + 1) * stride]
            coords = el[:ncoord].reshape(3, nnodes)  # [x-block, y-block, z-block]
            if np.all(np.abs(coords[axis] - plane) <= _PLANE_TOL):
                continue  # lies in the symmetry plane -> interior once completed
            kept.append(el)

            order = list(range(nnodes))[::-1]  # reflection reverses orientation
            mc = coords.copy()
            mc[axis] = 2.0 * plane - mc[axis]
            mv = el[ncoord:].reshape(nsteps, nnodes, 3).copy()
            mv[:, :, axis] *= normal_sign
            for c in range(3):
                if c != axis:
                    mv[:, :, c] *= tangential_sign
            mirrored.append(
                np.concatenate([mc[:, order].ravel(), mv[:, order, :].ravel()])
            )

        if kept or mirrored:
            block = np.concatenate(kept + mirrored)
            gmsh.view.addListData(out, dtype, len(kept) + len(mirrored), list(block))
    return out


def _parse_matlab_grid(path: str | Path, nphi: int, ntheta: int) -> tuple | None:
    """Read the written pattern file into ``(phi, theta, value)`` grids.

    Parameters
    ----------
    path : str | Path
        Path to the pattern file.
    nphi, ntheta : int
        Number of azimuth and elevation angles the file should hold.

    Returns
    -------
    tuple | None
        The three ``(nphi + 1, ntheta + 1)`` grids, or ``None`` if the file is
        missing or does not hold the expected number of angles.
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
    """Read scattered ``(theta, phi, value)`` samples out of a pattern view.

    Used only when the regular grid cannot be read from the pattern file.

    Parameters
    ----------
    view_tag : int
        The pattern view to read.
    center : tuple[float, float, float]
        Centre of the sampling box, in metres, which the sample positions are
        measured relative to.

    Returns
    -------
    tuple[list, list, list]
        The elevation angles, azimuth angles, and values of every sample.
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

    # gmsh list data stores coordinates per axis, not per node: an element's
    # block is x1..xn, y1..yn, z1..zn, then one value per node. Reading it as
    # interleaved triples silently transposes the sample positions, which
    # scrambles the interpolated pattern rather than failing.
    nodes_per_type = {"SP": 1, "ST": 3, "SQ": 4}
    for dt, _tag, arr in zip(dtypes, tags, data, strict=True):
        nnodes = nodes_per_type.get(dt)
        if nnodes is None:
            continue
        arr = np.asarray(arr, dtype=float)
        ncoord = 3 * nnodes
        for row in arr.reshape(-1, ncoord + nnodes):
            coords = row[:ncoord].reshape(3, nnodes)
            values = row[ncoord:]
            for n in range(nnodes):
                add(coords[0, n], coords[1, n], coords[2, n], values[n])
    return theta, phi, val


def _check_farfield_margin(
    bbox: tuple, domain_bbox: tuple, freq: float, symmetry_axis: int | None
) -> None:
    """Check the air padding is wide enough for an accurate far field at ``freq``.

    Parameters
    ----------
    bbox : tuple
        Extents of the structure, in metres.
    domain_bbox : tuple
        Extents of the meshed region, in metres.
    freq : float
        Frequency the pattern is wanted at, in Hz.
    symmetry_axis : int | None
        Axis the structure is mirrored across, whose lower face is a symmetry
        plane rather than open air. ``None`` if there is no symmetry plane.

    Raises
    ------
    ValueError
        If any face has less than a quarter-wavelength of air between the
        structure and the edge of the mesh, naming the padding needed.
    """
    # Only structures meshed with an explicit FEM_air_pad_mm can fail this --
    # the default padding already targets a quarter-wavelength. So it is
    # checked here, when a pattern is actually asked for, rather than at mesh
    # time: tight padding is fine for problems that never want a far field.
    min_gap = 0.25 * (C0 / freq)  # lambda/4 at the requested frequency
    gaps = {}
    for i, axis in enumerate("xyz"):
        if symmetry_axis != i:
            gaps[f"{axis}-"] = bbox[i] - domain_bbox[i]
        gaps[f"{axis}+"] = domain_bbox[3 + i] - bbox[3 + i]
    face, gap = min(gaps.items(), key=lambda kv: kv[1])
    if gap < min_gap:
        raise ValueError(
            f"Air padding too small for an accurate far field at {freq / 1e9:.4f} GHz: "
            f"the gap between the structure and the meshed domain boundary on the "
            f"{face} face is {gap * 1e3:.2f} mm, but at least {min_gap * 1e3:.2f} mm "
            f"(a quarter-wavelength, lambda/4 = c / (4 * f)) is needed for the "
            f"near-to-far-field transform to be valid. Re-run setup_simulation/"
            f"build_mesh with FEM_air_pad_mm >= {min_gap * 1e3:.2f}, or drop "
            f"FEM_air_pad_mm entirely to fall back to the automatic "
            f"lambda/4-based padding."
        )


class FEMNF2FF:
    """
    Far-field calculator for the FEM backend.

    Takes no arguments to create. Each :meth:`CalcNF2FF` call reads what it
    needs from the output directory it is given, and offers the same interface
    as the openEMS near-field-to-far-field box so the ``SimTools`` radiation
    plots work unchanged.
    """

    def __init__(self) -> None:
        """Create a far-field calculator with no patterns computed yet."""
        self._cache: dict[tuple, tuple] = {}  # (workdir, freq) -> pattern data

    def _pattern(self, output_path: str, freq: float) -> tuple:
        """Solve the fields and powers at ``freq`` and build the pattern, once."""
        key = (output_path, freq)
        if key not in self._cache:
            meta = json.loads((Path(output_path) / _MESH_META).read_text())
            pro, msh, bbox = meta["pro_path"], meta["msh_path"], tuple(meta["bbox"])
            domain_bbox = tuple(meta["domain_bbox"]) if "domain_bbox" in meta else None
            if domain_bbox is None:
                console.print(
                    "[warning]fem_mesh.json has no domain_bbox (stale mesh); "
                    "falling back to a structure-relative Huygens box, which "
                    "can land outside the meshed domain and zero out the far "
                    "field. Re-run build_mesh to regenerate it.[/warning]"
                )
            else:
                _check_farfield_margin(
                    bbox, domain_bbox, freq, meta.get("symmetry_axis")
                )
            console.print(
                f"[info]FEM far-field: solving fields at {freq / 1e9:.4f} GHz[/info]"
            )
            e_pos, h_pos, p_loss, p_rad = fem_solver.solve_fields_and_power(
                pro, msh, output_path, freq
            )
            sym_axis = meta.get("symmetry_axis")
            symmetry = (
                (sym_axis, meta["symmetry_plane"], meta["symmetry_kind"])
                if sym_axis is not None and meta.get("symmetry_plane") is not None
                else None
            )
            if sym_axis is not None and symmetry is None:
                raise RuntimeError(
                    "This mesh was built with a symmetry plane but predates the "
                    "far-field mirroring, so fem_mesh.json has no symmetry_plane/"
                    "symmetry_kind. Only half the Huygens surface can be sampled "
                    "and the pattern would be wrong; re-run build_mesh."
                )
            theta_axis, phi_axis, u_grid, dir_db = compute_pattern(
                e_pos,
                h_pos,
                freq,
                bbox,
                output_path,
                domain_bbox=domain_bbox,
                symmetry=symmetry,
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
    ) -> FEMFarField:
        """
        Evaluate the far field over a grid of angles.

        Matches ``openEMS.nf2ff.nf2ff.CalcNF2FF``. Negative ``theta`` is read
        as ``(|theta|, phi + 180)``, so principal-plane cuts spanning -180 to
        180 degrees work.

        Parameters
        ----------
        output_path : str | Path
            Directory the simulation results were written to.
        freq : float
            Frequency to evaluate the far field at, in Hz.
        theta, phi : NDArray
            Elevation and azimuth angles, in degrees, to evaluate the pattern
            at. The pattern is returned over every combination of the two.
        read_cached, outfile, verbose
            Accepted to match ``openEMS.nf2ff.nf2ff.CalcNF2FF`` but unused:
            results are never written to a file, and each frequency is solved
            only once regardless.

        Returns
        -------
        FEMFarField
            The far field over the requested angles. See
            :class:`FEMFarField`.
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
        return FEMFarField(
            E_norm=np.sqrt(u_vals),
            Dmax=np.array([10.0 ** (dir_db / 10.0)]),
            Prad=np.array([float(p_rad)]),
            P_rad=u_vals,
            theta=np.deg2rad(theta_deg),
            phi=np.deg2rad(phi_deg),
            Ploss=np.array([float(p_loss)]),
        )
