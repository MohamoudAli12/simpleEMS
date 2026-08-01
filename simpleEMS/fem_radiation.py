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
far field. :class:`FEMNF2FF` wraps that computation behind the same
``CalcNF2FF(output_path, freq, theta, phi, ...)`` interface that
:class:`openEMS.nf2ff.nf2ff` exposes, returning a :class:`FEMFarField` with the
attributes (``E_norm``, ``Dmax``, ``Prad``, ``P_rad``, ``theta``, ``phi``) that
the existing :class:`~simpleEMS.sim_tools.SimTools` 2D/3D radiation plots read.
That way the FDTD and FEM backends share the same plotting code unchanged.
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

# A quad counts as lying in the symmetry plane if every node is this close to
# it. CutBox places nodes exactly on the box faces, so the tolerance only has
# to absorb float noise.
_PLANE_TOL = 1e-12


@dataclass
class FEMFarField:
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
    Transform near fields to a far-field intensity pattern on a regular grid.

    Parameters
    ----------
    e_pos, h_pos : str | Path
        Paths to the GetDP ``e.pos``/``h.pos`` field views.
    freq : float
        Frequency in Hz.
    bbox : tuple
        Structure extents ``(xmin, ymin, zmin, xmax, ymax, zmax)`` in metres.
    workdir : str | Path
        Directory for the intermediate NearToFarField output.
    domain_bbox : tuple | None
        Meshed E/H field extents (the air/PML interface, from
        ``fem_mesh.json``'s ``domain_bbox``). The Huygens box is placed
        ``safety_frac`` of the way from ``bbox`` to this boundary on each
        axis, so ``CutBox`` always samples inside the mesh. If ``None``
        (older ``fem_mesh.json`` without the field), falls back to
        ``margin_frac`` of the largest structure dimension -- which can
        place the box outside the mesh if the FEM air padding is smaller,
        silently zeroing the far field (``CutBox``/``OctreePost`` return 0
        for points outside every element, unlike ``gmsh.view.probe``).
    nphi, ntheta : int
        Far-field angular sampling (azimuth, elevation). Defaults ``72`` and
        ``36``. The ``NearToFarField`` plugin is a triple loop over
        ``nphi * ntheta * num_surface_elements`` (one surface integral per
        requested direction), so cost scales with their product. Every angle
        ``SimTools`` ever requests -- even a 0.1 deg sweep -- is interpolated
        from this one grid
        (:class:`~scipy.interpolate.RegularGridInterpolator` in
        :meth:`FEMNF2FF.CalcNF2FF`), so raising these buys smoother
        interpolation of an already-smooth pattern, not more physical
        information; the near field is only resolved at the FEM mesh's own
        element density regardless.
    npts : tuple[int, int, int]
        CutBox sampling density on each axis of the Huygens box. Default
        ``(14, 14, 14)``. Determines ``num_surface_elements`` above
        (``~6*(npts-1)**2`` boundary quads), so it multiplies directly into
        the ``NearToFarField`` cost.
    margin_frac : float
        Fallback Huygens-box margin as a fraction of the largest structure
        dimension, used only when ``domain_bbox`` is ``None``. Default
        ``0.15``.
    safety_frac : float
        Fraction of the per-axis gap between ``bbox`` and ``domain_bbox``
        used for the Huygens-box margin. Default ``0.6``.
    symmetry : tuple[int, float, str] | None
        ``(axis, plane, kind)`` when only half the structure was meshed --
        the mirrored axis, the plane coordinate in metres, and ``"pec"`` or
        ``"pmc"``. The Huygens box is then closed on the plane and completed
        by reflection (see :func:`_mirror_boundary_view`), so the pattern is
        the whole antenna's rather than half of it. Default ``None``.
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
            # Only half the domain is meshed, and `domain_bbox` predates the
            # cut, so the box would otherwise reach across the symmetry plane
            # into empty space -- where CutBox samples zeros. Start it exactly
            # on the plane so the mirrored copy joins it seamlessly.
            axis_i, sym_plane, _kind = symmetry
            lo[axis_i] = sym_plane
        x0, y0, z0 = lo
        x1, y1, z1 = hi
    else:
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
    # The .pro is written in the exp(-i w t) convention (see fem_formulation),
    # so the transform must be told that: NegativeTime selects the plugin's
    # exp(-i w t) branch, whose integrand carries the matching exp(-i k r.r')
    # phase factor. Left at its default of 0 the plugin instead assumes
    # exp(+i w t) and applies the conjugate factor to the equivalent currents,
    # which is simply a different (wrong) pattern -- not a fixed rotation of
    # the right one, since the J/M combination differs between the two
    # branches as well. On the 24 GHz inset-fed patch it moved the peak's
    # azimuth from 270 to 90 degrees and changed the front-to-back ratio by
    # 10 dB, with both still beaming into the hemisphere the ground plane
    # allows. That branch returns |E_inf|^2, an arbitrarily scaled quantity
    # (the plugin's RFar option belongs to the exp(+i w t) branch and is not
    # read here), so the grid below is only meaningful up to a constant --
    # callers normalise by the peak and take the absolute level from the
    # directivity, which is a ratio.
    gmsh.plugin.setNumber("NearToFarField", "NegativeTime", 1)
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
    """Complete a half-model Huygens surface by reflecting it in the symmetry plane.

    ``CutBox`` samples the half domain, so its box is only half a Huygens
    surface and the pattern it yields is that of half an antenna. Image theory
    gives the missing half exactly: reflecting the sampled quads in the
    symmetry plane, with the field parity the wall imposes, reconstructs the
    surface that would have been sampled around the whole structure.

    Three things have to happen together, and getting any one wrong yields a
    plausible but wrong pattern:

    * **Quads on the plane are dropped.** They are interior to the completed
      surface. (For an electric wall they also carry no current: ``M = -n x E``
      vanishes because ``E_tan = 0``, and the two halves' ``J = n x H`` are
      equal and opposite, so they cancel in the sum anyway.)
    * **Node order is reversed.** A reflection reverses orientation, and
      ``NearToFarField`` takes each element's normal from its first three
      nodes (``normal3points``). Left as-is, every mirrored quad would face
      inwards and contribute ``J`` and ``M`` with flipped signs.
    * **Components follow the wall's parity.** At an electric wall the normal
      ``E`` and tangential ``H`` are even and the rest odd; at a magnetic wall
      it is the other way round.

    Parameters
    ----------
    tag : int
        View tag of the half Huygens surface (a ``CutBox`` result).
    axis : int
        Mirrored axis, ``0``/``1``/``2`` for x/y/z.
    plane : float
        Symmetry plane coordinate along ``axis``, in metres.
    kind : str
        ``"pec"`` (electric wall) or ``"pmc"`` (magnetic wall).
    field : str
        ``"E"`` or ``"H"`` -- which parity rule applies.

    Returns
    -------
    int
        View tag of the completed surface: the original quads (minus those on
        the plane) plus their mirror images.
    """
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


def _check_farfield_margin(
    bbox: tuple, domain_bbox: tuple, freq: float, symmetry_axis: int | None
) -> None:
    """Raise if the air padding is too tight for an accurate far field at ``freq``.

    The Huygens surface needs open air of at least a quarter-wavelength on
    every face that isn't a symmetry plane; anything tighter risks sitting in
    the reactive near field, corrupting the near-to-far-field transform. This
    only bites structures meshed with an explicit ``FEM_air_pad_mm`` (the
    default ``air_pad_frac`` sizing already targets ~lambda/4, see
    ``fem_geometry``'s module docstring), so it fires once, at the point a
    far-field pattern is actually requested, rather than at mesh time -- a
    small ``FEM_air_pad_mm`` is fine for non-radiating problems that never
    call ``CalcNF2FF``.
    """
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
    openEMS-``nf2ff``-compatible far-field calculator for the FEM backend.

    Like the openEMS box it is created without an ``output_path``; each
    :meth:`CalcNF2FF` call reads ``fem_mesh.json`` from the ``output_path`` it is
    given (the ``.pro``/``.msh`` paths and structure bbox) and mirrors
    ``openEMS.nf2ff.nf2ff.CalcNF2FF`` so the ``SimTools`` radiation plots work
    unchanged.
    """

    def __init__(self) -> None:
        """Create an empty far-field calculator with no cached patterns."""
        self._cache: dict[tuple, tuple] = {}  # (workdir, freq) -> pattern data

    def _pattern(self, output_path: str, freq: float) -> tuple:
        """Solve fields/power at ``freq`` and build the pattern (cached)."""
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
        Evaluate the far field on a ``theta``/``phi`` grid (degrees in).

        Matches ``openEMS.nf2ff.nf2ff.CalcNF2FF``. ``theta``/``phi`` are in
        degrees (as ``SimTools`` passes them); negative ``theta`` is mapped to
        ``(|theta|, phi + 180)`` so principal-plane cuts spanning -180..180 work.

        Parameters
        ----------
        output_path : str | Path
            FEM output directory containing ``fem_mesh.json`` (written by
            :func:`~simpleEMS.fem_geometry.build_mesh`); also used as the
            solve/pattern cache key together with ``freq``.
        freq : float
            Frequency in Hz to solve and evaluate the far field at.
        theta, phi : NDArray
            Angles in degrees at which to evaluate the pattern; broadcast
            against each other on a ``[theta, phi]`` grid.
        read_cached, outfile, verbose
            Accepted for signature compatibility with
            ``openEMS.nf2ff.nf2ff.CalcNF2FF`` but not used here: results are
            never written to ``outfile``, and caching is handled internally
            (keyed on ``output_path``/``freq``) regardless of
            ``read_cached``.

        Returns
        -------
        FEMFarField
            The far-field result on the requested grid; see
            :class:`FEMFarField` for the attributes and their shapes.
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
