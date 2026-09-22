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
Solve the transverse mode that a wave port launches.

A lumped port drives one constant field direction across a small gap. A
transmission line does not work that way: its cross-section carries a guided
mode whose shape is part of the answer, not something that can be assumed. This
module computes that mode.

For each wave port it cuts the port's cross-section out of the 3D mesh, solves a
2D transverse eigenproblem on it with GetDP, and reports the propagation
constant, the characteristic impedance, and the field profile itself -- written
back out as a Gmsh view that :mod:`~simpleEMS.fem_formulation` hands to the 3D
solve as the port's excitation and as the mode it takes S-parameters against.

One mode solve belongs to one frequency: a line's effective permittivity
disperses, so the sweep re-solves the mode at each of its solve points.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh
import numpy as np
from numpy.typing import NDArray

from .fem_materials import AIR, MU0, PEC, port_region

if TYPE_CHECKING:
    from .fem_geometry import Mesh, PortMesh
    from .fem_backend import Problem


__all__ = [
    "PortMode",
    "PortModeSetup",
    "prepare_port_mode",
    "solve_port_mode",
]

# A mode is guided only if beta is real and positive; the 2D problem also has a
# large gradient null space sitting at beta ~ 0, which this rejects.
_MIN_BETA_FRAC = 1e-3

# Parallel paths swept across the cross-section when measuring the modal
# voltage; see _characteristic_impedance.
_VOLTAGE_PATHS = 41

# A measured impedance outside this band means the voltage path found no
# conductor, not that the line really is that impedance.
_ZC_BOUNDS = (1.0, 1000.0)


@dataclass
class PortMode:
    """
    The transverse mode of one wave port at one frequency.

    Parameters
    ----------
    number : int
        One-based port number the mode belongs to.
    freq : float
        Frequency the mode was solved at, in Hz.
    beta : complex
        Propagation constant in rad/m.
    n_eff : complex
        Modal index ``beta / k0``, i.e. ``sqrt(eps_eff)`` for a quasi-TEM line.
        This is the Robin coefficient the 3D port term is scaled by.
    zc : float
        Characteristic impedance in ohms, by the power-voltage definition
        ``|V|^2 / (2P)``.
    pos_path : str
        Path to the written Gmsh view holding the mode profile, positioned in
        the 3D mesh's own coordinates.
    """

    number: int
    freq: float
    beta: complex
    n_eff: complex
    zc: float
    pos_path: str

    @property
    def eps_eff(self) -> complex:
        """Effective permittivity of the line, ``n_eff**2``."""
        return self.n_eff**2

    @property
    def alpha_np_per_m(self) -> float:
        """Attenuation constant in nepers per metre.

        Under the ``e^{jwt}`` convention a wave travelling in ``+z`` goes as
        ``e^{-j beta z}``, so a lossy line puts the loss in a negative imaginary
        part of ``beta`` and this is ``-Im(beta)``. It is zero for a lossless
        cross-section.
        """
        return max(0.0, -float(self.beta.imag))

    @property
    def alpha_db_per_m(self) -> float:
        """Attenuation constant in dB per metre."""
        return 20.0 / math.log(10.0) * self.alpha_np_per_m


@dataclass
class _Section:
    """The port cross-section, flattened into the xy plane for the 2D solve."""

    msh_path: str
    prop_axis: int  # 0/1/2: the axis the line runs along (the face normal)
    plane_at: float  # position of the port plane along `prop_axis`, in metres
    axes: tuple[int, int]  # the 3D axes that became local x and y
    nodes: NDArray  # (N, 2) local xy coordinates, indexed by `node_index`
    node_index: dict[int, int]  # gmsh node tag -> row of `nodes`
    tris: NDArray  # (M, 3) rows of `nodes`
    areas: NDArray  # (M,) triangle areas in m^2
    eps_r: NDArray  # (M,) relative permittivity of each triangle
    bounds: tuple[float, float, float, float] = field(default=(0.0, 0.0, 0.0, 0.0))


def _local_axes(prop_axis: int) -> tuple[int, int]:
    """The two 3D axes that become local x and y, right-handed about ``prop_axis``.

    Parameters
    ----------
    prop_axis : int
        Axis the port face is normal to: ``0``, ``1``, or ``2``.

    Returns
    -------
    tuple[int, int]
        The 3D axis indices used as local x and y.
    """
    # (p+1, p+2) mod 3 keeps u x v = +p for every p, so the flattened frame has
    # the same handedness as the 3D one and the mode's sign survives the trip.
    return (prop_axis + 1) % 3, (prop_axis + 2) % 3


def _tri_edges(tri: NDArray) -> list[frozenset]:
    """The three edges of a triangle, as node-tag pairs."""
    a, b, c = tri
    return [frozenset((a, b)), frozenset((b, c)), frozenset((c, a))]


def extract_cross_section(
    msh_path: str | Path,
    port: PortMesh,
    prop_axis: int,
    diel_bboxes: dict[str, tuple],
    diel_regions: dict[str, int],
    out_path: str | Path,
) -> _Section:
    """
    Cut one port's cross-section out of the 3D mesh as a standalone 2D mesh.

    The cross-section is taken straight from the 3D mesh rather than re-meshed,
    so its triangulation is node-identical with the port face the 3D solve uses
    and the mode needs no interpolation to line up with it. It is written out
    flattened into the xy plane, which is the frame
    :func:`~simpleEMS.fem_formulation.write_mode_problem`'s ``Form1P`` basis
    assumes.

    Parameters
    ----------
    msh_path : str | Path
        Path to the 3D ``.msh`` file.
    port : PortMesh
        The port to extract, supplying its region tag.
    prop_axis : int
        Axis the port face is normal to: ``0``, ``1``, or ``2``.
    diel_bboxes : dict[str, tuple]
        Extents of each dielectric solid, keyed by name, used to give each
        triangle its permittivity.
    diel_regions : dict[str, int]
        Region tag of each dielectric, keyed by the same names.
    out_path : str | Path
        Path to write the 2D ``.msh`` file to.

    Returns
    -------
    _Section
        The extracted cross-section and the transform back to 3D.

    Raises
    ------
    RuntimeError
        If the port's region holds no surface elements.
    """
    ax_u, ax_v = _local_axes(prop_axis)
    tag = port_region(port.number)

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh_path))

        coord_of: dict[int, NDArray] = {}
        port_tris: list[tuple[int, int, int]] = []
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, tag):
            types, _tags, nodes = gmsh.model.mesh.getElements(2, ent)
            for etype, enodes in zip(types, nodes, strict=True):
                if etype != 2:  # 3-node triangle
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, 3)
                port_tris.extend(map(tuple, arr))
        if not port_tris:
            raise RuntimeError(
                f"port {port.number} region {tag} holds no triangles in {msh_path}"
            )

        # Every edge of every PEC element. The 3D mesh is conforming, so a
        # conductor cutting the port plane shares its edges with the port face
        # exactly -- no geometric tolerance needed to find the trace or ground.
        pec_edges: set[frozenset] = set()
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, PEC):
            types, _tags, nodes = gmsh.model.mesh.getElements(2, ent)
            for etype, enodes in zip(types, nodes, strict=True):
                if etype != 2:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, 3)
                for tri in arr:
                    pec_edges.update(_tri_edges(tri))

        used = sorted({n for tri in port_tris for n in tri})
        for ntag in used:
            c, _p, _d, _t = gmsh.model.mesh.getNode(ntag)
            coord_of[ntag] = np.asarray(c, dtype=float)
    finally:
        gmsh.finalize()

    node_index = {ntag: i for i, ntag in enumerate(used)}
    xyz = np.array([coord_of[n] for n in used])
    nodes2d = np.column_stack([xyz[:, ax_u], xyz[:, ax_v]])
    plane_at = float(np.mean(xyz[:, prop_axis]))

    tris = np.array([[node_index[n] for n in tri] for tri in port_tris], dtype=np.int64)
    p0, p1, p2 = nodes2d[tris[:, 0]], nodes2d[tris[:, 1]], nodes2d[tris[:, 2]]
    areas = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )

    # A triangle's permittivity is that of the dielectric whose extents contain
    # its centroid; anything outside every dielectric is air. The port sheet is
    # planar and the dielectrics are boxes, so this is exact.
    centroid3 = (xyz[tris[:, 0]] + xyz[tris[:, 1]] + xyz[tris[:, 2]]) / 3.0
    region = np.full(len(tris), AIR, dtype=np.int64)
    eps_r = np.ones(len(tris))
    for name, bb in sorted(diel_bboxes.items()):
        if name not in diel_regions:
            continue
        tol = 1e-9
        inside = np.ones(len(tris), dtype=bool)
        for i in range(3):
            inside &= centroid3[:, i] >= bb[i] - tol
            inside &= centroid3[:, i] <= bb[i + 3] + tol
        region[inside] = diel_regions[name]

    # Edges on PEC: the conductors cut by the plane, plus the outline of the
    # cross-section itself (the mode box wall), which is a boundary edge -- one
    # that only one triangle owns.
    edge_count: dict[frozenset, int] = {}
    for tri in port_tris:
        for e in _tri_edges(np.asarray(tri)):
            edge_count[e] = edge_count.get(e, 0) + 1
    wall: set[frozenset] = {e for e, n in edge_count.items() if n == 1}
    wall |= {e for e in edge_count if e in pec_edges}

    _write_section_msh(out_path, used, nodes2d, tris, region, wall, node_index)

    return _Section(
        msh_path=str(out_path),
        prop_axis=prop_axis,
        plane_at=plane_at,
        axes=(ax_u, ax_v),
        nodes=nodes2d,
        node_index=node_index,
        tris=tris,
        areas=areas,
        eps_r=eps_r if eps_r.size else eps_r,
        bounds=(
            float(nodes2d[:, 0].min()),
            float(nodes2d[:, 1].min()),
            float(nodes2d[:, 0].max()),
            float(nodes2d[:, 1].max()),
        ),
    )


def _write_section_msh(
    out_path: str | Path,
    node_tags: list[int],
    nodes2d: NDArray,
    tris: NDArray,
    region: NDArray,
    wall: set[frozenset],
    node_index: dict[int, int],
) -> None:
    """Write the flattened cross-section as a version-2.2 Gmsh mesh.

    The material regions carry the same tags as the 3D mesh, so both problem
    files assign permittivity the same way; this is a separate file, so the
    numbering cannot collide.

    Parameters
    ----------
    out_path : str | Path
        Path to write to.
    node_tags : list[int]
        Original gmsh node tags, kept so the mesh stays traceable to the 3D one.
    nodes2d : NDArray
        ``(N, 2)`` local coordinates.
    tris : NDArray
        ``(M, 3)`` triangles, as rows of ``nodes2d``.
    region : NDArray
        ``(M,)`` region tag of each triangle.
    wall : set[frozenset]
        Edges that are perfect conductors, as original node-tag pairs.
    node_index : dict[int, int]
        Original node tag to row of ``nodes2d``.
    """
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat"]
    lines.append("$Nodes")
    lines.append(str(len(node_tags)))
    for i in range(len(node_tags)):
        # float() first: repr of a numpy scalar is "np.float64(...)", which
        # the mesh parser silently reads as zero nodes
        lines.append(f"{i + 1} {float(nodes2d[i, 0])!r} {float(nodes2d[i, 1])!r} 0")
    lines.append("$EndNodes")

    elems: list[str] = []
    n = 0
    for e in sorted(wall, key=lambda s: sorted(s)):
        a, b = sorted(e)
        n += 1
        elems.append(f"{n} 1 2 {PEC} {PEC} {node_index[a] + 1} {node_index[b] + 1}")
    for tri, rid in zip(tris, region, strict=True):
        n += 1
        elems.append(f"{n} 2 2 {rid} {rid} {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}")
    lines.append("$Elements")
    lines.append(str(n))
    lines.extend(elems)
    lines.append("$EndElements")
    Path(out_path).write_text("\n".join(lines) + "\n")


def read_eigenvalues(res_path: str | Path) -> list[complex]:
    """
    Read the eigenvalues out of a GetDP ``.res`` file.

    GetDP stores one ``$Solution`` block per eigenpair and puts the eigenvalue
    in its header, in the slot a time-domain run would use for the time. That
    is the eigenvalue's only machine-readable home -- the solver prints it to
    stdout only at high verbosity.

    Parameters
    ----------
    res_path : str | Path
        Path to the ``.res`` file.

    Returns
    -------
    list[complex]
        One eigenvalue per solution stored, in the order GetDP wrote them.
    """
    text = Path(res_path).read_text()
    out: list[complex] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("$Solution"):
            continue
        if i + 1 >= len(lines):
            break
        parts = lines[i + 1].split()
        if len(parts) >= 3:
            out.append(complex(float(parts[1]), float(parts[2])))
    return out


_VT_RE = re.compile(r"VT\(([^)]*)\)\{([^}]*)\}")


def read_pos_steps(pos_path: str | Path) -> tuple[NDArray, NDArray]:
    """
    Read a parsed Gmsh vector-triangle view.

    Parameters
    ----------
    pos_path : str | Path
        Path to the ``.pos`` file.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(coords, values)`` -- ``coords`` is ``(M, 3, 3)``, the xyz of each
        triangle's three nodes; ``values`` is ``(M, S, 3, 3)``, the vector at
        each node for each of the ``S`` steps. GetDP writes a complex field as
        consecutive real and imaginary steps, so mode ``k`` is steps ``2k`` and
        ``2k + 1``.
    """
    text = Path(pos_path).read_text()
    if text.lstrip().startswith("$MeshFormat"):
        raise RuntimeError(
            f"{pos_path} is a mesh-based view, not a parsed one. The mode "
            "problem's Print needs 'Format GmshParsed'; regenerate it with "
            "fem_formulation.write_mode_problem."
        )
    coords, vals = [], []
    for m in _VT_RE.finditer(text):
        c = np.fromstring(m.group(1), sep=",")
        v = np.fromstring(m.group(2), sep=",")
        coords.append(c.reshape(3, 3))
        vals.append(v)
    if not coords:
        return np.zeros((0, 3, 3)), np.zeros((0, 0, 3, 3))
    arr = np.array(vals)
    nsteps = arr.shape[1] // 9
    return np.array(coords), arr.reshape(len(coords), nsteps, 3, 3)


def _integrate_sq(values: NDArray, areas: NDArray) -> float:
    """Integral of ``|v|^2`` over a linear field on triangles.

    Exact for the product of two linear interpolants:
    ``int f g* = (A/12) (sum f_i g_i* + (sum f_i)(sum g_i)*)``.

    Parameters
    ----------
    values : NDArray
        ``(M, 3, 3)`` complex vector at each triangle's three nodes.
    areas : NDArray
        ``(M,)`` triangle areas.

    Returns
    -------
    float
        The integral.
    """
    per_node = np.sum(values * np.conj(values), axis=2).real  # (M, 3)
    summed = np.sum(values, axis=1)  # (M, 3)
    sq_of_sum = np.sum(summed * np.conj(summed), axis=1).real  # (M,)
    return float(np.sum(areas / 12.0 * (np.sum(per_node, axis=1) + sq_of_sum)))


def _tri_areas(coords: NDArray) -> NDArray:
    """Areas of the triangles in a ``(M, 3, 3)`` coordinate array (local frame)."""
    p0, p1, p2 = coords[:, 0, :2], coords[:, 1, :2], coords[:, 2, :2]
    return 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )


def _guided_modes(betas: list[complex], k0: float, n_max: float) -> list[int]:
    """
    The guided modes among the computed eigenpairs, largest ``beta`` first.

    Parameters
    ----------
    betas : list[complex]
        Propagation constants reported by the eigensolver.
    k0 : float
        Free-space wavenumber, in rad/m.
    n_max : float
        Largest modal index physically available, ``sqrt(eps_r_max)``.

    Returns
    -------
    list[int]
        Indices into ``betas``, ordered by descending propagation constant.
    """
    # Two things have to be filtered out: the gradient null space, which is
    # large and sits at beta ~ 0, and the modes below cutoff, whose beta is
    # imaginary. What is left travels.
    keep = []
    for i, b in enumerate(betas):
        re, im = b.real, abs(b.imag)
        if re <= _MIN_BETA_FRAC * k0 or im > 0.5 * abs(re):
            continue
        # beta above sqrt(eps_max)*k0 is not a mode of this cross-section
        if re > 1.05 * n_max * k0:
            continue
        keep.append(i)
    keep.sort(key=lambda i: betas[i].real, reverse=True)
    return keep


def _select_mode(
    betas: list[complex],
    k0: float,
    n_max: float,
    index: int = 0,
    target_eps_eff: float | None = None,
) -> int:
    """
    Pick which guided mode the port runs in.

    A single-conductor line such as microstrip has one quasi-TEM mode and
    neither argument is needed. A cross-section with more than one conductor
    above the ground plane has several -- a conductor-backed coplanar waveguide
    carries the CPW mode the line is meant to run in *and* a microstrip-like
    mode between the trace and the backside ground -- and the one with the
    largest ``beta`` need not be the wanted one.

    ``target_eps_eff`` is the better way to say which: it names the mode by a
    property of the line, so it keeps meaning the same mode as the frequency
    moves and the mesh changes. ``index`` counts from the largest propagation
    constant, which is only stable as long as every mode stays resolved -- a
    coarser mesh or a lower frequency can drop one and silently renumber the
    rest.

    Parameters
    ----------
    betas : list[complex]
        Propagation constants reported by the eigensolver.
    k0 : float
        Free-space wavenumber, in rad/m.
    n_max : float
        Largest modal index physically available, ``sqrt(eps_r_max)``.
    index : int
        Which guided mode to take, largest ``beta`` first. Default ``0``.
        Ignored when ``target_eps_eff`` is given.
    target_eps_eff : float, optional
        Effective permittivity to match; the guided mode closest to it is
        taken. Default ``None`` (select by ``index``).

    Returns
    -------
    int
        Index into ``betas`` of the chosen mode.

    Raises
    ------
    RuntimeError
        If no eigenpair is a guided mode, or if there are fewer guided modes
        than ``index`` asks for.
    """
    guided = _guided_modes(betas, k0, n_max)
    if not guided:
        raise RuntimeError(
            "the port mode solve found no guided mode. Computed beta values: "
            f"{[f'{b:.4g}' for b in betas]} (k0 = {k0:.4g} rad/m, "
            f"expected beta between {_MIN_BETA_FRAC * k0:.4g} and "
            f"{n_max * k0:.4g}). Widen the port cross-section, refine the mesh, "
            "or raise FEM_port_mode_modes."
        )
    if target_eps_eff is not None:
        return min(
            guided, key=lambda i: abs((betas[i] / k0).real ** 2 - target_eps_eff)
        )
    if index >= len(guided):
        found = [f"{(betas[i] / k0).real ** 2:.4g}" for i in guided]
        raise RuntimeError(
            f"port_mode_index {index} was asked for but only {len(guided)} "
            f"guided mode(s) were found, with eps_eff {found}. Raise "
            "FEM_port_mode_modes so the eigensolver computes more, or name "
            "the mode by its effective permittivity with FEM_port_mode_eps_eff, "
            "which does not depend on how many modes happen to be resolved."
        )
    return guided[index]


def _locate_and_sample(coords: NDArray, values: NDArray, pts: NDArray) -> NDArray:
    """
    Sample a piecewise-linear field on triangles at arbitrary points.

    Parameters
    ----------
    coords : NDArray
        ``(M, 3, 3)`` triangle node coordinates in the local frame.
    values : NDArray
        ``(M, 3, 3)`` complex vector at each node.
    pts : NDArray
        ``(P, 2)`` sample points in the local frame.

    Returns
    -------
    NDArray
        ``(P, 3)`` complex vector at each point; zero where a point falls
        outside every triangle.
    """
    p0 = coords[:, 0, :2]
    v0 = coords[:, 1, :2] - p0
    v1 = coords[:, 2, :2] - p0
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)

    out = np.zeros((len(pts), 3), dtype=complex)
    for i, p in enumerate(pts):
        d = p[None, :] - p0
        b1 = (d[:, 0] * v1[:, 1] - v1[:, 0] * d[:, 1]) / den
        b2 = (v0[:, 0] * d[:, 1] - d[:, 0] * v0[:, 1]) / den
        b0 = 1.0 - b1 - b2
        tol = -1e-9
        hit = np.flatnonzero((b0 >= tol) & (b1 >= tol) & (b2 >= tol))
        if hit.size == 0:
            continue
        t = hit[0]
        out[i] = b0[t] * values[t, 0] + b1[t] * values[t, 1] + b2[t] * values[t, 2]
    return out


def _modal_voltage(
    coords: NDArray,
    values: NDArray,
    section: PortModeSetup,
    direction: str,
    samples: int = 201,
) -> complex:
    """
    The signed modal voltage: the largest potential swing across the section.

    This is the ground-to-conductor voltage, found without being told which
    conductor is the signal one: a conductor is an equipotential, so the running
    integral of ``-E.dl`` flattens out on it and turns around after it.

    The sign is kept, and it is what makes the mode's own sign reproducible.
    The path is chosen by the geometry rather than by wherever the field happens
    to peak, so the voltage it reads varies smoothly with frequency -- see
    :func:`solve_port_mode`, which uses it to orient the mode.

    Parameters
    ----------
    coords, values : NDArray
        The mode profile, as returned by :func:`read_pos_steps`, in the local
        frame and already power-normalised.
    section : PortModeSetup
        The prepared port, supplying the local frame and its extents.
    direction : str
        Axis the voltage path runs along: ``"x"``, ``"y"``, or ``"z"``.
    samples : int
        Number of points along the path. Default ``201``.

    Returns
    -------
    complex
        The voltage in volts, or ``0`` if the path lies along the port normal,
        where there is no transverse voltage to integrate.
    """
    axis3 = {"x": 0, "y": 1, "z": 2}[direction]
    if axis3 == section.prop_axis:
        return 0j
    local = 0 if axis3 == section.axes[0] else 1
    other = 1 - local

    xmin, ymin, xmax, ymax = section.bounds
    lo = (xmin, ymin)[local]
    hi = (xmax, ymax)[local]
    o_lo = (xmin, ymin)[other]
    o_hi = (xmax, ymax)[other]

    s = np.linspace(lo, hi, samples)
    span = hi - lo
    dl = np.diff(s)

    # Which line across the cross-section to integrate along is not obvious in
    # general. For microstrip it is the one through the trace; for a coplanar
    # waveguide the field lives at the substrate surface, nowhere near the
    # middle of the box. Rather than guess, sweep parallel paths and keep the
    # largest voltage: a conductor is an equipotential, so the path that cuts
    # closest to it reads the full ground-to-conductor swing and the rest read
    # less.
    best = 0j
    for frac in np.linspace(0.02, 0.98, _VOLTAGE_PATHS):
        pts = np.zeros((samples, 2))
        pts[:, local] = s
        pts[:, other] = o_lo + frac * (o_hi - o_lo)
        # Nudge off the boundary so the endpoints land inside a triangle.
        pts[0, local] += 1e-6 * span
        pts[-1, local] -= 1e-6 * span

        e = _locate_and_sample(coords, values, pts)
        # the sampled vector is in the local frame, so the path component is
        # the local index -- not axis3, which numbers the 3D axes
        comp = e[:, local]
        # running V(s) = -int E.dl, trapezoid
        seg = -0.5 * (comp[:-1] + comp[1:]) * dl
        running = np.concatenate([[0.0 + 0j], np.cumsum(seg)])
        peak = running[int(np.argmax(np.abs(running)))]
        if abs(peak) > abs(best):
            best = complex(peak)
    return best


def _characteristic_impedance(
    coords: NDArray,
    values: NDArray,
    section: PortModeSetup,
    direction: str,
    samples: int = 201,
) -> float:
    """
    Characteristic impedance of a power-normalised mode, ``|V|^2 / (2P)``.

    Parameters
    ----------
    coords, values : NDArray
        The mode profile, in the local frame and already power-normalised.
    section : PortModeSetup
        The prepared port, supplying the local frame and its extents.
    direction : str
        Axis the voltage path runs along: ``"x"``, ``"y"``, or ``"z"``.
    samples : int
        Number of points along the path. Default ``201``.

    Returns
    -------
    float
        The impedance in ohms, or ``0.0`` if the path lies along the port
        normal, where there is no transverse voltage to integrate.
    """
    voltage = abs(_modal_voltage(coords, values, section, direction, samples))
    return voltage * voltage / 2.0


def _write_mode_view(
    out_path: str | Path,
    coords: NDArray,
    values: NDArray,
    section: PortModeSetup,
    name: str,
) -> None:
    """
    Write the mode as a Gmsh view in the 3D mesh's own coordinates.

    Written as consecutive real and imaginary steps, which is how GetDP stores
    a complex field and how ``ComplexVectorField[XYZ[]]`` expects to read one
    back.

    Parameters
    ----------
    out_path : str | Path
        Path to write to.
    coords : NDArray
        ``(M, 3, 3)`` triangle node coordinates in the local frame.
    values : NDArray
        ``(M, 3, 3)`` complex vector at each node, in the local frame.
    section : PortModeSetup
        The prepared port, supplying the transform back to 3D.
    name : str
        View name.
    """
    ax_u, ax_v = section.axes
    p = section.prop_axis

    xyz = np.zeros((len(coords), 3, 3))
    xyz[:, :, ax_u] = coords[:, :, 0]
    xyz[:, :, ax_v] = coords[:, :, 1]
    xyz[:, :, p] = section.plane_at

    vec = np.zeros((len(coords), 3, 3), dtype=complex)
    vec[:, :, ax_u] = values[:, :, 0]
    vec[:, :, ax_v] = values[:, :, 1]
    # the longitudinal component is normal to the port face, so the tangential
    # port terms drop it anyway; leaving it out keeps the view purely modal

    out = [f'View "{name}" {{']
    for t in range(len(coords)):
        c = ",".join(repr(float(x)) for x in xyz[t].reshape(-1))
        re = ",".join(repr(float(x)) for x in vec[t].real.reshape(-1))
        im = ",".join(repr(float(x)) for x in vec[t].imag.reshape(-1))
        out.append(f"VT({c}){{{re},{im}}};")
    out.append("TIME{0,1};")
    out.append("};")
    Path(out_path).write_text("\n".join(out) + "\n")


# GetDP's Arpack driver reads its settings from an eigen.par beside the run and
# prompts on stdin when there is not one, which would hang an unattended sweep.
# These are its own defaults; the trailing comment is the block it writes itself.
_EIGEN_PAR = """0.0001
0
50
/*
   The numbers above are the parameters for the numerical
   eigenvalue problem:

   prec = aimed accuracy for eigenvectors (default=1.e-4)
   reortho = reorthogonalisation of Krylov basis: yes=1, no=0 (default=0)
   size = size of the Krylov basis
*/
"""


def _write_eigen_par(workdir: Path) -> None:
    """Write Arpack's parameter file, so the solver never stops to ask."""
    par = workdir / "eigen.par"
    if par.exists() and par.stat().st_size > 0:
        return
    par.write_text(_EIGEN_PAR)


@dataclass
class PortModeSetup:
    """
    Everything the per-frequency mode solve needs, and nothing that changes with
    frequency.

    Cutting the cross-section out of the 3D mesh and writing its problem file
    are geometry work, so they happen once when the mesh is built. The sweep
    then re-solves the mode at each of its frequencies from this, which is
    plain data and travels through ``fem_mesh.json`` like the rest of the
    backend's stage-to-stage state.

    Parameters
    ----------
    number : int
        One-based port number.
    pro_path : str
        Path to the 2D mode problem file.
    msh_path : str
        Path to the flattened cross-section mesh.
    direction : str
        Axis the port's voltage is measured along.
    prop_axis : int
        Axis the port face is normal to.
    plane_at : float
        Position of the port plane along ``prop_axis``, in metres.
    axes : tuple[int, int]
        The 3D axes that became local x and y.
    bounds : tuple[float, float, float, float]
        Extents of the cross-section in the local frame, as
        ``(xmin, ymin, xmax, ymax)``.
    eps_max : float
        Largest relative permittivity in the problem, which sets where the
        eigensolver looks.
    z0 : float
        Reference impedance in ohms, used as a fallback if the mode's own
        impedance cannot be measured.
    nmodes : int
        Eigenpairs to compute. Default ``6``.
    mode_index : int
        Which guided mode the port runs in, largest ``beta`` first. Default
        ``0``.
    zc_override : float, optional
        Characteristic impedance in ohms to report instead of the one measured
        from the mode. Default ``None`` (measure it).
    target_eps_eff : float, optional
        Effective permittivity naming which guided mode the port runs in.
        Default ``None`` (select by ``mode_index``).
    """

    number: int
    pro_path: str
    msh_path: str
    direction: str
    prop_axis: int
    plane_at: float
    axes: tuple[int, int]
    bounds: tuple[float, float, float, float]
    eps_max: float
    z0: float
    nmodes: int = 6
    mode_index: int = 0
    zc_override: float | None = None
    target_eps_eff: float | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy."""
        return {
            "number": self.number,
            "pro_path": self.pro_path,
            "msh_path": self.msh_path,
            "direction": self.direction,
            "prop_axis": self.prop_axis,
            "plane_at": self.plane_at,
            "axes": list(self.axes),
            "bounds": list(self.bounds),
            "eps_max": self.eps_max,
            "z0": self.z0,
            "nmodes": self.nmodes,
            "mode_index": self.mode_index,
            "zc_override": self.zc_override,
            "target_eps_eff": self.target_eps_eff,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PortModeSetup:
        """Rebuild from :meth:`to_dict` output.

        Parameters
        ----------
        d : dict
            The serialised setup.

        Returns
        -------
        PortModeSetup
            The rebuilt setup.
        """
        return cls(
            number=int(d["number"]),
            pro_path=str(d["pro_path"]),
            msh_path=str(d["msh_path"]),
            direction=str(d["direction"]),
            prop_axis=int(d["prop_axis"]),
            plane_at=float(d["plane_at"]),
            axes=(int(d["axes"][0]), int(d["axes"][1])),
            bounds=tuple(float(x) for x in d["bounds"]),  # type: ignore[arg-type]
            eps_max=float(d["eps_max"]),
            z0=float(d["z0"]),
            nmodes=int(d.get("nmodes", 6)),
            mode_index=int(d.get("mode_index", 0)),
            zc_override=(
                None if d.get("zc_override") is None else float(d["zc_override"])
            ),
            target_eps_eff=(
                None if d.get("target_eps_eff") is None else float(d["target_eps_eff"])
            ),
        )


def prepare_port_mode(
    problem: Problem,
    mesh: Mesh,
    port: PortMesh,
    workdir: str | Path,
) -> PortModeSetup:
    """
    Do a wave port's geometry work once, ahead of the sweep.

    Cuts the port's cross-section out of the 3D mesh and writes the 2D
    eigenproblem for it. Neither depends on frequency, so neither is repeated
    per solve point.

    Parameters
    ----------
    problem : Problem
        The problem being solved.
    mesh : Mesh
        The generated 3D mesh.
    port : PortMesh
        The wave port to prepare.
    workdir : str | Path
        Directory to write the cross-section mesh and problem file into.

    Returns
    -------
    PortModeSetup
        The prepared setup, ready for :func:`solve_port_mode`.
    """
    from . import fem_formulation

    workdir = Path(workdir).absolute()
    section = extract_cross_section(
        mesh.msh_path,
        port,
        {"x": 0, "y": 1, "z": 2}[port.prop_dir],
        getattr(mesh, "dielectric_bboxes", {}) or {},
        mesh.dielectric_regions,
        workdir / f"{problem.name}_port_{port.number}.msh",
    )
    pro_path = fem_formulation.write_mode_problem(problem, mesh, port.number, workdir)
    eps_max = max(
        [s.dielectric.eps_r for s in problem.dielectrics() if s.dielectric] + [1.0]
    )
    return PortModeSetup(
        number=port.number,
        pro_path=pro_path,
        msh_path=section.msh_path,
        direction=port.direction,
        prop_axis=section.prop_axis,
        plane_at=section.plane_at,
        axes=section.axes,
        bounds=section.bounds,
        eps_max=float(eps_max),
        z0=float(port.z0),
        nmodes=int(problem.port_mode_modes),
        mode_index=int(problem.port_mode_index),
        zc_override=problem.port_mode_zc,
        target_eps_eff=problem.port_mode_eps_eff,
    )


def solve_port_mode(
    setup: PortModeSetup,
    freq: float,
    workdir: str | Path,
    *,
    nmodes: int | None = None,
    verbose: bool = True,
) -> PortMode:
    """
    Solve one wave port's transverse mode at one frequency.

    Runs the 2D eigenproblem prepared by :func:`prepare_port_mode`, picks the
    fundamental guided mode out of the spectrum, and writes it back as a Gmsh
    view positioned in the 3D mesh's own coordinates -- which is what the 3D
    problem file reads with ``GmshRead`` and uses both as the port's excitation
    and as the mode it takes S-parameters against.

    The mode is normalised to carry one watt, so its characteristic impedance
    falls straight out of its voltage and the reported ``V``/``I`` are on a
    physical scale.

    Parameters
    ----------
    setup : PortModeSetup
        The prepared port, from :func:`prepare_port_mode`.
    freq : float
        Frequency to solve at, in Hz.
    workdir : str | Path
        Working directory; the view lands in its ``output`` subdirectory,
        beside everything else the solver writes.
    nmodes : int, optional
        Number of eigenpairs to compute, overriding ``setup.nmodes``. Default
        ``None`` (use the setup's).
    verbose : bool
        Report the solve's progress and how long it took. Default ``True``;
        a sweep passes ``False``, having its own line for the frequency this
        mode belongs to.

    Returns
    -------
    PortMode
        The solved mode.

    Raises
    ------
    RuntimeError
        If the eigensolver reports no guided mode, writes no eigenvalues, or
        writes no profile.
    """
    from . import fem_solver

    workdir = Path(workdir).absolute()
    outdir = workdir / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    nmodes = setup.nmodes if nmodes is None else nmodes
    n_max = math.sqrt(setup.eps_max)
    k0 = 2 * math.pi * float(freq) / 299792458.0
    # The shift has to sit near beta^2. A shift near zero converges onto the
    # gradient null space, which is large and sits at beta ~ 0, and returns
    # nothing usable.
    shift = (n_max * k0) ** 2

    _write_eigen_par(workdir)

    res_path = workdir / f"{Path(setup.pro_path).stem}.res"
    pos_path = outdir / f"et_{setup.number}.pos"
    for stale in (res_path, pos_path):
        stale.unlink(missing_ok=True)

    fem_solver.run_getdp(
        setup.pro_path,
        setup.msh_path,
        workdir,
        {"FREQ": freq, "NMODES": nmodes, "SHIFT_RE": shift},
        "Get_Mode",
        resolution="ModeAnalysis",
        label=f"port {setup.number} mode @ {freq / 1e9:.4g} GHz",
        verbose=verbose,
    )

    betas = read_eigenvalues(res_path)
    if not betas:
        raise RuntimeError(f"the port mode solve wrote no eigenvalues to {res_path}")
    k = _select_mode(betas, k0, n_max, setup.mode_index, setup.target_eps_eff)
    beta = betas[k]

    coords, steps = read_pos_steps(pos_path)
    nsteps = steps.shape[1] if coords.size else 0
    if coords.size == 0 or nsteps < 2 * k + 2:
        raise RuntimeError(
            f"{pos_path} holds {nsteps} step(s); mode {k} needs steps "
            f"{2 * k} and {2 * k + 1}"
        )
    values = steps[:, 2 * k, :, :] + 1j * steps[:, 2 * k + 1, :, :]

    # The eigenvector carries an arbitrary complex scale and phase. Rotate the
    # dominant component onto the real axis first, so a lossless mode comes out
    # essentially real and its sign is reproducible from frequency to frequency.
    flat = values.reshape(-1)
    pivot = flat[int(np.argmax(np.abs(flat)))]
    if abs(pivot) > 0:
        values = values * (abs(pivot) / pivot)

    # Normalise to one watt: for a mode with H = (beta/(omega mu0)) z^ x E the
    # axial Poynting flux is (Re[beta] / (2 omega mu0)) * int |E|^2.
    areas = _tri_areas(coords)
    integral = _integrate_sq(values, areas)
    omega = 2 * math.pi * float(freq)
    flux = beta.real / (2.0 * omega * MU0)
    if integral <= 0 or flux <= 0:
        raise RuntimeError(
            f"port {setup.number}: the mode carries no power "
            f"(int|E|^2 = {integral:.4g}, beta = {beta:.4g})"
        )
    values = values * math.sqrt(1.0 / (flux * integral))

    # The pivot above fixes the phase but not the sign: which component of the
    # eigenvector is largest can change from one solve frequency to the next,
    # and when it does the whole mode flips. That puts a 180 degree step in the
    # middle of the sweep, which the rational fit then interpolates across. So
    # the sign is taken from the modal voltage instead -- its path is set by the
    # geometry, not by where the field peaks, so it varies smoothly with
    # frequency. The convention is the physical one: the signal conductor sits
    # at positive potential.
    v_mode = _modal_voltage(coords, values, setup, setup.direction)
    if v_mode.real < 0.0:
        values = -values

    if setup.zc_override is not None:
        zc = float(setup.zc_override)
    else:
        zc = _characteristic_impedance(coords, values, setup, setup.direction)
    if not math.isfinite(zc) or not _ZC_BOUNDS[0] <= zc <= _ZC_BOUNDS[1]:
        # Better to reference the S-parameters to what the user asked for than
        # to a number the voltage path plainly did not measure.
        zc = setup.z0

    view = outdir / f"mode_{setup.number}.pos"
    _write_mode_view(view, coords, values, setup, f"mode_{setup.number}")

    return PortMode(
        number=setup.number,
        freq=float(freq),
        beta=beta,
        n_eff=beta / k0,
        zc=float(zc),
        pos_path=str(view),
    )
