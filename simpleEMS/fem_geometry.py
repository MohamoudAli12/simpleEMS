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
STEP -> tagged 3D mesh for the FEM backend, via the Gmsh (OpenCASCADE) API.

Planar microwave conductors are electrically thin (tens of microns), so
meshing them as 3D volumes forces microscopic elements. Instead every
conductor and port is reduced to a zero-thickness sheet (its footprint face)
that is imprinted into the dielectric/air mesh:

1. Import the STEP solids (converted to metres) and identify each by name.
2. For every metal/port solid, copy its footprint face (largest face, snapped
   to the nearest dielectric surface) and delete the solid.
3. Wrap the structure in an air box padded by ~lambda/4; the outer faces
   become the Silver-Muller absorbing boundary (or PML volumes).
4. ``fragment`` the dielectric volumes + air box with the conductor sheets so
   the sheets become conforming interior faces.
5. Classify each resulting volume back to its origin solid.
6. The outer absorbing boundary is the domain shell minus the PEC/port faces.
7. Assign integer physical groups, refine near conductors, mesh, and write.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import gmsh

from .console import console
from .fem_materials import (
    ABC,
    AIR,
    C0,
    IMPEDANCE,
    PEC,
    PML,
    SYM,
    dielectric_region,
    microstrip_zc,
    port_region,
)

if TYPE_CHECKING:
    from .fem_backend import Problem

__all__ = ["PortMesh", "Mesh", "list_solids", "build_mesh"]


@dataclass
class PortMesh:
    """Geometric description of a meshed port sheet."""

    number: int
    region: int
    direction: str
    z0: float
    gap: float  # electrical gap length along `direction` (m)
    width: float  # transverse width of the port sheet (m)
    center: tuple[float, float, float]
    zc: float = 50.0  # computed line characteristic impedance (port reference)
    eps_eff: float = 1.0  # effective permittivity of the line (for de-embedding)
    port_type: str = "lumped"  # 'lumped' (impedance z0) or 'wave' (matched to Zc)

    @property
    def ref_impedance(self) -> float:
        """Impedance the port references: ``z0`` (lumped) or ``Zc`` (wave)."""
        return self.zc if self.port_type == "wave" else self.z0

    @property
    def sheet_impedance(self) -> float:
        """Surface-impedance-sheet value for the port term."""
        return self.zc if self.port_type == "wave" else self.z0 * self.width / self.gap


@dataclass
class Mesh:
    """A generated FEM mesh plus the region maps shared with the ``.pro``."""

    msh_path: str
    dielectric_regions: dict[str, int]  # solid name -> region id
    air_region: int
    pec_region: int
    port_regions: dict[int, PortMesh]  # port number -> info
    abc_region: int
    boundary: str
    bbox: tuple[float, float, float, float, float, float]  # structure extents (m)
    box_bbox: tuple[float, float, float, float, float, float]  # air-box extents (m)
    lambda_min: float = field(default=0.0)
    impedance_regions: list = field(default_factory=list)  # [(region_id, sigma), ...]
    pml_region: int = 0  # 0 if no PML
    inner_bbox: tuple = ()  # air/PML interface extents (m); PML damping starts here
    pml_thick: float = 0.0  # PML shell thickness (m)
    sym_region: int = 0  # symmetry-plane surface region (0 if none)
    sym_kind: str = ""  # 'pec' or 'pmc'


# ----------------------------
# helpers
# ----------------------------
def _short_name(entity_name: str) -> str:
    """'Shapes/<uuid>/patch_inset/patch_inset' -> 'patch_inset'."""
    return entity_name.rstrip("/").split("/")[-1] if entity_name else ""


def _bbox_inside(inner: tuple, outer: tuple, tol: float) -> bool:
    """True if bbox ``inner`` is contained in bbox ``outer`` (within ``tol``)."""
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] >= outer[2] - tol
        and inner[3] <= outer[3] + tol
        and inner[4] <= outer[4] + tol
        and inner[5] <= outer[5] + tol
    )


def _bbox_volume(bb: tuple) -> float:
    """Volume of a Gmsh bounding box (x0,y0,z0,x1,y1,z1); used to rank solids."""
    return max(bb[3] - bb[0], 0) * max(bb[4] - bb[1], 0) * max(bb[5] - bb[2], 0)


def _dist_point_bbox(bb: tuple, p: tuple) -> float:
    """Shortest distance from point ``p`` to bbox ``bb`` (0 if inside)."""
    dx = max(bb[0] - p[0], 0.0, p[0] - bb[3])
    dy = max(bb[1] - p[1], 0.0, p[1] - bb[4])
    dz = max(bb[2] - p[2], 0.0, p[2] - bb[5])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _dielectric_bbox(originals: list) -> tuple:
    """Combined bounding box of all dielectric solids (the board footprint)."""
    bb = [1e30, 1e30, 1e30, -1e30, -1e30, -1e30]
    for role, _name, obb, _v in originals:
        if role == "dielectric":
            for i in range(3):
                bb[i] = min(bb[i], obb[i])
                bb[i + 3] = max(bb[i + 3], obb[i + 3])
    return tuple(bb)


def _footprint_face(solid_tag: int, diel_bbox: tuple) -> int:
    """Largest face of a thin solid; on a tie (top vs bottom plate) pick the one
    nearest the dielectric, so the imprinted PEC sheet lands on its surface."""
    faces = [t for _, t in gmsh.model.getBoundary([(3, solid_tag)], oriented=False)]
    info = []
    for ft in faces:
        area = gmsh.model.occ.getMass(2, ft)
        com = gmsh.model.occ.getCenterOfMass(2, ft)
        info.append((area, _dist_point_bbox(diel_bbox, com), ft))
    max_area = max(a for a, _d, _t in info)
    plate = [(d, t) for a, d, t in info if a >= 0.99 * max_area]
    plate.sort()  # by distance to dielectric, ascending
    return plate[0][1]


def _clip_sheet_to_dielectric(face_tag: int, direction: str, diel_bbox: tuple) -> int:
    """Clip a port sheet to the dielectric extent along its excitation
    ``direction`` so it lands flush between the two conductor sheets instead of
    protruding past them. Exact bounds along ``direction``, generous elsewhere."""
    ai = {"x": 0, "y": 1, "z": 2}[direction]
    big = 0.1
    lo = [diel_bbox[0] - big, diel_bbox[1] - big, diel_bbox[2] - big]
    hi = [diel_bbox[3] + big, diel_bbox[4] + big, diel_bbox[5] + big]
    lo[ai], hi[ai] = diel_bbox[ai], diel_bbox[ai + 3]  # exact span along direction
    box = gmsh.model.occ.addBox(
        lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    )
    out, _ = gmsh.model.occ.intersect(
        [(2, face_tag)], [(3, box)], removeObject=True, removeTool=True
    )
    faces = [t for d, t in out if d == 2]
    return faces[0] if faces else face_tag


def _init() -> None:
    """Initialise a fresh, quiet Gmsh session."""
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)


def list_solids(step_file: str) -> list[str]:
    """
    Return the short product names of the solids in a STEP file (no meshing).

    Parameters
    ----------
    step_file : str
        Path to the STEP file.

    Returns
    -------
    list[str]
        Short solid names.
    """
    _init()
    try:
        gmsh.model.add("probe")
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        names = []
        for dim, tag in gmsh.model.getEntities(3):
            names.append(_short_name(gmsh.model.getEntityName(dim, tag)))
        return names
    finally:
        gmsh.finalize()


# ----------------------------
# main entry point
# ----------------------------
def build_mesh(problem: Problem, workdir: str, verbose: bool = True) -> Mesh:
    """
    Mesh a :class:`~simpleEMS.fem_backend.Problem`'s STEP geometry.

    Parameters
    ----------
    problem : Problem
        The FEM problem definition (solids, ports, boundary, mesh controls).
    workdir : str
        Directory to write the ``.msh`` file into.
    verbose : bool
        Print progress through the shared console. Default ``True``.

    Returns
    -------
    Mesh
        The mesh plus the region maps shared with the GetDP problem file.
    """
    fmin = float(problem.freqs.min())
    fmax = float(problem.freqs.max())
    # smallest wavelength anywhere (inside the highest-eps dielectric)
    eps_max = max([d.dielectric.eps_r for d in problem.dielectrics()] + [1.0])
    lambda_min = C0 / (fmax * (eps_max**0.5))
    lambda0_max = C0 / fmin  # longest free-space wavelength -> sets air padding

    os.makedirs(workdir, exist_ok=True)
    _init()
    if verbose:
        gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(problem.name)
    # STEP may be in mm; OCCTargetUnit converts everything to metres so all the
    # geometry below (and the wavelengths above) share one unit system.
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.model.occ.importShapes(problem.step_file)
    gmsh.model.occ.synchronize()

    # ---- record each imported solid's role + bbox before any boolean op -----
    # The boolean ops below (copy/remove/fragment) destroy and renumber solids,
    # so snapshot each one's role and bounding box now while identities are known;
    # `struct` accumulates the overall extent of the real (non-ignored) structure.
    originals = []  # (role, name, bbox, bbox_volume)
    struct = [1e30, 1e30, 1e30, -1e30, -1e30, -1e30]
    for dim, tag in gmsh.model.getEntities(3):
        short = _short_name(gmsh.model.getEntityName(dim, tag))
        spec = problem.solids.get(short)
        role = spec.role if spec else "ignore"
        bb = gmsh.model.getBoundingBox(dim, tag)
        originals.append((role, short, bb, _bbox_volume(bb)))
        if role != "ignore":
            for i in range(3):
                struct[i] = min(struct[i], bb[i])
                struct[i + 3] = max(struct[i + 3], bb[i + 3])
    struct = tuple(struct)

    diel_bbox = _dielectric_bbox(originals)
    port_geo = {name: bb for role, name, bb, _v in originals if role == "port"}

    # ---- reduce each thin conductor/port solid to a footprint sheet ---------
    # (copy the sheet so it survives deletion of the parent solid)
    port_direction = {p.solid: p.direction for p in problem.ports}
    sheets = []  # (role, name, face_tag)
    metal_solids = []
    for dim, tag in gmsh.model.getEntities(3):
        short = _short_name(gmsh.model.getEntityName(dim, tag))
        spec = problem.solids.get(short)
        role = spec.role if spec else "ignore"
        if role in ("pec", "impedance", "port"):
            face = _footprint_face(tag, diel_bbox)
            cp = gmsh.model.occ.copy([(2, face)])
            face_tag = cp[0][1]
            if role == "port":
                # clip flush to the dielectric so the port doesn't protrude
                face_tag = _clip_sheet_to_dielectric(
                    face_tag, port_direction.get(short, "z"), diel_bbox
                )
                gmsh.model.occ.synchronize()
                port_geo[short] = gmsh.model.getBoundingBox(2, face_tag)
            sheets.append((role, short, face_tag))
            metal_solids.append((3, tag))
    # delete the metal/port solids; dielectric solids remain and are fragmented
    if metal_solids:
        gmsh.model.occ.remove(metal_solids, recursive=True)
    gmsh.model.occ.synchronize()

    # ---- air box (+ optional outer PML shell) around the structure ----------
    # A radiating structure lives in open space, but FEM needs a finite domain.
    # Wrap it in an air box padded by ~a wavelength; its outer faces later carry
    # the absorbing boundary. With PML, add a second outer shell that damps
    # outgoing waves instead of a first-order absorbing condition.
    is_pml = problem.boundary == "pml"
    pad = max(problem.air_pad_frac * lambda0_max, 3.0 * (struct[5] - struct[2]))
    ax0, ay0, az0 = struct[0] - pad, struct[1] - pad, struct[2] - pad  # inner box
    ax1, ay1, az1 = struct[3] + pad, struct[4] + pad, struct[5] + pad
    inner_bbox = (ax0, ay0, az0, ax1, ay1, az1)
    gmsh.model.occ.addBox(ax0, ay0, az0, ax1 - ax0, ay1 - ay0, az1 - az0)
    pml_thick = 0.0
    if is_pml:
        pml_thick = max(0.2 * lambda0_max, 2.0 * pad / 3.0)
        ox0, oy0, oz0 = ax0 - pml_thick, ay0 - pml_thick, az0 - pml_thick
        ox1, oy1, oz1 = ax1 + pml_thick, ay1 + pml_thick, az1 + pml_thick
        gmsh.model.occ.addBox(ox0, oy0, oz0, ox1 - ox0, oy1 - oy0, oz1 - oz0)
        box_bbox = (ox0, oy0, oz0, ox1, oy1, oz1)
    else:
        box_bbox = inner_bbox
    gmsh.model.occ.synchronize()

    # ---- symmetry cut: keep the half-space (coord >= plane) -----------------
    sym_axis_i, sym_plane = None, None
    if problem.symmetry:
        s_axis, _s_kind, s_at = problem.symmetry
        sym_axis_i = {"x": 0, "y": 1, "z": 2}[s_axis]
        sym_plane = (
            s_at
            if s_at is not None
            else 0.5 * (struct[sym_axis_i] + struct[sym_axis_i + 3])
        )
        bb = box_bbox
        lo = [bb[0] - 1.0, bb[1] - 1.0, bb[2] - 1.0]
        hi = [bb[3] + 1.0, bb[4] + 1.0, bb[5] + 1.0]
        lo[sym_axis_i] = sym_plane  # keep coord >= plane
        halfbox = gmsh.model.occ.addBox(
            lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
        )
        gmsh.model.occ.synchronize()
        vol_targets = [(3, t) for _, t in gmsh.model.getEntities(3) if t != halfbox]
        sheet_targets = [(2, f) for _, _, f in sheets]
        _o, omap = gmsh.model.occ.intersect(
            vol_targets + sheet_targets,
            [(3, halfbox)],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()
        new_sheets = []  # sheets were clipped -> pick up new tags from the map
        for i, (role, name, _f) in enumerate(sheets):
            faces = [t for d, t in omap[len(vol_targets) + i] if d == 2]
            if faces:
                new_sheets.append((role, name, faces[0]))
        sheets = new_sheets

    # ---- fragment dielectric volumes + air box with the conductor sheets ----
    # `fragment` imprints the conductor sheets into the volumes: it splits the
    # volumes along the sheets so the sheets become shared, conforming interior
    # faces (the mesh will put element faces exactly on the conductors) while the
    # volumes stay watertight. This is what lets us mesh zero-thickness metals.
    vols = [(3, t) for _, t in gmsh.model.getEntities(3)]  # dielectrics + box
    sheet_dimtags = [(2, f) for _, _, f in sheets]
    _out, outmap = gmsh.model.occ.fragment(vols, sheet_dimtags)
    gmsh.model.occ.synchronize()

    # fragment map: first len(vols) entries are volumes, then the sheets
    sheet_faces = []  # aligned with `sheets`
    for i in range(len(sheets)):
        entry = outmap[len(vols) + i]
        sheet_faces.append({t for (d, t) in entry if d == 2})

    # ---- classify every resulting volume back to an origin solid ------------
    # A volume is dielectric iff its own bounding box fits inside a dielectric
    # solid's box (a centroid test fails: the enclosing air shell is centred on
    # the structure too). Tolerance is relative to each dielectric's size.
    diel_vols: dict[str, list[int]] = {}
    air_vols: list[int] = []
    pml_vols: list[int] = []
    inner_diag = (
        (inner_bbox[3] - inner_bbox[0]) ** 2
        + (inner_bbox[4] - inner_bbox[1]) ** 2
        + (inner_bbox[5] - inner_bbox[2]) ** 2
    ) ** 0.5
    for dim, tag in gmsh.model.getEntities(3):
        vbb = gmsh.model.getBoundingBox(dim, tag)
        best = None  # (bbox_volume, name)
        for role, name, bb, bvol in originals:
            if role != "dielectric":
                continue
            diag = (
                (bb[3] - bb[0]) ** 2 + (bb[4] - bb[1]) ** 2 + (bb[5] - bb[2]) ** 2
            ) ** 0.5
            tol = max(1e-6, 1e-3 * diag)
            if _bbox_inside(vbb, bb, tol) and (best is None or bvol < best[0]):
                best = (bvol, name)
        if best is not None:
            diel_vols.setdefault(best[1], []).append(tag)
        elif is_pml and not _bbox_inside(vbb, inner_bbox, 1e-3 * inner_diag):
            pml_vols.append(tag)  # extends beyond the inner air box -> PML shell
        else:
            air_vols.append(tag)

    # ---- collect PEC / impedance / port sheet faces (still in the model) ----
    existing_faces = {t for _, t in gmsh.model.getEntities(2)}
    pec_faces: set[int] = set()
    conductor_faces_by_name: dict[str, set[int]] = {}  # pec + impedance
    imped_faces_by_sigma: dict[float, set[int]] = {}  # lossy conductors by sigma
    port_faces_by_name: dict[str, set[int]] = {}
    for (role, name, _f), faces in zip(sheets, sheet_faces, strict=True):
        faces = faces & existing_faces
        if role == "pec":
            pec_faces |= faces
            conductor_faces_by_name.setdefault(name, set()).update(faces)
        elif role == "impedance":
            sig = problem.solids[name].sigma
            imped_faces_by_sigma.setdefault(sig, set()).update(faces)
            conductor_faces_by_name.setdefault(name, set()).update(faces)
        else:
            port_faces_by_name.setdefault(name, set()).update(faces)
    all_port_faces = (
        set().union(*port_faces_by_name.values()) if port_faces_by_name else set()
    )
    all_imped_faces = (
        set().union(*imped_faces_by_sigma.values()) if imped_faces_by_sigma else set()
    )

    # Identify the ground plane (largest-area conductor sheet). Excluding it from
    # the mesh-refinement set localises the fine mesh to the signal conductors.
    ground_faces: set[int] = set()
    if len(conductor_faces_by_name) > 1:

        def _area(fs: set) -> float:
            return sum(gmsh.model.occ.getMass(2, f) for f in fs)

        ground_name = max(
            conductor_faces_by_name,
            key=lambda n: _area(conductor_faces_by_name[n]),
        )
        ground_faces = conductor_faces_by_name[ground_name]

    # ---- outer absorbing shell ----------------------------------------------
    domain_dimtags = gmsh.model.getEntities(3)
    shell = gmsh.model.getBoundary(domain_dimtags, combined=True, oriented=False)
    shell_faces = {t for _, t in shell}
    abc_faces = shell_faces - pec_faces - all_port_faces - all_imped_faces

    # symmetry-plane faces are boundary faces at the cut plane -> not radiating
    sym_faces: set[int] = set()
    if problem.symmetry:
        eps = max(1e-6, 1e-4 * (box_bbox[3 + sym_axis_i] - box_bbox[sym_axis_i]))
        lo = list(box_bbox[:3])
        hi = list(box_bbox[3:])
        lo[sym_axis_i] = sym_plane - eps
        hi[sym_axis_i] = sym_plane + eps
        at_plane = {
            t
            for _, t in gmsh.model.getEntitiesInBoundingBox(
                lo[0] - eps,
                lo[1] - eps,
                lo[2] - eps,
                hi[0] + eps,
                hi[1] + eps,
                hi[2] + eps,
                2,
            )
        }
        sym_faces = at_plane & abc_faces
        abc_faces -= sym_faces

    # ---- assign physical groups (integer tags shared with the .pro) ---------
    # These integer tags are the ONLY contract between mesh and solver: GetDP
    # references each region purely by number (Region[{id}]), so the IDs here
    # (from fem_materials) must match exactly what fem_formulation emits.
    diel_regions = {}
    for i, (name, tags) in enumerate(sorted(diel_vols.items())):
        rid = dielectric_region(i)
        gmsh.model.addPhysicalGroup(3, tags, rid)
        gmsh.model.setPhysicalName(3, rid, name)
        diel_regions[name] = rid

    if air_vols:
        gmsh.model.addPhysicalGroup(3, air_vols, AIR)
        gmsh.model.setPhysicalName(3, AIR, "air")

    pml_region = 0
    if pml_vols:
        pml_region = PML
        gmsh.model.addPhysicalGroup(3, pml_vols, pml_region)
        gmsh.model.setPhysicalName(3, pml_region, "pml")

    if pec_faces:
        gmsh.model.addPhysicalGroup(2, sorted(pec_faces), PEC)
        gmsh.model.setPhysicalName(2, PEC, "pec")

    impedance_regions = []  # [(region_id, sigma), ...]
    for i, (sig, faces) in enumerate(sorted(imped_faces_by_sigma.items())):
        rid = IMPEDANCE + i
        gmsh.model.addPhysicalGroup(2, sorted(faces), rid)
        gmsh.model.setPhysicalName(2, rid, f"impedance_{i}")
        impedance_regions.append((rid, sig))

    port_regions: dict[int, PortMesh] = {}
    for pspec in problem.ports:
        faces = port_faces_by_name.get(pspec.solid, set())
        if not faces:
            raise RuntimeError(f"port solid {pspec.solid!r} produced no boundary faces")
        rid = port_region(pspec.number)
        gmsh.model.addPhysicalGroup(2, sorted(faces), rid)
        gmsh.model.setPhysicalName(2, rid, f"port_{pspec.number}")
        bb = port_geo[pspec.solid]
        ax = {"x": 0, "y": 1, "z": 2}[pspec.direction]
        extents = [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
        gap = extents[ax]
        width = max(extents[i] for i in range(3) if i != ax)  # transverse width
        center = ((bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2)
        eps_sub = max([d.dielectric.eps_r for d in problem.dielectrics()] + [1.0])
        zc, eps_eff = microstrip_zc(width, gap, eps_sub)  # line char. impedance
        port_regions[pspec.number] = PortMesh(
            number=pspec.number,
            region=rid,
            direction=pspec.direction,
            z0=pspec.z0,
            gap=gap,
            width=width,
            center=center,
            zc=zc,
            eps_eff=eps_eff,
            port_type=pspec.port_type,
        )

    if abc_faces:
        gmsh.model.addPhysicalGroup(2, sorted(abc_faces), ABC)
        gmsh.model.setPhysicalName(2, ABC, "abc")

    sym_region = 0
    if sym_faces:
        sym_region = SYM
        gmsh.model.addPhysicalGroup(2, sorted(sym_faces), sym_region)
        gmsh.model.setPhysicalName(2, sym_region, "sym")

    # ---- mesh sizing: fine near the signal conductors, coarse in open air ---
    diel_min_thick = min(
        [
            min(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
            for role, _n, bb, _v in originals
            if role == "dielectric"
        ]
        or [struct[5] - struct[2]]
    )
    # element size that puts `min_layers` elements through the dielectric;
    # `mesh_fine_scale` (>=1) coarsens for cost, but never below one layer.
    lc_fine = max(
        diel_min_thick / max(problem.min_layers, 1) * problem.mesh_fine_scale, 20e-6
    )
    lc_fine = min(lc_fine, diel_min_thick)  # keep at least ~1 layer even if coarsened
    lc_coarse = lambda_min / problem.elems_per_wavelength
    # Refine near the signal conductors + ports only (exclude the ground plane);
    # extend the fine zone through the substrate so trace-to-ground is resolved.
    refine_faces = ((pec_faces | all_imped_faces) - ground_faces) | all_port_faces
    dist_max = max(15.0 * lc_fine, 4.0 * diel_min_thick)
    _apply_size_field(refine_faces, lc_fine, lc_coarse, dist_max)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_coarse)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc_fine / 5.0)
    gmsh.option.setNumber("Mesh.Optimize", 1)

    gmsh.model.mesh.generate(3)
    msh_path = os.path.join(os.path.abspath(workdir), f"{problem.name}.msh")
    gmsh.write(msh_path)
    # also write a legacy VTK for pyvista display (no meshio dependency needed)
    gmsh.write(os.path.splitext(msh_path)[0] + ".vtk")

    if verbose:
        nn = len(gmsh.model.mesh.getNodes()[0])
        console.print(f"[info]mesh written: {msh_path}  ({nn} nodes)[/info]")
        console.print(
            f"[info]dielectrics={diel_regions} air={AIR} pec={PEC} "
            f"ports={list(port_regions)} abc={ABC}[/info]"
        )
    gmsh.finalize()

    return Mesh(
        msh_path=msh_path,
        dielectric_regions=diel_regions,
        air_region=AIR,
        pec_region=PEC,
        port_regions=port_regions,
        abc_region=ABC,
        boundary=problem.boundary,
        bbox=struct,
        box_bbox=box_bbox,
        lambda_min=lambda_min,
        impedance_regions=impedance_regions,
        pml_region=pml_region,
        inner_bbox=inner_bbox,
        pml_thick=pml_thick,
        sym_region=sym_region,
        sym_kind=(problem.symmetry[1] if problem.symmetry else ""),
    )


def _apply_size_field(
    refine_faces: set, lc_fine: float, lc_coarse: float, dist_max: float
) -> None:
    """Distance-from-conductors -> Threshold size field for graded refinement.

    Builds a Gmsh size field that is ``lc_fine`` on the refine surfaces and grows
    linearly to ``lc_coarse`` by ``dist_max`` away from them, so elements are
    small only where the fields vary fastest (near the signal conductors).
    """
    if not refine_faces:
        return
    # Distance field: distance from any point to the nearest refine surface.
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "SurfacesList", sorted(refine_faces))
    # Threshold field: map that distance to an element size (fine near, coarse far).
    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", lc_fine)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", lc_coarse)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", lc_fine)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)
    # Disable Gmsh's other size heuristics so this field alone drives the sizing.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
