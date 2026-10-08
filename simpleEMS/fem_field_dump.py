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
Field dumps for the FEM backend.

Writes the same files an openEMS frequency-domain VTK dump writes, so ParaView
treats both backends alike. The solved field is sampled on the FDTD mesh lines
inside the dump box, a rectilinear grid, and written as 21 phase frames
(``<name>_f=<freq>_p=000.vtr`` to ``_p=342.vtr``, which ParaView plays as an
animation) plus the magnitude (``_abs``) and phase (``_arg``) of each
component.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np
from numpy.typing import NDArray

from . import fem_solver
from .console import console

__all__ = ["FieldDumpRequest", "write_field_dumps"]

_MESH_META = "fem_mesh.json"

# openEMS's ProcessFieldsFD writes this many phases of one period per frequency.
_PHASE_COUNT = 21

# DumpType file prefix -> the array name openEMS gives that field
# (ProcessFields::GetFieldNameByType).
_FIELD_NAME_BY_PREFIX = {"Ef": "E-Field", "Hf": "H-Field", "Jf": "RotH-Field"}


@dataclass(frozen=True)
class FieldDumpRequest:
    """
    One field dump to write after a FEM simulation.

    Parameters
    ----------
    prefix : str
        ``"Ef"`` (electric field), ``"Hf"`` (magnetic field) or ``"Jf"``
        (total current density, curl H), the file prefixes
        :class:`~simpleEMS.sim_tools.DumpType` uses.
    freq : float
        Frequency to solve the fields at, in Hz.
    box : tuple[float, ...]
        ``(xmin, ymin, zmin, xmax, ymax, zmax)`` of the dump box, in metres.
    lines : tuple[tuple[float, ...], ...]
        The x, y and z FDTD mesh lines, in metres. The dump samples the lines
        inside ``box``; a ``"Jf"`` dump also reads the line just outside each
        face, to take the curl there.
    """

    prefix: str
    freq: float
    box: tuple[float, ...]
    lines: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        """Reject a prefix the FEM backend cannot dump."""
        if self.prefix not in _FIELD_NAME_BY_PREFIX:
            raise ValueError(
                "FEM field dump prefix must be one of "
                f"{sorted(_FIELD_NAME_BY_PREFIX)}, got {self.prefix!r}"
            )


@dataclass
class _VolumeFields:
    """The solved fields on the volume mesh, read back from GetDP's views.

    ``efield`` and ``hfield`` hold each tetrahedron's own value at each of its
    four nodes, ``(n_tets, 4, 3)`` complex: the fields jump across material
    interfaces and conductor sheets, so a node has no single value.
    """

    points: NDArray  # (n_points, 3) in metres
    tets: NDArray  # (n_tets, 4) indices into points
    efield: NDArray
    hfield: NDArray


def _start_gmsh() -> None:
    """Start a fresh, quiet gmsh session; gmsh is a process-wide singleton."""
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)


def _read_view(pos_path: Path) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Read one complex field file written by GetDP's ``Get_Fields``.

    The file carries its own mesh, followed by the field at each element's
    nodes: the real part as the first view step, the imaginary part as the
    second.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        ``(node_tags, coordinates, element_nodes, values)`` -- every node's
        tag and position, each tetrahedron's node tags, and the complex field
        at its nodes, ``(n_tets, 4, 3)``.
    """
    gmsh.open(str(pos_path))
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    _types, element_tags, element_nodes = gmsh.model.mesh.getElements(3)
    element_tags = np.asarray(element_tags[0])
    element_nodes = np.asarray(element_nodes[0]).reshape(-1, 4)

    view = gmsh.view.getTags()[0]
    parts = []
    for step in (0, 1):
        _kind, view_tags, data, _time, _components = gmsh.view.getModelData(view, step)
        if not np.array_equal(np.asarray(view_tags), element_tags):
            raise RuntimeError(f"{pos_path}: field data does not follow its mesh")
        parts.append(np.asarray(data).reshape(-1, 4, 3))
    gmsh.clear()
    return (
        np.asarray(node_tags),
        np.asarray(coordinates).reshape(-1, 3),
        element_nodes,
        parts[0] + 1j * parts[1],
    )


def _read_volume_fields(e_pos: Path, h_pos: Path) -> _VolumeFields:
    """Read the electric and magnetic field files into one :class:`_VolumeFields`."""
    node_tags, points, element_nodes, efield = _read_view(e_pos)
    h_node_tags, _points, h_element_nodes, hfield = _read_view(h_pos)
    if not (
        np.array_equal(node_tags, h_node_tags)
        and np.array_equal(element_nodes, h_element_nodes)
    ):
        raise RuntimeError("e.pos and h.pos were written on different meshes")

    index_by_tag = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    index_by_tag[node_tags.astype(np.int64)] = np.arange(len(node_tags))
    return _VolumeFields(
        points=points,
        tets=index_by_tag[element_nodes.astype(np.int64)],
        efield=efield,
        hfield=hfield,
    )


def _exploded_mesh(fields: _VolumeFields) -> object:
    """A pyvista mesh in which every tetrahedron has its own four points.

    Probing it interpolates inside the one tetrahedron that holds the probe
    point, from that tetrahedron's own nodal values, so the sampled field
    keeps its jumps across interfaces instead of averaging over them. VTK has
    no complex arrays, so each field is stored as its six real components.
    """
    import pyvista as pv

    corner_count = 4 * len(fields.tets)
    mesh = pv.UnstructuredGrid(
        {pv.CellType.TETRA: np.arange(corner_count).reshape(-1, 4)},
        fields.points[fields.tets].reshape(-1, 3),
    )
    for name, values in (("E", fields.efield), ("H", fields.hfield)):
        values = values.reshape(-1, 3)
        mesh.point_data[name] = np.hstack([values.real, values.imag])
    return mesh


def _sample(
    mesh: object, name: str, lines: tuple[NDArray, NDArray, NDArray]
) -> NDArray:
    """
    Sample one field of :func:`_exploded_mesh` on a rectilinear grid.

    A grid point on a face shared by two tetrahedra, a conductor sheet for
    one, belongs to both. Nudging every point a hair towards +x, +y and +z
    hands it to the same side each time, which keeps the jump in H across a
    sheet in one cell for the curl to find. A point the nudge pushes off the
    mesh, on the domain's upper faces, is sampled nudged the other way.

    Returns
    -------
    NDArray
        ``(nx, ny, nz, 3)`` complex field. Zero at a point outside the mesh.
    """
    import pyvista as pv

    x_lines, y_lines, z_lines = lines
    spacings = [np.diff(axis) for axis in lines if len(axis) > 1]
    nudge = 1e-6 * min((float(spacing.min()) for spacing in spacings), default=1.0)

    values = np.zeros((len(x_lines) * len(y_lines) * len(z_lines), 6))
    missing = np.ones(len(values), dtype=bool)
    for direction in (1.0, -1.0):
        grid = pv.RectilinearGrid(
            x_lines + direction * nudge,
            y_lines + direction * nudge,
            z_lines + direction * nudge,
        )
        sampled = grid.sample(mesh)
        found = missing & sampled.point_data["vtkValidPointMask"].astype(bool)
        values[found] = sampled.point_data[name][found]
        missing &= ~found

    # VTK orders the points x fastest, then y, then z.
    shape = (len(z_lines), len(y_lines), len(x_lines), 3)
    complex_values = values[:, :3] + 1j * values[:, 3:]
    return complex_values.reshape(shape).transpose(2, 1, 0, 3)


def _curl(field: NDArray, lines: tuple[NDArray, NDArray, NDArray]) -> NDArray:
    """Curl of an ``(nx, ny, nz, 3)`` field on a rectilinear grid.

    An axis with a single line has no derivative along it, taken as zero.
    """

    def derivative(component: int, axis: int) -> NDArray:
        if len(lines[axis]) < 2:
            return np.zeros(field.shape[:3], dtype=field.dtype)
        return np.gradient(field[..., component], lines[axis], axis=axis)

    return np.stack(
        [
            derivative(2, 1) - derivative(1, 2),
            derivative(0, 2) - derivative(2, 0),
            derivative(1, 0) - derivative(0, 1),
        ],
        axis=-1,
    )


def _dump_lines(
    request: FieldDumpRequest, domain_bbox: tuple[float, ...], margin: int
) -> tuple[tuple[NDArray, ...], tuple[slice, ...]]:
    """
    Choose the mesh lines to sample a dump on.

    Keeps the lines inside both the dump box and the meshed domain, plus up to
    ``margin`` lines past each face of the box that are still in the domain.

    Returns
    -------
    tuple[tuple[NDArray, ...], tuple[slice, ...]]
        The x, y and z lines to sample, and the slice of each that lies in
        the box.

    Raises
    ------
    ValueError
        If no line along some axis lies in both the box and the domain.
    """
    sample_lines, box_slices = [], []
    for axis in range(3):
        lines = np.asarray(request.lines[axis], dtype=float)
        low, high = request.box[axis], request.box[axis + 3]
        domain_low, domain_high = domain_bbox[axis], domain_bbox[axis + 3]
        tolerance = 1e-9 * max(high - low, domain_high - domain_low, 1.0)
        in_domain = lines[
            (lines >= domain_low - tolerance) & (lines <= domain_high + tolerance)
        ]
        in_box = np.flatnonzero(
            (in_domain >= low - tolerance) & (in_domain <= high + tolerance)
        )
        if not len(in_box):
            raise ValueError(
                f"no {'xyz'[axis]} mesh line lies in both the dump box "
                f"[{low}, {high}] and the FEM domain [{domain_low}, {domain_high}]"
            )
        first = max(int(in_box[0]) - margin, 0)
        last = min(int(in_box[-1]) + margin, len(in_domain) - 1)
        sample_lines.append(in_domain[first : last + 1])
        box_slices.append(slice(int(in_box[0]) - first, int(in_box[-1]) - first + 1))
    return tuple(sample_lines), tuple(box_slices)


def _frequency_label(freq: float) -> str:
    """Write ``freq`` the way openEMS puts it in a dump's file name."""
    if freq == int(freq):
        return str(int(freq))
    return f"{freq:.6f}"


def _write_openems_files(
    field: NDArray,
    lines: tuple[NDArray, ...],
    field_name: str,
    stem: Path,
) -> list[Path]:
    """
    Write a complex field as openEMS writes a frequency-domain VTK dump.

    Parameters
    ----------
    field : NDArray
        ``(nx, ny, nz, 3)`` complex field.
    lines : tuple[NDArray, ...]
        The x, y and z grid lines, in metres.
    field_name : str
        Name of the vector array in each file, e.g. ``"E-Field"``.
    stem : Path
        ``field_dump/<prefix>_f=<freq>``; each file appends its own suffix.

    Returns
    -------
    list[Path]
        The phase frames, then the magnitude and phase files.
    """
    import pyvista as pv

    grid = pv.RectilinearGrid(*lines)

    def write(suffix: str, values: NDArray) -> Path:
        grid.point_data.clear()
        # back to VTK's point order: x fastest, then y, then z
        grid.point_data[field_name] = (
            values.transpose(2, 1, 0, 3).reshape(-1, 3).astype(np.float32)
        )
        path = stem.with_name(f"{stem.name}_{suffix}.vtr")
        grid.save(path)
        return path

    written = []
    # One period, as openEMS samples it: Re{F e^(j angle)} with the angle in
    # whole degrees in the name, truncated as openEMS truncates it.
    for phase in range(_PHASE_COUNT):
        angle = 2 * math.pi * phase / _PHASE_COUNT
        degrees = int(angle * 180 / math.pi)
        written.append(write(f"p={degrees:03d}", np.real(field * np.exp(1j * angle))))
    written.append(write("abs", np.abs(field)))
    written.append(write("arg", np.angle(field)))
    return written


def write_field_dumps(
    requests: list[FieldDumpRequest],
    output_path: str | Path,
    verbose: bool = True,
) -> list[Path]:
    """
    Solve the fields and write each requested dump to ``output_path/field_dump``.

    Runs one GetDP solve per distinct frequency, driving the reference port
    (the lowest-numbered one), on the mesh a FEM simulation left in
    ``output_path``. Each dump is written as openEMS writes a frequency-domain
    VTK dump: 21 phase frames ``<prefix>_f=<freq>_p=<degrees>.vtr`` that
    ParaView plays as an animation, then ``_abs`` and ``_arg`` files holding
    each component's magnitude and phase.

    The dump samples the FDTD mesh lines inside the dump box. Lines outside
    the meshed FEM domain (the air box inside any PML) are dropped. The field
    amplitudes follow GetDP's port excitation, so only relative values and
    phases mean anything, as in an openEMS dump. A model cut by a symmetry
    plane has field on its meshed half only.

    Parameters
    ----------
    requests : list[FieldDumpRequest]
        The dumps to write.
    output_path : str | Path
        Directory holding the ``fem_mesh.json`` the simulation wrote.
    verbose : bool
        Report the solves and the files written. Default ``True``.

    Returns
    -------
    list[Path]
        Every file written, request by request.

    Raises
    ------
    RuntimeError
        If the simulation's mesh metadata is missing.
    """
    output_path = Path(output_path)
    meta_file = output_path / _MESH_META
    if not meta_file.exists():
        raise RuntimeError(
            f"FEM mesh metadata not found at {meta_file}; run the simulation first."
        )
    meta = json.loads(meta_file.read_text())
    driven_port = min(meta["port_numbers"])
    domain_bbox = tuple(meta.get("domain_bbox") or meta["bbox"])
    dump_dir = output_path / "field_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)

    written: dict[FieldDumpRequest, list[Path]] = {}
    for freq in sorted({request.freq for request in requests}):
        fem_solver.run_getdp(
            meta["pro_path"],
            meta["msh_path"],
            output_path,
            {"FREQ": freq, "ACTIVE_PORT": driven_port},
            "Get_Fields",
            resolution="AnalysisSinglePort",
            label=f"field dump @ {freq / 1e9:.4f} GHz",
            verbose=verbose,
        )
        result_dir = output_path.absolute() / "output"
        _start_gmsh()
        try:
            fields = _read_volume_fields(result_dir / "e.pos", result_dir / "h.pos")
        finally:
            gmsh.finalize()
        mesh = _exploded_mesh(fields)

        for request in requests:
            if request.freq != freq or request in written:
                continue
            is_curl = request.prefix == "Jf"
            lines, box = _dump_lines(request, domain_bbox, margin=int(is_curl))
            if is_curl:
                field = _curl(_sample(mesh, "H", lines), lines)
            else:
                field = _sample(mesh, request.prefix[0], lines)
            stem = dump_dir / f"{request.prefix}_f={_frequency_label(freq)}"
            box_lines = tuple(
                axis_lines[axis_box]
                for axis_lines, axis_box in zip(lines, box, strict=True)
            )
            written[request] = _write_openems_files(
                field[box],
                box_lines,
                _FIELD_NAME_BY_PREFIX[request.prefix],
                stem,
            )
            if verbose:
                console.print(
                    f"[success]FEM field dump: {stem}_p=*.vtr "
                    f"({_PHASE_COUNT} phases), _abs.vtr, _arg.vtr[/success]"
                )
    return [path for request in requests for path in written[request]]
