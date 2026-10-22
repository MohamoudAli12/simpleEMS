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
Interactive 3-D viewer for field dumps.

:func:`view_field_dump` opens a window that animates a field dump written by
:meth:`~simpleEMS.sim_tools.SimTools.add_field_dump` over the structure it was
taken from. It reads the electric field, the magnetic field, the conduction
current density and the total current density (curl H), in the time domain or
the frequency domain, from either backend. A control panel switches between
arrow, surface, height-map, cut-plane, iso-surface and volume plots, and shows
only the controls that apply to the chosen plot.

The viewer reads the openEMS VTK files in ``output_path / "field_dump"``:

- a frequency-domain dump gives ``<prefix>_f=<freq>_abs.vtr`` and ``_arg.vtr``,
  the magnitude and phase of each component, which the viewer animates over
  one period;
- a time-domain dump gives ``<prefix>_<timestep>.vtr``, one file per saved
  timestep, which the viewer plays in order.

The structure comes from ``output_path / "structure.xml"``, so each body keeps
its material: metal draws as copper, dielectric as translucent substrate.
"""

from __future__ import annotations

import re
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets, sip
from pyvistaqt import QtInteractor
from scipy import sparse

from .console import console
from .plot_theme import (
    PYVISTA_THEME_NAME,
    SECONDARY_TEXT,
    SERIES_COLORS,
    SURFACE,
    _apply_dark_qt_palette,
)

__all__ = ["view_field_dump"]

# File prefix -> (DumpType name, symbol, unit, description). The prefixes are the
# ones DumpType gives each dump's output files.
_FIELD_KINDS = {
    "Et": ("efield_time", "E", "V/m", "Electric field"),
    "Ht": ("hfield_time", "H", "A/m", "Magnetic field"),
    "It": ("current_time", "J", "A/m^2", "Conduction current density"),
    "Jt": ("current_density_time", "J", "A/m^2", "Total current density (curl H)"),
    "Ef": ("efield_frequency", "E", "V/m", "Electric field"),
    "Hf": ("hfield_frequency", "H", "A/m", "Magnetic field"),
    "If": ("current_frequency", "J", "A/m^2", "Conduction current density"),
    "Jf": (
        "current_density_frequency",
        "J",
        "A/m^2",
        "Total current density (curl H)",
    ),
}

_FREQUENCY_FILE = re.compile(r"^(?P<prefix>.+?)_f=(?P<frequency>[0-9.eE+-]+)_abs\.vtr$")
_TIME_FILE = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<timestep>\d+)\.vtr$")

# Reading a time-domain frame takes about 20 ms, so long dumps are thinned to
# this many evenly spaced frames to keep the window opening in a few seconds.
_MAX_TIME_FRAMES = 200

PLOT_TYPES = [
    "Vector - plane",
    "Vector - volume",
    "Surface - plane",
    "Surface - on structure",
    "Height map - plane",
    "Three planes",
    "Iso-surfaces",
    "Volume",
]
# These plots show the field plane on its own, without the board around it.
PLANE_ONLY_PLOT_TYPES = ("Surface - plane", "Height map - plane")
COLOR_MAPS = {
    "Rainbow": "jet",
    "Turbo": "turbo",
    "Inferno": "inferno",
    "Viridis": "viridis",
    "Cool-warm": "coolwarm",
    "Berlin (dark diverging)": "berlin",
}
# Background name -> (bottom colour, top colour of a gradient, text colour).
BACKGROUNDS = {
    "Dark": (SURFACE, None, SECONDARY_TEXT),
    "Light": ("white", None, "black"),
    "Gradient": ("white", "#8fa9d6", "black"),
}


@dataclass
class _FieldDump:
    """
    One field dump, cropped to where the field is, with coordinates in mm.

    A frequency-domain dump stores the in-phase and quadrature parts of the
    complex field, so that F(phase) = in_phase * cos(phase) - quadrature *
    sin(phase). A time-domain dump stores its frames. Field arrays have shape
    ``(nz, ny, nx, 3)``, or ``(frames, nz, ny, nx, 3)`` for ``frames``.
    """

    prefix: str
    symbol: str
    unit: str
    description: str
    x_lines: NDArray
    y_lines: NDArray
    z_lines: NDArray
    frequency: float | None = None
    in_phase: NDArray | None = None
    quadrature: NDArray | None = None
    frames: NDArray | None = None
    timesteps: tuple[int, ...] = ()

    @property
    def is_time_domain(self) -> bool:
        """Whether the dump holds time-domain frames rather than a phasor."""
        return self.frames is not None

    @property
    def step_count(self) -> int:
        """Number of animation steps: frames, or degrees of phase."""
        return len(self.frames) if self.frames is not None else 360

    def field_at(self, step: int) -> NDArray:
        """Return the field at a frame index or a phase in degrees."""
        if self.frames is not None:
            return self.frames[step]
        phase = np.deg2rad(step)
        return self.in_phase * np.cos(phase) - self.quadrature * np.sin(phase)

    def peak_magnitude(self) -> NDArray:
        """Return the largest |F| each grid point reaches, shape (nz, ny, nx)."""
        if self.frames is not None:
            return np.linalg.norm(self.frames, axis=-1).max(axis=0)
        return np.sqrt((self.in_phase**2 + self.quadrature**2).sum(axis=-1))

    def peak_component(self) -> NDArray:
        """Return the largest absolute value of each component, shape (3,)."""
        if self.frames is not None:
            return np.abs(self.frames).max(axis=(0, 1, 2, 3))
        return np.sqrt(self.in_phase**2 + self.quadrature**2).max(axis=(0, 1, 2))


def _read_vector_field(path: Path) -> tuple[pv.RectilinearGrid, NDArray]:
    """Read one openEMS VTK dump file; return its grid and field (nz, ny, nx, 3)."""
    grid = pv.read(path)
    values = np.asarray(grid.point_data[grid.point_data.keys()[0]])
    x_count, y_count, z_count = grid.dimensions
    # VTK stores x fastest, then y, then z.
    return grid, values.reshape(z_count, y_count, x_count, 3)


def _crop_to_field(dump: _FieldDump) -> _FieldDump:
    """Crop the dump in x and y to where the field exceeds 2 % of its peak."""
    peak = dump.peak_magnitude()
    strong = peak > 0.02 * peak.max()
    strong_x = dump.x_lines[strong.any(axis=(0, 1))]
    strong_y = dump.y_lines[strong.any(axis=(0, 2))]
    margin = 0.05 * max(np.ptp(strong_x), np.ptp(strong_y))
    keep_x = (dump.x_lines >= strong_x.min() - margin) & (
        dump.x_lines <= strong_x.max() + margin
    )
    keep_y = (dump.y_lines >= strong_y.min() - margin) & (
        dump.y_lines <= strong_y.max() + margin
    )

    def crop(field: NDArray | None) -> NDArray | None:
        if field is None:
            return None
        return field[..., keep_y, :, :][..., keep_x, :]

    dump.x_lines = dump.x_lines[keep_x]
    dump.y_lines = dump.y_lines[keep_y]
    dump.in_phase = crop(dump.in_phase)
    dump.quadrature = crop(dump.quadrature)
    dump.frames = crop(dump.frames)
    return dump


def _scan_dump_files(
    dump_directory: Path,
) -> tuple[dict[str, dict[float, Path]], dict[str, list[tuple[int, Path]]]]:
    """
    Sort the dump files the viewer can show by dump.

    Returns
    -------
    frequency_files : dict[str, dict[float, Path]]
        The ``_abs`` file of each frequency, by file prefix.
    time_files : dict[str, list[tuple[int, Path]]]
        The ``(timestep, file)`` pairs, by file prefix.
    """
    frequency_files: dict[str, dict[float, Path]] = {}
    time_files: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(dump_directory.glob("*.vtr")):
        frequency_match = _FREQUENCY_FILE.match(path.name)
        time_match = _TIME_FILE.match(path.name)
        if frequency_match and frequency_match["prefix"] in _FIELD_KINDS:
            frequency = float(frequency_match["frequency"])
            frequency_files.setdefault(frequency_match["prefix"], {})[frequency] = path
        elif time_match and time_match["prefix"] in _FIELD_KINDS:
            time_files.setdefault(time_match["prefix"], []).append(
                (int(time_match["timestep"]), path)
            )
    return frequency_files, time_files


def _field_dump_choices(dump_directory: Path) -> list[tuple[str, float | None]]:
    """
    List the dumps in ``dump_directory`` the viewer can show.

    Returns
    -------
    list[tuple[str, float | None]]
        One ``(prefix, frequency)`` pair per dump, with one pair per frequency
        of a frequency-domain dump and ``None`` as the frequency of a
        time-domain dump, sorted by prefix and frequency.
    """
    frequency_files, time_files = _scan_dump_files(dump_directory)
    choices: list[tuple[str, float | None]] = [
        (prefix, frequency)
        for prefix, files in frequency_files.items()
        for frequency in files
    ]
    choices += [(prefix, None) for prefix in time_files]
    return sorted(choices, key=lambda choice: (choice[0], choice[1] or 0.0))


def _choice_label(prefix: str, frequency: float | None) -> str:
    """Name a dump for the dump picker, e.g. "Electric field - 2.45 GHz"."""
    description = _FIELD_KINDS[prefix][3]
    if frequency is None:
        return f"{description} - time domain"
    return f"{description} - {frequency / 1e9:.6g} GHz"


def _load_field_dump(
    dump_directory: Path, prefix: str | None = None, dump_freq: float | None = None
) -> _FieldDump:
    """
    Find and read one field dump in ``dump_directory``.

    Parameters
    ----------
    dump_directory : Path
        The ``field_dump`` directory of a simulation.
    prefix : str, optional
        File prefix of the dump to read (``"Ef"``, ``"Jt"``, ...). Defaults to
        the only dump in the directory.
    dump_freq : float, optional
        Frequency of a frequency-domain dump, in Hz; the nearest dumped
        frequency is used. Defaults to the only frequency dumped.

    Returns
    -------
    _FieldDump
        The dump, cropped to where the field is, in millimetres.

    Raises
    ------
    FileNotFoundError
        If the directory holds no readable dump, or none with ``prefix``.
    ValueError
        If ``prefix`` names a dump the viewer cannot show, if the choice of
        dump or frequency is ambiguous, or if ``dump_freq`` was not dumped.
    """
    frequency_files, time_files = _scan_dump_files(dump_directory)
    available = sorted({*frequency_files, *time_files})
    available_names = ", ".join(_FIELD_KINDS[found][0] for found in available)

    if prefix is None:
        if not available:
            raise FileNotFoundError(f"no field dump files found in {dump_directory}")
        if len(available) > 1:
            raise ValueError(
                f"{dump_directory} holds several dumps ({available_names}); "
                "pass dump_type to choose one"
            )
        prefix = available[0]
    elif prefix not in _FIELD_KINDS:
        raise ValueError(
            f"the field viewer cannot show the {prefix!r} dump; it shows E-field, "
            "H-field, current and current-density dumps"
        )
    elif prefix not in available:
        raise FileNotFoundError(
            f"no {_FIELD_KINDS[prefix][0]} dump in {dump_directory}"
            + (f" (found {available_names})" if available else "")
        )
    _, symbol, unit, description = _FIELD_KINDS[prefix]

    if prefix in frequency_files:
        frequencies = sorted(frequency_files[prefix])
        listed = ", ".join(f"{frequency / 1e9:g} GHz" for frequency in frequencies)
        if dump_freq is None:
            if len(frequencies) > 1:
                raise ValueError(
                    f"the {prefix} dump holds several frequencies ({listed}); "
                    "pass dump_freq to choose one"
                )
            frequency = frequencies[0]
        else:
            frequency = min(frequencies, key=lambda found: abs(found - dump_freq))
            if abs(frequency - dump_freq) > 0.01 * dump_freq:
                raise ValueError(
                    f"no {prefix} dump at {dump_freq / 1e9:g} GHz; found {listed}"
                )
        magnitude_path = frequency_files[prefix][frequency]
        phase_path = magnitude_path.with_name(
            magnitude_path.name.replace("_abs.vtr", "_arg.vtr")
        )
        grid, magnitude = _read_vector_field(magnitude_path)
        _, phase = _read_vector_field(phase_path)
        magnitude = magnitude.astype(float)
        phase = phase.astype(float)
        dump = _FieldDump(
            prefix=prefix,
            symbol=symbol,
            unit=unit,
            description=description,
            x_lines=np.asarray(grid.x) * 1e3,
            y_lines=np.asarray(grid.y) * 1e3,
            z_lines=np.asarray(grid.z) * 1e3,
            frequency=frequency,
            in_phase=magnitude * np.cos(phase),
            quadrature=magnitude * np.sin(phase),
        )
        return _crop_to_field(dump)

    # A rerun into the same directory overwrites the files it writes and leaves
    # the rest of the old run behind. Its first file is written first, so keep
    # only the files written since then.
    timestep_paths = sorted(time_files[prefix])
    run_start = timestep_paths[0][1].stat().st_mtime
    latest_run = [
        (timestep, path)
        for timestep, path in timestep_paths
        if path.stat().st_mtime >= run_start - 1.0
    ]
    if len(latest_run) < len(timestep_paths):
        console.print(
            f"[warning]Skipping {len(timestep_paths) - len(latest_run)} {prefix} "
            "files left over from an earlier run.[/warning]"
        )
    stride = int(np.ceil(len(latest_run) / _MAX_TIME_FRAMES))
    chosen = latest_run[::stride]
    if stride > 1:
        console.print(
            f"Reading {len(chosen)} of the {len(latest_run)} {prefix} frames, "
            "evenly spaced."
        )
    grid, first_frame = _read_vector_field(chosen[0][1])
    frames = [first_frame]
    timesteps = [chosen[0][0]]
    for timestep, path in chosen[1:]:
        _, frame = _read_vector_field(path)
        if frame.shape == first_frame.shape:
            frames.append(frame)
            timesteps.append(timestep)
    frames = np.asarray(frames, dtype=np.float32)
    # Drop the near-silent frames before and after the excitation pulse.
    frame_energy = np.sqrt((frames.astype(float) ** 2).mean(axis=(1, 2, 3, 4)))
    active = np.flatnonzero(frame_energy > 0.05 * frame_energy.max())
    first, last = active[0], active[-1] + 1
    dump = _FieldDump(
        prefix=prefix,
        symbol=symbol,
        unit=unit,
        description=description,
        x_lines=np.asarray(grid.x) * 1e3,
        y_lines=np.asarray(grid.y) * 1e3,
        z_lines=np.asarray(grid.z) * 1e3,
        frames=frames[first:last],
        timesteps=tuple(timesteps[first:last]),
    )
    return _crop_to_field(dump)


def _primitive_mesh(primitive: ElementTree.Element, scale: float) -> pv.PolyData | None:
    """Build the surface of one CSXCAD primitive in mm, or ``None`` if unsupported."""

    def point(name: str) -> NDArray:
        element = primitive.find(name)
        return np.array([float(element.get(axis)) for axis in "XYZ"]) * scale

    if primitive.tag == "Box":
        first, second = point("P1"), point("P2")
        low, high = np.minimum(first, second), np.maximum(first, second)
        return pv.Box(bounds=(low[0], high[0], low[1], high[1], low[2], high[2]))
    if primitive.tag == "Cylinder":
        start, stop = point("P1"), point("P2")
        return pv.Cylinder(
            center=(start + stop) / 2,
            direction=stop - start,
            radius=float(primitive.get("Radius")) * scale,
            height=float(np.linalg.norm(stop - start)),
            resolution=24,
        )
    if primitive.tag in ("LinPoly", "Polygon"):
        # The vertices lie in the plane normal to NormDir; X1 and X2 are the next
        # two axes in cyclic order.
        normal_axis = int(primitive.get("NormDir"))
        first_axis, second_axis = (normal_axis + 1) % 3, (normal_axis + 2) % 3
        vertices = primitive.findall("Vertex")
        points = np.zeros((len(vertices), 3))
        points[:, first_axis] = [float(vertex.get("X1")) for vertex in vertices]
        points[:, second_axis] = [float(vertex.get("X2")) for vertex in vertices]
        points[:, normal_axis] = float(primitive.get("Elevation"))
        points *= scale
        face = pv.PolyData(points, faces=[len(vertices), *range(len(vertices))])
        if primitive.tag == "Polygon":
            return face.triangulate()
        extrusion = np.zeros(3)
        extrusion[normal_axis] = float(primitive.get("Length")) * scale
        return face.extrude(extrusion, capping=True).triangulate()
    return None


def _load_structure(xml_path: Path) -> dict[str, list[pv.PolyData]]:
    """
    Read the bodies of a CSXCAD ``structure.xml``, grouped by material, in mm.

    Parameters
    ----------
    xml_path : Path
        The ``structure.xml`` written next to the simulation results.

    Returns
    -------
    dict[str, list[pyvista.PolyData]]
        ``"copper"``, ``"ground"`` and ``"dielectric"`` bodies. A metal whose
        name contains "ground" or "gnd" counts as ground. Ports, probes and
        dump boxes are left out.
    """
    root = ElementTree.parse(xml_path).getroot()
    grid = root.find(".//RectilinearGrid")
    # Coordinates are in drawing units of DeltaUnit metres; convert them to mm.
    scale = float(grid.get("DeltaUnit", "0.001")) * 1e3 if grid is not None else 1.0
    bodies: dict[str, list[pv.PolyData]] = {
        "copper": [],
        "ground": [],
        "dielectric": [],
    }
    skipped = set()
    for prop in root.find(".//Properties"):
        if prop.tag in ("Metal", "ConductingSheet"):
            name = prop.get("Name", "").lower()
            group = "ground" if "ground" in name or "gnd" in name else "copper"
        elif prop.tag == "Material":
            group = "dielectric"
        else:
            continue
        primitives = prop.find("Primitives")
        for primitive in [] if primitives is None else primitives:
            transformation = primitive.find("Transformation")
            mesh = None
            if transformation is None or len(transformation) == 0:
                mesh = _primitive_mesh(primitive, scale)
            if mesh is None:
                skipped.add(primitive.tag)
                continue
            bodies[group].append(mesh)
    if skipped:
        console.print(
            "[warning]The field viewer does not draw these primitives: "
            f"{', '.join(sorted(skipped))}.[/warning]"
        )
    return bodies


def _interpolation_matrix(
    x_lines: NDArray, y_lines: NDArray, z_lines: NDArray, points: NDArray
) -> sparse.csr_matrix:
    """
    Build the trilinear weights that carry grid values to arbitrary points.

    Multiplying the matrix by a field flattened in (z, y, x) order gives the
    field at ``points``; points outside the grid get zero. Building it once
    lets every animation frame interpolate with a single sparse product.

    Parameters
    ----------
    x_lines, y_lines, z_lines : NDArray
        The grid lines, ascending.
    points : NDArray
        Points of shape ``(n, 3)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n, nz * ny * nx)``.
    """
    lower_indices, fractions = [], []
    inside = np.ones(len(points), dtype=bool)
    for axis, lines in enumerate((x_lines, y_lines, z_lines)):
        coordinate = points[:, axis]
        inside &= (coordinate >= lines[0] - 1e-9) & (coordinate <= lines[-1] + 1e-9)
        if len(lines) == 1:
            lower_indices.append(np.zeros(len(points), dtype=int))
            fractions.append(np.zeros(len(points)))
            continue
        lower = np.clip(np.searchsorted(lines, coordinate) - 1, 0, len(lines) - 2)
        span = lines[lower + 1] - lines[lower]
        fractions.append(np.clip((coordinate - lines[lower]) / span, 0.0, 1.0))
        lower_indices.append(lower)
    x_count, y_count, z_count = len(x_lines), len(y_lines), len(z_lines)
    rows, columns, weights = [], [], []
    for x_step in (0, 1):
        for y_step in (0, 1):
            for z_step in (0, 1):
                x_index = np.minimum(lower_indices[0] + x_step, x_count - 1)
                y_index = np.minimum(lower_indices[1] + y_step, y_count - 1)
                z_index = np.minimum(lower_indices[2] + z_step, z_count - 1)
                weight = (
                    (fractions[0] if x_step else 1 - fractions[0])
                    * (fractions[1] if y_step else 1 - fractions[1])
                    * (fractions[2] if z_step else 1 - fractions[2])
                    * inside
                )
                rows.append(np.arange(len(points)))
                columns.append((z_index * y_count + y_index) * x_count + x_index)
                weights.append(weight)
    return sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(points), x_count * y_count * z_count),
    )


def _top_on_grid(body: pv.PolyData, x_lines: NDArray, y_lines: NDArray) -> pv.PolyData:
    """
    Rebuild a body's top face on the field grid lines.

    The field varies on the scale of the grid, so a top face cut along the grid
    lines carries all of it, without splitting a large board into tiny
    triangles. Cells whose centre lies inside the face are kept.
    """
    body = body.triangulate().compute_normals(
        cell_normals=True, point_normals=False, split_vertices=False
    )
    triangles = body.faces.reshape(-1, 4)[:, 1:]
    corners = body.points[triangles]
    top = body.bounds.z_max
    is_top = (np.abs(body.cell_data["Normals"][:, 2]) > 0.99) & np.all(
        np.isclose(corners[:, :, 2], top), axis=1
    )
    top_corners = corners[is_top][:, :, :2]
    if len(top_corners) == 0:
        return pv.PolyData()
    low, high = (
        top_corners.reshape(-1, 2).min(axis=0),
        top_corners.reshape(-1, 2).max(axis=0),
    )
    cut_x = np.unique(
        np.r_[x_lines[(x_lines > low[0]) & (x_lines < high[0])], low[0], high[0]]
    )
    cut_y = np.unique(
        np.r_[y_lines[(y_lines > low[1]) & (y_lines < high[1])], low[1], high[1]]
    )
    centre_x, centre_y = np.meshgrid(
        (cut_x[:-1] + cut_x[1:]) / 2, (cut_y[:-1] + cut_y[1:]) / 2, indexing="xy"
    )
    centres = np.column_stack([centre_x.ravel(), centre_y.ravel()])
    # Barycentric test of every cell centre against every top triangle.
    first, second, third = (top_corners[:, index][None] for index in range(3))
    offset = centres[:, None] - first
    edge_one, edge_two = second - first, third - first
    determinant = (
        edge_one[..., 0] * edge_two[..., 1] - edge_one[..., 1] * edge_two[..., 0]
    )
    determinant = np.where(np.abs(determinant) < 1e-15, np.nan, determinant)
    weight_one = (
        offset[..., 0] * edge_two[..., 1] - offset[..., 1] * edge_two[..., 0]
    ) / determinant
    weight_two = (
        edge_one[..., 0] * offset[..., 1] - edge_one[..., 1] * offset[..., 0]
    ) / determinant
    inside = (
        (weight_one >= -1e-9)
        & (weight_two >= -1e-9)
        & (weight_one + weight_two <= 1 + 1e-9)
    ).any(axis=1)
    plane = pv.RectilinearGrid(cut_x, cut_y, [top]).cast_to_unstructured_grid()
    return plane.extract_cells(inside).extract_surface().triangulate()


def _sides_and_bottom(body: pv.PolyData) -> pv.PolyData:
    """Return a body's surface without its top face."""
    body = body.triangulate()
    triangles = body.faces.reshape(-1, 4)[:, 1:]
    on_top = np.all(
        np.isclose(body.points[triangles][:, :, 2], body.bounds.z_max), axis=1
    )
    return body.extract_cells(~on_top).extract_surface()


def _without_shared_bottom(body: pv.PolyData, metal_heights: NDArray) -> pv.PolyData:
    """Drop a dielectric's bottom face where it lies on metal, to stop flicker."""
    bottom = body.bounds.z_min
    if not np.any(np.isclose(metal_heights, bottom)):
        return body
    body = body.triangulate()
    triangles = body.faces.reshape(-1, 4)[:, 1:]
    on_bottom = np.all(np.isclose(body.points[triangles][:, :, 2], bottom), axis=1)
    return body.extract_cells(~on_bottom).extract_surface()


def _merged(meshes: list[pv.PolyData]) -> pv.PolyData:
    """Merge surfaces into one, or return an empty surface."""
    meshes = [mesh for mesh in meshes if mesh.n_points]
    return pv.merge(meshes).extract_surface() if meshes else pv.PolyData()


class FieldViewer(QtWidgets.QMainWindow):
    """
    Window that animates a field dump over its structure.

    A PyVista view fills the window, with a control panel docked on the left.
    Create it with :func:`view_field_dump`.

    Parameters
    ----------
    dump : _FieldDump
        The field to show first.
    structure : dict[str, list[pyvista.PolyData]]
        The ``"copper"``, ``"ground"`` and ``"dielectric"`` bodies to draw.
    screenshot_directory : Path
        Where "Save screenshot" writes its images.
    off_screen : bool, optional
        Render into an off-screen plotter instead of a window, for checks
        without a display. Default is ``False``.
    dump_directory : Path, optional
        The ``field_dump`` directory the other dumps are read from when picked.
    dump_choices : list[tuple[str, float | None]], optional
        The ``(prefix, frequency)`` of every dump the picker offers, from
        :func:`_field_dump_choices`. Defaults to just ``dump``.
    """

    def __init__(
        self,
        dump: _FieldDump,
        structure: dict[str, list[pv.PolyData]],
        screenshot_directory: Path,
        off_screen: bool = False,
        dump_directory: Path | None = None,
        dump_choices: list[tuple[str, float | None]] | None = None,
    ) -> None:
        super().__init__()
        self.resize(1500, 900)
        self.screenshot_directory = screenshot_directory
        self.dump_directory = dump_directory
        self.dump_choices = dump_choices or [(dump.prefix, dump.frequency)]
        # Dumps read so far, so switching back to one is instant.
        self.loaded_dumps = {(dump.prefix, dump.frequency): dump}
        self.field_actor = None
        self.displayed_mesh = None
        self.geometry_actors = {}

        metal_heights = np.array(
            [
                height
                for body in structure["copper"] + structure["ground"]
                for height in (body.bounds.z_min, body.bounds.z_max)
            ]
        )
        self.structure = structure
        self.geometry = {
            "copper": _merged(structure["copper"]),
            "ground": _merged(structure["ground"]),
            "dielectric": _merged(
                [
                    _without_shared_bottom(body, metal_heights)
                    for body in structure["dielectric"]
                ]
            ),
        }
        self.use_dump(dump)

        if off_screen:
            self.plotter = pv.Plotter(off_screen=True, window_size=(1100, 850))
            self.setCentralWidget(QtWidgets.QLabel("off-screen"))
        else:
            self.plotter = QtInteractor(self)
            self.setCentralWidget(self.plotter.interactor)
        self.use_dark_theme = pv.global_theme.name == PYVISTA_THEME_NAME
        self.build_control_panel()
        self.add_geometry()
        self.heading_text = self.plotter.add_text(
            _choice_label(dump.prefix, dump.frequency),
            position="upper_right",
            font_size=10,
        )
        self.step_text = self.plotter.add_text(
            self.step_label(0), position="lower_right", font_size=10
        )
        self.plotter.add_axes()
        # Sort the translucent substrate and field surfaces correctly.
        self.plotter.enable_depth_peeling(number_of_peels=8)
        self.apply_projection()
        self.apply_background()
        self.rebuild_field()
        self.update_visible_controls()
        self.set_camera("Iso")

        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.setInterval(40)
        self.animation_timer.timeout.connect(self.advance_step)

        # Close on "q" like a plain PyVista window. A shortcut catches the key even
        # while the 3-D view has keyboard focus.
        quit_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Q"), self)
        quit_shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        quit_shortcut.activated.connect(self.close)

        if self.use_dark_theme:
            _apply_dark_qt_palette(self)
            self.style_check_boxes()

    def use_dump(self, dump: _FieldDump) -> None:
        """Set up everything that depends on the dump shown."""
        self.dump = dump
        self.setWindowTitle(
            f"simpleEMS field viewer - {_choice_label(dump.prefix, dump.frequency)}"
        )
        self.x_lines, self.y_lines, self.z_lines = (
            dump.x_lines,
            dump.y_lines,
            dump.z_lines,
        )
        symbol = dump.symbol
        self.magnitude_name = f"|{symbol}|"
        self.quantities = [
            self.magnitude_name,
            f"{symbol}x",
            f"{symbol}y",
            f"{symbol}z",
        ]
        # Largest |F| each grid point reaches, used to skip weak arrows.
        self.cycle_peak_magnitude = dump.peak_magnitude()
        peak_component = dump.peak_component()
        self.peak_by_quantity = {
            self.magnitude_name: float(self.cycle_peak_magnitude.max()),
            f"{symbol}x": float(peak_component[0]),
            f"{symbol}y": float(peak_component[1]),
            f"{symbol}z": float(peak_component[2]),
        }
        # Arrow and height sizes scale with the field region, so they suit a
        # 60 GHz patch and a 1.5 GHz filter alike.
        self.size_unit = (
            max(np.ptp(self.x_lines), np.ptp(self.y_lines)) / 180
            if len(self.x_lines) > 1
            else 0.1
        )
        self.step = 0
        # Each dump has its own grid, so the interpolation weights start afresh.
        self.volume_image = None
        self.volume_weights = None
        self.structure_surface = None
        self.structure_weights = None
        self.plane_index = self.choose_plane_index()
        # Arrows in a plane inside the copper would hide in it, so the arrow plane
        # plot draws them on the copper's top face, the way surface current shows.
        plane_height = self.z_lines[self.plane_index]
        copper_tops = [
            body.bounds.z_max
            for body in self.structure["copper"]
            if body.bounds.z_min - 1e-9 <= plane_height <= body.bounds.z_max + 1e-9
        ]
        self.arrow_plane_height = max(copper_tops, default=plane_height)

    def choose_plane_index(self) -> int:
        """
        Pick the grid plane the plane plots show.

        An electric field is shown in the middle of the dielectric, where a
        patch or line holds its field. A current is shown in the copper, where
        the surface current flows: curl H also counts the displacement current
        in the substrate, which can outweigh it there. Otherwise the plane where
        the field is strongest is shown.
        """
        dielectric = self.geometry["dielectric"]
        if self.dump.symbol == "E" and dielectric.n_points:
            middle = (dielectric.bounds.z_min + dielectric.bounds.z_max) / 2
            if self.z_lines[0] <= middle <= self.z_lines[-1]:
                return int(np.argmin(np.abs(self.z_lines - middle)))
        plane_strength = (self.cycle_peak_magnitude**2).sum(axis=(1, 2))
        if self.dump.symbol == "J":
            in_copper = np.zeros(len(self.z_lines), dtype=bool)
            for body in self.structure["copper"]:
                in_copper |= (self.z_lines >= body.bounds.z_min - 1e-9) & (
                    self.z_lines <= body.bounds.z_max + 1e-9
                )
            if in_copper.any():
                return int(np.argmax(np.where(in_copper, plane_strength, -1.0)))
        return int(np.argmax(plane_strength))

    # ------------------------------------------------------------------ controls

    def build_control_panel(self) -> None:
        """Build the docked control panel."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        plot_group = QtWidgets.QGroupBox("Field plot")
        plot_form = self.plot_form = QtWidgets.QFormLayout(plot_group)
        self.dump_box = QtWidgets.QComboBox()
        self.dump_box.addItems([_choice_label(*choice) for choice in self.dump_choices])
        self.dump_box.setCurrentIndex(
            self.dump_choices.index((self.dump.prefix, self.dump.frequency))
        )
        self.dump_box.currentIndexChanged.connect(lambda _index: self.on_dump_changed())
        self.plot_type_box = self.make_combo(PLOT_TYPES, self.on_plot_type_changed)
        self.quantity_box = self.make_combo(self.quantities, self.on_plot_type_changed)
        self.arrow_spacing_slider, spacing_row = self.make_slider(
            1, 8, 3, self.refresh_field, lambda value: f"every {value}"
        )
        self.arrow_size_slider, size_row = self.make_slider(
            1,
            40,
            15,
            self.refresh_field,
            lambda value: f"{value * self.size_unit:.2g} mm",
        )
        self.arrow_scaling_box = QtWidgets.QCheckBox("Scale arrows by magnitude")
        self.arrow_scaling_box.setChecked(True)
        self.arrow_scaling_box.toggled.connect(self.refresh_field)
        self.iso_level_slider, iso_row = self.make_slider(
            5, 95, 35, self.refresh_field, lambda value: f"{value} % of peak"
        )
        self.height_slider, height_row = self.make_slider(
            1,
            50,
            20,
            self.refresh_field,
            lambda value: f"{value * self.size_unit:.2g} mm",
        )
        plot_form.addRow("Field dump", self.dump_box)
        plot_form.addRow("Plot type", self.plot_type_box)
        plot_form.addRow("Quantity", self.quantity_box)
        plot_form.addRow("Arrow spacing", spacing_row)
        plot_form.addRow("Arrow size", size_row)
        plot_form.addRow("", self.arrow_scaling_box)
        plot_form.addRow("Iso level", iso_row)
        plot_form.addRow("Height scale", height_row)
        layout.addWidget(plot_group)

        color_group = QtWidgets.QGroupBox("Colour scale")
        color_form = self.color_form = QtWidgets.QFormLayout(color_group)
        self.color_map_box = self.make_combo(list(COLOR_MAPS), self.rebuild_field)
        self.scale_box = self.make_combo(
            ["Linear", "Log (dB-like)"], self.rebuild_field
        )
        self.range_slider, range_row = self.make_slider(
            5, 100, 100, self.rebuild_field, lambda value: f"{value} % of peak"
        )
        self.field_opacity_slider, field_opacity_row = self.make_slider(
            10, 100, 100, self.rebuild_field, lambda value: f"{value} %"
        )
        color_form.addRow("Colour map", self.color_map_box)
        color_form.addRow("Scale", self.scale_box)
        color_form.addRow("Range max", range_row)
        color_form.addRow("Opacity", field_opacity_row)
        layout.addWidget(color_group)

        animation_group = QtWidgets.QGroupBox("Animation")
        animation_form = QtWidgets.QFormLayout(animation_group)
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self.on_play_toggled)
        # A time-domain dump steps through frames, a frequency-domain one through
        # degrees of phase; set_animation_range() sets the ranges for the dump.
        self.step_slider, step_row = self.make_slider(
            0,
            0,
            0,
            self.on_step_slider_moved,
            lambda value: (
                f"step {self.dump.timesteps[value]}"
                if self.dump.is_time_domain
                else f"{value} deg"
            ),
        )
        self.speed_slider, speed_row = self.make_slider(
            1,
            1,
            1,
            lambda: None,
            lambda value: (
                f"{value} frame / tick"
                if self.dump.is_time_domain
                else f"{value} deg / frame"
            ),
        )
        self.step_name_label = QtWidgets.QLabel()
        self.set_animation_range()
        animation_form.addRow(self.play_button)
        animation_form.addRow(self.step_name_label, step_row)
        animation_form.addRow("Speed", speed_row)
        layout.addWidget(animation_group)

        geometry_group = self.geometry_group = QtWidgets.QGroupBox("Geometry")
        geometry_form = self.geometry_form = QtWidgets.QFormLayout(geometry_group)
        self.visibility_boxes = {}
        for name, label in [
            ("copper", "Copper"),
            ("ground", "Ground plane"),
            ("dielectric", "Substrate"),
            ("outline", "Field region outline"),
        ]:
            check_box = QtWidgets.QCheckBox(label)
            check_box.setChecked(name != "outline")
            check_box.toggled.connect(self.apply_geometry_visibility)
            self.visibility_boxes[name] = check_box
            geometry_form.addRow(check_box)
        self.copper_opacity_slider, copper_row = self.make_slider(
            0, 100, 100, self.apply_geometry_visibility, lambda value: f"{value} %"
        )
        self.substrate_opacity_slider, substrate_row = self.make_slider(
            0, 100, 45, self.apply_geometry_visibility, lambda value: f"{value} %"
        )
        geometry_form.addRow("Copper opacity", copper_row)
        geometry_form.addRow("Substrate opacity", substrate_row)
        layout.addWidget(geometry_group)

        view_group = QtWidgets.QGroupBox("View")
        view_layout = QtWidgets.QGridLayout(view_group)
        for index, name in enumerate(["Top", "Iso", "Front", "Side"]):
            button = QtWidgets.QPushButton(name)
            button.clicked.connect(
                lambda _checked=False, view=name: self.set_camera(view)
            )
            view_layout.addWidget(button, 0, index)
        self.background_box = self.make_combo(
            list(BACKGROUNDS), self.on_background_changed
        )
        if not self.use_dark_theme:
            # The labels it recolours do not exist yet, so skip the change signal.
            self.background_box.blockSignals(True)
            self.background_box.setCurrentText("Light")
            self.background_box.blockSignals(False)
        screenshot_button = QtWidgets.QPushButton("Save screenshot")
        screenshot_button.clicked.connect(self.save_screenshot)
        view_layout.addWidget(QtWidgets.QLabel("Background"), 1, 0)
        view_layout.addWidget(self.background_box, 1, 1, 1, 3)
        self.orthographic_box = QtWidgets.QCheckBox("Orthographic (CAD) projection")
        self.orthographic_box.setChecked(True)
        self.orthographic_box.toggled.connect(self.apply_projection)
        view_layout.addWidget(self.orthographic_box, 2, 0, 1, 4)
        view_layout.addWidget(screenshot_button, 3, 0, 1, 4)
        layout.addWidget(view_group)
        layout.addStretch()
        # Let each group shrink when rows are hidden instead of keeping the gap.
        for group in panel.findChildren(QtWidgets.QGroupBox):
            group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(panel)
        scroll_area.setWidgetResizable(True)
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setWidget(scroll_area)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setMinimumWidth(360)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def set_animation_range(self) -> None:
        """Fit the step and speed sliders to the dump: frames or degrees."""
        if self.dump.is_time_domain:
            self.step_name_label.setText("Timestep")
            self.step_slider.setRange(0, self.dump.step_count - 1)
            self.speed_slider.setRange(1, 10)
            self.speed_slider.setValue(1)
        else:
            self.step_name_label.setText("Phase")
            self.step_slider.setRange(0, 359)
            self.speed_slider.setRange(1, 30)
            self.speed_slider.setValue(6)
        self.step_slider.setValue(0)

    def make_combo(
        self, items: list[str], on_change: Callable[[], None]
    ) -> QtWidgets.QComboBox:
        """Create a drop-down list that calls ``on_change`` when the choice changes."""
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.currentIndexChanged.connect(lambda _index: on_change())
        return combo

    def make_slider(
        self,
        minimum: int,
        maximum: int,
        value: int,
        on_change: Callable[[], None],
        describe: Callable[[int], str],
    ) -> tuple[QtWidgets.QSlider, QtWidgets.QWidget]:
        """Create a slider with a value label; return the slider and its row."""
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        value_label = QtWidgets.QLabel(describe(value))
        value_label.setMinimumWidth(110)

        def on_value_changed(new_value: int) -> None:
            value_label.setText(describe(new_value))
            on_change()

        slider.valueChanged.connect(on_value_changed)
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider)
        row_layout.addWidget(value_label)
        return slider, row

    def style_check_boxes(self) -> None:
        """
        Give the check boxes a light border and a ticked accent fill.

        The dark palette leaves them with a dark border on a dark panel. The
        style goes on each check box: a window-wide style sheet would override
        the dark palette.
        """
        checkmark_path = Path(tempfile.gettempdir()) / "simpleems_checkmark.svg"
        checkmark_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
            '<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="#ffffff" '
            'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
        check_box_style = f"""
            QCheckBox {{ color: {SECONDARY_TEXT}; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px; border-radius: 3px;
                border: 1.5px solid {SECONDARY_TEXT}; background: transparent;
            }}
            QCheckBox::indicator:hover {{ border-color: {SERIES_COLORS[0]}; }}
            QCheckBox::indicator:checked {{
                border-color: {SERIES_COLORS[0]}; background: {SERIES_COLORS[0]};
                image: url({checkmark_path.as_posix()});
            }}
        """
        for check_box in self.findChildren(QtWidgets.QCheckBox):
            check_box.setStyleSheet(check_box_style)

    def update_visible_controls(self) -> None:
        """Show only the controls that change the current plot; hide the rest."""
        plot_type = self.plot_type_box.currentText()
        is_vector = plot_type.startswith("Vector")
        is_volume = plot_type == "Volume"
        geometry_shown = plot_type not in PLANE_ONLY_PLOT_TYPES
        on_structure = plot_type == "Surface - on structure"
        # The opaque field plane covers the ground in "Three planes".
        ground_seen = (
            geometry_shown
            and not on_structure
            and plot_type != "Three planes"
            and self.geometry["ground"].n_points > 0
        )
        row_visibility = [
            (self.plot_form, self.dump_box, len(self.dump_choices) > 1),
            (self.plot_form, self.quantity_box, not is_vector and not is_volume),
            (self.plot_form, self.arrow_spacing_slider, is_vector),
            (self.plot_form, self.arrow_size_slider, is_vector),
            (self.plot_form, self.arrow_scaling_box, is_vector),
            (self.plot_form, self.iso_level_slider, plot_type == "Iso-surfaces"),
            (self.plot_form, self.height_slider, plot_type == "Height map - plane"),
            # A log scale only makes sense for a magnitude, not a signed component.
            (
                self.color_form,
                self.scale_box,
                self.current_quantity() == self.magnitude_name,
            ),
            # Opacity suits surfaces; on thin arrows it just looks broken.
            (
                self.color_form,
                self.field_opacity_slider,
                not is_volume and not is_vector,
            ),
            # On-structure plots paint the copper and substrate themselves.
            (
                self.geometry_form,
                self.visibility_boxes["copper"],
                not on_structure and self.geometry["copper"].n_points > 0,
            ),
            (
                self.geometry_form,
                self.visibility_boxes["dielectric"],
                not on_structure and self.geometry["dielectric"].n_points > 0,
            ),
            (self.geometry_form, self.visibility_boxes["ground"], ground_seen),
            (self.geometry_form, self.copper_opacity_slider, not on_structure),
            (self.geometry_form, self.substrate_opacity_slider, not on_structure),
        ]
        for form, control, visible in row_visibility:
            # Sliders sit inside a row widget together with their value label.
            field = control if form.indexOf(control) >= 0 else control.parentWidget()
            form.setRowVisible(field, visible)
        self.geometry_group.setVisible(geometry_shown)

    # ------------------------------------------------------------------ geometry

    def add_geometry(self) -> None:
        """Add the structure and the field-region outline to the view."""
        field_region = pv.RectilinearGrid(self.x_lines, self.y_lines, self.z_lines)
        geometry_meshes = {
            "copper": self.geometry["copper"],
            "copper_edges": self.geometry["copper"].extract_feature_edges()
            if self.geometry["copper"].n_points
            else pv.PolyData(),
            "ground": self.geometry["ground"],
            "dielectric": self.geometry["dielectric"],
            "outline": field_region.outline(),
        }
        styles = {
            "copper": {"color": "#d9a441", "specular": 0.6},
            "copper_edges": {"color": "#5a3d0c", "line_width": 1},
            "ground": {"color": "#9c6a2a", "specular": 0.4},
            "dielectric": {"color": "#3f8f5a", "opacity": 0.45},
            "outline": {"color": "#9aa3b5"},
        }
        for name, mesh in geometry_meshes.items():
            if mesh.n_points == 0:
                # Keep an invisible placeholder so the visibility code stays simple.
                mesh = pv.PolyData(np.zeros((1, 3)))
                styles[name] = {"opacity": 0.0}
            self.geometry_actors[name] = self.plotter.add_mesh(
                mesh, reset_camera=False, **styles[name]
            )
        self.apply_geometry_visibility()

    def apply_projection(self) -> None:
        """Switch between orthographic and perspective projection."""
        if self.orthographic_box.isChecked():
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.plotter.render()

    def apply_geometry_visibility(self) -> None:
        """Show, hide and fade the structure to match the controls."""
        plot_type = self.plot_type_box.currentText()
        plane_only = plot_type in PLANE_ONLY_PLOT_TYPES
        for name, check_box in self.visibility_boxes.items():
            self.geometry_actors[name].SetVisibility(
                check_box.isChecked() and not plane_only
            )
        # On-structure plots draw painted copies of the copper and substrate.
        if plot_type == "Surface - on structure":
            self.geometry_actors["copper"].SetVisibility(False)
            self.geometry_actors["dielectric"].SetVisibility(False)
        self.geometry_actors["copper_edges"].SetVisibility(
            self.visibility_boxes["copper"].isChecked() and not plane_only
        )
        if self.geometry["dielectric"].n_points:
            self.geometry_actors["dielectric"].GetProperty().SetOpacity(
                self.substrate_opacity_slider.value() / 100
            )
        if self.geometry["copper"].n_points:
            self.geometry_actors["copper"].GetProperty().SetOpacity(
                self.copper_opacity_slider.value() / 100
            )
        self.plotter.render()

    # ------------------------------------------------------------------ field

    def step_label(self, step: int) -> str:
        """Describe an animation step for the on-screen label."""
        if self.dump.is_time_domain:
            return f"Timestep = {self.dump.timesteps[step]}"
        return f"Phase = {step} deg"

    def quantity_values(self, field: NDArray, quantity: str) -> NDArray:
        """Return the magnitude or one component of a field array."""
        if quantity == self.magnitude_name:
            return np.linalg.norm(field, axis=-1)
        return field[..., "xyz".index(quantity[-1])]

    def current_quantity(self) -> str:
        """Return the quantity the current plot colours by."""
        plot_type = self.plot_type_box.currentText()
        if plot_type.startswith("Vector") or plot_type == "Volume":
            return self.magnitude_name
        return self.quantity_box.currentText()

    def color_limits(self) -> tuple[list[float], bool]:
        """Return the colour range and whether it is logarithmic."""
        quantity = self.current_quantity()
        upper = self.peak_by_quantity[quantity] * self.range_slider.value() / 100
        use_log = self.scale_box.currentIndex() == 1
        if quantity == self.magnitude_name:
            return ([upper * 1e-3, upper] if use_log else [0.0, upper]), use_log
        return [-upper, upper], False  # signed components stay linear

    def painted_structure(self) -> tuple[pv.PolyData, sparse.csr_matrix]:
        """
        Build the copper and substrate surfaces the on-structure plot paints.

        Top faces are rebuilt on the field grid, so large boards stay light;
        sides keep the shape of the structure. Built once, with the weights
        that interpolate the field onto it.
        """
        if self.structure_surface is None:
            pieces = []
            for body in self.structure["dielectric"] + self.structure["copper"]:
                pieces.append(_top_on_grid(body, self.x_lines, self.y_lines))
                pieces.append(_sides_and_bottom(body))
            surface = _merged(pieces)
            self.structure_surface = pv.PolyData(surface.points, surface.faces)
            self.structure_weights = _interpolation_matrix(
                self.x_lines, self.y_lines, self.z_lines, surface.points
            )
        return self.structure_surface, self.structure_weights

    def build_field_mesh(self) -> pv.DataSet:
        """Build the mesh of the current plot at the current animation step."""
        plot_type = self.plot_type_box.currentText()
        quantity = self.current_quantity()
        field = self.dump.field_at(self.step)
        values = self.quantity_values(field, quantity)
        clim, use_log = self.color_limits()
        if use_log:
            values = np.maximum(values, clim[0])
        plane_index = self.plane_index

        if plot_type.startswith("Vector"):
            spacing = self.arrow_spacing_slider.value()
            z_indices = (
                [plane_index]
                if plot_type.endswith("plane")
                else range(len(self.z_lines))
            )
            x_mesh, y_mesh = np.meshgrid(
                self.x_lines[::spacing], self.y_lines[::spacing], indexing="xy"
            )
            # Skip arrows that stay below 2 % of the peak at every step: they draw
            # as specks but made up most of the arrows and most of the frame time.
            arrow_threshold = 0.02 * self.peak_by_quantity[self.magnitude_name]
            arrow_length = self.arrow_size_slider.value() * self.size_unit
            points, vectors = [np.zeros((0, 3))], [np.zeros((0, 3))]
            for z_index in z_indices:
                height = self.z_lines[z_index]
                if plot_type.endswith("plane") and self.arrow_plane_height > height:
                    # Lift the arrows clear of the copper by their tip radius.
                    height = self.arrow_plane_height + 0.12 * arrow_length
                strong = (
                    self.cycle_peak_magnitude[z_index, ::spacing, ::spacing].ravel()
                    > arrow_threshold
                )
                points.append(
                    np.column_stack(
                        [
                            x_mesh.ravel(),
                            y_mesh.ravel(),
                            np.full(x_mesh.size, height),
                        ]
                    )[strong]
                )
                vectors.append(
                    field[z_index, ::spacing, ::spacing].reshape(-1, 3)[strong]
                )
            arrow_cloud = pv.PolyData(np.vstack(points))
            arrow_cloud["vectors"] = np.vstack(vectors)
            arrow_cloud[quantity] = np.linalg.norm(arrow_cloud["vectors"], axis=1)
            arrow_shape = pv.Arrow(
                tip_length=0.3,
                tip_radius=0.12,
                shaft_radius=0.04,
                tip_resolution=8,
                shaft_resolution=8,
            )
            if arrow_cloud.n_points == 0:
                glyphs = pv.PolyData(np.zeros((1, 3)))
                glyphs[quantity] = np.zeros(1)
            elif self.arrow_scaling_box.isChecked():
                # Arrows reach full length at "Range max" and stop growing above it,
                # so lowering the range enlarges weak arrows as it brightens them.
                # Edge and corner hot spots otherwise shrink the rest to specks.
                glyphs = arrow_cloud.glyph(
                    orient="vectors",
                    scale=quantity,
                    factor=arrow_length,
                    geom=arrow_shape,
                    clamping=True,
                    rng=(0.0, clim[1]),
                )
            else:
                glyphs = arrow_cloud.glyph(
                    orient="vectors", scale=False, factor=arrow_length, geom=arrow_shape
                )
            if use_log:
                glyphs[quantity] = np.maximum(glyphs[quantity], clim[0])
            return glyphs

        if plot_type in ("Surface - plane", "Height map - plane"):
            plane = pv.RectilinearGrid(
                self.x_lines, self.y_lines, [self.z_lines[plane_index]]
            )
            plane[quantity] = values[plane_index].ravel()
            plane = plane.cast_to_structured_grid().extract_surface()
            if plot_type == "Height map - plane":
                height = self.height_slider.value() * self.size_unit
                plane = plane.warp_by_scalar(
                    quantity, factor=height / self.peak_by_quantity[quantity]
                )
            return plane

        if plot_type == "Three planes":
            x_index = len(self.x_lines) // 2
            y_index = len(self.y_lines) // 2
            planes = []
            plane = pv.RectilinearGrid(
                self.x_lines, self.y_lines, [self.z_lines[plane_index]]
            )
            plane[quantity] = values[plane_index].ravel()
            planes.append(plane)
            plane = pv.RectilinearGrid(
                [self.x_lines[x_index]], self.y_lines, self.z_lines
            )
            plane[quantity] = values[:, :, x_index].ravel()
            planes.append(plane)
            plane = pv.RectilinearGrid(
                self.x_lines, [self.y_lines[y_index]], self.z_lines
            )
            plane[quantity] = values[:, y_index, :].ravel()
            planes.append(plane)
            return pv.merge([plane.cast_to_structured_grid() for plane in planes])

        if plot_type == "Surface - on structure":
            surface, weights = self.painted_structure()
            surface_values = self.quantity_values(
                weights @ field.reshape(-1, 3), quantity
            )
            if use_log:
                surface_values = np.maximum(surface_values, clim[0])
            painted = surface.copy()
            painted[quantity] = surface_values
            return painted

        if plot_type == "Iso-surfaces":
            volume_grid = pv.RectilinearGrid(self.x_lines, self.y_lines, self.z_lines)
            volume_grid[quantity] = values.ravel()
            level = (
                self.iso_level_slider.value() / 100 * self.peak_by_quantity[quantity]
            )
            levels = [level] if quantity == self.magnitude_name else [-level, level]
            surfaces = volume_grid.contour(levels, scalars=quantity)
            if surfaces.n_points == 0:  # nothing crosses the level at this step
                surfaces = pv.PolyData(np.zeros((1, 3)))
                surfaces[quantity] = np.zeros(1)
            return surfaces

        # Volume: a uniform grid, with interpolation weights built once.
        if self.volume_image is None:
            self.volume_image = pv.ImageData(
                dimensions=(110, 110, 32),
                origin=(self.x_lines[0], self.y_lines[0], self.z_lines[0]),
                spacing=(
                    np.ptp(self.x_lines) / 109,
                    np.ptp(self.y_lines) / 109,
                    np.ptp(self.z_lines) / 31,
                ),
            )
            self.volume_weights = _interpolation_matrix(
                self.x_lines, self.y_lines, self.z_lines, self.volume_image.points
            )
        volume_values = np.linalg.norm(
            self.volume_weights @ field.reshape(-1, 3), axis=1
        )
        if use_log:
            volume_values = np.maximum(volume_values, clim[0])
        image = self.volume_image.copy()
        image[quantity] = volume_values
        return image

    def rebuild_field(self) -> None:
        """Recreate the field actor, for a new plot type or colour style."""
        if self.field_actor is not None:
            self.plotter.remove_actor(self.field_actor, render=False)
        # PyVista reuses the colour range of an existing scalar bar with the same
        # title, which would freeze "Range max" and the log scale; start each bar
        # fresh.
        for title in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title, render=False)
        quantity = self.current_quantity()
        clim, use_log = self.color_limits()
        self.displayed_mesh = self.build_field_mesh()
        self.displayed_mesh.set_active_scalars(quantity)
        text_color = BACKGROUNDS[self.background_box.currentText()][2]
        scalar_bar_args = {
            "title": f"{quantity} [{self.dump.unit}]",
            "vertical": True,
            "position_x": 0.02,
            "position_y": 0.35,
            "height": 0.55,
            "width": 0.07,
            "fmt": "%.2e",
            "color": text_color,
            "title_font_size": 14,
            "label_font_size": 12,
            "n_labels": 9,
        }
        style = {
            "scalars": quantity,
            "cmap": COLOR_MAPS[self.color_map_box.currentText()],
            "clim": clim,
            "log_scale": use_log,
            "scalar_bar_args": scalar_bar_args,
            "reset_camera": False,
        }
        if self.plot_type_box.currentText() == "Volume":
            self.field_actor = self.plotter.add_volume(
                self.displayed_mesh, opacity="sigmoid_6", **style
            )
        else:
            self.field_actor = self.plotter.add_mesh(
                self.displayed_mesh,
                opacity=self.field_opacity_slider.value() / 100,
                smooth_shading=True,
                **style,
            )
        self.plotter.render()

    def refresh_field(self) -> None:
        """Push new data into the existing actor: step, plane or arrow changes."""
        if self.displayed_mesh is None:
            return
        self.displayed_mesh.copy_from(self.build_field_mesh())
        self.displayed_mesh.set_active_scalars(
            self.current_quantity()
        )  # copy_from resets it
        self.step_text.SetText(1, self.step_label(self.step))
        self.plotter.render()

    # ------------------------------------------------------------------ events

    def on_dump_changed(self) -> None:
        """Read the picked dump, if not read yet, and show it in the same plot."""
        choice = self.dump_choices[self.dump_box.currentIndex()]
        if choice not in self.loaded_dumps:
            self.statusBar().showMessage(f"Reading {_choice_label(*choice)} ...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            QtWidgets.QApplication.processEvents()
            try:
                self.loaded_dumps[choice] = _load_field_dump(
                    self.dump_directory, *choice
                )
            except (OSError, ValueError) as error:
                QtWidgets.QApplication.restoreOverrideCursor()
                self.statusBar().clearMessage()
                QtWidgets.QMessageBox.warning(self, "Field dump", str(error))
                # Go back to the dump on screen.
                self.dump_box.blockSignals(True)
                self.dump_box.setCurrentIndex(
                    self.dump_choices.index((self.dump.prefix, self.dump.frequency))
                )
                self.dump_box.blockSignals(False)
                return
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()

        # Leave the old mesh alone while the controls change; rebuild_field()
        # draws the new dump once they are set.
        self.displayed_mesh = None
        self.use_dump(self.loaded_dumps[choice])
        quantity_index = self.quantity_box.currentIndex()
        self.quantity_box.blockSignals(True)
        self.quantity_box.clear()
        self.quantity_box.addItems(self.quantities)
        self.quantity_box.setCurrentIndex(quantity_index)
        self.quantity_box.blockSignals(False)
        self.set_animation_range()
        # Refresh the value labels, which depend on the dump's steps and size.
        for slider in (
            self.step_slider,
            self.speed_slider,
            self.arrow_size_slider,
            self.height_slider,
        ):
            slider.valueChanged.emit(slider.value())
        self.heading_text.SetText(3, _choice_label(*choice))
        self.geometry_actors["outline"].mapper.dataset = pv.RectilinearGrid(
            self.x_lines, self.y_lines, self.z_lines
        ).outline()
        self.update_visible_controls()
        self.apply_geometry_visibility()
        self.rebuild_field()

    def on_plot_type_changed(self) -> None:
        """Rebuild the plot and its controls for a new plot type or quantity."""
        self.update_visible_controls()
        self.apply_geometry_visibility()
        self.rebuild_field()

    def on_play_toggled(self, playing: bool) -> None:
        """Start or pause the animation."""
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self.animation_timer.start()
        else:
            self.animation_timer.stop()

    def advance_step(self) -> None:
        """Move the animation on by the speed setting."""
        next_step = (
            self.step_slider.value() + self.speed_slider.value()
        ) % self.dump.step_count
        self.step_slider.setValue(next_step)  # triggers on_step_slider_moved

    def on_step_slider_moved(self) -> None:
        """Show the field at the slider's step."""
        self.step = self.step_slider.value()
        self.refresh_field()

    def on_background_changed(self) -> None:
        """Apply a new background and recolour the labels to match."""
        self.apply_background()
        self.rebuild_field()

    def apply_background(self) -> None:
        """Set the background and the colour of the on-screen labels."""
        bottom, top, text_color = BACKGROUNDS[self.background_box.currentText()]
        self.plotter.set_background(bottom, top=top)
        rgb = pv.Color(text_color).float_rgb
        for annotation in (self.heading_text, self.step_text):
            annotation.GetTextProperty().SetColor(rgb)
        self.plotter.render()

    def set_camera(self, view: str) -> None:
        """Look from the top, front, side or an isometric angle."""
        if view == "Top":
            self.plotter.view_xy()
        elif view == "Front":
            self.plotter.view_xz()
        elif view == "Side":
            self.plotter.view_yz()
        else:
            self.plotter.view_isometric()
        self.frame_camera()

    def frame_camera(self) -> None:
        """Fit where the field is strong, keeping the current view direction."""
        # Weak field often fills the whole dump box, so frame the region above
        # 10 % of the peak, plus a margin.
        strong = self.cycle_peak_magnitude > 0.1 * self.cycle_peak_magnitude.max()
        strong_x = self.x_lines[strong.any(axis=(0, 1))]
        strong_y = self.y_lines[strong.any(axis=(0, 2))]
        margin = 0.4 * max(np.ptp(strong_x), np.ptp(strong_y))
        self.plotter.reset_camera(
            bounds=(
                strong_x.min() - margin,
                strong_x.max() + margin,
                strong_y.min() - margin,
                strong_y.max() + margin,
                self.z_lines[0],
                self.z_lines[-1],
            )
        )
        # Stop the first render from fitting the camera to the whole board again.
        self.plotter.camera_set = True
        self.plotter.render()

    def save_screenshot(self) -> None:
        """Save the view as a PNG next to the other 3-D plots."""
        self.screenshot_directory.mkdir(parents=True, exist_ok=True)
        label = (
            f"step{self.dump.timesteps[self.step]}"
            if self.dump.is_time_domain
            else f"{self.step:03d}deg"
        )
        path = self.screenshot_directory / f"field_dump_{self.dump.prefix}_{label}.png"
        self.plotter.screenshot(path)
        self.statusBar().showMessage(f"Saved {path}", 5000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Stop the animation and release the plotter when the window closes."""
        self.animation_timer.stop()
        self.plotter.close()
        super().closeEvent(event)


def view_field_dump(
    output_path: Path | None = None,
    show: bool = True,
) -> FieldViewer:
    """
    Open an interactive 3-D viewer for a field dump.

    The window animates the field over its structure: over one period for a
    frequency-domain dump, or through the saved timesteps for a time-domain
    dump. When the simulation saved several dumps, such as the electric and
    the magnetic field, or one dump at several frequencies, a "Field dump"
    picker in the window switches between them, starting from the first dump
    and its lowest frequency. It opens without blocking; call
    :meth:`~simpleEMS.sim_tools.SimTools.show_plots` to keep it on screen. Press
    ``q`` to close it.

    Parameters
    ----------
    output_path : Path, optional
        The simulation directory passed to
        :meth:`~simpleEMS.sim_tools.SimTools.add_field_dump`; the dump is read
        from its ``field_dump`` subdirectory and the structure from its
        ``structure.xml``. Defaults to ``cwd / "Sim_Path"``.
    show : bool, optional
        Show the window now. Pass ``False`` to build it hidden and show it
        later, as :meth:`~simpleEMS.sim_tools.SimTools.show_plots` does.
        Defaults to ``True``.

    Returns
    -------
    FieldViewer
        The viewer window.

    Raises
    ------
    FileNotFoundError
        If the directory holds no field dump the viewer can show.
    """
    if output_path is None:
        output_path = Path.cwd() / "Sim_Path"
    dump_directory = output_path / "field_dump"
    dump_choices = _field_dump_choices(dump_directory)
    if not dump_choices:
        raise FileNotFoundError(f"no field dump files found in {dump_directory}")
    prefix, dump_freq = dump_choices[0]
    dump = _load_field_dump(dump_directory, prefix, dump_freq)
    xml_path = output_path / "structure.xml"
    if xml_path.exists():
        structure = _load_structure(xml_path)
    else:
        console.print(
            f"[warning]No structure.xml in {output_path}; showing the field "
            "without the structure.[/warning]"
        )
        structure = {"copper": [], "ground": [], "dielectric": []}

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = FieldViewer(
        dump,
        structure,
        output_path / "3D_plots",
        dump_directory=dump_directory,
        dump_choices=dump_choices,
    )
    # Python frees an unreferenced QApplication, and show_plots() would then find
    # no event loop to run, so the viewer holds on to it.
    viewer.application = application
    # Hand the window to Qt, so it stays open when the caller drops it, and
    # free it once it closes.
    sip.transferto(viewer, None)
    viewer.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    if show:
        viewer.show()
        application.processEvents()
    return viewer
