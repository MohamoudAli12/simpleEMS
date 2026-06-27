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
EMerge FEM simulation backend for simpleEMS.

Provides the FEM simulation pipeline (CSX → STEP → emerge → SimData)
and a duck-typed NF2FF wrapper for far-field plotting compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .sim_params import SimParams
from .sim_tools import SimData

import gmsh

import matplotlib.pyplot as plt

_plt_rc = plt.rc
_plt_update = plt.rcParams.update
plt.rc = lambda *a, **kw: None
plt.rcParams.update = lambda _: None

from emerge._emerge.geo import STEPItems, Plate  # noqa: E402
from emerge._emerge.simmodel import Simulation  # noqa: E402
from emerge._emerge.cs import XAX, YAX, ZAX  # noqa: E402
from emsutil.material import Material  # noqa: E402
from emsutil.lib import PEC, EISO  # noqa: E402

plt.rc = _plt_rc
plt.rcParams.update = _plt_update

MM = 0.001


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emerge_load_step(
    step_path: str | Path,
    unit: float = 0.001,
    label: str = "Model",
) -> dict:
    """Load a STEP file into emerge.

    Parameters
    ----------
    step_path : str | Path
        Path to the ``.step`` / ``.stp`` file.
    unit : float
        Model-unit-to-meter conversion (default 0.001 → mm).
    label : str
        Prefix for volume names in the STEP (default ``"Model"``).

    Returns
    -------
    tuple
        ``(step, vols)`` — the ``STEPItems`` object and its
        ``.dictionary`` of named volumes.
    """

    step = STEPItems(label, str(step_path), unit=unit)
    vols = step.dictionary

    return vols, step


def emerge_assign_materials(vols: dict, er: float, tand: float) -> None:
    """Assign materials to STEP volumes by naming convention.

    * ``substrate`` in name → dielectric with given ``er`` / ``tand``
    * ``port_resist`` in name → skipped
    * everything else → ``PEC``
    """

    for name, vol in vols.items():
        lower_name = name.lower()
        if "port_resist" in lower_name:
            continue
        if "substrate" in lower_name:
            vol.set_material(
                Material(
                    er=er,
                    tand=tand,
                    color="#0F8B00",
                    opacity=0.9,
                )
            )
        else:
            vol.set_material(PEC)


class PortInfo(NamedTuple):
    port_num: int
    origin: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    width: float
    height: float
    direction: object  # XAX/YAX/ZAX
    center: tuple[float, float, float]
    normal_axis: object  # XAX/YAX/ZAX


def detect_ports(vols: dict, mm: float = 0.001) -> list[PortInfo]:
    """Detect lumped-port parameters from ``port_resist_*`` volumes.

    Returns a list of ``PortInfo`` named tuples with fields:

    * ``port_num`` — integer port number
    * ``origin`` — (3,) array of the port face corner in model units
    * ``v1``, ``v2`` — (3,) edge vectors spanning the port face
    * ``width``, ``height`` — face dimensions along the non-normal axes
    * ``direction`` — ``ZAX`` (for use with ``LumpedPort``)
    * ``center`` — ``(x, y, z)`` tuple for ``inplane`` selection
    * ``normal_axis`` — ``XAX``, ``YAX``, or ``ZAX``
    """

    AXES = [XAX, YAX, ZAX]
    ports: list[PortInfo] = []
    for name in sorted(vols.keys()):
        if "port_resist" not in name.lower():
            continue

        vol = vols[name]
        dim, tag = vol.dimtags[0]
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        zmin = -0.035 * mm
        spans = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        na = int(np.argmin(spans))
        port_num = int(name.split("_")[-1])

        axes = [0, 1, 2]
        a, b = [ax for ax in axes if ax != na]

        ports.append(
            PortInfo(
                port_num=port_num,
                center=(xmin, (ymin + ymax) / 2, (zmin + zmax) / 2),
                origin=np.array([xmin, ymin, zmin]),
                v1=np.array([spans[a] if i == a else 0.0 for i in range(3)]),
                v2=np.array([spans[b] if i == b else 0.0 for i in range(3)]),
                width=spans[a],
                height=spans[b],
                direction=ZAX,
                normal_axis=AXES[na],
            )
        )

    return ports


def _emerge_data_to_simdata(
    data: object,
    ports_info: list[dict],
    sim_params: SimParams,
    model: object,
) -> SimData:
    Z0 = sim_params.charac_imp
    min, max = sim_params.freq_range
    freqs = np.linspace(min, max, sim_params.num_points)
    s11 = data.scalar.grid.model_S(1, 1, freqs)

    n_ports = len(ports_info)
    s21 = data.scalar.grid.model_S(2, 1, freqs) if n_ports > 1 else None

    z11 = Z0 * (1 + s11) / (1 - s11)
    s11_mag = np.abs(s11)
    s11_mag = np.clip(s11_mag, 0, 0.999)
    vswr = (1 + s11_mag) / (1 - s11_mag)
    input_power = 0.5 / Z0 * (1.0 - s11_mag**2)

    data.emerge_model = model
    data.emerge_ports_info = ports_info

    return SimData(freqs, s11, s21, z11, vswr, input_power)


@dataclass
class EmergeNF2FFResult:
    """Duck-typed nf2ff_results for SimTools plotting compatibility."""

    theta: np.ndarray
    phi: np.ndarray
    E_norm: list
    Dmax: np.ndarray
    Prad: np.ndarray
    P_rad: list
    E_theta: list
    E_phi: list


class EmergeNF2FF:
    """Duck-typed nf2ff — provides CalcNF2FF() from emerge field data."""

    def __init__(self, field: object, boundary: object) -> None:
        self._field = field
        self._boundary = boundary

    def CalcNF2FF(
        self,
        sim_path: str | Path,
        freq: float,
        theta: np.ndarray | float,
        phi: np.ndarray | float,
        radius: float = 1,
        center: tuple | None = None,
        outfile: str | None = None,
        read_cached: bool = False,
        verbose: int = 0,
    ) -> EmergeNF2FFResult:
        theta_deg = np.atleast_1d(np.asarray(theta, float))
        phi_deg = np.atleast_1d(np.asarray(phi, float))

        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)

        ff3d = self._field.farfield_3d(
            self._boundary,
            thetas=theta_rad,
            phis=phi_rad,
            origin=(0, 0, 0),
        )

        # farfield_3d uses default meshgrid indexing="xy" → shape (Np, Nt)
        # transpose to (Nt, Np) for plotting compatibility
        E_theta_2d = ff3d.Etheta.T
        E_phi_2d = ff3d.Ephi.T
        E_norm_2d = np.sqrt(np.abs(E_theta_2d) ** 2 + np.abs(E_phi_2d) ** 2)
        Ptot = ff3d.Ptot
        dir_norm = E_norm_2d / (EISO * np.sqrt(Ptot))

        return EmergeNF2FFResult(
            theta=theta_deg,
            phi=phi_deg,
            E_norm=[E_norm_2d],
            Dmax=np.array([float(np.max(dir_norm))]),
            Prad=np.array([Ptot]),
            P_rad=[E_norm_2d**2],
            E_theta=[E_theta_2d],
            E_phi=[E_phi_2d],
        )


@dataclass
class FEMResult:
    sim_data: SimData
    nf2ff: EmergeNF2FF
    nf2ff_3d_result: EmergeNF2FFResult
    input_power: float


def run_fem_simulation(
    step_path: str | Path,
    params: SimParams,
    label: str = "Model",
    air_padding_mm: float = 4.0,
    mesh_size_factor: float = 0.20,
    port_face_size_mm: float = 0.01,
) -> FEMResult:
    model = Simulation(f"{label}_FEM")
    vols, step = emerge_load_step(step_path, label=label)
    print(vols)
    emerge_assign_materials(vols, params.substrate_eps_r, params.substrate_tand)
    air = step.enclose(air_padding_mm * MM)

    ports = detect_ports(vols)
    if not ports:
        raise ValueError("No port_resist volumes found in STEP file")
    p = ports[0]
    port_plate = Plate(p.origin, p.v1, p.v2)

    fmin, fmax = params.freq_range
    model.mw.set_frequency_range(fmin, fmax, 10)
    model.mw.set_resolution(mesh_size_factor)
    model.commit_geometry()
    model.mesher.set_face_size(port_plate, port_face_size_mm * MM)
    model.generate_mesh()
    model.view(plot_mesh=True, volume_mesh=True)

    model.mw.bc.LumpedPort(
        port_plate,
        p.port_num,
        width=p.width,
        height=p.height,
        direction=p.direction,
        Z0=params.charac_imp,
    )
    model.mw.bc.AbsorbingBoundary(air.boundary())

    data = model.mw.run_sweep()

    sim_data = _emerge_data_to_simdata(
        data, ports_info=ports, sim_params=params, model=model
    )

    field = data.field.find(freq=params.resonant_freq)
    boundary = air.boundary()
    nf2ff = EmergeNF2FF(field, boundary)

    thetas = np.arange(0, 181, 2)
    phis = np.arange(0, 361, 2)
    nf2ff_3d = nf2ff.CalcNF2FF("", params.resonant_freq, thetas, phis)

    return FEMResult(
        sim_data=sim_data,
        nf2ff=nf2ff,
        nf2ff_3d_result=nf2ff_3d,
        input_power=sim_data.input_power,
    )
