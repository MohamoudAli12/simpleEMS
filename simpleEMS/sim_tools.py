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
Simulation orchestration, post-processing, and utility tools.

Provides the `SimTools` class (namespace for static methods) and
standalone functions for setting up openEMS simulations, running
optimization, parameter sweeps, plotting (S-parameters, VSWR,
Smith chart, radiation patterns, 3D patterns), NF2FF computation,
field dumps, and exports (Touchstone, Gerber, STL). Also includes
unit conversion helpers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Literal, NamedTuple
from collections.abc import Callable
from dataclasses import fields
from enum import Enum
from itertools import product
from pathlib import Path

# ----------------------------
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.ticker import EngFormatter, MaxNLocator
from PyQt6.QtCore import QCoreApplication, QTimer
from PyQt6.QtWidgets import QApplication
from pysmithchart import S_PARAMETER
from pyvistaqt import BackgroundPlotter

# ----------------------------
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

# ----------------------------
from CSXCAD import AppCSXCAD_BIN, ContinuousStructure
from openEMS.openEMS import openEMS
from openEMS.ports import LumpedPort
from openEMS.nf2ff import nf2ff
from openEMS.nf2ff import nf2ff_results

# ----------------------------
from .console import console
from .export_layout import export_dxf, export_gerber, export_kicad_footprint
from .export_cad import export_stl, export_step, export_csxcad_xml_to_step
from .export_touchstone import write_touchstone
from .fdtd_mesh import Mesh, auto_simulation_bounds, geometry_extent
from .fem_field_dump import FieldDumpRequest
from .plot_theme import HIGHLIGHT, style_background_plotter, themed_cursor
from .report import generate_report
from .sim_params import SimParams
from .visualise_field_dump import view_field_dump

# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "DumpType",
    "SimTools",
    "setup_simulation",
    "optimize_s_params",
    "minimize_s_param",
    "maximize_s_param",
    "param_sweep",
    "mm_to_m",
    "m_to_mm",
    "mil_to_mm",
    "mm_to_mil",
    "cm_to_mm",
    "mm_to_cm",
]

freq_formatter = EngFormatter(unit="Hz", places=3)
time_formatter = EngFormatter(unit="s", places=3)

# 3D windows carry this Qt property until show_plots() opens them, so they open
# together with the matplotlib figures instead of sitting frozen while the
# script keeps running. Its value is the time the window was made, so they open
# in that order and the last plot ends up on top.
SHOW_WITH_PLOTS = "simpleems_show_with_plots"


class DumpType(Enum):
    """
    Field and other dump types provided by openEMS.

    Each member's value is a ``(dump_type_code, file_prefix)`` pair, where
    ``dump_type_code`` is the openEMS ``AddDump`` dump-type integer and
    ``file_prefix`` names the dump's output file.

    Attributes
    ----------
    efield_time : tuple[int, str]
        Electric field time-domain dump.
    hfield_time : tuple[int, str]
        Magnetic field time-domain dump.
    current_time : tuple[int, str]
        Electric current time-domain dump.
    current_density_time : tuple[int, str]
        Total current density (rot(H)) time-domain dump.
    efield_frequency : tuple[int, str]
        Electric field frequency-domain dump.
    hfield_frequency : tuple[int, str]
        Magnetic field frequency-domain dump.
    current_frequency : tuple[int, str]
        Electric current frequency-domain dump.
    current_density_frequency : tuple[int, str]
        Total current density (rot(H)) frequency-domain dump.
    local_sar_frequency : tuple[int, str]
        Local SAR frequency-domain dump.
    average_sar_frequency_1g : tuple[int, str]
        1g-averaging SAR frequency-domain dump.
    average_sar_frequency_10g : tuple[int, str]
        10g-averaging SAR frequency-domain dump.
    raw_data : tuple[int, str]
        Raw data needed for SAR calculations (electric field FD, cell
        volume, conductivity, and density).

    Notes
    -----
    This class was adapted from pyems.
    """

    efield_time = (0, "Et")
    hfield_time = (1, "Ht")
    current_time = (2, "It")
    current_density_time = (3, "Jt")
    efield_frequency = (10, "Ef")
    hfield_frequency = (11, "Hf")
    current_frequency = (12, "If")
    current_density_frequency = (13, "Jf")
    local_sar_frequency = (20, "SAR_f")
    average_sar_frequency_1g = (21, "SAR_1g_f")
    average_sar_frequency_10g = (22, "SAR_10g_f")
    raw_data = (29, "raw")


class SimData(NamedTuple):
    """Named tuple holding simulation data/results.

    Attributes
    ----------
    freqs : NDArray
        Frequency points used in post-processing.
    s11 : NDArray
        Complex S11 values across the frequency range.
    s21 : NDArray | None
        Complex S21 values (None if single-port).
    s31 : NDArray | None
        Complex S31 values (None with fewer than three ports).
    s41 : NDArray | None
        Complex S41 values (None with fewer than four ports).
    z11 : NDArray
        Complex Z11 values across the frequency range.
    vswr : NDArray
        VSWR values across the frequency range.
    input_power : float
        Calculated input power at the port.
    port_voltage : NDArray
        Total complex voltage at the driven port across the frequency range.
    port_current : NDArray
        Total complex current at the driven port across the frequency range.
    ref_impedance : float
        Reference impedance S11/Z11 were computed against. For the FEM
        backend's wave ports this is the port's computed line characteristic
        impedance (``Zc``), not necessarily ``charac_imp``; pass it to
        ``plot_smith_chart(..., charac_imp=...)`` to normalize the chart to
        the impedance the data was actually referenced to.
    """

    freqs: NDArray
    s11: NDArray
    s21: NDArray | None
    s31: NDArray | None
    s41: NDArray | None
    z11: NDArray
    vswr: NDArray
    input_power: float
    port_voltage: NDArray
    port_current: NDArray
    ref_impedance: float = 50.0


class SweepData(NamedTuple):
    """Named tuple holding the results of a :func:`param_sweep`.

    It has the same fields as :class:`SimData`, with every run stacked into
    one array so the ``SimTools.plot_*`` functions overlay the whole sweep in
    one call, e.g. ``plot_vswr(sweep_data.freqs, sweep_data.vswr,
    label=sweep_data.labels)``. Row ``i`` of each array belongs to run ``i``.

    Attributes
    ----------
    freqs : NDArray
        Frequency points, shared by every run.
    s11 : NDArray
        Complex S11 values, shape ``(runs, freqs)``.
    s21 : NDArray | None
        Complex S21 values, shape ``(runs, freqs)`` (None if single-port).
    s31 : NDArray | None
        Complex S31 values, shape ``(runs, freqs)`` (None with fewer than
        three ports).
    s41 : NDArray | None
        Complex S41 values, shape ``(runs, freqs)`` (None with fewer than
        four ports).
    z11 : NDArray
        Complex Z11 values, shape ``(runs, freqs)``.
    vswr : NDArray
        VSWR values, shape ``(runs, freqs)``.
    input_power : NDArray
        Input power at the port, one value per run.
    port_voltage : NDArray
        Total complex voltage at the driven port, shape ``(runs, freqs)``.
    port_current : NDArray
        Total complex current at the driven port, shape ``(runs, freqs)``.
    ref_impedance : NDArray
        Reference impedance S11/Z11 were computed against, one value per run.
    labels : list[str]
        Legend label for each run, such as ``"width=1.0, length=20.0"``.
    values : list[dict[str, float]]
        The swept parameter values of each run, keyed by parameter name.
    """

    freqs: NDArray
    s11: NDArray
    s21: NDArray | None
    s31: NDArray | None
    s41: NDArray | None
    z11: NDArray
    vswr: NDArray
    input_power: NDArray
    port_voltage: NDArray
    port_current: NDArray
    ref_impedance: NDArray
    labels: list[str]
    values: list[dict[str, float]]


class EfficiencyCurve(NamedTuple):
    """Named tuple holding an antenna's efficiency across its matched band.

    Attributes
    ----------
    freqs : NDArray
        Frequencies (in Hz) the curve is reported at.
    radiation : NDArray
        Radiation efficiency ``Prad / accepted power`` at ``freqs``, as a
        fraction, fitted through the computed points.
    total : NDArray
        Total efficiency at ``freqs``: the radiation efficiency derated by the
        port mismatch ``1 - |S11|^2``.
    computed_freqs : NDArray
        The frequencies inside the band the efficiency was actually computed
        at; the curve between them is interpolated.
    computed_radiation : NDArray
        Radiation efficiency at ``computed_freqs``.
    computed_total : NDArray
        Total efficiency at ``computed_freqs``.
    """

    freqs: NDArray
    radiation: NDArray
    total: NDArray
    computed_freqs: NDArray
    computed_radiation: NDArray
    computed_total: NDArray


class NF2FFData(NamedTuple):
    """Named tuple holding the far-field results.

    ``compute_nf2ff_data`` fills it in one go, and every radiation plot reads
    from it, so the far field is solved once. Everything but
    ``efficiency_curve`` belongs to the one frequency ``freq``; the curve
    spans the antenna's matched band.

    Attributes
    ----------
    freq : float
        Frequency (in Hz) the far field was computed at.
    sphere : nf2ff_results
        The full sphere: theta in [0, 180] deg and phi in [0, 360] deg, in
        2 deg steps. Its ``Dmax`` and peak ``E_norm`` set the absolute level
        of every directivity and gain plot.
    xz_cut : nf2ff_results
        The xz-plane: theta swept over [-180, 180] deg in 0.1 deg steps at
        phi = 0.
    yz_cut : nf2ff_results
        The yz-plane: theta swept over [-180, 180] deg in 0.1 deg steps at
        phi = 90 deg.
    xy_cut : nf2ff_results
        The xy-plane: phi swept over [-180, 180] deg in 0.1 deg steps at
        theta = 90 deg.
    efficiency : float
        Radiation efficiency ``Prad / accepted power`` at ``freq``.
    mismatch_efficiency : float
        Port mismatch efficiency ``1 - |S11|^2`` at ``freq``.
    efficiency_curve : EfficiencyCurve
        Radiation and total efficiency across the -10 dB band.
    """

    freq: float
    sphere: nf2ff_results
    xz_cut: nf2ff_results
    yz_cut: nf2ff_results
    xy_cut: nf2ff_results
    efficiency: float
    mismatch_efficiency: float
    efficiency_curve: EfficiencyCurve


class SimSetup(NamedTuple):
    """
    Named tuple holding the simulation setup objects.

    Attributes
    ----------
    CSX : ContinuousStructure
        CSXCAD geometry object.
    FDTD : openEMS
        openEMS FDTD object.
    freqs : NDArray
        Frequency points used in simulation and post-processing. For the FEM
        backend these are the interpolated output points, not the solve points.
    backend_engine : str
        Solver backend, either ``"FDTD"`` (default) or ``"FEM"``.
    params : SimParams | None
        The parameters the setup was built from. The FEM backend reads its
        ``FEM_*`` solver and mesh settings off them. ``None`` for a setup
        built without one, such as a standalone FDTD model.
    charac_imp : float
        Reference impedance of the ports, in ohms.
    FEM_field_dumps : list[FieldDumpRequest] | None
        Field dumps :meth:`SimTools.add_field_dump` queued for the FEM
        backend to write after its sweep. ``None`` for FDTD, where the dumps
        live in ``CSX``.
    """

    CSX: ContinuousStructure
    FDTD: openEMS
    freqs: NDArray
    backend_engine: str = "FDTD"
    params: SimParams | None = None
    charac_imp: float = 50
    FEM_field_dumps: list[FieldDumpRequest] | None = None


def setup_simulation(
    params: SimParams,
    FDTD_boundary: list[str] | None = None,
) -> SimSetup:
    """
    Build the CSXCAD geometry container and configure the openEMS/FDTD
    (or FEM) solver from a parameter object.

    Creates the ``ContinuousStructure`` and ``openEMS`` objects, applies the
    boundary conditions and Gaussian excitation derived from
    ``params.freq_range``, and carries ``params`` through to the returned
    :class:`SimSetup`, which is passed to most other ``SimTools`` methods.

    Parameters
    ----------
    params : SimParams
        Parameter object that holds all simulation parameters, including
        ``freq_range``, ``num_points``, ``FDTD_timestep``,
        ``FDTD_end_criteria``, ``backend_engine``, the flat ``FEM_*``
        mesh/solver tuning fields (including ``FEM_num_solve_points``), and
        ``charac_imp``.
    FDTD_boundary : list[str], optional
        Six openEMS boundary-condition strings, one per box face in the
        order ``[xmin, xmax, ymin, ymax, zmin, zmax]``. Only used by the
        FDTD backend. Defaults to ``["PML_8"] * 6``.

    Returns
    -------
    SimSetup
        Named tuple with ``CSX`` (CSXCAD geometry), ``FDTD`` (openEMS FDTD
        object), ``freqs`` (frequency array), ``backend_engine``,
        ``params``, ``charac_imp``, and
        ``FEM_field_dumps`` (an empty list for FEM, ``None`` for FDTD).
    """
    if FDTD_boundary is None:
        FDTD_boundary = [
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8",
        ]
    CSX = ContinuousStructure()
    FDTD = openEMS(NrTS=params.FDTD_timestep, EndCriteria=params.FDTD_end_criteria)

    FDTD.SetBoundaryCond(FDTD_boundary)
    FDTD.SetCSX(CSX)

    fmin, fmax = params.freq_range
    centre_freq = (fmin + fmax) / 2
    span_freq = (fmax - fmin) / 2

    FDTD.SetGaussExcite(centre_freq, span_freq)

    freqs = np.linspace(fmin, fmax, params.num_points)

    return SimSetup(
        CSX=CSX,
        FDTD=FDTD,
        freqs=freqs,
        backend_engine=params.backend_engine,
        params=params,
        charac_imp=params.charac_imp,
        FEM_field_dumps=[] if params.backend_engine == "FEM" else None,
    )


class SimTools:
    """
    Namespace of static methods for simulation orchestration, plotting,
    and export.

    This class is not meant to be instantiated; all methods are
    ``@staticmethod`` and are called directly on the class (e.g.
    ``SimTools.run_simulation(...)``).

    All structure classes (e.g. ``InsetFedPatchAntenna``) inherit this
    class so its methods are available on structure instances too.
    """

    @staticmethod
    def write_and_show_structure(
        sim: SimSetup,
        output_path: Path | None = None,
        theme: str | None = None,
    ) -> None:
        """
        Display the simulation geometry.

        For the FDTD backend this opens the structure in AppCSXCAD. For the
        FEM backend it (re-)meshes the current ``CSX``/``params`` with
        Gmsh -- reusing the existing mesh instead of rebuilding if neither
        has changed since the last call on this ``output_path`` -- and
        shows it with :func:`simpleEMS.fem_backend.show_fem_mesh`: one
        coloured, switchable region per material or boundary, plus a table
        of mesh statistics.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        output_path : Path, optional
            Directory to write/read the structure XML (FDTD) or mesh files
            (FEM). Defaults to ``cwd / "Sim_Path"``, created if missing.
        theme : str, optional
            FEM mesh only. ``None`` (default) keeps the active simpleEMS theme
            (see :func:`simpleEMS.plot_theme.use_dark_theme`). A PyVista theme
            name -- ``"dark"``, ``"default"``, ``"document"`` or ``"paraview"``
            -- is applied globally via :func:`pyvista.set_plot_theme`.

        Raises
        ------
        ValueError
            If ``theme`` is not one of PyVista's native theme names.
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"
            output_path.mkdir(parents=True, exist_ok=True)

        if sim.backend_engine == "FEM":
            from . import fem_backend

            if theme is not None:
                pv.set_plot_theme(theme)

            msh_path = fem_backend.build_mesh(
                sim.CSX, sim.freqs, output_path, params=sim.params
            )
            fem_backend.show_fem_mesh(msh_path)
            return

        structure_3d = output_path / "structure.xml"

        sim.FDTD.Write2XML(
            str(structure_3d)
        )  # str in this fixes an error encountered on Windows.
        subprocess.run([AppCSXCAD_BIN, str(structure_3d)], check=True)

    @staticmethod
    def export_stl(
        sim: SimSetup,
        output_path: Path | None = None,
    ) -> None:
        """
        Export the CSXCAD structure to STL files (FDTD backend only).

        Reads the ``structure.xml`` previously written by
        ``write_and_show_structure`` and exports it to STL via AppCSXCAD.

        Parameters
        ----------
        output_path : Path, optional
            Directory containing ``structure.xml``. STL files are written
            to a ``stl`` subdirectory. Defaults to ``cwd / "Sim_Path"``.

        Raises
        ------
        ValueError
            If ``structure.xml`` does not exist in ``output_path`` (i.e.
            ``write_and_show_structure`` was not called first).
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        stl_path = output_path / "stl"
        stl_path.mkdir(parents=True, exist_ok=True)
        export_stl(CSX=sim.CSX, output_path=stl_path)

    @staticmethod
    def run_simulation(sim: SimSetup, output_path: Path | None = None) -> None:
        """
        Run the simulation and write results to ``output_path``.

        Dispatches to the FDTD engine (``FDTD.Run``) or, for the FEM
        backend, to the adaptive GetDP frequency sweep
        (``fem_backend.run_sweep``), which performs up to
        ``sim.params.FEM_num_solve_points`` full solves. For the FEM backend
        this always (re-)meshes first, reusing the existing mesh if
        ``sim.CSX``/``sim.params`` haven't changed since the last mesh
        in this ``output_path`` (e.g. a prior ``write_and_show_structure``
        call) and rebuilding if they have -- so a sweep/optimize loop that
        calls this directly, varying the geometry across iterations into
        the same ``output_path``, still gets a fresh mesh each time the
        geometry actually changes. The FEM backend then writes any field
        dumps :meth:`add_field_dump` queued, with one extra solve per dump
        frequency.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        output_path : Path, optional
            Directory to write simulation results to. Defaults to
            ``cwd / "Sim_Path"``.
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        # openEMS's Run() chdirs into this path and then asserts that it
        # matches os.getcwd(), so a relative path never compares equal and
        # trips its internal "this should never happen" check.
        output_path = Path(output_path).resolve()

        if sim.backend_engine == "FEM":
            from . import fem_backend

            fem_backend.run_sweep(
                sim.CSX,
                sim.freqs,
                output_path,
                params=sim.params,
            )
            if sim.FEM_field_dumps:
                from . import fem_field_dump

                fem_field_dump.write_field_dumps(sim.FEM_field_dumps, output_path)
            return

        cwd = os.getcwd()
        try:
            sim.FDTD.Run(output_path)
        finally:
            os.chdir(cwd)

    @staticmethod
    def create_nf2ff(sim: SimSetup | None = None) -> nf2ff:
        """
        Create the near-field-to-far-field (NF2FF) recording box.

        Parameters
        ----------
        sim : SimSetup, optional
            Simulation setup named tuple returned by ``setup_simulation``.
            Omit for the standalone STEP-FEM workflow (``simulate_step_FEM``),
            which has no ``SimSetup`` -- ``None`` always returns a
            ``FEMNF2FF`` adapter.

        Returns
        -------
        nf2ff : object
            NF2FF object. For the FEM backend (or when ``sim`` is omitted)
            this is a ``FEMNF2FF`` adapter that exposes the same
            ``CalcNF2FF`` interface, so the radiation-plotting methods work
            identically for both backends.
        """
        if sim is None or sim.backend_engine == "FEM":
            from .fem_radiation import FEMNF2FF

            return FEMNF2FF()

        nf2ff = sim.FDTD.CreateNF2FFBox()
        return nf2ff

    @staticmethod
    def compute_sim_data(
        sim: SimSetup,
        port: LumpedPort | list[LumpedPort],
        output_path: Path | None = None,
    ) -> SimData:
        """
        Compute S-parameters, impedance, VSWR, and port power/voltage/current
        from simulation results, for plotting and post-processing.

        For the FDTD backend this calls ``CalcPort`` on the given port(s) and
        derives S11, the transmission terms S21/S31/S41 for each extra port,
        Z11, VSWR, and input power from the port voltage/current waves. For
        the FEM backend the ``port`` argument is ignored and results are
        instead read from the GetDP sweep output previously written to
        ``output_path`` by ``run_simulation``.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``;
            supplies ``freqs``, ``backend_engine``, and ``charac_imp``.
        port : LumpedPort or list of LumpedPort
            The openEMS port object of a single-port simulation, or a list
            of up to four ports ``[port1, ..., port4]``. The first port is
            the driven one. Ignored when ``sim.backend_engine == "FEM"``.
        output_path : Path, optional
            Directory the simulation results were written to. Defaults to
            ``cwd`` (FDTD backend) or ``cwd / "Sim_Path"`` (FEM backend, to
            match the default used by ``run_simulation``).

        Returns
        -------
        SimData
            Named tuple containing:

            - freqs : NDArray
                Frequency points used in post-processing.
            - s11 : NDArray
                Complex S11 values across the frequency range.
            - s21 : NDArray or None
                Complex S21 values across the frequency range, or ``None``
                for a single-port simulation.
            - s31 : NDArray or None
                Complex S31 values, or ``None`` with fewer than three ports.
            - s41 : NDArray or None
                Complex S41 values, or ``None`` with fewer than four ports.
            - z11 : NDArray
                Complex Z11 values across the frequency range.
            - vswr : NDArray
                VSWR values across the frequency range, computed from S11
                (clipped to ``|S11| <= 0.999`` to avoid division by zero).
            - input_power : float
                Time-averaged input power at the driven port.
            - port_voltage : NDArray
                Total complex voltage at the driven port across the
                frequency range.
            - port_current : NDArray
                Total complex current at the driven port across the
                frequency range.
            - ref_impedance : float
                Reference impedance ``charac_imp`` (from ``sim``) that S11
                and Z11 were computed against.

        Raises
        ------
        ValueError
            If more than four ports are passed.
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        if sim.backend_engine == "FEM":
            from . import fem_backend

            return fem_backend.compute_sim_data(sim.freqs, sim.charac_imp, output_path)

        ports = port if isinstance(port, list) else [port]
        if len(ports) > 4:
            raise ValueError(
                f"compute_sim_data supports up to 4 ports, got {len(ports)}"
            )
        for each_port in ports:
            each_port.CalcPort(
                str(output_path), sim.freqs, ref_impedance=sim.charac_imp
            )
        driven_port = ports[0]
        s11 = driven_port.uf_ref / driven_port.uf_inc
        # Port 1 is the only excited port, so every other port's outgoing
        # wave over port 1's incident wave gives the first S-matrix column.
        transmission = [
            each_port.uf_ref / driven_port.uf_inc for each_port in ports[1:]
        ]
        transmission += [None] * (3 - len(transmission))
        s21, s31, s41 = transmission
        z11 = driven_port.uf_tot / driven_port.if_tot
        s11_mag = np.clip(np.abs(s11), 0, 0.999)  # prevent division by zero error
        vswr = (1 + s11_mag) / (1 - s11_mag)
        input_power = 0.5 * np.real(driven_port.uf_tot * np.conj(driven_port.if_tot))
        return SimData(
            sim.freqs,
            s11,
            s21,
            s31,
            s41,
            z11,
            vswr,
            input_power,
            driven_port.uf_tot,
            driven_port.if_tot,
            sim.charac_imp,
        )

    @staticmethod
    def plot_s_param(
        freqs: NDArray,
        s11: NDArray,
        s21: NDArray | None = None,
        s31: NDArray | None = None,
        s41: NDArray | None = None,
        x_label: str = "Frequency",
        y_label: str = "S-parameter (dB)",
        title: str = "S-parameters vs Frequency",
        label_s11: str | list[str] | None = None,
        label_s21: str | list[str] | None = None,
        label_s31: str | list[str] | None = None,
        label_s41: str | list[str] | None = None,
    ) -> None:
        """
        Plot S-parameters (S11 and optionally S21, S31, S41) against frequency.

        Opens a new figure. Every S-parameter is expected as complex/linear
        values; each is converted to dB (``20*log10(|.|)``) internally before
        plotting.

        Each S-parameter may also be a 2D array with one row per run, such as
        ``SweepData.s11`` from :func:`param_sweep`; every row then gets its
        own trace, labelled by the matching entry of the label list.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        s11 : NDArray
            A 1D array of complex (linear) S11 values, or a 2D array with one
            row per run.
        s21 : NDArray, optional
            Complex (linear) S21 values, shaped like ``s11``. If ``None``
            (default), no S21 trace is drawn.
        s31 : NDArray, optional
            Complex (linear) S31 values, shaped like ``s11``. If ``None``
            (default), no S31 trace is drawn.
        s41 : NDArray, optional
            Complex (linear) S41 values, shaped like ``s11``. If ``None``
            (default), no S41 trace is drawn.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"S-parameter (dB)"``.
        title : str, optional
            Title of the plot. Default is ``"S-parameters vs Frequency"``.
        label_s11 : str or list[str], optional
            Legend label for the S11 trace, or one label per row of a 2D
            ``s11``. Defaults to ``"S11"``.
        label_s21 : str or list[str], optional
            Legend label(s) for the S21 trace(s). Defaults to ``"S21"``.
        label_s31 : str or list[str], optional
            Legend label(s) for the S31 trace(s). Defaults to ``"S31"``.
        label_s41 : str or list[str], optional
            Legend label(s) for the S41 trace(s). Defaults to ``"S41"``.

        Returns
        -------
        None
            Displays the S-parameter plot in a new figure; does not return a
            value.
        """
        plt.figure()
        traces = [
            (s11, label_s11 or "S11", "S11"),
            (s21, label_s21 or "S21", "S21"),
            (s31, label_s31 or "S31", "S31"),
            (s41, label_s41 or "S41", "S41"),
        ]
        for s_param, label, name in traces:
            if s_param is None:
                continue
            s_param_rows = np.atleast_2d(s_param)
            labels = label if isinstance(label, list) else [label] * len(s_param_rows)
            for s_param_row, row_label in zip(s_param_rows, labels, strict=True):
                s_param_db = 20.0 * np.log10(np.abs(s_param_row))
                lines = plt.plot(freqs, s_param_db, label=row_label)
                cursor = themed_cursor(lines)
                cursor.connect(
                    "add",
                    lambda sel, name=name: sel.annotation.set_text(
                        f"Freq={freq_formatter(sel.target[0])}\n"
                        f"{name}={sel.target[1]:.2f} dB"
                    ),
                )

        plt.gca().xaxis.set_major_formatter(freq_formatter)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()

    @staticmethod
    def plot_vswr(
        freqs: NDArray,
        vswr: NDArray,
        x_label: str = "Frequency",
        y_label: str = "VSWR",
        label: str | list[str] = "VSWR",
        title: str = "VSWR vs Frequency",
    ) -> None:
        """
        Plot the voltage standing wave ratio (VSWR) as a function of frequency.

        Opens a new figure and plots VSWR over the given frequency range, to
        evaluate impedance matching performance. A 2D ``vswr``, such as
        ``SweepData.vswr`` from :func:`param_sweep`, draws one trace per row.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        vswr : NDArray
            A 1D array of VSWR values corresponding to each frequency, or a 2D
            array with one row per run.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"VSWR"``.
        label : str or list[str], optional
            Label used for the plot legend, or one label per row of a 2D
            ``vswr``. Default is ``"VSWR"``.
        title : str, optional
            Title of the plot. Default is ``"VSWR vs Frequency"``.

        Returns
        -------
        None
            Displays the VSWR plot in a new figure; does not return a value.
        """
        plt.figure()
        vswr_rows = np.atleast_2d(vswr)
        labels = label if isinstance(label, list) else [label] * len(vswr_rows)
        for vswr_row, row_label in zip(vswr_rows, labels, strict=True):
            lines = plt.plot(freqs, vswr_row, label=row_label)
            cursor = themed_cursor(lines)

            cursor.connect(
                "add",
                lambda sel: sel.annotation.set_text(
                    f"Freq={freq_formatter(sel.target[0])}\nVSWR={sel.target[1]:.2f}"
                ),
            )

        plt.gca().xaxis.set_major_formatter(freq_formatter)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)

    @staticmethod
    def plot_phase(
        freqs: NDArray,
        s21: NDArray | None,
        s31: NDArray | None = None,
        s41: NDArray | None = None,
        x_label: str = "Frequency",
        y_label: str = "Phase (deg)",
        title: str = "Phase vs Frequency",
        label_s21: str | list[str] | None = None,
        label_s31: str | list[str] | None = None,
        label_s41: str | list[str] | None = None,
    ) -> None:
        """
        Plot the transmission phase (angle of S21, S31, S41) against frequency.

        Opens a new figure and plots the phase, in degrees, of every
        transmission term given, one trace each, over the frequency range.
        A 2D S-parameter, such as ``SweepData.s21`` from :func:`param_sweep`,
        draws one trace per row.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        s21 : NDArray
            A 1D array of complex S21 values corresponding to each frequency,
            or a 2D array with one row per run.
        s31 : NDArray, optional
            Complex S31 values, shaped like ``s21``. If ``None`` (default), no
            S31 trace is drawn.
        s41 : NDArray, optional
            Complex S41 values, shaped like ``s21``. If ``None`` (default), no
            S41 trace is drawn.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"Phase (deg)"``.
        title : str, optional
            Title of the plot. Default is ``"Phase vs Frequency"``.
        label_s21 : str or list[str], optional
            Legend label for the S21 trace, or one label per row of a 2D
            ``s21``. Defaults to ``"S21"``.
        label_s31 : str or list[str], optional
            Legend label(s) for the S31 trace(s). Defaults to ``"S31"``.
        label_s41 : str or list[str], optional
            Legend label(s) for the S41 trace(s). Defaults to ``"S41"``.

        Returns
        -------
        None
            Displays the phase plot in a new figure; does not return a value.
        """
        plt.figure()
        traces = [
            (s21, label_s21 or "S21", "S21"),
            (s31, label_s31 or "S31", "S31"),
            (s41, label_s41 or "S41", "S41"),
        ]
        for s_param, label, name in traces:
            if s_param is None:
                continue
            s_param_rows = np.atleast_2d(s_param)
            labels = label if isinstance(label, list) else [label] * len(s_param_rows)
            for s_param_row, row_label in zip(s_param_rows, labels, strict=True):
                lines = plt.plot(
                    freqs, np.angle(s_param_row, deg=True), label=row_label
                )
                cursor = themed_cursor(lines)
                cursor.connect(
                    "add",
                    lambda sel, name=name: sel.annotation.set_text(
                        f"Freq={freq_formatter(sel.target[0])}\n"
                        f"{name} phase={sel.target[1]:.2f} Deg"
                    ),
                )

        plt.gca().xaxis.set_major_formatter(freq_formatter)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()

    @staticmethod
    def plot_group_delay(
        freqs: NDArray,
        s21: NDArray | None,
        x_label: str = "Frequency",
        y_label: str = "Group Delay",
        title: str = "Group Delay vs Frequency",
        label: str | list[str] = "group_delay",
    ) -> None:
        """
        Compute and plot the group delay as a function of frequency.

        The group delay is computed from the frequency derivative of the
        unwrapped phase of S21 and displayed in seconds (via an engineering
        time-unit axis formatter). A 2D ``s21``, such as ``SweepData.s21``
        from :func:`param_sweep`, draws one trace per row.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        s21 : NDArray
            A 1D array of complex S21 values corresponding to each frequency,
            or a 2D array with one row per run.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"Group Delay"``.
        title : str, optional
            Title of the plot. Default is ``"Group Delay vs Frequency"``.
        label : str or list[str], optional
            Label used for the plot legend, or one label per row of a 2D
            ``s21``. Default is ``"group_delay"``.

        Returns
        -------
        None
            Displays the group delay plot in a new figure; does not return
            a value.
        """
        s21_rows = np.atleast_2d(s21)
        delta_phi_df = np.gradient(np.unwrap(np.angle(s21_rows)), freqs, axis=-1)
        group_delay_rows = -delta_phi_df / (2 * np.pi)
        labels = label if isinstance(label, list) else [label] * len(s21_rows)
        plt.figure()
        for group_delay, row_label in zip(group_delay_rows, labels, strict=True):
            lines_group = plt.plot(freqs, group_delay, label=row_label)
            cursor_group = themed_cursor(lines_group)

            cursor_group.connect(
                "add",
                lambda sel: sel.annotation.set_text(
                    f"Freq={freq_formatter(sel.target[0])}\n"
                    f"Group Delay={time_formatter(sel.target[1])}"
                ),
            )

        plt.gca().xaxis.set_major_formatter(freq_formatter)
        plt.gca().yaxis.set_major_formatter(time_formatter)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()

    @staticmethod
    def plot_impedance(
        freqs: NDArray,
        z11: NDArray,
        x_label: str = "Frequency",
        y_label: str = "Z11",
        title: str = "Z11 vs Frequency",
        label: str | list[str] = "Z11",
    ) -> None:
        """
        Plot the input impedance (Z11) as a function of frequency.

        Opens a new figure and plots the real and imaginary parts of Z11,
        in ohms, as separate traces over the given frequency range. A 2D
        ``z11``, such as ``SweepData.z11`` from :func:`param_sweep`, draws a
        real and an imaginary trace per row.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        z11 : NDArray
            A 1D array of complex Z11 values (in ohms) corresponding to
            each frequency, or a 2D array with one row per run.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"Z11"``.
        title : str, optional
            Title of the plot. Default is ``"Z11 vs Frequency"``.
        label : str or list[str], optional
            Legend label, or one label per row of a 2D ``z11``. The traces
            are named ``"Real <label>"`` and ``"Imag <label>"``. Default is
            ``"Z11"``.

        Returns
        -------
        None
            Displays the impedance plot in a new figure; does not return
            a value.
        """
        z11_rows = np.atleast_2d(z11)
        labels = label if isinstance(label, list) else [label] * len(z11_rows)
        plt.figure()
        for z11_row, row_label in zip(z11_rows, labels, strict=True):
            lines_real = plt.plot(freqs, np.real(z11_row), label=f"Real {row_label}")
            cursor_real = themed_cursor(lines_real)

            cursor_real.connect(
                "add",
                lambda sel: sel.annotation.set_text(
                    f"Freq={freq_formatter(sel.target[0])}\nZ11={sel.target[1]:.2f} Ω"
                ),
            )

            lines_imag = plt.plot(freqs, np.imag(z11_row), label=f"Imag {row_label}")
            cursor_imag = themed_cursor(lines_imag)

            cursor_imag.connect(
                "add",
                lambda sel: sel.annotation.set_text(
                    f"Freq={freq_formatter(sel.target[0])}\nZ11={sel.target[1]:.2f} Ω"
                ),
            )

        plt.gca().xaxis.set_major_formatter(freq_formatter)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()

    @staticmethod
    def plot_smith_chart(
        freqs: NDArray,
        s11: NDArray,
        label: str | list[str] = "",
        charac_imp: float = 50.0,
    ) -> None:
        """
        Plot S11 data on a Smith chart.

        Opens a new figure with a Smith-chart projection and plots the
        complex reflection coefficient (S11) over the given frequency
        range, marking the value at the midpoint frequency. A 2D ``s11``,
        such as ``SweepData.s11`` from :func:`param_sweep`, draws one trace
        per row and leaves out the midpoint marker.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) corresponding to the S11 data.
        s11 : NDArray
            A 1D array of complex S11 (reflection coefficient) values, or a 2D
            array with one row per run.
        label : str or list[str], optional
            Label used for the plot legend, or one label per row of a 2D
            ``s11``. Default is an empty string.
        charac_imp : float, optional
            Characteristic (normalizing) impedance of the chart, in ohms.
            Default is ``50.0``.

        Returns
        -------
        None
            Displays the Smith chart in a new figure; does not return a
            value.

        Notes
        -----
        pysmithchart is used to render the Smith chart.
        """
        plt.figure()
        surface_color = plt.rcParams["axes.facecolor"]
        plt.subplot(
            1,
            1,
            1,
            projection="smith",
            grid_major_fancy=True,
            grid_minor_enable=True,
            grid_minor_fancy=True,
            grid_minor_fancy_threshold=10,
            axes_normalize=True,
            axes_impedance=charac_imp,
            grid_major_color=plt.rcParams["axes.edgecolor"],
            grid_minor_color=plt.rcParams["grid.color"],
            # pysmithchart puts a white box behind each real-axis number; fill it
            # with the surface colour so only the number shows.
            axes_xlabel_fancybox={
                "boxstyle": "round,pad=0.2,rounding_size=0.2",
                "facecolor": surface_color,
                "edgecolor": surface_color,
                "mutation_aspect": 0.75,
                "alpha": 1,
            },
        )
        s11_rows = np.atleast_2d(s11)
        labels = label if isinstance(label, list) else [label] * len(s11_rows)
        for s11_row, row_label in zip(s11_rows, labels, strict=True):
            s11_lines = plt.plot(
                s11_row, datatype=S_PARAMETER, marker="", label=row_label
            )
            cursor = themed_cursor(s11_lines)
            cursor.connect(
                "add",
                lambda sel, s11_row=s11_row: sel.annotation.set_text(
                    f"Freq={freq_formatter(freqs[int(round(sel.index))])}\n"
                    f"S11={s11_row[int(round(sel.index))]:.2f}"
                ),
            )

        # One midpoint marker per run would bury an overlay in legend entries,
        # so only a single trace gets it.
        if np.ndim(s11) == 1:
            tar_idx = len(freqs) // 2
            plt.plot(
                s11[tar_idx],
                color=plt.rcParams["text.color"],
                datatype=S_PARAMETER,
                label=(
                    f"S11 is {s11[tar_idx]:.2f} at {freq_formatter(freqs[tar_idx])}"
                ),
            )

        plt.title("Smith Chart - S11")
        plt.legend(loc="lower right", bbox_to_anchor=(1.5, 1.0))

    @staticmethod
    def compute_nf2ff_data(
        nf2ff: nf2ff,
        freq: float,
        sim_data: SimData,
        output_path: Path | None = None,
        read_cached: bool = False,
        efficiency_freqs: NDArray | None = None,
        efficiency_points: int = 5,
    ) -> NF2FFData:
        """
        Compute every far-field result the radiation plots need.

        Solves the far field at ``freq`` once over the full sphere (theta in
        [0, 180] deg, phi in [0, 360] deg, 2 deg steps) and once over each
        principal cut (xz, yz and xy planes, 0.1 deg steps), and works out the
        radiation and mismatch efficiencies there. Hand the result to
        ``plot_2d_rad_pattern``, ``plot_2d_directivity``, ``plot_2d_gain``,
        ``plot_3d_directivity``, ``plot_3d_gain`` and ``plot_efficiency``.

        It also works out the efficiency across the antenna's -10 dB band:
        the run of frequencies around the S11 minimum where ``|S11|`` stays
        below -10 dB, or the whole simulated band if S11 never gets that low.
        The efficiency is computed at a few frequencies and a smooth curve is
        fitted through them:

        - FDTD transforms ``efficiency_points`` frequencies in one far-field
          pass over a 5 deg sphere. The radiated power is an integral over it,
          and a finer grid changes it by far less than the solve is accurate
          to.
        - FEM reads the efficiency the sweep recorded at each of its solves,
          which costs nothing. Results from before the sweep recorded it, or a
          band reaching past the swept one, fall back to one field solve per
          frequency at ``efficiency_points`` frequencies.

        The curve is a rational fit through the computed points, as the FEM
        sweep fits its S-parameters. A smooth efficiency does not overshoot
        its own samples by more than their spread, so a fit that does -- a
        stray pole -- is replaced by straight lines between the points, with
        a warning.

        Parameters
        ----------
        nf2ff : object
            The near-field-to-far-field (NF2FF) object containing the
            simulation data.
        freq : float
            Frequency (in Hz) at which the far field is evaluated. Must be a
            scalar.
        sim_data : SimData
            The simulation results of the same run. Both backends take the
            port mismatch from its S11 at ``freq``. The FDTD backend also
            takes the accepted power at ``freq`` from them; the FEM backend
            reports its losses in the far-field result instead.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to ``cwd / "Sim_Path"``.
        read_cached : bool, optional
            If True, read cached NF2FF results instead of re-computing.
            Default is False.
        efficiency_freqs : NDArray, optional
            Frequencies (in Hz) to report the efficiency curve at. FDTD, and
            the FEM fallback, compute at exactly these; FEM with sweep data
            interpolates onto them. Defaults to 101 points across the -10 dB
            band.
        efficiency_points : int, optional
            Number of frequencies to compute the efficiency at when
            ``efficiency_freqs`` is not given. Unused when the FEM sweep
            recorded the efficiency. Default is 5.

        Returns
        -------
        NF2FFData
            The full-sphere result, the three principal cuts, the radiation
            and mismatch efficiencies at ``freq``, and the efficiency curve
            across the band.

        Raises
        ------
        TypeError
            If ``freq`` is not a scalar (e.g. an array of frequencies).
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency, not an array, "
                "for far-field calculation"
            )

        # 0.1° steps from -180° to 180°; 180° repeats -180° so each cut closes.
        sweep = np.linspace(-180.0, 180.0, 3601)

        # (theta, phi, outfile) for the full sphere and each principal cut
        requests = [
            (np.arange(0, 181, 2), np.arange(0, 361, 2), "nf2ff_3d.h5"),
            (sweep, 0, "nf2ff_xz.h5"),
            (sweep, 90, "nf2ff_yz.h5"),
            (90, sweep, "nf2ff_xy.h5"),
        ]
        console.print("Calculating Far Field.........", style="info")
        sphere, xz_cut, yz_cut, xy_cut = [
            nf2ff.CalcNF2FF(
                output_path,
                freq,
                theta,
                phi,
                read_cached=read_cached,
                outfile=outfile,
                verbose=0,
            )
            for theta, phi, outfile in requests
        ]

        # Prad is at freq, so the accepted power has to be taken at freq too.
        # Only the full sphere integrates Prad over every direction.
        radiated_power = sphere.Prad[0]
        ploss = getattr(sphere, "Ploss", None)
        if ploss is not None:
            accepted_power = radiated_power + ploss[0]  # FEM reports its losses
        else:
            accepted_power = np.interp(freq, sim_data.freqs, sim_data.input_power)

        # The share of the incident power the port mismatch lets through
        s11_magnitude = np.interp(freq, sim_data.freqs, np.abs(sim_data.s11))

        if efficiency_freqs is None:
            # Walk out from the S11 minimum while |S11| stays below -10 dB
            matched = np.abs(sim_data.s11) < 10.0 ** (-10.0 / 20.0)
            resonance_index = int(np.argmin(np.abs(sim_data.s11)))
            low_index = high_index = resonance_index
            if matched[resonance_index]:
                while low_index > 0 and matched[low_index - 1]:
                    low_index -= 1
                while high_index < matched.size - 1 and matched[high_index + 1]:
                    high_index += 1
            # Unmatched, or matched at a single sample: take the whole band
            if low_index == high_index:
                low_index, high_index = 0, matched.size - 1
            band_low = sim_data.freqs[low_index]
            band_high = sim_data.freqs[high_index]
            curve_freqs = np.linspace(band_low, band_high, 101)
            computed_freqs = np.linspace(band_low, band_high, efficiency_points)
        else:
            curve_freqs = np.atleast_1d(np.asarray(efficiency_freqs, dtype=float))
            computed_freqs = curve_freqs
            band_low, band_high = curve_freqs.min(), curve_freqs.max()

        from openEMS.nf2ff import nf2ff as openems_nf2ff
        from scipy.interpolate import AAA

        from .fem_solver import describe_solve, format_duration, last_solve

        band_text = f"{band_low / 1e9:.3f}-{band_high / 1e9:.3f} GHz"
        started = time.perf_counter()
        solved_one_by_one = False
        from_sweep = False

        # openEMS takes the whole frequency array in one call and returns Prad
        # per frequency. Like the pattern above it follows read_cached, which
        # is off by default: a cached file from an earlier design would
        # otherwise be read back as this one's result.
        if isinstance(nf2ff, openems_nf2ff):
            console.rule(
                f"[info]Efficiency · {band_text} · {computed_freqs.size} "
                f"frequencies in one far-field transform[/info]"
            )
            efficiency_sphere = nf2ff.CalcNF2FF(
                output_path,
                computed_freqs,
                np.arange(0, 181, 5),
                np.arange(0, 361, 5),
                read_cached=read_cached,
                outfile="nf2ff_efficiency.h5",
                verbose=0,
            )
            computed_accepted_power = np.interp(
                computed_freqs, sim_data.freqs, sim_data.input_power
            )
            computed_efficiency = efficiency_sphere.Prad / computed_accepted_power
        else:
            sweep_points = nf2ff.sweep_efficiency(output_path)
            # The sweep's points cover the band only if they bracket it
            if (
                sweep_points is not None
                and sweep_points[0].size >= 2
                and sweep_points[0][0] <= band_low
                and sweep_points[0][-1] >= band_high
            ):
                computed_freqs, computed_efficiency = sweep_points
                from_sweep = True
                console.rule(
                    f"[info]Efficiency · {band_text} · interpolated from the "
                    f"FEM sweep's {computed_freqs.size} solves[/info]"
                )
            # The powers come from the port and the losses, not the far
            # field, so they need no pattern and hold below the frequency the
            # air box was sized for. One aligned line per solve, as the FEM
            # sweep prints its own.
            else:
                solved_one_by_one = True
                console.rule(
                    f"[info]Efficiency · {band_text} · "
                    f"{computed_freqs.size} field solves[/info]"
                )
                computed_efficiency = np.empty(computed_freqs.size)
                for index, solve_freq in enumerate(computed_freqs):
                    solve_started = time.perf_counter()
                    solve_radiated_power, solve_lost_power = nf2ff.calc_power(
                        output_path, solve_freq, verbose=False
                    )
                    computed_efficiency[index] = solve_radiated_power / (
                        solve_radiated_power + solve_lost_power
                    )
                    elapsed = time.perf_counter() - started
                    left = (computed_freqs.size - index - 1) * elapsed / (index + 1)
                    elapsed_text = f"elapsed {format_duration(elapsed)}"
                    left_text = f"~{format_duration(left)} left" if left > 0 else ""
                    total = computed_efficiency[index] * (
                        1.0
                        - np.interp(solve_freq, sim_data.freqs, np.abs(sim_data.s11))
                        ** 2
                    )
                    console.print(
                        f"[info]{index + 1:>3}/{computed_freqs.size}  "
                        f"{solve_freq / 1e9:8.4f} GHz  "
                        f"{format_duration(time.perf_counter() - solve_started):>7}  "
                        f"{elapsed_text:<18}{left_text:<17}"
                        f"radiation {100.0 * computed_efficiency[index]:5.1f} %   "
                        f"total {100.0 * total:5.1f} %[/info]"
                    )
                size = describe_solve(last_solve())
                console.rule(
                    f"[info]{computed_freqs.size} solves in "
                    f"{format_duration(time.perf_counter() - started)}"
                    f"{f' · {size}' if size else ''}[/info]"
                )

        computed_mismatch = 1.0 - (
            np.interp(computed_freqs, sim_data.freqs, np.abs(sim_data.s11)) ** 2
        )
        # The sweep's points can reach past the band; list (and later mark)
        # only the ones inside it
        shown = (computed_freqs >= band_low) & (computed_freqs <= band_high)
        if not solved_one_by_one:
            for count, index in enumerate(np.flatnonzero(shown), 1):
                total = computed_efficiency[index] * computed_mismatch[index]
                console.print(
                    f"[info]{count:>3}/{shown.sum()}  "
                    f"{computed_freqs[index] / 1e9:8.4f} GHz  "
                    f"radiation {100.0 * computed_efficiency[index]:5.1f} %   "
                    f"total {100.0 * total:5.1f} %[/info]"
                )
            if from_sweep:
                console.rule(
                    f"[info]{shown.sum()} of the {computed_freqs.size} sweep "
                    f"solves fall in this band · no extra solves[/info]"
                )
            else:
                console.rule(
                    f"[info]{computed_freqs.size} frequencies in "
                    f"{format_duration(time.perf_counter() - started)}[/info]"
                )

        # Rational fit in a [-1, 1] frequency coordinate, which AAA is far
        # better conditioned in than raw Hz
        if computed_freqs.size >= 2:
            fit_low, fit_high = computed_freqs.min(), computed_freqs.max()
            fit = AAA(
                (2.0 * computed_freqs - (fit_low + fit_high)) / (fit_high - fit_low),
                computed_efficiency,
            )
            radiation_efficiency = np.real(
                fit((2.0 * curve_freqs - (fit_low + fit_high)) / (fit_high - fit_low))
            )
            spread = np.ptp(computed_efficiency) + 0.01
            if not (
                np.all(np.isfinite(radiation_efficiency))
                and radiation_efficiency.min() >= computed_efficiency.min() - spread
                and radiation_efficiency.max() <= computed_efficiency.max() + spread
            ):
                console.print(
                    "[warning]Efficiency: the rational fit overshoots the "
                    "computed points; drawing straight lines between them "
                    "instead.[/warning]"
                )
                radiation_efficiency = np.interp(
                    curve_freqs, computed_freqs, computed_efficiency
                )
        else:
            radiation_efficiency = np.interp(
                curve_freqs, computed_freqs, computed_efficiency
            )
        radiation_efficiency = np.clip(radiation_efficiency, 0.0, 1.0)

        curve_s11_magnitude = np.interp(
            curve_freqs, sim_data.freqs, np.abs(sim_data.s11)
        )
        total_efficiency = radiation_efficiency * (1.0 - curve_s11_magnitude**2)

        return NF2FFData(
            freq=freq,
            sphere=sphere,
            xz_cut=xz_cut,
            yz_cut=yz_cut,
            xy_cut=xy_cut,
            efficiency=radiated_power / accepted_power,
            mismatch_efficiency=1.0 - s11_magnitude**2,
            efficiency_curve=EfficiencyCurve(
                freqs=curve_freqs,
                radiation=radiation_efficiency,
                total=total_efficiency,
                computed_freqs=computed_freqs[shown],
                computed_radiation=computed_efficiency[shown],
                computed_total=(computed_efficiency * computed_mismatch)[shown],
            ),
        )

    @staticmethod
    def plot_2d_rad_pattern(
        nf2ff_data: NF2FFData,
        floor_db: float | None = None,
    ) -> None:
        """
        Plot the 2D E-field radiation pattern.

        Plots the far-field E-field cuts in the xz-plane (phi=0), the
        yz-plane (phi=90) and the xy-plane (theta=90) as normalized polar
        patterns (in dB) side by side. The three cuts hold both principal
        planes of any antenna whose axis lies along x, y or z: its E-plane
        and its omnidirectional H-plane.

        Parameters
        ----------
        nf2ff_data : NF2FFData
            The far-field results returned by ``compute_nf2ff_data``.
        floor_db : float or None, optional
            Floor of each radial axis relative to the 0 dB peak, as a
            negative number (e.g. ``-30``). When set, the axis runs from
            ``floor_db`` to 0 dB, so an omnidirectional cut draws as a
            near-circle rather than filling the plot, and deeper nulls clip
            at the axis floor. Default is None, which autoscales each axis to
            its pattern.

        Returns
        -------
        None
            Generates the 2D radiation pattern plot; does not return a
            value.

        Raises
        ------
        ValueError
            If ``floor_db`` is zero or positive.
        """
        if floor_db is not None and floor_db >= 0:
            raise ValueError(f"floor_db must be negative, got {floor_db}")

        plt.figure(figsize=(15, 5))

        # (label, cut, swept angle name, swept angles in radians) for each cut
        cuts = [
            ("xz-plane", nf2ff_data.xz_cut, "theta", nf2ff_data.xz_cut.theta),
            ("yz-plane", nf2ff_data.yz_cut, "theta", nf2ff_data.yz_cut.theta),
            ("xy-plane", nf2ff_data.xy_cut, "phi", nf2ff_data.xy_cut.phi),
        ]
        for subplot_index, (label, cut, sweep_name, sweep) in enumerate(cuts, 1):
            ax = plt.subplot(1, len(cuts), subplot_index, polar=True)

            efield = np.squeeze(cut.E_norm)
            efield_norm_dB = 20 * np.log10(efield / np.max(efield))

            # Autoscale to the pattern unless the user sets a floor, and clip
            # deeper nulls (an exact null is -inf dB) to the floor so they
            # don't wrap through the centre.
            if floor_db is None:
                axis_floor = np.min(efield_norm_dB[np.isfinite(efield_norm_dB)])
            else:
                axis_floor = floor_db
            lines = ax.plot(
                sweep,
                np.maximum(efield_norm_dB, axis_floor),
                linewidth=2,
                label=label,
            )
            if floor_db is not None:
                ax.set_rlim(axis_floor, 0)

            cursor = themed_cursor(lines)
            cursor.connect(
                "add",
                lambda sel, sweep_name=sweep_name: sel.annotation.set_text(
                    f"{sweep_name}={np.rad2deg(sel.target[0]):.2f}°"
                    f"\ne_field = {sel.target[1]:.2f} dB"
                ),
            )

            ax.grid(True)
            ax.set_xlabel(f"{sweep_name} (deg)")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.legend()

        plt.suptitle(
            f"Radiation Pattern at: {freq_formatter(nf2ff_data.freq)}",
            fontsize=14,
        )

    @staticmethod
    def plot_2d_directivity(
        nf2ff_data: NF2FFData,
        floor_db: float | None = None,
        style: Literal["polar", "cartesian", "both"] = "both",
    ) -> None:
        """
        Plot the 2D directivity pattern (xz-plane, phi=0), annotated with the
        half-power beamwidth (HPBW).

        Computes directivity (in dBi) from the xz-plane cut, referenced to the
        full sphere's peak, and draws it as a polar pattern, a Cartesian (dBi
        versus theta) plot, or both in separate figures. Each view marks the
        -3 dB beamwidth, main-lobe direction, and peak magnitude. The
        beamwidth search wraps through +-180 degrees; an omnidirectional cut,
        which never drops 3 dB, gets no beamwidth marker.

        Parameters
        ----------
        nf2ff_data : NF2FFData
            The far-field results returned by ``compute_nf2ff_data``.
        floor_db : float or None, optional
            Floor of the directivity axis relative to the peak, in dB, as a
            negative number (e.g. ``-30``). When set, the axis starts at
            ``peak + floor_db``, so an omnidirectional cut draws as a
            near-circle rather than filling the plot, and deeper nulls clip
            at the axis floor. Default is None, which starts the axis at the
            minimum of the pattern.
        style : {"polar", "cartesian", "both"}, optional
            Which view to draw. ``"both"`` draws the polar and Cartesian
            views in two separate figures. Default is ``"both"``.

        Returns
        -------
        None
            Plots the 2D directivity pattern; does not return a value.

        Raises
        ------
        ValueError
            If ``style`` is not ``"polar"``, ``"cartesian"`` or ``"both"``,
            or ``floor_db`` is zero or positive.
        """
        if style not in ("polar", "cartesian", "both"):
            raise ValueError(
                f"style must be 'polar', 'cartesian' or 'both', got {style!r}"
            )
        if floor_db is not None and floor_db >= 0:
            raise ValueError(f"floor_db must be negative, got {floor_db}")

        xz_cut = nf2ff_data.xz_cut
        sphere = nf2ff_data.sphere
        theta = np.rad2deg(xz_cut.theta)

        # Reference the cut to the sphere's peak, not its own, so a cut that
        # misses the main lobe sits below Dmax instead of being lifted to it.
        e_field_norm = np.squeeze(xz_cut.E_norm) / np.max(sphere.E_norm)
        e_field_norm_db = 20 * np.log10(e_field_norm)

        max_directivity_db = 10.0 * np.log10(sphere.Dmax[0])
        directivity_dbi = e_field_norm_db + max_directivity_db

        # Start the axis at the pattern minimum unless the user sets a floor
        # below the peak, and clip deeper nulls (an exact null is -inf dB) to
        # the floor so they don't wrap through the centre.
        if floor_db is None:
            axis_floor = np.min(directivity_dbi[np.isfinite(directivity_dbi)])
        else:
            axis_floor = np.max(directivity_dbi) + floor_db
        clipped_directivity_dbi = np.maximum(directivity_dbi, axis_floor)

        # ---- HPBW calculation ----
        # The cut is a full circle: drop the last sample, which repeats the
        # first direction, then roll it so the peak sits in the middle and the
        # −3 dB search can pass through ±180°.
        theta_step = theta[1] - theta[0]
        pattern = directivity_dbi[:-1]
        peak_idx = np.argmax(pattern)
        # Report the direction in (-180°, 180°], so a backward lobe reads 180°
        peak_theta = 180.0 - (180.0 - theta[peak_idx]) % 360.0
        peak_val = pattern[peak_idx]

        # Half-power level (−3 dB)
        hpbw_level = peak_val - 3.0

        centre_idx = pattern.size // 2
        centred_pattern = np.roll(pattern, centre_idx - peak_idx)

        # Find −3 dB crossings
        left_idx = np.where(centred_pattern[:centre_idx] <= hpbw_level)[0]
        right_idx = np.where(centred_pattern[centre_idx:] <= hpbw_level)[0]

        # An omnidirectional cut never drops 3 dB, so it has no beamwidth.
        has_beamwidth = len(left_idx) > 0 and len(right_idx) > 0
        annotation = ""
        if has_beamwidth:
            left_theta = peak_theta - (centre_idx - left_idx[-1]) * theta_step
            right_theta = peak_theta + right_idx[0] * theta_step
            left_val = centred_pattern[left_idx[-1]]
            right_val = centred_pattern[centre_idx + right_idx[0]]
            hpbw = right_theta - left_theta
            annotation = f"HPBW (3dB) = {hpbw:.2f}°\n"

        main_lobe_mag = np.round(np.max(directivity_dbi), 2)
        annotation += (
            f"Main Lobe Direction = {peak_theta:.2f}°"
            f"\nMain Lobe Magnitude = {main_lobe_mag:.2f} dBi"
        )

        # A lobe pointing near ±180° would split across both edges of the
        # cartesian view, so run that axis from 0° to 360° to keep it whole.
        axis_start = 0.0 if abs(peak_theta) > 90.0 else -180.0
        axis_theta = (theta[:-1] - axis_start) % 360.0 + axis_start
        axis_order = np.argsort(axis_theta)
        axis_theta = np.append(axis_theta[axis_order], axis_theta[axis_order[0]] + 360)
        axis_directivity_dbi = clipped_directivity_dbi[:-1][axis_order]
        axis_directivity_dbi = np.append(axis_directivity_dbi, axis_directivity_dbi[0])

        # ---- Plotting ----
        views = ["polar", "cartesian"] if style == "both" else [style]

        for view in views:
            figure = plt.figure()

            if view == "polar":
                ax = figure.add_subplot(1, 1, 1, polar=True)
                lines = ax.plot(
                    np.deg2rad(theta),
                    clipped_directivity_dbi,
                    linewidth=2,
                    label="xz-plane",
                )
                ax.set_rlim(axis_floor)

                if has_beamwidth:
                    ax.plot(
                        [np.deg2rad(left_theta), np.deg2rad(left_theta)],
                        [axis_floor, left_val],
                        "--",
                        color=HIGHLIGHT,
                        linewidth=1,
                    )
                    ax.plot(
                        [np.deg2rad(right_theta), np.deg2rad(right_theta)],
                        [axis_floor, right_val],
                        "--",
                        color=HIGHLIGHT,
                        linewidth=1,
                    )
                    ax.plot(
                        [np.deg2rad(peak_theta), np.deg2rad(peak_theta)],
                        [axis_floor, hpbw_level],
                        "--",
                        color=HIGHLIGHT,
                        linewidth=1,
                    )
                    # HPBW arc
                    hpbw_arc = np.linspace(left_theta, right_theta, 360)
                    ax.plot(
                        np.deg2rad(hpbw_arc),
                        hpbw_level * np.ones_like(hpbw_arc),
                        color=HIGHLIGHT,
                        linewidth=2,
                    )

                text_anchor = (1.0, 0.1)
                cursor = themed_cursor(lines)
                cursor.connect(
                    "add",
                    lambda sel: sel.annotation.set_text(
                        f"theta={np.rad2deg(sel.target[0]):.2f}°"
                        f"\nDirectivity = {sel.target[1]:.2f} dBi"
                    ),
                )
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.legend()

            else:
                ax = figure.add_subplot(1, 1, 1)
                lines = ax.plot(
                    axis_theta,
                    axis_directivity_dbi,
                    linewidth=2,
                    label="xz-plane",
                )

                if has_beamwidth:
                    # Fold the peak and both beam edges into the plotted range
                    peak_line, left_edge, right_edge = (
                        (angle - axis_start) % 360.0 + axis_start
                        for angle in (peak_theta, left_theta, right_theta)
                    )
                    ax.axhline(hpbw_level, linestyle="--", color=HIGHLIGHT, lw=1)
                    ax.axvline(peak_line, linestyle="--", color=HIGHLIGHT, lw=1)
                    ax.axvline(left_edge, linestyle="--", color=HIGHLIGHT, lw=1)
                    ax.axvline(right_edge, linestyle="--", color=HIGHLIGHT, lw=1)

                text_anchor = (0.02, 0.98)
                cursor = themed_cursor(lines)
                cursor.connect(
                    "add",
                    lambda sel: sel.annotation.set_text(
                        f"theta={sel.target[0]:.2f}°"
                        f"\nDirectivity = {sel.target[1]:.2f} dBi"
                    ),
                )
                ax.set_xlim(axis_start, axis_start + 360.0)
                ax.set_xticks(np.arange(axis_start, axis_start + 361.0, 45.0))
                ax.set_ylim(axis_floor, peak_val + 3.0)
                ax.set_ylabel("Directivity (dBi)")
                ax.legend(loc="lower right")

            ax.text(
                *text_anchor,
                annotation,
                transform=ax.transAxes,
                fontsize=12,
                color=plt.rcParams["text.color"],
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor=plt.rcParams["axes.facecolor"],
                    edgecolor=plt.rcParams["axes.edgecolor"],
                ),
            )
            ax.grid(True)
            ax.set_xlabel("theta (deg)")
            figure.suptitle(
                f" Directivity at {freq_formatter(nf2ff_data.freq)}", fontsize=14
            )

    @staticmethod
    def plot_2d_gain(
        nf2ff_data: NF2FFData,
        floor_db: float | None = None,
        style: Literal["polar", "cartesian", "both"] = "both",
    ) -> None:
        """
        Plot the 2D realized gain pattern (xz-plane, phi=0).

        Computes the realized gain (in dBi) from the xz-plane cut: the
        directivity pattern scaled by the radiation efficiency and by the
        port mismatch ``1 - |S11|^2``. Draws it as a polar pattern, a
        Cartesian (dBi versus theta) plot, or both in separate figures. Each
        view reports the main-lobe direction and peak gain. The efficiencies
        shift the whole pattern by a constant, so the beamwidth matches the
        one ``plot_2d_directivity`` reports.

        Parameters
        ----------
        nf2ff_data : NF2FFData
            The far-field results returned by ``compute_nf2ff_data``.
        floor_db : float or None, optional
            Floor of the gain axis relative to the peak, in dB, as a
            negative number (e.g. ``-30``). When set, the axis starts at
            ``peak + floor_db``, so an omnidirectional cut draws as a
            near-circle rather than filling the plot, and deeper nulls clip
            at the axis floor. Default is None, which starts the axis at the
            minimum of the pattern.
        style : {"polar", "cartesian", "both"}, optional
            Which view to draw. ``"both"`` draws the polar and Cartesian
            views in two separate figures. Default is ``"both"``.

        Returns
        -------
        None
            Plots the 2D gain pattern; does not return a value.

        Raises
        ------
        ValueError
            If ``style`` is not ``"polar"``, ``"cartesian"`` or ``"both"``,
            or ``floor_db`` is zero or positive.
        """
        if style not in ("polar", "cartesian", "both"):
            raise ValueError(
                f"style must be 'polar', 'cartesian' or 'both', got {style!r}"
            )
        if floor_db is not None and floor_db >= 0:
            raise ValueError(f"floor_db must be negative, got {floor_db}")

        xz_cut = nf2ff_data.xz_cut
        sphere = nf2ff_data.sphere
        theta = np.rad2deg(xz_cut.theta)

        # Reference the cut to the sphere's peak, not its own, so a cut that
        # misses the main lobe sits below the peak gain instead of at it.
        e_field_norm = np.squeeze(xz_cut.E_norm) / np.max(sphere.E_norm)
        e_field_norm_db = 20 * np.log10(e_field_norm)

        # Realized gain also loses the power the port mismatch reflects
        max_gain_db = 10.0 * np.log10(
            nf2ff_data.efficiency * nf2ff_data.mismatch_efficiency * sphere.Dmax[0]
        )
        gain_dbi = e_field_norm_db + max_gain_db

        # Start the axis at the pattern minimum unless the user sets a floor
        # below the peak, and clip deeper nulls (an exact null is -inf dB) to
        # the floor so they don't wrap through the centre.
        if floor_db is None:
            axis_floor = np.min(gain_dbi[np.isfinite(gain_dbi)])
        else:
            axis_floor = np.max(gain_dbi) + floor_db
        clipped_gain_dbi = np.maximum(gain_dbi, axis_floor)

        # The last sample repeats the first direction, so leave it out.
        peak_idx = np.argmax(gain_dbi[:-1])
        # Report the direction in (-180°, 180°], so a backward lobe reads 180°
        peak_theta = 180.0 - (180.0 - theta[peak_idx]) % 360.0
        peak_val = gain_dbi[peak_idx]

        annotation = (
            f"Main Lobe Direction = {peak_theta:.2f}°"
            f"\nMain Lobe Magnitude = {peak_val:.2f} dBi"
        )

        # A lobe pointing near ±180° would split across both edges of the
        # cartesian view, so run that axis from 0° to 360° to keep it whole.
        axis_start = 0.0 if abs(peak_theta) > 90.0 else -180.0
        axis_theta = (theta[:-1] - axis_start) % 360.0 + axis_start
        axis_order = np.argsort(axis_theta)
        axis_theta = np.append(axis_theta[axis_order], axis_theta[axis_order[0]] + 360)
        axis_gain_dbi = clipped_gain_dbi[:-1][axis_order]
        axis_gain_dbi = np.append(axis_gain_dbi, axis_gain_dbi[0])

        # ---- Plotting ----
        views = ["polar", "cartesian"] if style == "both" else [style]

        for view in views:
            figure = plt.figure()

            if view == "polar":
                ax = figure.add_subplot(1, 1, 1, polar=True)
                lines = ax.plot(
                    np.deg2rad(theta),
                    clipped_gain_dbi,
                    linewidth=2,
                    label="xz-plane",
                )
                ax.set_rlim(axis_floor)

                text_anchor = (1.0, 0.1)
                cursor = themed_cursor(lines)
                cursor.connect(
                    "add",
                    lambda sel: sel.annotation.set_text(
                        f"theta={np.rad2deg(sel.target[0]):.2f}°"
                        f"\nRealized Gain = {sel.target[1]:.2f} dBi"
                    ),
                )
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.legend()

            else:
                ax = figure.add_subplot(1, 1, 1)
                lines = ax.plot(
                    axis_theta,
                    axis_gain_dbi,
                    linewidth=2,
                    label="xz-plane",
                )

                text_anchor = (0.02, 0.98)
                cursor = themed_cursor(lines)
                cursor.connect(
                    "add",
                    lambda sel: sel.annotation.set_text(
                        f"theta={sel.target[0]:.2f}°"
                        f"\nRealized Gain = {sel.target[1]:.2f} dBi"
                    ),
                )
                ax.set_xlim(axis_start, axis_start + 360.0)
                ax.set_xticks(np.arange(axis_start, axis_start + 361.0, 45.0))
                ax.set_ylim(axis_floor, peak_val + 3.0)
                ax.set_ylabel("Realized Gain (dBi)")
                ax.legend(loc="lower right")

            ax.text(
                *text_anchor,
                annotation,
                transform=ax.transAxes,
                fontsize=12,
                color=plt.rcParams["text.color"],
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor=plt.rcParams["axes.facecolor"],
                    edgecolor=plt.rcParams["axes.edgecolor"],
                ),
            )
            ax.grid(True)
            ax.set_xlabel("theta (deg)")
            figure.suptitle(
                f" Realized Gain at {freq_formatter(nf2ff_data.freq)}", fontsize=14
            )

    @staticmethod
    def plot_efficiency(
        nf2ff_data: NF2FFData,
        unit: Literal["percent", "db", "both"] = "both",
    ) -> None:
        """
        Plot an antenna's radiation and total efficiency against frequency.

        Draws the efficiency curve ``compute_nf2ff_data`` worked out across
        the antenna's -10 dB band, with markers on the frequencies it was
        actually computed at, so the line between them reads as interpolated.
        The efficiency shows in percent, in dB (``10 log10`` of the power
        ratio), or both in separate figures.

        Parameters
        ----------
        nf2ff_data : NF2FFData
            The far-field results returned by ``compute_nf2ff_data``.
        unit : {"percent", "db", "both"}, optional
            Which scale to draw. ``"both"`` draws the percent and dB views in
            two separate figures. Default is ``"both"``.

        Returns
        -------
        None
            Plots the efficiency curves; does not return a value.

        Raises
        ------
        ValueError
            If ``unit`` is not ``"percent"``, ``"db"`` or ``"both"``.
        """
        if unit not in ("percent", "db", "both"):
            raise ValueError(f"unit must be 'percent', 'db' or 'both', got {unit!r}")

        curve = nf2ff_data.efficiency_curve
        views = ["percent", "db"] if unit == "both" else [unit]

        for view in views:
            if view == "percent":
                curves = [
                    (100.0 * values, 100.0 * computed, label)
                    for values, computed, label in [
                        (curve.radiation, curve.computed_radiation, "Radiation"),
                        (curve.total, curve.computed_total, "Total"),
                    ]
                ]
                value_unit = "%"
                readout = "{:.1f} %"
            else:
                # A zero efficiency has no dB value; floor it at -60 dB
                curves = [
                    (
                        10.0 * np.log10(np.maximum(values, 1e-6)),
                        10.0 * np.log10(np.maximum(computed, 1e-6)),
                        label,
                    )
                    for values, computed, label in [
                        (curve.radiation, curve.computed_radiation, "Radiation"),
                        (curve.total, curve.computed_total, "Total"),
                    ]
                ]
                value_unit = "dB"
                readout = "{:.2f} dB"

            plt.figure()
            for values, computed, label in curves:
                label = f"{label} Efficiency"
                lines = plt.plot(curve.freqs, values, label=label)
                plt.plot(
                    curve.computed_freqs,
                    computed,
                    linestyle="none",
                    marker="o",
                    color=lines[0].get_color(),
                )
                cursor = themed_cursor(lines)
                cursor.connect(
                    "add",
                    lambda sel, label=label, readout=readout: sel.annotation.set_text(
                        f"Freq={freq_formatter(sel.target[0])}\n"
                        f"{label}={readout.format(sel.target[1])}"
                    ),
                )

            plt.gca().xaxis.set_major_formatter(freq_formatter)
            # A narrow band in GHz makes long tick labels; five keep them apart
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
            if view == "percent":
                plt.ylim(0.0, 100.0)
            else:
                plt.ylim(top=0.0)
            plt.xlabel("Frequency")
            plt.ylabel(f"Efficiency ({value_unit})")
            plt.title("Efficiency vs Frequency")
            plt.grid(True)
            plt.legend()

    @staticmethod
    def plot_3d_directivity(
        nf2ff_data: NF2FFData,
        output_path: Path | None = None,
    ) -> None:
        """
        Plot the 3D directivity pattern of an antenna.

        Visualizes the directivity pattern (in dBi) of the antenna in 3D
        space from the full-sphere result of ``compute_nf2ff_data``, and saves
        the mesh as a VTK file under ``output_path / "3D_plots"``.

        Parameters
        ----------
        nf2ff_data : NF2FFData
            The far-field results returned by ``compute_nf2ff_data``.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to ``cwd / "Sim_Path"``.

        Returns
        -------
        None
            Opens an interactive 3D plot window and saves the mesh to
            ``3D_plots/3D_directivity.vtk``; does not return a value.
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        plots_3d_path = output_path / "3D_plots"
        plots_3d_path.mkdir(parents=True, exist_ok=True)

        sphere = nf2ff_data.sphere
        e_field = np.squeeze(sphere.E_norm)
        e_field = e_field / np.max(e_field)  # normalize without mutating the result

        # radiated power density scales with |E|^2, so square the field ratio
        max_directivity = sphere.Dmax[0]
        directivity = max_directivity * e_field**2

        directivity_dbi = 10 * np.log10(directivity)

        theta = sphere.theta
        phi = sphere.phi

        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        R = 10 ** (directivity_dbi / 10)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        mesh = pv.StructuredGrid(X, Y, Z)
        mesh["Directivity (dBi)"] = directivity_dbi.ravel(order="F")
        mesh.save(plots_3d_path / "3D_directivity.vtk")

        plotter = BackgroundPlotter(
            title="Antenna 3D Pattern - Directivity (dBi)",
            show=False,
            auto_update=False,
        )
        plotter.app_window.setProperty(SHOW_WITH_PLOTS, time.monotonic())
        style_background_plotter(plotter)
        plotter.add_mesh(
            mesh,
            scalars="Directivity (dBi)",
            cmap="turbo",
            clim=[np.min(directivity_dbi), np.max(directivity_dbi)],
            smooth_shading=True,
        )
        plotter.add_text(
            "Antenna 3D Pattern - Directivity (dBi) at "
            f"{freq_formatter(nf2ff_data.freq)}",
            position="upper_edge",
        )

    @staticmethod
    def plot_3d_gain(
        nf2ff_data: NF2FFData,
        output_path: Path | None = None,
    ) -> None:
        """
        Plot the 3D realized gain pattern of an antenna.

        Visualizes the realized gain pattern (in dBi) of the antenna in 3D
        space: the full-sphere directivity scaled by the radiation and
        mismatch efficiencies that ``compute_nf2ff_data`` worked out. Saves
        the mesh under ``output_path / "3D_plots"``.

        Parameters
        ----------
        nf2ff_data : NF2FFData
            The far-field results returned by ``compute_nf2ff_data``.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to ``cwd / "Sim_Path"``.

        Returns
        -------
        None
            Opens an interactive 3D plot window and saves the mesh to
            ``3D_plots/3D_Realized_Gain.vtk``; does not return a value.
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        sphere = nf2ff_data.sphere
        e_field = np.squeeze(sphere.E_norm)
        e_field = e_field / np.max(e_field)  # normalize without mutating the result

        plots_3d_path = output_path / "3D_plots"
        plots_3d_path.mkdir(parents=True, exist_ok=True)

        # radiated power density scales with |E|^2, so square the field ratio
        max_directivity = sphere.Dmax[0]
        directivity = max_directivity * e_field**2

        # Realized gain also loses the power the port mismatch reflects
        realized_gain = (
            nf2ff_data.efficiency * nf2ff_data.mismatch_efficiency * directivity
        )
        gain_dbi = 10 * np.log10(realized_gain)

        theta = sphere.theta
        phi = sphere.phi

        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        R = 10 ** (gain_dbi / 10)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        mesh = pv.StructuredGrid(X, Y, Z)
        mesh["Realized Gain (dBi)"] = gain_dbi.ravel(order="F")
        mesh.save(plots_3d_path / "3D_Realized_Gain.vtk")
        plotter = BackgroundPlotter(
            title="Antenna 3D Pattern - Realized Gain (dBi)",
            show=False,
            auto_update=False,
        )
        plotter.app_window.setProperty(SHOW_WITH_PLOTS, time.monotonic())
        style_background_plotter(plotter)
        plotter.add_mesh(
            mesh,
            scalars="Realized Gain (dBi)",
            cmap="turbo",
            clim=[np.min(gain_dbi), np.max(gain_dbi)],
            smooth_shading=True,
        )
        plotter.add_text(
            "Antenna 3D Pattern - Realized Gain (dBi) at "
            f"{freq_formatter(nf2ff_data.freq)}",
            position="upper_edge",
        )

    @staticmethod
    def save_plots(output_path: Path | None = None, file_format: str = "png") -> None:
        """
        Save all currently open matplotlib figures to ``output_path / "plots"``.

        Each open figure is resized to 12x6 inches and saved as
        ``plot_<n>.<file_format>``, in figure order.

        Parameters
        ----------
        output_path : Path, optional
            Directory to save the ``plots`` subdirectory under. Defaults
            to ``cwd / "Sim_Path"``.
        file_format : str, optional
            File format to save the plots as (e.g. ``"png"``, ``"jpg"``,
            ``"pdf"``). Default is ``"png"``.

        Returns
        -------
        None

        Notes
        -----
        Call this before ``show_plots``, since ``show_plots`` blocks until
        the plot windows are closed.
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        plot_path = output_path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)
        # Get all open figure numbers
        figures = plt.get_fignums()

        for i, fig_num in enumerate(figures):
            fig = plt.figure(fig_num)
            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(12, 6)
            file_path = plot_path / f"plot_{i + 1}.{file_format}"
            fig.savefig(file_path, format=file_format, dpi=300)
            console.print(f"Plot {i + 1} saved as {file_path}", style="info")

    @staticmethod
    def add_field_dump(
        sim: SimSetup,
        params: SimParams,
        output_path: Path | None = None,
        dump_freq: float | None = None,
        dump_type: DumpType = DumpType.efield_frequency,
    ) -> None:
        """
        Add a field dump box to the simulation setup.

        Configures and attaches a field/current dump of the requested
        ``dump_type`` to the CSXCAD structure. The box follows the simulation
        box in x and y -- ``params.simulation_box`` when defined, or the same
        geometry-padded box the FDTD mesher builds when it is ``None`` (see
        :func:`simpleEMS.fdtd_mesh.auto_simulation_bounds`) -- and spans z
        from 0 to the top of the copper layer
        (``substrate_thickness_mm + copper_thickness_mm``).

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        params : SimParams
            Parameter object supplying ``simulation_box``, ``lambda0``,
            ``substrate_thickness_mm``, ``copper_thickness_mm`` and
            ``main_freq``, used to size and configure the dump.
        output_path : Path, optional
            Directory the dump data will be written under (a ``field_dump``
            subdirectory is created). Defaults to ``cwd / "Sim_Path"``.
        dump_freq : float, optional
            Frequency of a frequency-domain dump. Defaults to
            ``params.main_freq``.
        dump_type : DumpType, optional
            Type of field/current dump to add (time-domain or
            frequency-domain E-field, H-field, current, current density, or
            SAR). Default is ``DumpType.efield_frequency``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Under the FEM backend, if ``dump_type`` is not
            ``efield_frequency``, ``hfield_frequency`` or
            ``current_density_frequency``.

        Notes
        -----
        The FEM backend queues the dump on ``sim``, and :meth:`run_simulation`
        writes it after the sweep in the same files an openEMS
        frequency-domain dump produces: 21 phase frames
        (``<prefix>_f=<freq>_p=<degrees>.vtr``) that ParaView animates, plus
        ``_abs`` and ``_arg`` files. The field is sampled on the FDTD mesh
        lines inside the box -- the grid already in ``sim.CSX`` if
        ``create_mesh`` built one, or else the one the FDTD auto-mesher would
        build -- so the dump box has the same straight faces as in openEMS.
        """
        simulation_bounds = params.simulation_bounds
        if simulation_bounds is None:
            simulation_bounds = auto_simulation_bounds(
                geometry_extent(sim.CSX), params.lambda0
            )
        copper_top = params.substrate_thickness_mm + params.copper_thickness_mm
        if dump_freq is None:
            dump_freq = params.main_freq

        if params.backend_engine == "FEM":
            if dump_type not in (
                DumpType.efield_frequency,
                DumpType.hfield_frequency,
                DumpType.current_density_frequency,
            ):
                raise ValueError(
                    f"the FEM backend cannot dump {dump_type.name}; use "
                    "efield_frequency, hfield_frequency or "
                    "current_density_frequency"
                )
            box_mm = (
                simulation_bounds[0][0],
                simulation_bounds[1][0],
                0.0,
                simulation_bounds[0][1],
                simulation_bounds[1][1],
                copper_top,
            )
            grid = sim.CSX.GetGrid()
            if not all(len(grid.GetLines(axis)) for axis in range(3)):
                Mesh(sim.CSX, params)
            sim.FEM_field_dumps.append(
                FieldDumpRequest(
                    prefix=dump_type.value[1],
                    freq=float(dump_freq),
                    box=tuple(value * params.unit for value in box_mm),
                    lines=tuple(
                        tuple(float(line) * params.unit for line in grid.GetLines(axis))
                        for axis in range(3)
                    ),
                )
            )
            return

        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        # TODO Add appropriate dump mode based on openEMS docs
        dump_path = output_path / "field_dump"
        dump_path.mkdir(parents=True, exist_ok=True)
        dump_name = str(Path("field_dump") / dump_type.value[1])
        dump = sim.CSX.AddDump(
            dump_name,
            file_type=0,
            dump_type=dump_type.value[0],
            dump_mode=0,
        )
        start = [simulation_bounds[0][0], simulation_bounds[1][0], 0.0]
        stop = [simulation_bounds[0][1], simulation_bounds[1][1], copper_top]
        dump.AddBox(start=start, stop=stop)
        if dump_type.value[0] >= 10:
            dump.AddFrequency(dump_freq)

    @staticmethod
    def visualise_field_dump(output_path: Path | None = None) -> None:
        """
        Open an interactive 3-D viewer for a field dump.

        Animates a dump added with :meth:`add_field_dump` over the structure,
        with a control panel for arrow, surface, height-map, cut-plane,
        iso-surface and volume plots. Shows E-field, H-field, current and
        current-density dumps, in the time or frequency domain, from either
        backend. When the simulation saved several dumps, a "Field dump" picker
        in the window switches between them, starting from the first dump and
        its lowest frequency. The window opens with the other
        plots when you call :meth:`show_plots`; press ``q`` in it to close it.
        Wraps :func:`simpleEMS.visualise_field_dump.view_field_dump`.

        Parameters
        ----------
        output_path : Path, optional
            Simulation directory holding the ``field_dump`` subdirectory and
            ``structure.xml``. Defaults to ``cwd / "Sim_Path"``.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the directory holds no field dump the viewer can show.
        """
        viewer = view_field_dump(output_path=output_path, show=False)
        viewer.setProperty(SHOW_WITH_PLOTS, time.monotonic())

    @staticmethod
    def export_touchstone(
        freqs: NDArray,
        s11: NDArray | None = None,
        *,
        s21: NDArray | None = None,
        s_matrix: NDArray | None = None,
        charac_imp: float = 50.0,
        output_path: Path | None = None,
        filename: str = "s_param",
    ) -> Path:
        """
        Export S-parameters to a Touchstone ``.sNp`` file.

        Writes the provided S-parameter data to ``output_path / "touchstone"``
        in Touchstone version 1 format. Pass either ``s11`` (plus ``s21`` for
        a two-port) or the full ``s_matrix``; the port count sets the file
        extension.

        Parameters
        ----------
        freqs : NDArray
            Frequency points (in Hz) to export.
        s11 : NDArray, optional
            Complex S11 values across the frequency range. Required unless
            ``s_matrix`` is given.
        s21 : NDArray, optional
            Complex S21 values across the frequency range. If provided, a
            two-port (``.s2p``) network is written; otherwise a single-port
            (``.s1p``) network is written. Default is ``None``.
        s_matrix : NDArray, optional
            Complex S-parameters of shape ``(F, P, P)``, with
            ``s_matrix[:, i, j]`` holding S(i+1)(j+1). Writes every
            S-parameter of a ``P``-port network. Cannot be combined with
            ``s11`` or ``s21``. Default is ``None``.
        charac_imp : float, optional
            Reference impedance (in ohms) recorded in the Touchstone file.
            Default is ``50.0``.
        output_path : Path, optional
            Directory to save the ``touchstone`` subdirectory under.
            Defaults to ``cwd / "Sim_Path"``.
        filename : str, optional
            Base name of the exported file (extension is added
            automatically). Default is ``"s_param"``.

        Returns
        -------
        Path
            Path of the written Touchstone file.

        Raises
        ------
        ValueError
            If neither ``s11`` nor ``s_matrix`` is given, or if ``s_matrix``
            is combined with ``s11`` / ``s21``.

        Notes
        -----
        With ``s11`` and ``s21``, only S11 and S21 are populated; S12 and
        S22 are written as zero (the reverse-direction parameters are not
        simulated). Pass ``s_matrix`` to export all of them.
        """
        console.print("-------------------------------------------", style="info")
        console.print("Exporting S-Parameters to Touchstone file", style="info")
        console.print("-------------------------------------------", style="info")

        if s_matrix is None:
            if s11 is None:
                raise ValueError("Pass either s11 (and optionally s21) or s_matrix.")
            port_count = 1 if s21 is None else 2
            s_matrix = np.zeros((len(s11), port_count, port_count), dtype=complex)
            s_matrix[:, 0, 0] = s11
            if s21 is not None:
                s_matrix[:, 1, 0] = s21
        elif s11 is not None or s21 is not None:
            raise ValueError("Pass s_matrix or s11/s21, not both.")

        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        return write_touchstone(
            output_path / "touchstone" / filename,
            freqs,
            s_matrix,
            ref_impedance=charac_imp,
        )

    @staticmethod
    def export_csv(
        freqs: NDArray | None = None,
        s11: NDArray | None = None,
        *,
        s21: NDArray | None = None,
        s_matrix: NDArray | None = None,
        nf2ff_data: NF2FFData | None = None,
        output_path: Path | None = None,
        prefix: str = "",
    ) -> list[Path]:
        """
        Export S-parameters and far-field results to CSV files.

        Writes comma-separated tables to ``output_path / "csv"`` that Excel,
        LibreOffice, MATLAB or pandas open directly. Each kind of result goes
        to its own file:

        - ``s_param.csv``, when ``freqs`` and S-parameters are given: the
          frequency in GHz, then for each S-parameter its magnitude in dB,
          its phase in degrees, and its real and imaginary parts. Pass
          either ``s11`` (plus ``s21`` for a two-port) or the full
          ``s_matrix``.
        - ``radiation_pattern.csv``, when ``nf2ff_data`` is given: the
          directivity and realized gain in dBi on the xz, yz and xy planes
          at ``nf2ff_data.freq``, in 0.1 deg steps from -180 to 180 deg. The
          angle column is theta for the xz and yz planes and phi for the xy
          plane.
        - ``efficiency.csv``, when ``nf2ff_data`` is given: the radiation and
          total efficiency in percent and in dB across the -10 dB band, read
          from ``nf2ff_data.efficiency_curve``.

        Parameters
        ----------
        freqs : NDArray, optional
            Frequency points (in Hz) of the S-parameters. Required with
            ``s11`` or ``s_matrix``.
        s11 : NDArray, optional
            Complex S11 values across the frequency range.
        s21 : NDArray, optional
            Complex S21 values across the frequency range. Adds the S21
            columns when provided. Default is ``None``.
        s_matrix : NDArray, optional
            Complex S-parameters of shape ``(F, P, P)``, with
            ``s_matrix[:, i, j]`` holding S(i+1)(j+1). Writes every
            S-parameter of a ``P``-port network. Cannot be combined with
            ``s11`` or ``s21``. Default is ``None``.
        nf2ff_data : NF2FFData, optional
            The far-field results returned by ``compute_nf2ff_data``. Writes
            the radiation pattern and efficiency files. Default is ``None``.
        output_path : Path, optional
            Directory to save the ``csv`` subdirectory under.
            Defaults to ``cwd / "Sim_Path"``.
        prefix : str, optional
            Text put in front of every file name, e.g. ``"patch_"`` writes
            ``patch_s_param.csv``. Default is ``""``.

        Returns
        -------
        list of Path
            Paths of the written CSV files.

        Raises
        ------
        ValueError
            If nothing is given to export, if ``s_matrix`` is combined with
            ``s11`` / ``s21``, if ``freqs`` and the S-parameters do not come
            together, or if the frequency count does not match the
            S-parameter data.
        """
        console.print("-------------------------------------------", style="info")
        console.print("Exporting results to CSV files", style="info")
        console.print("-------------------------------------------", style="info")

        has_s_params = s11 is not None or s21 is not None or s_matrix is not None
        if not has_s_params and freqs is None and nf2ff_data is None:
            raise ValueError("Pass S-parameters with freqs, nf2ff_data, or both.")
        if has_s_params != (freqs is not None):
            raise ValueError("Pass freqs together with s11 or s_matrix.")

        # Each entry is one file: (name, column headers, column values).
        tables = []
        # Spreadsheets cannot parse -inf, so every dB conversion clips an
        # exact zero to the smallest positive float first.
        tiny = np.finfo(float).tiny

        if has_s_params:
            frequencies = np.asarray(freqs, dtype=float).ravel()
            if s_matrix is None:
                if s11 is None:
                    raise ValueError(
                        "Pass either s11 (and optionally s21) or s_matrix."
                    )
                named_params = {"S11": np.asarray(s11, dtype=complex).ravel()}
                if s21 is not None:
                    named_params["S21"] = np.asarray(s21, dtype=complex).ravel()
            elif s11 is not None or s21 is not None:
                raise ValueError("Pass s_matrix or s11/s21, not both.")
            else:
                matrix = np.asarray(s_matrix, dtype=complex)
                port_count = matrix.shape[1]
                named_params = {
                    f"S{row + 1}{column + 1}": matrix[:, row, column]
                    for row in range(port_count)
                    for column in range(port_count)
                }

            header = ["Frequency (GHz)"]
            columns = [frequencies / 1e9]
            for name, values in named_params.items():
                if values.size != frequencies.size:
                    raise ValueError(
                        f"{frequencies.size} frequencies but {values.size} "
                        f"{name} values"
                    )
                header += [
                    f"|{name}| (dB)",
                    f"{name} phase (deg)",
                    f"Re({name})",
                    f"Im({name})",
                ]
                columns += [
                    20 * np.log10(np.maximum(np.abs(values), tiny)),
                    np.angle(values, deg=True),
                    values.real,
                    values.imag,
                ]
            tables.append(("s_param", header, columns))

        if nf2ff_data is not None:
            # Reference each cut to the sphere's peak, as plot_2d_directivity
            # and plot_2d_gain do, so a cut that misses the main lobe sits
            # below Dmax instead of being lifted to it.
            sphere = nf2ff_data.sphere
            sphere_peak = np.max(sphere.E_norm)
            max_directivity_db = 10.0 * np.log10(sphere.Dmax[0])
            gain_offset_db = 10.0 * np.log10(
                nf2ff_data.efficiency * nf2ff_data.mismatch_efficiency
            )
            # All three cuts share the same 3601-point angle sweep.
            header = ["Angle (deg)"]
            columns = [np.rad2deg(np.ravel(nf2ff_data.xz_cut.theta))]
            for plane, cut in (
                ("xz", nf2ff_data.xz_cut),
                ("yz", nf2ff_data.yz_cut),
                ("xy", nf2ff_data.xy_cut),
            ):
                e_field_norm = np.maximum(np.squeeze(cut.E_norm) / sphere_peak, tiny)
                directivity_dbi = 20 * np.log10(e_field_norm) + max_directivity_db
                header += [
                    f"{plane}-plane directivity (dBi)",
                    f"{plane}-plane realized gain (dBi)",
                ]
                columns += [directivity_dbi, directivity_dbi + gain_offset_db]
            tables.append(("radiation_pattern", header, columns))

            curve = nf2ff_data.efficiency_curve
            radiation = np.asarray(curve.radiation, dtype=float)
            total = np.asarray(curve.total, dtype=float)
            header = [
                "Frequency (GHz)",
                "Radiation efficiency (%)",
                "Total efficiency (%)",
                "Radiation efficiency (dB)",
                "Total efficiency (dB)",
            ]
            columns = [
                np.asarray(curve.freqs, dtype=float) / 1e9,
                100 * radiation,
                100 * total,
                10 * np.log10(np.maximum(radiation, tiny)),
                10 * np.log10(np.maximum(total, tiny)),
            ]
            tables.append(("efficiency", header, columns))

        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"
        csv_dir = output_path / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        written = []
        for name, header, columns in tables:
            csv_path = csv_dir / f"{prefix}{name}.csv"
            np.savetxt(
                csv_path,
                np.column_stack(columns),
                delimiter=",",
                header=",".join(header),
                comments="",
                fmt="%.10g",
            )
            console.print(f"CSV file saved to {csv_path}", style="info")
            written.append(csv_path)
        return written

    @staticmethod
    def export_gerber(
        sim: SimSetup,
        output_path: Path | None = None,
        options: dict[str, list] | None = None,
        prefix: str = "layout",
    ) -> list[Path]:
        """
        Export CSXCAD geometry to Gerber format, one file per copper layer.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        output_path : Path, optional
            Directory to save the ``gerber`` subdirectory under. Defaults
            to ``cwd / "Sim_Path"``.
        options : dict[str, list], optional
            Dictionary of export options. Supported key ``"ignore"``: a
            list of property names to skip. Defaults to ``{}``.
        prefix : str, optional
            File name prefix. Default ``"layout"``.

        Returns
        -------
        list of Path
            The Gerber and drill files written.

        Notes
        -----
        CSXCAD carries no layer information, so layers are inferred from
        the Z position of the metal: the highest metal is the top layer
        (``F_Cu``), the lowest the bottom layer (``B_Cu``), and anything
        in between an inner layer. Z-axis cylinders (vias) are written to
        an Excellon drill file and the substrate footprint to the board
        outline. See :func:`simpleEMS.export_layout.export_gerber`.

        """
        if options is None:
            options = {}

        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        gerber_path = output_path / "gerber"
        gerber_path.mkdir(parents=True, exist_ok=True)

        return export_gerber(
            CSX=sim.CSX,
            output_path=gerber_path,
            options=options,
            prefix=prefix,
        )

    @staticmethod
    def export_dxf(
        sim: SimSetup,
        output_path: Path | None = None,
        options: dict[str, list] | None = None,
        filename: str = "layout",
    ) -> Path:
        """
        Export CSXCAD geometry to a DXF drawing for PCB or mechanical CAD.

        Writes ``output_path / "dxf" / "<filename>.dxf"`` in millimetres, with
        one DXF layer per copper layer (``F_Cu``, ``In<n>_Cu``, ``B_Cu``), plus
        ``Edge_Cuts`` for the board outline, ``Drills`` for the vias and a
        ``<layer>_Clearance`` layer for any antipads.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        output_path : Path, optional
            Directory to save the ``dxf`` subdirectory under. Defaults
            to ``cwd / "Sim_Path"``.
        options : dict[str, list], optional
            Dictionary of export options. Supported key ``"ignore"``: a
            list of property names to skip. Defaults to ``{}``.
        filename : str, optional
            Base name of the file (``.dxf`` is added). Default ``"layout"``.

        Returns
        -------
        Path
            The DXF file written.

        Notes
        -----
        Layers are inferred from the Z position of the metal, as for
        ``export_gerber``. See :func:`simpleEMS.export_layout.export_dxf`.
        """
        if options is None:
            options = {}

        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        dxf_path = output_path / "dxf"
        dxf_path.mkdir(parents=True, exist_ok=True)

        return export_dxf(
            CSX=sim.CSX,
            output_path=dxf_path,
            options=options,
            filename=filename,
        )

    @staticmethod
    def export_kicad_footprint(
        sim: SimSetup,
        port: LumpedPort,
        output_path: Path | None = None,
        options: dict[str, list] | None = None,
        name: str = "antenna",
        ground: list[str] | tuple[str, ...] = ("ground",),
    ) -> Path:
        """
        Export CSXCAD geometry as a KiCad footprint to place on a board.

        Writes ``output_path / "kicad_footprint" / "<name>.kicad_mod"``. To use
        it, add the ``kicad_footprint`` folder to KiCad as a footprint library.
        The footprint's origin is the feed point. Pad ``1`` carries the feed
        and every copper region not named in ``ground``; pad ``2`` carries
        the ground copper and the vias. Assign the feed trace's net to pad
        ``1`` and GND to pad ``2``.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        port : openEMS.ports.LumpedPort
            The excitation port the structure's build method returned. It
            sets the feed point and the feed pad size.
        output_path : Path, optional
            Directory to save the ``kicad_footprint`` subdirectory under.
            Defaults to ``cwd / "Sim_Path"``.
        options : dict[str, list], optional
            Dictionary of export options. Supported key ``"ignore"``: a
            list of property names to skip. Defaults to ``{}``.
        name : str, optional
            Footprint name, also the file name. Default ``"antenna"``.
        ground : list of str, optional
            Property names of the copper that belongs to ground. Default
            ``("ground",)``, the name every simpleEMS structure uses.

        Returns
        -------
        Path
            The footprint file written.

        Notes
        -----
        Antipads are not written, since a pad cannot cut a hole in another
        pad. See :func:`simpleEMS.export_layout.export_kicad_footprint`.
        """
        if options is None:
            options = {}

        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        library_path = output_path / "kicad_footprint"
        library_path.mkdir(parents=True, exist_ok=True)

        return export_kicad_footprint(
            CSX=sim.CSX,
            port=port,
            output_path=library_path,
            options=options,
            name=name,
            ground=ground,
        )

    @staticmethod
    def export_step(
        sim: SimSetup,
        output_path: Path | None = None,
    ) -> None:
        """
        Export CSXCAD geometry to a colored, multi-layer STEP AP242 file
        using CadQuery.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        output_path : Path, optional
            Directory to save the ``step`` subdirectory under. Defaults to
            ``cwd / "Sim_Path"``.

        Returns
        -------
        None
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        step_path = output_path / "step"
        step_path.mkdir(parents=True, exist_ok=True)

        export_step(
            CSX=sim.CSX,
            output_path=step_path,
        )

    @staticmethod
    def export_csxcad_xml_to_step(
        structure_xml_path: str | Path,
        output_path: Path | None = None,
    ) -> None:
        """
        Load a ``structure.xml`` file and export it to STEP AP242 format.

        Useful for re-exporting a structure to STEP without re-running the
        simulation, given a ``structure.xml`` previously written by
        ``write_and_show_structure``.

        Parameters
        ----------
        structure_xml_path : str or Path
            Path to the ``structure.xml`` file exported by CSXCAD / openEMS.
        output_path : Path, optional
            Directory to save the ``step`` subdirectory under. Defaults to
            ``cwd / "Sim_Path"``.

        Returns
        -------
        None
        """

        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        step_path = output_path / "step"
        step_path.mkdir(parents=True, exist_ok=True)

        export_csxcad_xml_to_step(structure_xml_path, output_path)

    @staticmethod
    def generate_report(
        params: SimParams,
        sim: SimSetup,
        sim_data: SimData,
        nf2ff_data: NF2FFData | None = None,
        output_path: Path | None = None,
        title: str | None = None,
        author: str | None = None,
    ) -> Path:
        """
        Write a PDF design report of the structure into a ``report`` folder
        under ``output_path``.

        The report holds the design target, the copper layout with its ports,
        the design dimensions and the simulated performance -- matching,
        transmission, input impedance and, with ``nf2ff_data``, the radiation
        pattern and efficiency -- with a short note on how to read each page.
        See :func:`simpleEMS.report.generate_report`.

        Parameters
        ----------
        params : SimParams
            Parameter object of the structure.
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        sim_data : SimData
            Network results returned by ``compute_sim_data``.
        nf2ff_data : NF2FFData, optional
            Far-field results returned by ``compute_nf2ff_data``. Adds the
            radiation page. Default is ``None``.
        output_path : Path, optional
            Directory to create the ``report`` folder under. Defaults to
            ``cwd / "Sim_Path"``.
        title : str, optional
            Report title. Defaults to ``"<Structure> design report"``.
        author : str, optional
            Name printed on the report and stored in the PDF metadata.

        Returns
        -------
        Path
            Path of the written PDF.
        """
        return generate_report(
            params,
            sim,
            sim_data,
            nf2ff_data,
            output_path=output_path,
            title=title,
            author=author,
        )

    @staticmethod
    def show_plots() -> None:
        """
        Display all generated plots. Blocks until the plot window(s) are
        closed.

        The 3D windows from :meth:`plot_3d_directivity`, :meth:`plot_3d_gain`
        and :meth:`visualise_field_dump` open here, on top of the matplotlib
        figures. If no matplotlib figures are open and a Qt application
        instance is already running (e.g. a PyVista/FEM mesh viewer opened by
        ``write_and_show_structure``), blocks on that Qt event loop instead
        and exits the process once it closes. Otherwise calls
        ``matplotlib.pyplot.show()`` to display any open matplotlib figures.

        Returns
        -------
        None
            Does not return a value.
        """

        def show_3d_windows() -> None:
            waiting = [
                window
                for window in QApplication.topLevelWidgets()
                if window.property(SHOW_WITH_PLOTS)
            ]
            for window in sorted(
                waiting, key=lambda window: window.property(SHOW_WITH_PLOTS)
            ):
                window.setProperty(SHOW_WITH_PLOTS, None)
                window.show()
                window.raise_()
                # Raising a window leaves the keyboard on the last figure, behind
                # it, so q would close a window you cannot see.
                window.activateWindow()

        if not plt.get_fignums():
            view_3d = QCoreApplication.instance()
            if view_3d is not None:
                show_3d_windows()
                sys.exit(view_3d.exec())
        # plt.show() shows the figures and then runs the event loop, so the
        # timer shows the 3D windows once the figures are up.
        if QCoreApplication.instance() is not None:
            QTimer.singleShot(0, show_3d_windows)
        plt.show()

    @staticmethod
    def print_and_save_params(
        params: SimParams,
        output_path: Path | None = None,
    ) -> None:
        """
        Print the simulation parameters to the console and save them to
        ``params/params.txt``.

        Parameters
        ----------
        params : SimParams
            Parameter object whose dataclass fields are printed and saved.
        output_path : Path, optional
            Directory to save the ``params`` subdirectory under. Defaults
            to ``cwd / "Sim_Path"``.

        Returns
        -------
        None
            Does not return a value.
        """

        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        params_path = output_path / "params"
        params_path.mkdir(parents=True, exist_ok=True)
        cls_name = params.__class__.__name__
        console.print(f"{cls_name}:", style="class_name")
        lines = []
        for field in fields(params):
            value = getattr(params, field.name)
            console.print(f"  [field]{field.name}:[/field] [value]{value}[/value]")
            lines.append(f"  {field.name}: {value}\n")
        with open(params_path / "params.txt", "w") as file:
            file.write(f"{cls_name}:\n")
            file.writelines(lines)


def minimize_s_param(
    freqs: NDArray,
    s_param: NDArray,
    target_freq: float | None = None,
    freq_band: tuple[float, float] | None = None,
    mode: str = "worst",
    threshold: float = -15,
) -> np.floating:
    """
    Cost function that drives an S-parameter down. Lower cost is better.

    Use it as the return value of the ``simulate_fn`` passed to
    :func:`optimize_s_params`, for S11 matching or a band-stop S21. It
    converts ``s_param`` to dB and reduces it to one number, either at a
    single frequency or over a band. ``target_freq`` takes precedence over
    ``freq_band``.

    Parameters
    ----------
    freqs : NDArray
        Frequencies in Hz, matching ``s_param``.
    s_param : NDArray
        Complex S-parameter to drive down, e.g. ``sim_data.s11``.
    target_freq : float, optional
        Evaluate at the frequency in ``freqs`` closest to this value,
        ignoring ``freq_band``, ``mode`` and ``threshold``.
    freq_band : tuple of (float, float), optional
        Band ``(start, stop)`` in Hz to evaluate over when ``target_freq``
        is not given.
    mode : str, optional
        How to reduce the band to one number:

        - ``"worst"`` : the highest value in dB.
        - ``"mean"`` : the mean value in dB.
        - ``"threshold"`` : the sum of how far, in dB, each point rises
          above ``threshold`` (0 when none do).

        Default is ``"worst"``.
    threshold : float, optional
        Ceiling in dB, used only when ``mode="threshold"``. Default is
        ``-15``.

    Returns
    -------
    float
        Scalar cost; lower means a lower S-parameter.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, or neither ``target_freq`` nor
        ``freq_band`` is given.
    """
    s_param_db = 20.0 * np.log10(np.abs(s_param))

    if target_freq is not None:
        closest_index = (np.abs(freqs - target_freq)).argmin()
        return s_param_db[closest_index]

    if freq_band is not None:
        band_start, band_stop = freq_band
        in_band = (freqs >= band_start) & (freqs <= band_stop)

        if mode == "worst":
            return np.max(s_param_db[in_band])

        elif mode == "mean":
            return np.mean(s_param_db[in_band])

        elif mode == "threshold":
            excess = np.maximum(s_param_db[in_band] - threshold, 0)
            return np.sum(excess)

        else:
            raise ValueError("mode must be 'mean', 'worst' or 'threshold'")

    raise ValueError("Provide target_freq or freq_band")


def maximize_s_param(
    freqs: NDArray,
    s_param: NDArray,
    target_freq: float | None = None,
    freq_band: tuple[float, float] | None = None,
    mode: str = "worst",
    threshold: float = -1,
) -> np.floating:
    """
    Cost function that drives an S-parameter up. Lower cost is better.

    Use it as the return value of the ``simulate_fn`` passed to
    :func:`optimize_s_params`, for example a band-pass S21. Because
    ``scipy.optimize.minimize`` minimizes, the cost is the *negated* value
    in dB, so a lower cost means a higher S-parameter. ``target_freq``
    takes precedence over ``freq_band``.

    Parameters
    ----------
    freqs : NDArray
        Frequencies in Hz, matching ``s_param``.
    s_param : NDArray
        Complex S-parameter to drive up, e.g. ``sim_data.s21``.
    target_freq : float, optional
        Evaluate at the frequency in ``freqs`` closest to this value,
        ignoring ``freq_band``, ``mode`` and ``threshold``.
    freq_band : tuple of (float, float), optional
        Band ``(start, stop)`` in Hz to evaluate over when ``target_freq``
        is not given.
    mode : str, optional
        How to reduce the band to one number:

        - ``"worst"`` : the negated lowest value in dB.
        - ``"mean"`` : the negated mean value in dB.
        - ``"threshold"`` : the sum of how far, in dB, each point drops
          below ``threshold`` (0 when none do).

        Default is ``"worst"``.
    threshold : float, optional
        Floor in dB, used only when ``mode="threshold"``. Default is
        ``-1``.

    Returns
    -------
    float
        Scalar cost; lower means a higher S-parameter.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, or neither ``target_freq`` nor
        ``freq_band`` is given.
    """
    s_param_db = 20.0 * np.log10(np.abs(s_param))

    if target_freq is not None:
        closest_index = (np.abs(freqs - target_freq)).argmin()
        return -s_param_db[closest_index]

    if freq_band is not None:
        band_start, band_stop = freq_band
        in_band = (freqs >= band_start) & (freqs <= band_stop)

        if mode == "worst":
            return -np.min(s_param_db[in_band])

        elif mode == "mean":
            return -np.mean(s_param_db[in_band])

        elif mode == "threshold":
            shortfall = np.maximum(threshold - s_param_db[in_band], 0)
            return np.sum(shortfall)

        else:
            raise ValueError("mode must be 'mean', 'worst' or 'threshold'")

    raise ValueError("Provide target_freq or freq_band")


def optimize_s_params(
    simulate_fn: Callable,
    x0: dict[str, float],
    output_path: Path,
    bounds: tuple | None = None,
) -> None:
    """
    Optimize simulation parameters using SciPy's Nelder-Mead algorithm.

    Runs ``scipy.optimize.minimize(..., method="Nelder-Mead")`` over
    ``simulate_fn``, starting from ``x0``. Stops on SciPy's own
    convergence criteria (``xatol=1e-3``, ``fatol=1e-3``), or early via a
    callback that stops the optimization once the parameter vector moves
    by less than ``1e-4`` (Euclidean norm) for 5 consecutive iterations.
    Prints the optimal parameter values to stdout when finished.

    Parameters
    ----------
    simulate_fn : callable
        Simulation function invoked at each optimization step as
        ``simulate_fn(output_path=output_path, optimize=True,
        optimize_val=x)``, where ``x`` is the current parameter vector (in
        the order of ``x0``'s values). Must return a scalar cost to
        minimize, e.g. via ``minimize_s_param`` or
        ``maximize_s_param``.
    x0 : dict[str, float]
        Initial parameter values, used as the optimization starting point.
        Keys are used only for the final printout; the order of the
        values defines the parameter vector order passed to
        ``simulate_fn``.
    output_path : Path
        Directory where simulation results are stored; forwarded to
        ``simulate_fn`` on every call.
    bounds : sequence of (float, float), optional
        Per-parameter ``(min, max)`` bounds, in the same order as
        ``x0``'s values, passed through to ``scipy.optimize.minimize``.
        ``None`` in a pair means no bound on that side. Default is
        ``None`` (unbounded).

    Returns
    -------
    None
        Does not return a value; prints the optimal parameters found.
    """
    console.print("-------------------------------------", style="info")
    console.print("Running Optimization", style="info")
    console.print("-------------------------------------", style="info")
    x0_values = list(x0.values())
    x0_keys = list(x0.keys())

    def make_callback(tol: float = 1e-4, patience: int = 5) -> Callable:
        """Create a callback that stops optimization when parameters stall."""
        prev_x = None
        stall = 0

        def callback(xk: OptimizeResult) -> None:
            """Raise StopIteration once the parameter vector stalls for
            ``patience`` consecutive calls."""
            nonlocal prev_x, stall
            if prev_x is not None:
                if np.linalg.norm(xk - prev_x) < tol:
                    stall += 1
                else:
                    stall = 0
            prev_x = xk.copy()

            if stall >= patience:
                raise StopIteration

        return callback

    callback = make_callback()

    def optimize_fun(x: NDArray) -> float:
        """Call the simulation with the current parameter values."""
        return simulate_fn(output_path=output_path, optimize=True, optimize_val=x)

    try:
        res = minimize(
            optimize_fun,
            x0_values,
            method="Nelder-Mead",
            bounds=bounds,
            callback=callback,
            options={"disp": True, "xatol": 1e-3, "fatol": 1e-3},
        )
        for i, key in enumerate(x0_keys):
            print(f"optimal {key} = {res.x[i]}")

    except StopIteration:
        pass


def param_sweep(
    simulate_fn: Callable,
    sweep_vals: dict[str, tuple],
    output_path: Path,
    sweep: bool = True,
) -> SweepData:
    """
    Run a simulation function over a Cartesian product of parameter values
    and collect the results of every combination.

    Nothing is plotted. Pass the returned :class:`SweepData` to the
    ``SimTools.plot_*`` functions to overlay the runs, e.g.
    ``SimTools.plot_s_param(sweep_data.freqs, sweep_data.s11,
    label_s11=sweep_data.labels)``.

    Supports sweeping multiple parameters at once; the total number of
    simulations run is the Cartesian product of each parameter's
    ``num_points``, which can get expensive quickly with more than one or
    two swept parameters.

    Parameters
    ----------
    simulate_fn : callable
        Simulation function invoked for each sweep combination as
        ``simulate_fn(sweep_path, sweep, values)``, where ``sweep_path``
        is a per-combination output directory, ``sweep`` is the ``sweep``
        argument passed through unchanged, and ``values`` is the tuple of
        parameter values for that combination (in ``sweep_vals`` key
        order). Must return a :class:`SimData`, with the same frequency
        points on every call.
    sweep_vals : dict[str, tuple]
        Dictionary defining the parameter sweep ranges. Keys are
        parameter names (for labeling only); values are
        ``(start, stop, num_points)`` tuples passed to ``numpy.linspace``.
    output_path : Path
        Directory under which a ``sweep/<param>_<value>_..." `` subfolder is
        created for each combination.
    sweep : bool, optional
        Passed through unchanged as the second positional argument to
        ``simulate_fn`` on every call. Default is ``True``.

    Returns
    -------
    SweepData
        The simulation data from every parameter combination, stacked one
        row per run in the same order as the Cartesian product of
        ``sweep_vals``, with a legend label and the swept values per run.

    Raises
    ------
    ValueError
        If ``sweep_vals`` has no combinations to run, or if a run returns
        different frequency points from the first run.
    """
    console.print("-------------------------------------", style="info")
    console.print("Running Parameter Sweep", style="info")
    console.print("-------------------------------------", style="info")

    sweep_values = {key: np.linspace(*val).tolist() for key, val in sweep_vals.items()}
    cartesian_sweep = list(product(*sweep_values.values()))
    if not cartesian_sweep:
        raise ValueError("The sweep has no parameter combinations to run.")

    sim_data_list = []
    labels = []
    values_per_run = []

    for values in cartesian_sweep:
        named_values = dict(zip(sweep_values.keys(), values, strict=True))
        key_value = "_".join(f"{k}_{v}" for k, v in named_values.items())
        sweep_path = output_path / "sweep" / key_value
        sweep_path.mkdir(parents=True, exist_ok=True)

        sim_data = simulate_fn(sweep_path, sweep, values)
        if sim_data_list and not np.array_equal(sim_data.freqs, sim_data_list[0].freqs):
            raise ValueError(
                f"The run with {key_value} returned different frequency points "
                "from the first run; every run must share the same frequencies."
            )
        sim_data_list.append(sim_data)
        labels.append(", ".join(f"{k}={v}" for k, v in named_values.items()))
        values_per_run.append(named_values)

    first_run = sim_data_list[0]
    stacked_fields = {
        field: None
        if getattr(first_run, field) is None
        else np.stack([getattr(sim_data, field) for sim_data in sim_data_list])
        for field in SimData._fields
        if field != "freqs"
    }
    return SweepData(
        freqs=first_run.freqs,
        labels=labels,
        values=values_per_run,
        **stacked_fields,
    )


def mm_to_m(mm: float) -> float:
    """
    Convert millimeters to meters.

    Parameters
    ----------
    mm : float
        Value in millimeters.

    Returns
    -------
    float
        Value in meters.
    """
    return mm / 1000.0


def m_to_mm(m: float) -> float:
    """
    Convert meters to millimeters.

    Parameters
    ----------
    m : float
        Value in meters.

    Returns
    -------
    float
        Value in millimeters.
    """
    return m * 1000.0


def mil_to_mm(mil: float) -> float:
    """
    Convert mils (thousandths of an inch) to millimeters.

    Parameters
    ----------
    mil : float
        Value in mils.

    Returns
    -------
    float
        Value in millimeters.
    """
    return mil * 0.0254


def mm_to_mil(mm: float) -> float:
    """
    Convert millimeters to mils (thousandths of an inch).

    Parameters
    ----------
    mm : float
        Value in millimeters.

    Returns
    -------
    float
        Value in mils.
    """
    return mm / 0.0254


def cm_to_mm(cm: float) -> float:
    """
    Convert centimeters to millimeters.

    Parameters
    ----------
    cm : float
        Value in centimeters.

    Returns
    -------
    float
        Value in millimeters.
    """
    return cm * 10.0


def mm_to_cm(mm: float) -> float:
    """
    Convert millimeters to centimeters.

    Parameters
    ----------
    mm : float
        Value in millimeters.

    Returns
    -------
    float
        Value in centimeters.
    """
    return mm / 10.0
