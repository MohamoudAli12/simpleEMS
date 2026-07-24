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

import subprocess
import sys
from typing import NamedTuple
from collections.abc import Callable
from dataclasses import fields
from enum import Enum
from itertools import product
from pathlib import Path

# ----------------------------
import matplotlib.pyplot as plt
import mplcursors
import pyvista as pv
from matplotlib.ticker import EngFormatter
from PyQt6.QtCore import QCoreApplication
from pysmithchart import S_PARAMETER
from pyvistaqt import BackgroundPlotter

# ----------------------------
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from skrf import Network

# ----------------------------
from CSXCAD import AppCSXCAD_BIN, ContinuousStructure
from openEMS.openEMS import openEMS
from openEMS.ports import LumpedPort
from openEMS.nf2ff import nf2ff
from openEMS.nf2ff import nf2ff_results

# ----------------------------
from .console import console
from .export_gerber import export_gerber
from .export_step import export_step, export_csxcad_xml_to_step
from .fem_materials import FEMOptions
from .sim_params import SimParams

# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "DumpType",
    "SimTools",
    "setup_simulation",
    "optimize_s_params",
    "optimize_s11",
    "optimize_s21",
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
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["savefig.dpi"] = 300


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
    z11: NDArray
    vswr: NDArray
    input_power: float
    port_voltage: NDArray
    port_current: NDArray
    ref_impedance: float = 50.0


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
    num_FEM_solve_points : int
        Number of full FEM solves the adaptive sweep may perform (FEM only).
    FEM_options : FEMOptions | None
        Global FEM solver/mesh options (boundary, symmetry, fe_order, mesh
        tuning, port type). ``None`` for FDTD.
    """

    CSX: ContinuousStructure
    FDTD: openEMS
    freqs: NDArray
    backend_engine: str
    num_FEM_solve_points: int = 10
    FEM_options: FEMOptions | None = None
    charac_imp: float = 50


def setup_simulation(
    params: SimParams,
    FDTD_boundary: list[str] | None = None,
    FEM_boundary: str = "silver_muller",
    FEM_symmetry: tuple | None = None,
    FEM_fe_order: int = 1,
    FEM_air_pad_frac: float = 0.2,
    FEM_elems_per_wavelength: float = 8.0,
    FEM_mesh_fine_scale: float = 1.0,
    FEM_min_layers: int = 3,
    FEM_port_type: str = "lumped",
) -> SimSetup:
    """
    Build the CSXCAD geometry container and configure the openEMS/FDTD
    (or FEM) solver from a parameter object.

    Creates the ``ContinuousStructure`` and ``openEMS`` objects, applies the
    boundary conditions and Gaussian excitation derived from
    ``params.freq_range``, and (for the FEM backend) bundles the mesh/solver
    tuning options into a :class:`~simpleEMS.fem_materials.FEMOptions`
    instance. The returned :class:`SimSetup` is passed to most other
    ``SimTools`` methods.

    Parameters
    ----------
    params : SimParams
        Parameter object that holds all simulation parameters, including
        ``freq_range``, ``num_points``, ``timestep``, ``end_criteria``,
        ``backend_engine``, ``num_FEM_solve_points``, and ``charac_imp``.
    FDTD_boundary : list[str], optional
        Six openEMS boundary-condition strings, one per box face in the
        order ``[xmin, xmax, ymin, ymax, zmin, zmax]``. Only used by the
        FDTD backend. Defaults to ``["PML_8"] * 6``.
    FEM_boundary : str, optional
        FEM backend only. Outer truncation: ``"silver_muller"`` (default)
        or ``"pml"``.
    FEM_symmetry : tuple, optional
        FEM backend only. Mirror-symmetry plane ``(axis, kind, at)`` used to
        halve the mesh. ``None`` (default) disables symmetry. See
        :class:`~simpleEMS.fem_materials.FEMOptions`.
    FEM_fe_order : int, optional
        FEM backend only. Nedelec edge-element order: ``1`` (default) or
        ``2``.
    FEM_air_pad_frac : float, optional
        FEM backend only. Air padding as a fraction of the longest
        wavelength. Default is ``0.2``.
    FEM_elems_per_wavelength : float, optional
        FEM backend only. Target coarse mesh density in open air. Default
        is ``8.0``.
    FEM_mesh_fine_scale : float, optional
        FEM backend only. Multiplier on the near-conductor element size.
        Default is ``1.0``.
    FEM_min_layers : int, optional
        FEM backend only. Element layers through the dielectric thickness.
        Default is ``3``.
    FEM_port_type : str, optional
        FEM backend only. ``"lumped"`` (default, impedance ``charac_imp``)
        or ``"wave"`` (matched to the line's characteristic impedance) for
        all ports.

    All ``FEM_*`` arguments are ignored when ``params.backend_engine`` is
    ``"FDTD"``.

    Returns
    -------
    SimSetup
        Named tuple with ``CSX`` (CSXCAD geometry), ``FDTD`` (openEMS FDTD
        object), ``freqs`` (frequency array), ``backend_engine``,
        ``num_FEM_solve_points``, ``FEM_options`` (``None`` for FDTD), and
        ``charac_imp``.
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
    FDTD = openEMS(NrTS=params.timestep, EndCriteria=params.end_criteria)

    FDTD.SetBoundaryCond(FDTD_boundary)
    FDTD.SetCSX(CSX)

    fmin, fmax = params.freq_range
    centre_freq = (fmin + fmax) / 2
    span_freq = (fmax - fmin) / 2

    FDTD.SetGaussExcite(centre_freq, span_freq)

    freqs = np.linspace(fmin, fmax, params.num_points)

    FEM_options = None
    if params.backend_engine == "FEM":
        FEM_options = FEMOptions(
            boundary=FEM_boundary,
            symmetry=FEM_symmetry,
            fe_order=FEM_fe_order,
            air_pad_frac=FEM_air_pad_frac,
            elems_per_wavelength=FEM_elems_per_wavelength,
            mesh_fine_scale=FEM_mesh_fine_scale,
            min_layers=FEM_min_layers,
            port_type=FEM_port_type,
        )

    return SimSetup(
        CSX=CSX,
        FDTD=FDTD,
        freqs=freqs,
        backend_engine=params.backend_engine,
        num_FEM_solve_points=params.num_FEM_solve_points,
        FEM_options=FEM_options,
        charac_imp=params.charac_imp,
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
        mesh_style: str = "wireframe",
        theme: str = "dark",
    ) -> None:
        """
        Display the simulation geometry.

        For the FDTD backend this opens the structure in AppCSXCAD. For the
        FEM backend it builds (or reuses) the Gmsh mesh and renders it with
        PyVista, coloring cells by their Gmsh physical-group id.

        For the FEM backend, if ``output_path`` already holds a mesh from a
        previous call (``fem_mesh.json`` + its ``.msh`` file), that mesh is
        reused instead of re-exporting STEP and re-meshing. Delete
        ``fem_mesh.json`` (or use a fresh ``output_path``) to force a rebuild
        after changing the geometry or ``FEM_options``.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        output_path : Path, optional
            Directory to write/read the structure XML (FDTD) or mesh files
            (FEM). Defaults to ``cwd / "Sim_Path"``, created if missing.
        mesh_style : str
            FEM mesh only. PyVista ``add_mesh`` representation: ``"surface"``,
            ``"wireframe"`` (default), or ``"points"``. Mesh cells are colored
            by their Gmsh physical-group id (``CellEntityIds``: dielectric,
            PEC, port, absorbing boundary, ...).
        theme : str
            FEM mesh only. PyVista plot theme, applied globally via
            :func:`pyvista.set_plot_theme`: ``"dark"`` (default),
            ``"default"``, ``"document"``, or ``"paraview"``.

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

            pv.set_plot_theme(theme)

            msh_path = fem_backend.existing_mesh_path(output_path)
            if msh_path is None:
                msh_path = fem_backend.build_mesh(
                    sim.CSX, sim.freqs, output_path, FEM_options=sim.FEM_options
                )
            vtk_path = Path(msh_path).with_suffix(".vtk")
            grid = pv.read(str(vtk_path if vtk_path.exists() else msh_path))

            plotter = pv.Plotter()
            plotter.add_mesh(
                grid,
                style=mesh_style,
                show_edges=mesh_style != "wireframe",
                opacity=0.5,
                scalars="CellEntityIds",
                cmap="tab20",
                show_scalar_bar=False,
            )
            plotter.add_axes()
            plotter.view_xy()
            plotter.show()
            return

        structure_3d = output_path / "structure.xml"

        sim.FDTD.Write2XML(
            str(structure_3d)
        )  # str in this fixes an error encountered on Windows.
        subprocess.run([AppCSXCAD_BIN, str(structure_3d)], check=True)

    @staticmethod
    def export_stl(output_path: Path | None = None) -> None:
        """
        Export the CSXCAD structure to STL files (FDTD backend only).

        Reads the ``structure.xml`` previously written by
        ``write_and_show_structure`` and exports it to STL via AppCSXCAD.

        Parameters
        ----------
        output_path : Path, optional
            Directory containing ``structure.xml``. STL files are written
            to a ``stl`` subdirectory. Defaults to ``cwd``.

        Raises
        ------
        ValueError
            If ``structure.xml`` does not exist in ``output_path`` (i.e.
            ``write_and_show_structure`` was not called first).
        """
        if output_path is None:
            output_path = Path.cwd()

        structure_3d = output_path / "structure.xml"
        if not structure_3d.exists():
            raise ValueError(
                "3D Structure does not exist. Make sure to call "
                "write_and_show_structure before exporting STL"
            )
        stl_path = output_path / "stl"
        stl_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            AppCSXCAD_BIN,
            f"--export-STL={stl_path}",
            str(structure_3d),
        ]

        subprocess.run(cmd, check=True)

    @staticmethod
    def run_simulation(sim: SimSetup, output_path: Path | None = None) -> None:
        """
        Run the simulation and write results to ``output_path``.

        Dispatches to the FDTD engine (``FDTD.Run``) or, for the FEM
        backend, to the adaptive GetDP frequency sweep
        (``fem_backend.run_sweep``), which performs up to
        ``sim.num_FEM_solve_points`` full solves.

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

        if sim.backend_engine == "FEM":
            from . import fem_backend

            fem_backend.run_sweep(
                sim.CSX,
                sim.freqs,
                sim.num_FEM_solve_points,
                output_path,
                FEM_options=sim.FEM_options,
            )
            return

        sim.FDTD.Run(output_path)

    @staticmethod
    def create_nf2ff(sim: SimSetup) -> nf2ff:
        """
        Create the near-field-to-far-field (NF2FF) recording box.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.

        Returns
        -------
        nf2ff : object
            NF2FF object. For the FEM backend this is a ``FEMNF2FF`` adapter that
            exposes the same ``CalcNF2FF`` interface, so the radiation-plotting
            methods work identically for both backends.
        """
        if sim.backend_engine == "FEM":
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
        derives S11 (and S21 for two-port setups), Z11, VSWR, and input
        power from the port voltage/current waves. For the FEM backend the
        ``port`` argument is ignored and results are instead read from the
        GetDP sweep output previously written to ``output_path`` by
        ``run_simulation``.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``;
            supplies ``freqs``, ``backend_engine``, and ``charac_imp``.
        port : LumpedPort or list of LumpedPort
            The openEMS port object representing a single-port simulation,
            or a two-element list ``[port1, port2]`` for a two-port
            simulation. Ignored when ``sim.backend_engine == "FEM"``.
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
        """
        if sim.backend_engine == "FEM":
            from . import fem_backend

            # match the default used by write_and_show_structure / run_simulation
            # so the FEM results are read from where they were written.
            if output_path is None:
                output_path = Path.cwd() / "Sim_Path"
            return fem_backend.compute_sim_data(sim.freqs, sim.charac_imp, output_path)

        if output_path is None:
            output_path = Path.cwd()

        if isinstance(port, list):
            for p in port:
                p.CalcPort(str(output_path), sim.freqs, ref_impedance=sim.charac_imp)
            s11 = port[0].uf_ref / port[0].uf_inc
            s21 = port[1].uf_ref / port[0].uf_inc
            z11 = port[0].uf_tot / port[0].if_tot
            s11_mag = np.abs(s11)
            s11_mag = np.clip(s11_mag, 0, 0.999)  # prevent division by zero error
            vswr = (1 + s11_mag) / (1 - s11_mag)
            input_power = 0.5 * np.real(port[0].uf_tot * np.conj(port[0].if_tot))
            return SimData(
                sim.freqs,
                s11,
                s21,
                z11,
                vswr,
                input_power,
                port[0].uf_tot,
                port[0].if_tot,
                sim.charac_imp,
            )
        else:
            port.CalcPort(str(output_path), sim.freqs, ref_impedance=sim.charac_imp)
            s21 = None
            z11 = port.uf_tot / port.if_tot
            s11 = port.uf_ref / port.uf_inc
            s11_mag = np.abs(s11)
            s11_mag = np.clip(s11_mag, 0, 0.999)  # prevent division by zero error
            vswr = (1 + s11_mag) / (1 - s11_mag)
            input_power = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
            return SimData(
                sim.freqs,
                s11,
                s21,
                z11,
                vswr,
                input_power,
                port.uf_tot,
                port.if_tot,
                sim.charac_imp,
            )

    @staticmethod
    def plot_s_param(
        freqs: NDArray,
        s11: NDArray,
        s21: NDArray | None = None,
        x_label: str = "Frequency",
        y_label: str = "S-parameter (dB)",
        title: str = "S-parameters vs Frequency",
        label_s11: str | None = None,
        label_s21: str | None = None,
    ) -> None:
        """
        Plot S-parameters (S11 and optionally S21) against frequency.

        Draws onto the current matplotlib axes (call ``plt.figure()``
        beforehand to start a new figure). Both ``s11`` and ``s21`` are
        expected as complex/linear values; they are converted to dB
        (``20*log10(|.|)``) internally before plotting.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        s11 : NDArray
            A 1D array of complex (linear) S11 values.
        s21 : NDArray, optional
            A 1D array of complex (linear) S21 values for two-port
            networks. If ``None`` (default), only S11 is plotted.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"S-parameter (dB)"``.
        title : str, optional
            Title of the plot. Default is ``"S-parameters vs Frequency"``.
        label_s11 : str, optional
            Legend label for the S11 trace. Defaults to ``"S11"``.
        label_s21 : str, optional
            Legend label for the S21 trace. Defaults to ``"S21"``.

        Returns
        -------
        None
            Draws the plot on the current axes; does not return a value.
        """
        if s21 is not None:
            if label_s21 is None:
                label_s21 = "S21"

            s21 = 20 * np.log10(np.abs(s21))
            s21_lines = plt.plot(freqs, s21, label=label_s21)
            cursor_s21 = mplcursors.cursor(s21_lines, multiple=True)

            cursor_s21.connect(
                "add",
                lambda sel: sel.annotation.set_text(
                    f"Freq={freq_formatter(sel.target[0])}\nS21={sel.target[1]:.2f} dB"
                ),
            )
        if label_s11 is None:
            label_s11 = "S11"

        s11 = 20.0 * np.log10(np.abs(s11))

        s11_lines = plt.plot(freqs, s11, label=label_s11)

        cursor_s11 = mplcursors.cursor(s11_lines, multiple=True)

        cursor_s11.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"Freq={freq_formatter(sel.target[0])}\nS11={sel.target[1]:.2f} dB"
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
        label: str = "VSWR",
        title: str = "VSWR vs Frequency",
    ) -> None:
        """
        Plot the voltage standing wave ratio (VSWR) as a function of frequency.

        Opens a new figure and plots VSWR over the given frequency range, to
        evaluate impedance matching performance.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        vswr : NDArray
            A 1D array of VSWR values corresponding to each frequency.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"VSWR"``.
        label : str, optional
            Label used for the plot legend. Default is ``"VSWR"``.
        title : str, optional
            Title of the plot. Default is ``"VSWR vs Frequency"``.

        Returns
        -------
        None
            Displays the VSWR plot in a new figure; does not return a value.
        """
        plt.figure()
        lines = plt.plot(freqs, vswr, label=label)
        cursor = mplcursors.cursor(lines, multiple=True)

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
        x_label: str = "Frequency",
        y_label: str = "Phase (deg)",
        title: str = "Phase vs Frequency",
    ) -> None:
        """
        Plot the transmission phase (angle of S21) as a function of frequency.

        Opens a new figure and plots the phase of S21, in degrees, over the
        given frequency range.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        s21 : NDArray
            A 1D array of complex S21 values corresponding to each frequency.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"Phase (deg)"``.
        title : str, optional
            Title of the plot. Default is ``"Phase vs Frequency"``.

        Returns
        -------
        None
            Displays the phase plot in a new figure; does not return a value.
        """
        phase = np.angle(s21, deg=True)
        plt.figure()
        lines_phase = plt.plot(freqs, phase, label="Phase (deg)")
        cursor_phase = mplcursors.cursor(lines_phase, multiple=True)

        cursor_phase.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"Freq={freq_formatter(sel.target[0])}\nPhase={sel.target[1]:.2f} Deg"
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
    ) -> None:
        """
        Compute and plot the group delay as a function of frequency.

        The group delay is computed from the frequency derivative of the
        unwrapped phase of S21 and displayed in seconds (via an engineering
        time-unit axis formatter).

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        s21 : NDArray
            A 1D array of complex S21 values corresponding to each frequency.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"Group Delay"``.
        title : str, optional
            Title of the plot. Default is ``"Group Delay vs Frequency"``.

        Returns
        -------
        None
            Displays the group delay plot in a new figure; does not return
            a value.
        """
        delta_phi_df = np.gradient(np.unwrap(np.angle(s21)), freqs)
        group_delay = -delta_phi_df / 2 * np.pi
        plt.figure()
        lines_group = plt.plot(freqs, group_delay, label="group_delay")
        cursor_group = mplcursors.cursor(lines_group, multiple=True)

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
    ) -> None:
        """
        Plot the input impedance (Z11) as a function of frequency.

        Opens a new figure and plots the real and imaginary parts of Z11,
        in ohms, as separate traces over the given frequency range.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.
        z11 : NDArray
            A 1D array of complex Z11 values (in ohms) corresponding to
            each frequency.
        x_label : str, optional
            Label for the x-axis. Default is ``"Frequency"``.
        y_label : str, optional
            Label for the y-axis. Default is ``"Z11"``.
        title : str, optional
            Title of the plot. Default is ``"Z11 vs Frequency"``.

        Returns
        -------
        None
            Displays the impedance plot in a new figure; does not return
            a value.
        """
        z11_real = np.real(z11)
        z11_imag = np.imag(z11)
        plt.figure()
        lines_real = plt.plot(freqs, z11_real, label="Real Z11")
        cursor_real = mplcursors.cursor(lines_real, multiple=True)

        cursor_real.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"Freq={freq_formatter(sel.target[0])}\nZ11={sel.target[1]:.2f} Ω"
            ),
        )

        lines_imag = plt.plot(freqs, z11_imag, label="Imag Z11")
        cursor_imag = mplcursors.cursor(lines_imag, multiple=True)

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
        label: str = "",
        charac_imp: float = 50.0,
    ) -> None:
        """
        Plot S11 data on a Smith chart.

        Opens a new figure with a Smith-chart projection and plots the
        complex reflection coefficient (S11) over the given frequency
        range, marking the value at the midpoint frequency.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) corresponding to the S11 data.
        s11 : NDArray
            A 1D array of complex S11 (reflection coefficient) values.
        label : str, optional
            Label used for the plot legend. Default is an empty string.
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
        )
        s11_lines = plt.plot(s11, datatype=S_PARAMETER, marker="", label=label)
        tar_idx = len(freqs) // 2
        plt.plot(
            s11[tar_idx],
            color="black",
            datatype=S_PARAMETER,
            label=(f"S11 is {s11[tar_idx]:.2f} at {freq_formatter(freqs[tar_idx])}"),
        )

        cursor = mplcursors.cursor(s11_lines, multiple=True)
        cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"Freq={freq_formatter(freqs[int(round(sel.index))])}\n"
                f"S11={s11[int(round(sel.index))]:.2f}"
            ),
        )

        plt.title("Smith Chart - S11")
        plt.legend(loc="lower right", bbox_to_anchor=(1.5, 1.0))

    @staticmethod
    def plot_2d_rad_pattern(
        nf2ff: nf2ff,
        freq: float,
        output_path: Path | None = None,
        read_cached: bool = False,
    ) -> None:
        """
        Plot the 2D E-field radiation pattern at a specified frequency.

        Computes far-field E-field cuts in the xz-plane (phi=0) and
        xy-plane (theta=90) from NF2FF data and plots both as normalized
        polar patterns (in dB) side by side.

        Parameters
        ----------
        nf2ff : object
            The near-field-to-far-field (NF2FF) object containing the
            simulation data.
        freq : float
            Frequency (in Hz) at which the radiation pattern is evaluated.
            Must be a scalar.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to the current working directory.
        read_cached : bool, optional
            If True, read cached NF2FF results instead of re-computing.
            Default is False.

        Returns
        -------
        None
            Generates the 2D radiation pattern plot; does not return a
            value.

        Raises
        ------
        TypeError
            If ``freq`` is not a scalar (e.g. an array of frequencies).
        """
        if output_path is None:
            output_path = Path.cwd()

        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency, not an array, "
                "for radiation pattern calculation"
            )

        theta = np.arange(-180.0, 181.0, 2.0)
        console.print("Calculating 2D Radiation Pattern.........", style="info")
        nf2ff_res_phi0 = nf2ff.CalcNF2FF(
            output_path,
            freq,
            theta,
            0,
            read_cached=read_cached,
            outfile="nf2ff_xz.h5",
            verbose=0,
        )

        plt.figure()
        ax = plt.subplot(121, polar=True)

        efield = np.squeeze(nf2ff_res_phi0.E_norm)
        efield_norm = efield / np.max(efield)
        efield_norm_dB = 20 * np.log10(efield_norm)

        lines = ax.plot(
            np.deg2rad(theta),
            efield_norm_dB,
            linewidth=2,
            label="xz-plane",
        )

        cursor = mplcursors.cursor(lines, multiple=True)
        cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"theta={np.rad2deg(sel.target[0]):.2f}°"
                f"\ne_field = {sel.target[1]:.2f} dB"
            ),
        )

        ax.grid(True)
        ax.set_xlabel("theta (deg)")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend()

        phi = theta
        nf2ff_res_theta90 = nf2ff.CalcNF2FF(
            output_path,
            freq,
            90,
            phi,
            read_cached=read_cached,
            outfile="nf2ff_xy.h5",
        )

        ax = plt.subplot(122, polar=True)

        efield = np.squeeze(nf2ff_res_theta90.E_norm)
        efield_norm = efield / np.max(efield)
        efield_norm_dB = 20 * np.log10(efield_norm)

        lines = ax.plot(
            np.deg2rad(phi),
            efield_norm_dB,
            linewidth=2,
            label="xy-plane",
        )

        cursor = mplcursors.cursor(lines, multiple=True)
        cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"phi={np.rad2deg(sel.target[0]):.2f}°\n efield= {sel.target[1]:.2f} dB"
            ),
        )

        ax.grid(True)
        ax.set_xlabel("phi (deg)")
        plt.suptitle(
            f"Radiation Pattern at: {freq_formatter(freq)}",
            fontsize=14,
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend()

    @staticmethod
    def plot_2d_directivity(
        nf2ff: nf2ff,
        freq: float,
        output_path: Path | None = None,
        read_cached: bool = False,
    ) -> None:
        """
        Plot the 2D directivity pattern (xz-plane, phi=0) at a specified
        frequency, annotated with the half-power beamwidth (HPBW).

        Computes directivity (in dBi) from NF2FF E-field data, plots it as
        a polar pattern, and marks the -3 dB beamwidth, main-lobe direction,
        and peak magnitude around the pattern's maximum.

        Parameters
        ----------
        nf2ff : object
            The near-field-to-far-field (NF2FF) object containing the
            simulation data.
        freq : float
            Frequency (in Hz) at which the directivity is evaluated. Must
            be a scalar.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to the current working directory.
        read_cached : bool, optional
            If True, read cached NF2FF results instead of re-computing.
            Default is False.

        Returns
        -------
        None
            Plots the 2D directivity pattern; does not return a value.

        Raises
        ------
        TypeError
            If ``freq`` is not a scalar (e.g. an array of frequencies).
        ValueError
            If the -3 dB HPBW crossings cannot be found on either side of
            the peak.
        """
        if output_path is None:
            output_path = Path.cwd()

        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency, not an array, "
                "for directivity calculation"
            )

        theta = np.arange(-180.0, 181.0, 0.1)
        console.print("Calculating 2D Directivity.........", style="info")
        nf2ff_res_phi0 = nf2ff.CalcNF2FF(
            output_path,
            freq,
            theta,
            0,
            read_cached=read_cached,
            outfile="nf2ff_xz.h5",
            verbose=0,
        )

        plt.figure()
        ax = plt.subplot(111, polar=True)

        e_field = np.squeeze(nf2ff_res_phi0.E_norm)
        e_field_norm = e_field / np.max(e_field)
        e_field_norm_db = 20 * np.log10(e_field_norm)

        max_directivity_db = 10.0 * np.log10(nf2ff_res_phi0.Dmax)
        directivity_dbi = e_field_norm_db + max_directivity_db

        lines = ax.plot(
            np.deg2rad(theta),
            directivity_dbi,
            linewidth=2,
            label="xz-plane",
        )

        # ---- HPBW calculation ----
        peak_idx = np.argmax(directivity_dbi)
        peak_theta = theta[peak_idx]
        peak_val = directivity_dbi[peak_idx]

        # Half-power level (−3 dB)
        hpbw_level = peak_val - 3.0
        # Find −3 dB crossings
        left_idx = np.where(directivity_dbi[:peak_idx] <= hpbw_level)[0]
        right_idx = np.where(directivity_dbi[peak_idx:] <= hpbw_level)[0]

        if len(left_idx) == 0 or len(right_idx) == 0:
            raise ValueError("HPBW could not be determined")

        left_theta = theta[left_idx[-1]]
        right_theta = theta[peak_idx + right_idx[0]]

        hpbw = right_theta - left_theta

        r_min = np.min(directivity_dbi)
        ax.set_rlim(r_min)
        ax.plot(
            [np.deg2rad(left_theta), np.deg2rad(left_theta)],
            [r_min, directivity_dbi[left_idx[-1]]],
            "r--",
            linewidth=1,
        )
        ax.plot(
            [np.deg2rad(right_theta), np.deg2rad(right_theta)],
            [r_min, directivity_dbi[peak_idx + right_idx[0]]],
            "r--",
            linewidth=1,
        )
        ax.plot(
            [np.deg2rad(peak_theta), np.deg2rad(peak_theta)],
            [r_min, hpbw_level],
            "r--",
            linewidth=1,
        )
        # HPBW arc
        hpbw_arc = np.linspace(left_theta, right_theta, 360)
        ax.plot(
            np.deg2rad(hpbw_arc),
            hpbw_level * np.ones_like(hpbw_arc),
            "r",
            linewidth=2,
        )
        main_lobe_mag = np.round(np.max(directivity_dbi), 2)
        ax.text(
            1.0,
            0.1,
            f"HPBW (3dB) = {hpbw:.2f}°"
            f"\nMain Lobe Direction = {peak_theta:.2f}°"
            f"\nMain Lobe Magnitude = {main_lobe_mag:.2f} dBi",
            transform=ax.transAxes,
            fontsize=12,
            color="black",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white"),
        )
        plt.suptitle(
            f" Directivity at {freq_formatter(freq)}",
            fontsize=14,
        )

        cursor = mplcursors.cursor(lines, multiple=True)
        cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"theta={np.rad2deg(sel.target[0]):.2f}°"
                f"\nDirectivity = {sel.target[1]:.2f} dBi"
            ),
        )

        ax.grid(True)
        ax.set_xlabel("theta (deg)")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend()

    @staticmethod
    def compute_nf2ff_3d(
        nf2ff: nf2ff,
        freq: float,
        output_path: Path | None = None,
        read_cached: bool = False,
    ) -> nf2ff_results:
        """
        Compute the 3D far-field radiation pattern over a full theta/phi
        sweep (theta in [0, 180] deg, phi in [0, 360] deg, 2 deg steps).

        The resulting far-field data can be used for visualization,
        directivity/gain/power analysis, and post-processing of antenna
        performance via ``plot_3d_directivity``, ``plot_3d_gain``, and
        ``plot_3d_power``.

        Parameters
        ----------
        nf2ff : object
            The near-field-to-far-field (NF2FF) object containing the
            simulation data.
        freq : float
            Frequency (in Hz) at which the 3D far-field pattern is
            evaluated. Must be a scalar.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to the current working directory.
        read_cached : bool, optional
            If True, read cached NF2FF results instead of re-computing.
            Default is False.

        Returns
        -------
        nf2ff_results
            The NF2FF result object for the full theta/phi sweep, holding
            (among other fields) ``E_norm``, ``Dmax``, ``Prad``, ``theta``,
            and ``phi``.

        Raises
        ------
        TypeError
            If ``freq`` is not a scalar (e.g. an array of frequencies).
        """
        if output_path is None:
            output_path = Path.cwd()

        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency, not an array, "
                "for 3D far-field calculation"
            )

        theta = np.arange(0, 181, 2)  # elevation
        phi = np.arange(0, 361, 2)  # azimuth

        console.print("Calculating 3D Radiation Pattern.........", style="info")
        nf2ff_3d_result = nf2ff.CalcNF2FF(
            output_path,
            freq,
            theta,
            phi,
            read_cached=read_cached,
            outfile="nf2ff_3d.h5",
            verbose=0,
        )
        return nf2ff_3d_result

    @staticmethod
    def plot_3d_directivity(
        nf2ff_3d_result: nf2ff_results,
        freq: float,
        output_path: Path | None = None,
    ) -> None:
        """
        Plot the 3D directivity pattern of an antenna at a specified frequency.

        Visualizes the directivity pattern (in dBi) of the antenna in 3D
        space, based on the far-field results from ``compute_nf2ff_3d``,
        and saves the mesh as a VTK file under ``output_path / "3D_plots"``.

        Parameters
        ----------
        nf2ff_3d_result : nf2ff_results
            The 3D far-field result object returned by ``compute_nf2ff_3d``.
        freq : float
            The frequency (in Hz) at which the 3D directivity pattern is
            evaluated. Must be a scalar; used only for plot/file labeling.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to the current working directory.

        Returns
        -------
        None
            Opens an interactive 3D plot window and saves the mesh to
            ``3D_plots/3D_directivity.vtk``; does not return a value.

        Raises
        ------
        TypeError
            If ``freq`` is not a scalar (e.g. an array of frequencies).
        """
        if output_path is None:
            output_path = Path.cwd()

        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency, not an array, "
                "for directivity calculation"
            )

        plots_3d_path = output_path / "3D_plots"
        plots_3d_path.mkdir(parents=True, exist_ok=True)

        e_field = np.squeeze(nf2ff_3d_result.E_norm)
        e_field /= np.max(e_field)  # normalize

        max_directivity = nf2ff_3d_result.Dmax[0]
        D = max_directivity * e_field

        directivity_dbi = 10 * np.log10(D)

        theta = nf2ff_3d_result.theta
        phi = nf2ff_3d_result.phi

        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        R = 10 ** (directivity_dbi / 10)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        mesh = pv.StructuredGrid(X, Y, Z)
        mesh["Directivity (dBi)"] = directivity_dbi.ravel(order="F")
        mesh.save(plots_3d_path / "3D_directivity.vtk")

        plotter = BackgroundPlotter(
            title=f"Antenna 3D Pattern - Directivity (dBi) at {freq_formatter(freq)} "
        )
        plotter.add_mesh(
            mesh,
            scalars="Directivity (dBi)",
            cmap="turbo",
            clim=[np.min(directivity_dbi), np.max(directivity_dbi)],
            smooth_shading=True,
        )
        plotter.add_text(
            "Antenna 3D Pattern - Directivity (dBi) ", position="upper_edge"
        )

    @staticmethod
    def plot_3d_gain(
        nf2ff_3d_result: nf2ff_results,
        freq: float,
        input_power: float,
        output_path: Path | None = None,
    ) -> None:
        """
        Plot the 3D gain pattern of an antenna at a specified frequency.

        Visualizes the realized gain pattern (in dBi) of the antenna in 3D
        space: gain is directivity (from ``compute_nf2ff_3d``) scaled by
        the radiation efficiency ``Prad / input_power``. Saves the mesh
        under ``output_path / "3D_plots"``.

        Parameters
        ----------
        nf2ff_3d_result : nf2ff_results
            The 3D far-field result object returned by ``compute_nf2ff_3d``.
        freq : float
            The frequency (in Hz) the pattern was computed at. Used only
            for plot/file labeling.
        input_power : float
            Input power at the port (same units as ``Prad``), used to
            normalize directivity into realized gain.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to the current working directory.

        Returns
        -------
        None
            Opens an interactive 3D plot window and saves the mesh to
            ``3D_plots/3D_Gain.vtk``; does not return a value.
        """
        if output_path is None:
            output_path = Path.cwd()

        e_field = np.squeeze(nf2ff_3d_result.E_norm)
        e_field /= np.max(e_field)  # normalize

        plots_3d_path = output_path / "3D_plots"
        plots_3d_path.mkdir(parents=True, exist_ok=True)

        max_directivity = nf2ff_3d_result.Dmax[0]
        directivity = max_directivity * e_field

        efficiency = nf2ff_3d_result.Prad[0] / (np.max(input_power))
        gain = efficiency * directivity
        gain_dbi = 10 * np.log10(gain)

        theta = nf2ff_3d_result.theta
        phi = nf2ff_3d_result.phi

        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        R = 10 ** (gain_dbi / 10)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        mesh = pv.StructuredGrid(X, Y, Z)
        mesh["Gain (dBi)"] = gain_dbi.ravel(order="F")
        mesh.save(plots_3d_path / "3D_Gain.vtk")
        plotter = BackgroundPlotter(
            title=f"Antenna 3D Pattern - Gain (dBi) at {freq_formatter(freq)} "
        )
        plotter.add_mesh(
            mesh,
            scalars="Gain (dBi)",
            cmap="turbo",
            clim=[np.min(gain_dbi), np.max(gain_dbi)],
            smooth_shading=True,
        )
        plotter.add_text("Antenna 3D Pattern - Gain (dBi) ", position="upper_edge")

    @staticmethod
    def plot_3d_power(
        nf2ff_3d_result: nf2ff_results,
        freq: float,
        output_path: Path | None = None,
    ) -> None:
        """
        Plot the 3D radiated power pattern of an antenna at a specified
        frequency.

        Visualizes the normalized radiated power pattern (in dB) of the
        antenna in 3D space, based on the ``P_rad`` field of the far-field
        results from ``compute_nf2ff_3d``. Saves the mesh under
        ``output_path / "3D_plots"``.

        Parameters
        ----------
        nf2ff_3d_result : nf2ff_results
            The 3D far-field result object returned by ``compute_nf2ff_3d``.
        freq : float
            The frequency (in Hz) the pattern was computed at. Used only
            for plot/file labeling.
        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to the current working directory.

        Returns
        -------
        None
            Opens an interactive 3D plot window and saves the mesh to
            ``3D_plots/3D_Power.vtk``; does not return a value.
        """
        if output_path is None:
            output_path = Path.cwd()

        plots_3d_path = output_path / "3D_plots"
        plots_3d_path.mkdir(parents=True, exist_ok=True)

        power = np.squeeze(nf2ff_3d_result.P_rad)

        power = power / np.max(power)  # normalize

        power_db = 10 * np.log10(power)

        theta = nf2ff_3d_result.theta
        phi = nf2ff_3d_result.phi

        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        R = 10 ** (power_db / 10)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        mesh = pv.StructuredGrid(X, Y, Z)
        mesh["Power (dB)"] = power_db.ravel(order="F")

        mesh.save(plots_3d_path / "3D_Power.vtk")
        plotter = BackgroundPlotter(
            title=f"Antenna 3D Pattern - Power (dB) at {freq_formatter(freq)} "
        )
        plotter.add_mesh(
            mesh,
            scalars="Power (dB)",
            cmap="turbo",
            clim=[np.min(power_db), np.max(power_db)],
            smooth_shading=True,
        )
        plotter.add_text("Antenna 3D Pattern - Power (dB)", position="upper_edge")

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
            to the current working directory.
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
            output_path = Path.cwd()

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
        dump_type: DumpType = DumpType.efield_time,
    ) -> None:
        """
        Add a field dump box to the simulation setup.

        Configures and attaches a field/current dump of the requested
        ``dump_type`` to the CSXCAD structure. The dump box spans each
        transverse (x, y) dimension found from the bounding box of all
        existing geometry primitives, padded by ``max(lambda0/2, 15%
        of span)`` (or by half of ``params.simulation_box`` if no geometry
        is found in that dimension yet), and spans z from 0 to the top of
        the copper layer (``substrate_thickness_mm + copper_thickness_mm``).

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        params : SimParams
            Parameter object supplying ``lambda0``, ``simulation_box``,
            ``substrate_thickness_mm``, and ``copper_thickness_mm``, used
            to size the dump box.
        output_path : Path, optional
            Directory the dump data will be written under (a ``field_dump``
            subdirectory is created). Defaults to ``cwd / "Sim_Path"``.
        dump_type : DumpType, optional
            Type of field/current dump to add (time-domain or
            frequency-domain E-field, H-field, current, current density, or
            SAR). Default is ``DumpType.efield_time``.

        Returns
        -------
        None
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        def _set_sim_bounds_from_geometry(
            params: SimParams, dim_bounds: list[list[float]]
        ) -> NDArray:
            new_sim_box = []
            for dim in range(3):
                if not dim_bounds[dim]:
                    half = params.simulation_box[dim] / 2.0
                    new_sim_box.append((-half, half))
                    continue
                geo_min = dim_bounds[dim][0]
                geo_max = dim_bounds[dim][-1]
                span = geo_max - geo_min
                padding = max(params.lambda0 / 2, span * 0.15)
                new_sim_box.append((geo_min - padding, geo_max + padding))
            return np.array(new_sim_box)

        bounds = [[], [], []]
        for prim in sim.CSX.GetAllPrimitives():
            try:
                bb = prim.GetBoundBox()
                tr = prim.GetTransform()
                p0 = np.array(tr.Transform(bb[0]))
                p1 = np.array(tr.Transform(bb[1]))
                for dim in range(3):
                    bounds[dim].append(min(p0[dim], p1[dim]))
                    bounds[dim].append(max(p0[dim], p1[dim]))
            except Exception:
                pass

        dim_bounds = [sorted(set(b)) for b in bounds]
        sim_box = _set_sim_bounds_from_geometry(params, dim_bounds)

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
        start = [sim_box[0][0], sim_box[1][0], 0]
        stop = [
            sim_box[0][1],
            sim_box[1][1],
            params.substrate_thickness_mm + params.copper_thickness_mm,
        ]
        dump.AddBox(start=start, stop=stop)

    @staticmethod
    def export_touchstone(
        freqs: NDArray,
        s11: NDArray,
        *,
        s21: NDArray | None = None,
        charac_imp: float = 50.0,
        output_path: Path | None = None,
        filename: str = "s_param",
    ) -> None:
        """
        Export S-parameters to a Touchstone file (``.s1p`` for single-port,
        ``.s2p`` for two-port).

        Writes the provided S-parameter data to ``output_path / "touchstone"``
        via :mod:`skrf`.

        Parameters
        ----------
        freqs : NDArray
            Frequency points (in Hz) to export.
        s11 : NDArray
            Complex S11 values across the frequency range.
        s21 : NDArray, optional
            Complex S21 values across the frequency range. If provided, a
            two-port (``.s2p``) network is written; otherwise a single-port
            (``.s1p``) network is written. Default is ``None``.
        charac_imp : float, optional
            Reference impedance (in ohms) recorded in the Touchstone file.
            Default is ``50.0``.
        output_path : Path, optional
            Directory to save the ``touchstone`` subdirectory under.
            Defaults to the current working directory.
        filename : str, optional
            Base name of the exported file (extension is added
            automatically). Default is ``"s_param"``.

        Returns
        -------
        None
            Does not return a value.

        Notes
        -----
        In the two-port case only S11 and S21 are populated from the given
        data; S12 and S22 are written as zero (the reverse-direction
        parameters are not measured/simulated).
        """
        console.print("-------------------------------------------", style="info")
        console.print("Exporting S-Parameters to Touchstone file", style="info")
        console.print("-------------------------------------------", style="info")

        if output_path is None:
            output_path = Path.cwd()

        touchstone_path = output_path / "touchstone"
        touchstone_path.mkdir(parents=True, exist_ok=True)

        if s21 is None:  # 1 port structure
            ntwk = Network(frequency=freqs, s=s11, z0=charac_imp)
            ntwk.write_touchstone(filename=filename, dir=touchstone_path)

        elif s21 is not None:  # 2port structure
            s_params = np.zeros((len(s11), 2, 2), dtype=complex)
            s_params[:, 0, 0] = s11
            s_params[:, 1, 0] = s21
            # s_params[:, 0, 1] = s21
            # s_params[:, 1, 1] = s11
            ntwk = Network(frequency=freqs, s=s_params, z0=charac_imp)
            ntwk.write_touchstone(filename=filename, dir=touchstone_path)

    @staticmethod
    def export_gerber(
        sim: SimSetup,
        output_path: Path | None = None,
        options: dict[str, list] | None = None,
    ) -> None:
        """
        Export CSXCAD geometry to Gerber format.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        output_path : Path, optional
            Directory to save the ``gerber`` subdirectory under. Defaults
            to the current working directory.
        options : dict[str, list], optional
            Dictionary of export options. Defaults to
            ``{"ignore": ["ground"]}``.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The gerber export is currently limited and might not be able
        to export all geometries.  Layers should be exported separately
        by specifying ignore options (e.g. to export only the top metal
        layer, ignore the ground).

        """
        if options is None:
            options = {"ignore": ["ground"]}

        if output_path is None:
            output_path = Path.cwd()

        gerber_path = output_path / "gerber"
        gerber_path.mkdir(parents=True, exist_ok=True)

        export_gerber(
            CSX=sim.CSX,
            output_path=gerber_path,
            options=options,
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
            the current working directory.

        Returns
        -------
        None
        """
        if output_path is None:
            output_path = Path.cwd()

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
            the current working directory.

        Returns
        -------
        None
        """

        if output_path is None:
            output_path = Path.cwd()

        step_path = output_path / "step"
        step_path.mkdir(parents=True, exist_ok=True)

        export_csxcad_xml_to_step(structure_xml_path, output_path)

    @staticmethod
    def show_plots() -> None:
        """
        Display all generated plots. Blocks until the plot window(s) are
        closed.

        If no matplotlib figures are open and a Qt application instance is
        already running (e.g. a PyVista/FEM mesh viewer opened by
        ``write_and_show_structure``), blocks on that Qt event loop instead
        and exits the process once it closes. Otherwise calls
        ``matplotlib.pyplot.show()`` to display any open matplotlib figures.

        Returns
        -------
        None
            Does not return a value.
        """
        if not plt.get_fignums():
            view_3d = QCoreApplication.instance()
            if view_3d is not None:
                sys.exit(view_3d.exec())
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
            to the current working directory.

        Returns
        -------
        None
            Does not return a value.
        """

        if output_path is None:
            output_path = Path.cwd()

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

    @staticmethod
    def run_all_post_processing(
        sim: SimSetup,
        freqs: NDArray,
        s11: NDArray,
        vswr: NDArray,
        z11: NDArray,
        input_power: float,
        nf2ff: nf2ff,
        nf2ff_3d_result: nf2ff_results,
        params: SimParams,
        s21: NDArray | None = None,
        output_path: Path | None = None,
    ) -> None:
        """
        Run the full post-processing pipeline for a completed simulation.

        Convenience method that plots S-parameters, Smith chart, VSWR, and
        impedance over the full frequency range; plots 2D/3D radiation
        pattern, directivity, gain, and power at ``params.main_freq``;
        saves and displays all plots; and exports STL, Touchstone, and
        Gerber (ground layer ignored) files.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple returned by ``setup_simulation``.
        freqs : NDArray
            Array of frequencies (in Hz) used in the simulation.
        s11 : NDArray
            Complex S11 parameter array across the frequency range.
        vswr : NDArray
            VSWR values across the frequency range.
        z11 : NDArray
            Complex input impedance (Z11) values across the frequency range.
        input_power : float
            Input power at the driven port, used to normalize the 3D gain
            plot.
        nf2ff : object
            Near-field-to-far-field object, e.g. from ``create_nf2ff``.
        nf2ff_3d_result : nf2ff_results
            3D far-field result object from ``compute_nf2ff_3d``.
        params : SimParams
            Parameter object; ``params.main_freq`` selects the frequency
            used for the radiation-pattern, directivity, gain, and power
            plots, and ``params.charac_imp`` is used for the Touchstone
            export.
        s21 : NDArray, optional
            Complex S21 parameter array across the frequency range.
            Default is ``None`` for single-port structures.
        output_path : Path, optional
            Directory where simulation results and exports will be saved.
            Defaults to the current working directory.

        Returns
        -------
        None

        Notes
        -----
        ``show_plots`` blocks until the plot windows are closed, so the
        STL/Touchstone/Gerber exports run only after the plots are closed.
        """
        if output_path is None:
            output_path = Path.cwd()

        target_freq = params.main_freq

        SimTools.plot_s_param(freqs, s11, s21)
        SimTools.plot_smith_chart(freqs, s11)
        SimTools.plot_vswr(freqs, vswr)
        SimTools.plot_impedance(freqs, z11)
        SimTools.plot_2d_directivity(nf2ff, target_freq, output_path)
        SimTools.plot_2d_rad_pattern(nf2ff, target_freq, output_path)
        SimTools.plot_3d_directivity(nf2ff_3d_result, target_freq, output_path)
        SimTools.plot_3d_gain(
            nf2ff_3d_result,
            target_freq,
            input_power,
            output_path,
        )
        SimTools.plot_3d_power(nf2ff_3d_result, target_freq, output_path)
        SimTools.save_plots(output_path)
        SimTools.show_plots()
        SimTools.export_stl(output_path)
        SimTools.export_touchstone(
            freqs,
            s11,
            output_path=output_path,
            charac_imp=params.charac_imp,
            s21=s21,
        )
        SimTools.export_gerber(sim, output_path, options={"ignore": ["ground"]})


def optimize_s11(
    freqs: NDArray,
    s11: NDArray,
    target_freq: float | None = None,
    freq_band: tuple[float, float] | None = None,
    mode: str = "worst",
    threshold: float = -15,
) -> np.floating:
    """
    Objective function for S11 (reflection) optimization. Lower is better.

    Intended as a scalar cost function for ``optimize_s_params``: converts
    S11 to dB and reduces it to a single number, either at one frequency
    or over a band. If ``target_freq`` is given it takes precedence over
    ``freq_band`` (and ``mode``/``threshold`` are ignored).

    Parameters
    ----------
    freqs : NDArray
        Array of frequencies in Hz, matching ``s11``.
    s11 : NDArray
        Complex S11 parameter array across the frequency range.
    target_freq : float, optional
        If given, evaluate S11 (in dB) at the frequency in ``freqs``
        closest to this value, ignoring ``freq_band``/``mode``/
        ``threshold``.
    freq_band : tuple of (float, float), optional
        Frequency band ``(f_min, f_max)`` in Hz over which to evaluate S11
        when ``target_freq`` is not given.
    mode : str, optional
        Evaluation mode within the frequency band:

        - ``"worst"`` : the maximum S11 in dB (worst-case reflection).
        - ``"mean"`` : the mean S11 in dB across the band.
        - ``"threshold"`` : sum, over the band, of the amount (in dB) each
          point exceeds ``threshold`` (0 if none exceed it).

        Default is ``"worst"``.
    threshold : float, optional
        S11 threshold in dB, used only when ``mode="threshold"``. Default
        is ``-15``.

    Returns
    -------
    float
        Scalar cost value (S11 in dB); lower means better matched.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``"mean"``, ``"worst"``, or
        ``"threshold"``, or if neither ``target_freq`` nor ``freq_band``
        is provided.
    """

    s11_db = 20.0 * np.log10(np.abs(s11))

    if target_freq is not None:
        idx = (np.abs(freqs - target_freq)).argmin()
        return s11_db[idx]

    if freq_band is not None:
        f_min, f_max = freq_band
        mask = (freqs >= f_min) & (freqs <= f_max)

        if mode == "worst":
            return np.max(s11_db[mask])  # -3, -13, -23 return -3

        elif mode == "mean":
            return np.mean(s11_db[mask])  # -3, -13, -23 return average

        elif mode == "threshold":
            penalty = np.maximum(s11_db[mask] - threshold, 0)
            return np.sum(penalty)

        else:
            raise ValueError("mode must be 'mean', 'worst', 'threshold'")

    raise ValueError("Provide target_freq or freq_band")


def optimize_s21(
    freqs: NDArray,
    s21: NDArray,
    target_freq: float | None = None,
    freq_band: tuple[float, float] | None = None,
    mode: str = "worst",
) -> np.floating:
    """
    Objective function for S21 (transmission) optimization. Higher S21 is
    better.

    Computes a scalar cost from S21 data for use with ``optimize_s_params``.
    Since ``scipy.optimize.minimize`` minimizes its objective, the cost is
    the *negated* S21 in dB, so maximizing S21 corresponds to minimizing
    the cost. If ``target_freq`` is given it takes precedence over
    ``freq_band`` (and ``mode`` is ignored).

    Parameters
    ----------
    freqs : NDArray
        Array of frequencies in Hz, matching ``s21``.
    s21 : NDArray
        Complex S21 parameter array across the frequency range.
    target_freq : float, optional
        If given, evaluate S21 (in dB) at the frequency in ``freqs``
        closest to this value, ignoring ``freq_band``/``mode``.
    freq_band : tuple of (float, float), optional
        Frequency band ``(f_min, f_max)`` in Hz over which to evaluate S21
        when ``target_freq`` is not given.
    mode : str, optional
        Evaluation mode within the frequency band:

        - ``"mean"`` : negative mean of S21 in dB across the band.
        - ``"worst"`` : negative minimum of S21 in dB across the band
          (worst-case transmission).

        Default is ``"worst"``.

    Returns
    -------
    float
        Scalar cost value (negated S21 in dB); lower means higher
        transmission.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"mean"`` or ``"worst"``, or if neither
        ``target_freq`` nor ``freq_band`` is provided.
    """
    s21_db = 20.0 * np.log10(np.abs(s21))

    if target_freq is not None:
        idx = (np.abs(freqs - target_freq)).argmin()
        return -s21_db[idx]

    if freq_band is not None:
        f_min, f_max = freq_band
        mask = (freqs >= f_min) & (freqs <= f_max)

        if mode == "mean":
            return -np.mean(s21_db[mask])

        elif mode == "worst":
            return -np.min(s21_db[mask])

        else:
            raise ValueError("mode must be 'mean' or 'worst'")

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
        minimize, e.g. via ``optimize_s11`` or ``optimize_s21``.
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
) -> list[SimData]:
    """
    Run a simulation function over a Cartesian product of parameter values
    and plot S11 for each combination.

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
        order). Must return a :class:`SimData` (or compatible object with
        ``freqs``/``s11`` attributes).
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
    list[SimData]
        The simulation data from every parameter combination in the sweep,
        in the same order as the Cartesian product of ``sweep_vals``. Empty
        if ``sweep_vals`` has no combinations to run.
    """
    console.print("-------------------------------------", style="info")
    console.print("Running Parameter Sweep", style="info")
    console.print("-------------------------------------", style="info")

    sweep_values = {key: np.linspace(*val).tolist() for key, val in sweep_vals.items()}
    cartesian_sweep = list(product(*sweep_values.values()))
    sim_data_list = []

    for values in cartesian_sweep:
        kv_pairs = list(zip(sweep_values.keys(), values, strict=True))
        key_value = "_".join(f"{k}_{v}" for k, v in kv_pairs)
        label = ", ".join(f"{k}={v}" for k, v in kv_pairs)
        sweep_path = output_path / "sweep" / key_value
        sweep_path.mkdir(parents=True, exist_ok=True)

        sim_data = simulate_fn(sweep_path, sweep, values)
        sim_data_list.append(sim_data)
        SimTools.plot_s_param(
            sim_data.freqs,
            sim_data.s11,
            label_s11=label,
        )
    SimTools.show_plots()
    return sim_data_list


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
