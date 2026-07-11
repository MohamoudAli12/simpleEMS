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
optimisation, parameter sweeps, plotting (S-parameters, VSWR,
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
from .fem_materials import FemOptions
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
    Represents field and other dump types provided by openEMS.

    Attributes
    ----------

    efield_time: tuple[int, str]
        Electric field time-domain dump
    hfield_time: tuple[int, str]
        Magnetic field time-domain dump
    current_time: tuple[int, str]
        Electric current time-domain dump
    current_density_time: tuple[int, str]
        Total current density (rot(H)) time-domain dump
    efield_frequency: tuple[int, str]
        Electric field frequency-domain dump
    hfield_frequency: tuple[int, str]
        Magnetic field frequency-domain dump
    current_frequency: tuple[int, str]
        Electric current frequency-domain dump
    current_density_frequency: tuple[int, str]
        Total current density (rot(H)) frequency-domain dump
    local_sar_frequency: tuple[int, str]
        Local SAR frequency-domain dump
    average_sar_frequency_1g: tuple[int, str]
        1g averaging SAR frequency-domain dump
    average_sar_frequency_10g: tuple[int, str]
        10g averaging SAR frequency-domain dump
    raw_data: tuple[int, str]
        raw data needed for SAR calculations
        (electric field FD, cell volume, conductivity and density)

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
    """

    freqs: NDArray
    s11: NDArray
    s21: NDArray | None
    z11: NDArray
    vswr: NDArray
    input_power: float


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
    num_fem_solve_points : int
        Number of full FEM solves the adaptive sweep may perform (FEM only).
    fem_options : FemOptions | None
        Global FEM solver/mesh options (boundary, symmetry, fe_order, mesh
        tuning, port type). ``None`` for FDTD.
    """

    CSX: ContinuousStructure
    FDTD: openEMS
    freqs: NDArray
    backend_engine: str = "FDTD"
    num_fem_solve_points: int = 10
    fem_options: FemOptions | None = None


def setup_simulation(
    params: SimParams,
    boundary_cond: list[str] | None = None,
) -> SimSetup:
    """
    Sets up the openEMS simulation.

    Parameters
    ----------

    params: SimParams
        parameter object that holds all simulation parameters.
    boundary_cond: list[str]
        boundary condition for the simulation.

    Returns
    -------
    SimSetup
        Named tuple with ``CSX`` (CSXCAD geometry), ``FDTD`` (openEMS FDTD
        object), and ``freqs`` (frequency array).
    """
    if boundary_cond is None:
        boundary_cond = [
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8",
        ]
    CSX = ContinuousStructure()
    FDTD = openEMS(NrTS=params.timestep, EndCriteria=params.end_criteria)

    FDTD.SetBoundaryCond(boundary_cond)
    FDTD.SetCSX(CSX)

    fmin, fmax = params.freq_range
    centre_freq = (fmin + fmax) / 2
    span_freq = (fmax - fmin) / 2

    FDTD.SetGaussExcite(centre_freq, span_freq)

    freqs = np.linspace(fmin, fmax, params.num_points)

    fem_options = None
    if params.backend_engine == "FEM":
        fem_options = FemOptions(
            boundary=params.fem_boundary,
            symmetry=params.fem_symmetry,
            fe_order=params.fem_fe_order,
            air_pad_frac=params.fem_air_pad_frac,
            elems_per_wavelength=params.fem_elems_per_wavelength,
            mesh_fine_scale=params.fem_mesh_fine_scale,
            min_layers=params.fem_min_layers,
            port_type=params.fem_port_type,
        )

    return SimSetup(
        CSX=CSX,
        FDTD=FDTD,
        freqs=freqs,
        backend_engine=params.backend_engine,
        num_fem_solve_points=params.num_fem_solve_points,
        fem_options=fem_options,
    )


class SimTools:
    """
    A collection of common tools and functions for simulations.
    This class is not intended to be instantiated. It provides a
    namespace for common simulation utilities and function.

    All structure classes i.e. InsetFedPatchAntenna inherit this class.

    """

    @staticmethod
    def write_and_show_structure(
        sim: SimSetup, output_path: Path | None = None
    ) -> None:
        """
        This method shows the CSXCAD structure.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple containing CSX, FDTD, and freqs.
        output_path: Path
            The output path where simulation results is stored.
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"
            output_path.mkdir(parents=True, exist_ok=True)

        if sim.backend_engine == "FEM":
            from . import fem_backend

            msh_path = fem_backend.build_mesh(
                sim.CSX, sim.freqs, output_path, fem_options=sim.fem_options
            )
            SimTools._display_mesh(msh_path)
            return

        structure_3d = output_path / "structure.xml"

        sim.FDTD.Write2XML(
            str(structure_3d)
        )  # str in this fixes an error encountered on Windows.
        subprocess.run([AppCSXCAD_BIN, str(structure_3d)], check=True)

    @staticmethod
    def _display_mesh(msh_path: str | Path) -> None:
        """
        Display a Gmsh ``.msh`` tetrahedral mesh with pyvista.

        Parameters
        ----------
        msh_path : str | Path
            Path to the ``.msh`` file to display.
        """
        try:
            vtk_path = Path(msh_path).with_suffix(".vtk")
            grid = pv.read(str(vtk_path if vtk_path.exists() else msh_path))
            plotter = pv.Plotter()
            plotter.add_mesh(grid, show_edges=True, opacity=0.5, color="lightblue")
            plotter.add_axes()
            plotter.show()
        except Exception as exc:  # display is non-critical (e.g. headless runs)
            console.print(f"[warning]Could not display mesh: {exc}[/warning]")

    @staticmethod
    def export_stl(output_path: Path | None = None) -> None:
        """
        This method exports the CSXCAD structure to STL objects.

        Parameters
        ----------
        output_path: Path
            The output path of the simulation results.
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
        This method starts and runs the openEMS simulation.

        Parameters
        ----------

        sim : SimSetup
            Simulation setup named tuple containing CSX, FDTD, and freqs.
        output_path: Path
            The output path of the simulation results.
        """
        if output_path is None:
            output_path = Path.cwd() / "Sim_Path"

        if sim.backend_engine == "FEM":
            from . import fem_backend

            fem_backend.run_sweep(
                sim.CSX,
                sim.freqs,
                sim.num_fem_solve_points,
                output_path,
                fem_options=sim.fem_options,
            )
            return

        sim.FDTD.Run(output_path)

    @staticmethod
    def create_nf2ff(sim: SimSetup) -> nf2ff:
        """
        Creates near field to far field (NF2FF) recording box.

        Parameters
        ----------

        sim : SimSetup
            Simulation setup named tuple containing CSX, FDTD, and freqs.

        Returns
        -------
        nf2ff : object
            NF2FF object. For the FEM backend this is a ``FemNF2FF`` adapter that
            exposes the same ``CalcNF2FF`` interface, so the radiation-plotting
            methods work identically for both backends.
        """
        if sim.backend_engine == "FEM":
            from .fem_radiation import FemNF2FF

            return FemNF2FF()

        nf2ff = sim.FDTD.CreateNF2FFBox()
        return nf2ff

    @staticmethod
    def compute_sim_data(
        port: LumpedPort | list[LumpedPort],
        freqs: NDArray,
        charac_imp: float,
        output_path: Path | None = None,
        backend_engine: str = "FDTD",
    ) -> SimData:
        """
        Compute several network parameters for plotting and
        post-processing of simulation results.

        This method calculates network parameters such as S11, Z11,
        VSWR, and input power based on the simulation results. It uses
        the provided port and simulation parameters to extract the
        necessary data for further analysis and plotting.

        Parameters
        ----------
        port : LumpedPort or list of LumpedPort
            The openEMS port object (or list of two ports) representing the
            simulation port(s).
        freqs: NDArray
            takes list of frequencies that was used for the simulation.
        charac_imp: float
            characteristic impedance of the input port.
        output_path : Path
            The path to the directory where the simulation results are saved.
        backend_engine : str
            Solver backend. ``"FDTD"`` (default) computes results from the
            openEMS port probes; ``"FEM"`` reads the GetDP sweep results
            written to ``output_path`` (the ``port`` argument is then ignored).

        Returns
        -------
        SimData
            A named tuple containing the simulation data:

            - freqs : NDArray
                A numpy array of frequencies used for post-processing.

            - s11 : NDArray
                Complex S11 values across the frequency range.

            - s21 : NDArray
                Complex S21 values across the frequency range.

            - z11 : NDArray
                Complex Z11 values across the frequency range.

            - vswr : NDArray
                VSWR values across the frequency range.

            - input_power : float
                The calculated input power at the port.

        """
        if backend_engine == "FEM":
            from . import fem_backend

            # match the default used by write_and_show_structure / run_simulation
            # so the FEM results are read from where they were written.
            if output_path is None:
                output_path = Path.cwd() / "Sim_Path"
            return fem_backend.compute_sim_data(freqs, charac_imp, output_path)

        if output_path is None:
            output_path = Path.cwd()

        if isinstance(port, list):
            for p in port:
                p.CalcPort(str(output_path), freqs, ref_impedance=charac_imp)
            s11 = port[0].uf_ref / port[0].uf_inc
            s21 = port[1].uf_ref / port[0].uf_inc
            z11 = port[0].uf_tot / port[0].if_tot
            s11_mag = np.abs(s11)
            s11_mag = np.clip(s11_mag, 0, 0.999)  # prevent division by zero error
            vswr = (1 + s11_mag) / (1 - s11_mag)
            input_power = 0.5 * np.real(port[0].uf_tot * np.conj(port[0].if_tot))
            return SimData(freqs, s11, s21, z11, vswr, input_power)
        else:
            port.CalcPort(str(output_path), freqs, ref_impedance=charac_imp)
            s21 = None
            z11 = port.uf_tot / port.if_tot
            s11 = port.uf_ref / port.uf_inc
            s11_mag = np.abs(s11)
            s11_mag = np.clip(s11_mag, 0, 0.999)  # prevent division by zero error
            vswr = (1 + s11_mag) / (1 - s11_mag)
            input_power = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
            return SimData(freqs, s11, s21, z11, vswr, input_power)

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

        Generate a 2D plot of the S-parameters (in dB) as a function
        of frequency.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to plot on the x-axis.

        s11 : NDArray
            A 1D array of the corresponding S11 values (in dB) to plot on the y-axis.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "S-Parameter (dB)".

        s21 : NDArray, optional
            A 1D array of complex S21 values for two-port networks.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "S-Parameter (dB)".

        title : str, optional
            Title of the plot. Default is "S-Parameters vs Frequency".

        label_s11 : str, optional
            Legend label for the S11 trace. Default is ``"S11"``.

        label_s21 : str, optional
            Legend label for the S21 trace. Default is ``"S21"``.

        Returns
        -------
        None
            This method does not return any value. It directly displays the plot.
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
        Plot the Voltage Standing Wave Ratio (VSWR) as a function of frequency.

        Generate a 2D plot of VSWR values over the specified frequency
        range, used to evaluate impedance matching performance.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to be plotted on the x-axis.

        vswr : NDArray
            A 1D array of VSWR values corresponding to each frequency.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "VSWR".

        label : str, optional
            Label used for the plot legend.

        title : str, optional
            Title of the plot. Default is "VSWR vs Frequency".

        Returns
        -------
        None
            This method does not return any value. It displays the VSWR plot.
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

        This method generates a 2D plot of the unwrapped phase of S21 over
        the specified frequency range.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to be plotted on the x-axis.

        s21 : NDArray
            A 1D array of complex S21 values corresponding to each frequency.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "Phase (deg)".

        title : str, optional
            Title of the plot. Default is "Phase vs Frequency".

        Returns
        -------
        None
            This method does not return any value. It displays the phase plot.
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
        Plot the group delay as a function of frequency.

        This method computes and plots the group delay from the derivative of
        the unwrapped phase of S21 over the specified frequency range.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to be plotted on the x-axis.

        s21 : NDArray
            A 1D array of complex S21 values corresponding to each frequency.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "Group Delay".

        title : str, optional
            Title of the plot. Default is "Group Delay vs Frequency".

        Returns
        -------
        None
            This method does not return any value. It displays the group delay plot.
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
        Plot the transmission phase (angle of S21) as a function of frequency.

        This method generates a 2D plot of the unwrapped phase of S21 over
        the specified frequency range.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) to be plotted on the x-axis.

        s21 : NDArray
            A 1D array of complex S21 values corresponding to each frequency.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "Phase (deg)".

        title : str, optional
            Title of the plot. Default is "Phase vs Frequency".

        Returns
        -------
        None
            This method does not return any value. It displays the phase plot.
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

        This method generates a Smith chart representation of the complex reflection
        coefficient (S11) over a specified frequency range.

        Parameters
        ----------
        freqs : NDArray
            A 1D array of frequencies (in Hz) corresponding to the S11 data.

        s11 : NDArray
            A 1D array of complex S11 (reflection coefficient) values.

        label : str, optional
            Label used for the plot legend. Default is an empty string.

        charac_imp : float, optional
            Characteristic impedance of the system in ohms. Default is 50 Ω.

        Returns
        -------
        None
            This method does not return any value. It displays the Smith chart.

        Notes
        -----
        - pysmithchart is used for plotting the smith chart.
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
        plt.plot(s11, datatype=S_PARAMETER, marker="", label=label)
        tar_idx = len(freqs) // 2
        plt.plot(
            s11[tar_idx],
            color="black",
            datatype=S_PARAMETER,
            label=(f"S11 is {s11[tar_idx]:.2f} at {freq_formatter(freqs[tar_idx])}"),
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
        Plot the 2D efield radiation pattern at a specified frequency.

        This method computes and visualizes a 2D radiation pattern from near-field
        to far-field (NF2FF) data for a given frequency.

        Parameters
        ----------
        nf2ff : object
            The near-field to far-field (NF2FF) object containing the simulation data.

        freq : float
            Frequency (in Hz) at which the far-field radiation pattern is evaluated.

        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to current working directory.

        read_cached : bool, optional
            If True, read cached NF2FF results instead of re-computing.
            Default is False.

        Returns
        -------
        None
            This method does not return any value. It generates the 2D
            radiation pattern plot.
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
        Plot the 2D directivity at a specified frequency.

        This method computes and visualizes a 2D directivity from near-field
        to far-field (NF2FF) data for a given frequency.

        Parameters
        ----------
        nf2ff : object
            The near-field to far-field (NF2FF) object containing the simulation data.

        freq : float
            Frequency (in Hz) at which the far-field radiation pattern is evaluated.

        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to current working directory.

        read_cached : bool, optional
            If True, read cached NF2FF results instead of re-computing.
            Default is False.

        Returns
        -------
        None
            This method does not return any value. It plots the 2D directivity.
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
        Compute the 3D far-field radiation pattern.

        The resulting far-field data can be used for visualization,
        directivity analysis, and post-processing of antenna performance.

        Parameters
        ----------
        nf2ff : object
            The near-field to far-field (NF2FF) object containing the simulation data.

        freq : float
            Frequency (in Hz) at which the 3D far-field radiation pattern is evaluated.

        output_path : Path, optional
            Path to the directory where the simulation result is saved.
            Defaults to current working directory.

        read_cached : bool, optional
            If True, read cached NF2FF results instead of re-computing.
            Default is False.

        Returns
        -------
        nf2ff_3d_result : object
            Returns nf2ff 3D result object.
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

        This method visualizes the directivity pattern of the antenna in 3D space,
        based on the far-field results computed from ``compute_nf2ff_3d``.

        Parameters
        ----------
        nf2ff_3d_result : object
            The 3D far-field result object containing the computed radiation data.

        freq : float
            The frequency (in Hz) at which the 3D directivity pattern is evaluated.

        output_path : Path
            Path to the directory where the simulation result is saved.

        Returns
        -------
        None
            This method does not return any value. It generates and saves a 3D plot of
            the directivity pattern.
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

        This method visualizes the gain pattern of the antenna in 3D space,
        based on the far-field results computed from ``compute_nf2ff_3d``.

        Parameters
        ----------
        nf2ff_3d_result : object
            The 3D far-field result object containing the computed radiation data.

        freq : float
            The frequency (in Hz) at which the 3D directivity pattern is evaluated.

        input_power : float
            Input power at the port used for gain normalisation.

        output_path : Path
            Path to the directory where the simulation result is saved.

        Returns
        -------
        None
            This method does not return any value. It generates and saves a 3D plot of
            the gain pattern.
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
        Plot the 3D power pattern of an antenna at a specified frequency.

        This method visualizes the power pattern of the antenna in 3D space,
        based on the far-field results computed from ``compute_nf2ff_3d``.

        Parameters
        ----------
        nf2ff_3d_result : object
            The 3D far-field result object containing the computed radiation data.

        freq : float
            The frequency (in Hz) at which the 3D directivity pattern is evaluated.

        output_path : Path
            Path to the directory where the simulation result is saved.

        Returns
        -------
        None
            This method does not return any value. It generates and saves a 3D plot of
            the power pattern.
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
        Save all currently open plots to a specified output path.

        Parameters
        ----------

        output_path : Path
            Path to the directory where the simulation result is saved.

        file_format: str
            File format to save the plot (e.g., 'png', 'jpg', 'pdf').

        Returns
        -------
        None

        Notes
        -----
        This method should be called before ``show_plots`` since
        ``show_plots`` blocks until plots are closed.
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
        Add a field dump to the simulation setup.

        This method configures and attaches a field dump to the openEMS simulation
        for storing electromagnetic field data (e.g., electric or magnetic fields)
        during the simulation run.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple containing CSX, FDTD, and freqs.

        params : SimParams
            Parameter object containing simulation parameters such as frequency,
            time steps, and dump configuration.

        output_path : Path
            Path to the directory where the field dump data will be saved.

        dump_type : DumpType, optional
            Type of field dump to add (e.g., time-domain or frequency-domain electric
            or magnetic fields). Default is `DumpType.efield_time`.

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
        Export S-parameters to a Touchstone file.

        This method writes the provided S-parameter data to a Touchstone
        file.

        Parameters
        ----------
        freqs : NDArray
            List of frequencies to be exported.
        s11 : NDArray
            Calculated S11 parameters over the frequency range.

        s21 : NDArray, optional
            Calculated S21 parameters over the frequency range.
            Default is None (single-port case).

        output_path : Path, optional
            Path to the directory where simulation result is saved.
            Defaults to current working directory.

        charac_imp : float, optional
            Characteristic impedance used for the Touchstone export. Default is 50.0.

        filename : str, optional
            Name of the file. Default is "s_param".

        Returns
        -------
        None
            This method does not return any value.

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
            Simulation setup named tuple containing CSX, FDTD, and freqs.

        output_path : Path
            Path to the directory where simulation result is stored.

        options : dict[str, list], optional
            Dictionary of export options.

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
        options: dict[str, list] | None = None,
    ) -> None:
        """
        Export CSXCAD geometry to STEP AP242 format using CadQuery.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple containing CSX, FDTD, and freqs.

        output_path : Path, optional
            Directory where the STEP file will be saved. Defaults to
            current working directory.

        options : dict[str, list], optional
            Dictionary of export options. Supported keys:
            - ``"ignore"`` : list of property names to skip.

        Returns
        -------
        None
        """
        if options is None:
            options = {}

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
        Load a ``structure.xml`` and export it to a STEP file.


        Parameters
        ----------
        structure_xml_path : str | Path
            Path to the ``structure.xml`` file exported by CSXCAD / openEMS.
        output_path : Path | None
            Directory where the STEP file will be saved.
        options : dict, optional
            Passed through to :func:`export_step`.
        """

        if output_path is None:
            output_path = Path.cwd()

        step_path = output_path / "step"
        step_path.mkdir(parents=True, exist_ok=True)

        export_csxcad_xml_to_step(structure_xml_path, output_path)

    @staticmethod
    def show_plots() -> None:
        """
        Display all generated plots.
        This method triggers the rendering of plots that have been created but not
        yet displayed.

        Returns
        -------
        None
            This method does not return any value.
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
        Print and save the simulation parameters.
        This method prints the simulation parameters to the terminal
        and saves them to a text file.

        Parameters
        ----------

        params: SimParams
            parameter object that contains all simulation parameters.
        output_path: Path
            The path to the directory where the simulation result
            is stored. Default is ``cwd``.

        Returns
        -------
        None
            This method does not return any value.
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
        Run all post-processing steps for a completed simulation.
        This convenience method executes the full suite of post-processing
        operations including plotting S-parameters, Smith chart, VSWR,
        impedance, radiation patterns, directivity, gain, power, exporting
        STL, Touchstone, and Gerber files, and saving/displaying all plots.

        Parameters
        ----------
        sim : SimSetup
            Simulation setup named tuple containing CSX, FDTD, and freqs.
        freqs : NDArray
            Array of frequencies used in the simulation.
        s11 : NDArray
            Complex S11 parameter array across the frequency range.
        vswr : NDArray
            VSWR values across the frequency range.
        z11 : NDArray
            Complex input impedance values across the frequency range.
        input_power : float
            Input power at the port.
        nf2ff : object
            Near-field to far-field transformation object.
        nf2ff_3d_result : object
            3D far-field result object from compute_nf2ff_3d.
        params : object
            Parameter object containing simulation settings.
        s21 : NDArray, optional
            Complex S21 parameter array across the frequency range.
            Default is None for single-port structures.
        output_path : Path, optional
            Directory where simulation results and exports will be saved.
            Defaults to current working directory.

        Returns
        -------
        None
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
    Objective function for S11 (reflection) optimisation. Lower is better.

    Parameters
    ----------
    freqs : NDArray
        Array of frequencies in Hz.
    s11 : NDArray
        Complex S11 parameter array across the frequency range.
    target_freq : float, optional
        Specific frequency at which to evaluate S11.
    freq_band : tuple of (float, float), optional
        Frequency band (f_min, f_max) over which to evaluate S11.
    mode : str, optional
        Evaluation mode within the frequency band.
        - "worst" : Returns the maximum S11 (worst-case reflection).
        - "mean" : Returns the mean S11 across the band.
        - "threshold" : Sum of S11 exceeding the threshold value.
        Default is "worst".
    threshold : float, optional
        S11 threshold in dB for the "threshold" mode. Default is -15.

    Returns
    -------
    float
        Scalar cost value (S11 in dB).

    Raises
    ------
    ValueError
        If mode is not "mean", "worst", or "threshold", or if neither
        target_freq nor freq_band is provided.
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
    Objective function for S21 (transmission) optimization. Higher S21 is better.
    This function computes a scalar cost from S21 data for use in
    optimization routines. Since optimizers minimize, the cost is
    negated so that maximizing S21 corresponds to minimizing the cost.

    Parameters
    ----------

    freqs : NDArray
        Array of frequencies in Hz.
    s21 : NDArray
        Complex S21 parameter array across the frequency range.
    target_freq : float, optional
        Specific frequency at which to evaluate S21.
    freq_band : tuple of (float, float), optional
        Frequency band (f_min, f_max) over which to evaluate S21.
    mode : str, optional
        Evaluation mode within the frequency band.
        - "mean" : Returns negative mean of S21 in dB across the band.
        - "worst" : Returns negative minimum of S21 in dB across the band.
        Default is "worst".

    Returns
    -------

    float
        Scalar cost value (negated S21 in dB).
    Raises
    ------

    ValueError
        If mode is not "mean" or "worst", or if neither target_freq
        nor freq_band is provided.
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
    Optimise S-parameters using SciPy's minimize function.
    Nelder-Mead algorithm is used to perform the optimisation.

    Parameters
    ----------
    simulate_fn : callable
        Simulation function that is invoked during optimization.
        The simulation function should return the S11 value at the
        resonant frequency; this return value is used as the
        optimisation goal.

    x0 : dict[str, float]
        Dictionary of initial parameter values used as the starting point for
        the optimization.

    output_path : Path
        Path to the directory where simulation results is stored.

    bounds : optional
        Bounds on the optimisation parameters. Optional — not used
        by Nelder-Mead algorithm.

    Returns
    -------
    None
        This function does not return any value.
    """
    console.print("-------------------------------------", style="info")
    console.print("Running Optimization", style="info")
    console.print("-------------------------------------", style="info")
    x0_values = list(x0.values())
    x0_keys = list(x0.keys())

    def make_callback(tol: float = 1e-4, patience: int = 5) -> Callable:
        """Create a callback that stops optimisation when parameters stall."""
        prev_x = None
        stall = 0

        def callback(xk: OptimizeResult) -> None:
            """StopIteration callback: raises StopIteration if parameters stall."""
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
    label: str = "",
    sweep: bool = True,
) -> SimData | None:
    """
    Perform a parameter sweep using a simulation function.
    The function supports multiple parameters; the sweep is a
    Cartesian product which can be expensive with many parameters.

    Parameters
    ----------
    simulate_fn : callable
        Simulation function to be invoked during the parameter sweep.

    sweep_vals : dict[str, tuple]
        Dictionary defining the parameter sweep ranges. Keys are
        parameter names; values are tuples of
        ``(min_value, max_value, num_points)`` passed to
        ``numpy.linspace``.

    output_path : Path
        Path to the directory where simulation result is saved.

    label : str, optional
        Label prefix used in plot legends for the sweep. Default is "".

    sweep : bool, optional
        Flag indicating whether the parameter sweep is performed. Default is True.

    Returns
    -------
    SimData or None
        The simulation data returned by the simulation function. Contains
        attributes such as freqs, s11, s21, vswr, z11, and input_power.
    """
    console.print("-------------------------------------", style="info")
    console.print("Running Parameter Sweep", style="info")
    console.print("-------------------------------------", style="info")

    sweep_values = {key: np.linspace(*val).tolist() for key, val in sweep_vals.items()}
    cartesian_sweep = list(product(*sweep_values.values()))
    sim_data = None

    for values in cartesian_sweep:
        kv_pairs = list(zip(sweep_values.keys(), values, strict=True))
        key_value = "_".join(f"{k}_{v}" for k, v in kv_pairs)
        label = ", ".join(f"{k}={v}" for k, v in kv_pairs)
        sweep_path = output_path / "sweep" / key_value
        sweep_path.mkdir(parents=True, exist_ok=True)

        sim_data = simulate_fn(sweep_path, sweep, values)
        SimTools.plot_s_param(
            sim_data.freqs,
            sim_data.s11,
            label_s11=label,
        )
    SimTools.show_plots()
    return sim_data


def mm_to_m(mm: float) -> float:
    """
    Converts millimeters to meters.

    Parameters
    ----------

    mm : float
        Value in millimeters

    Returns
    -------
    float
        Value in meters

    """
    return mm / 1000.0


def m_to_mm(m: float) -> float:
    """
    Converts meters to millimeters.

    Parameters
    ----------

    m : float
        Value in meters

    Returns
    -------
    float
        Value in millimeters

    """
    return m * 1000.0


def mil_to_mm(mil: float) -> float:
    """
    Converts mils (thousandths of an inch) to millimeters.

    Parameters
    ----------

    mil : float
        Value in mils

    Returns
    -------
    float
        Value in millimeters

    """
    return mil * 0.0254


def mm_to_mil(mm: float) -> float:
    """
    Converts millimeters to mils (thousandths of an inch).

    Parameters
    ----------

    mm : float
        Value in millimeters

    Returns
    -------
    float
        Value in mils

    """
    return mm / 0.0254


def cm_to_mm(cm: float) -> float:
    """
    Converts centimeters to millimeters.

    Parameters
    ----------

    cm : float
        Value in centimeters

    Returns
    -------
    float
        Value in millimeters

    """
    return cm * 10.0


def mm_to_cm(mm: float) -> float:
    """
    Converts millimeters to centimeters.

    Parameters
    ----------

    mm : float
        Value in millimeters

    Returns
    -------
    float
        Value in centimeters

    """
    return mm / 10.0
