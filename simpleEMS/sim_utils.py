import sys
import subprocess
from pathlib import Path
from collections import namedtuple
from dataclasses import fields
from openEMS.openEMS import openEMS
from CSXCAD import ContinuousStructure
from CSXCAD import AppCSXCAD_BIN
import numpy as np
import matplotlib.pyplot as plt
from pysmithchart import S_PARAMETER
from matplotlib.ticker import EngFormatter
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt6.QtWidgets import QApplication
from enum import Enum
from skrf import Network
from scipy.optimize import minimize
from itertools import product
from simpleEMS import export_gerber


freq_formatter = EngFormatter(unit="Hz", places=2)
plt.rcParams["figure.constrained_layout.use"] = True


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
        raw data needed for SAR calculations (electric field FD, cell volume, conductivity and density)
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


def setup_simulation(
    params,
    boundary_cond: list[str] = ["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"],
):
    """
    Sets up the openEMS simulation.

    Parameters
    ----------

    params: object
        parameter object that holds all simulation parameters.
    boundary_cond: list[str]
        boundary condition for the simulation.

    Returns
    -------
    CSX:object
        CSXCAD geometry object.
    FDTD: object
        openEMS FDTD object.
    """
    CSX = ContinuousStructure()
    FDTD = openEMS(NrTS=params.timestep, EndCriteria=params.end_criteria)
    FDTD.SetGaussExcite(params.resonant_freq, params.corner_freq)
    FDTD.SetBoundaryCond(boundary_cond)
    FDTD.SetCSX(CSX)
    return CSX, FDTD


class SimUtils:
    """
    A collection of common utility functions for simulations.
    This class is not intended to be instantiated. It provides a
    namespace for common simulation utilities and function.

    All structure classes i.e. InsetFedPatchAntenna inherit this class.

    """

    @staticmethod
    def write_and_show_structure(CSX, output_path: Path) -> None:
        """
        This method shows the CSXCAD structure.

        Parameters
        ----------
        CSX:object
            CSXCAD geometry object.
        output_path: Path
            The output path where simulation results is stored.
        """
        if not output_path.exists():
            raise ValueError(
                "output path does not exist. Make sure to provide valid output path."
            )
        structure_3d = output_path / "structure.xml"
        CSX.Write2XML(structure_3d)
        subprocess.run([AppCSXCAD_BIN, str(structure_3d)], check=True)

    @staticmethod
    def export_stl(output_path: Path) -> None:
        """
        This method exports the CSXCAD structure to STL objects.

        Parameters
        ----------
        output_path: Path
            The output path of the simulation results.
        """
        structure_3d = output_path / "structure.xml"
        if not structure_3d.exists():
            raise ValueError(
                "3D Structure does not exist. Make sure to call write_and_show_structure method before exporting STL"
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
    def run_simulation(FDTD, output_path: Path) -> None:
        """
        This method starts and runs the openEMS simulation.

        Paramaters
        ----------
        FDTD: object
            openEMS FDTD object.
        output_path: Path
            The output path of the simulation results.
        """
        FDTD.Run(output_path)

    @staticmethod
    def create_nf2ff(FDTD):
        """
        Creates near field to far field (NF2FF) recording box.

        Parameters
        ----------
        FDTD: object
            openEMS FDTD object.

        Returns
        -------
        nf2ff : object
            NF2FF object
        """
        nf2ff = FDTD.CreateNF2FFBox()
        return nf2ff

    @staticmethod
    def compute_network_params(port, params, output_path: Path):
        """
        Computes several network parameters for plotting and post-processing of simulation results.

        This method calculates various network parameters such as S11, Z11, VSWR, and input power
        based on the simulation results. It uses the provided port and simulation parameters to
        extract the necessary data, processes it, and prepares it for further analysis and plotting.

        Parameters
        ----------
        port : object
            The openEMS port object representing the simulation port.

        params : object
            The parameter object that holds all simulation parameters, including frequency range and other settings.

        output_path : Path
            The path to the directory where the simulation results are saved.

        Returns
        -------
        NetworkParams
            A named tuple containing the following attributes:

            - freqs : ndarray
                A numpy array of frequencies used for post-processing. The array is generated
                from `resonant_freq - corner_freq` to `resonant_freq + corner_freq`.

            - s11 : ndarray
                A numpy array of calculated complex S11 values across the entire frequency range.

            - z11 : ndarray
                A numpy array of calculated complex Z11 values across the entire frequency range.

            - vswr : ndarray
                A numpy array of calculated VSWR values across the entire frequency range.

            - input_power : float
                The calculated input power at the port.

        Notes
        -----
        - The method assumes the simulation results are available at the specified `output_path`.
        - The frequency range for post-processing is determined by the resonant frequency and the corner frequency.
        """
        NetworkParams = namedtuple(
            "NetworkParams", ["freqs", "s11", "z11", "vswr", "input_power"]
        )
        freqs = np.linspace(
            params.resonant_freq - params.corner_freq,
            params.resonant_freq + params.corner_freq,
            params.num_points,
        )
        port.CalcPort(output_path, freqs, ref_impedance=params.charac_imp)
        z11 = port.uf_tot / port.if_tot
        s11 = port.uf_ref / port.uf_inc
        s11_mag = np.abs(s11)
        s11_mag = np.clip(s11_mag, 0, 0.999)  # prevent division by zero error
        vswr = (1 + s11_mag) / (1 - s11_mag)
        input_power = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
        return NetworkParams(freqs, s11, z11, vswr, input_power)

    @staticmethod
    def plot_s11(
        freqs: np.ndarray,
        s11: np.ndarray,
        x_label: str = "Frequency",
        y_label: str = "S11 (dB)",
        label: str = "",
        title: str = "S11 vs Frequency",
    ) -> None:
        """
        Plot the S11 parameter (reflection coefficient) against frequency.

        This method generates a 2D plot of the S11 parameter (in dB) as a function of frequency.

        Parameters
        ----------
        freqs : np.ndarray
            A 1D array of frequencies (in Hz) to plot on the x-axis.

        s11 : np.ndarray
            A 1D array of the corresponding S11 values (in dB) to plot on the y-axis.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "S11 (dB)".

        label : str, optional
            Label for the plot legend. Default is an empty string (no label).

        title : str, optional
            Title of the plot. Default is "S11 vs Frequency".

        Returns
        -------
        None
            This method does not return any value. It directly displays the plot.
        """
        s11 = 20.0 * np.log10(np.abs(s11))
        plt.plot(freqs, s11, label=label)
        min_idx = np.nanargmin(s11)
        plt.scatter(
            freqs[min_idx],
            s11[min_idx],
            color="black",
            label=f"Min S11 = {s11[min_idx]:.2f} at {freq_formatter(freqs[min_idx])}",
        )
        tar_idx = len(freqs) // 2
        plt.scatter(
            freqs[tar_idx],
            s11[tar_idx],
            color="red",
            label=f"Target S11 = {s11[tar_idx]:.2f} at {freq_formatter(freqs[tar_idx])} ",
        )
        plt.gca().xaxis.set_major_formatter(EngFormatter(unit="Hz"))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    @staticmethod
    def plot_vswr(
        freqs: np.ndarray,
        vswr: np.ndarray,
        x_label: str = "Frequency",
        y_label: str = "VSWR",
        title: str = "VSWR vs Frequency",
        label: str = "VSWR",
    ):
        """
        Plot the Voltage Standing Wave Ratio (VSWR) as a function of frequency.

        This method generates a 2D plot of VSWR values over the specified frequency range,
        which is used to evaluate impedance matching performance.

        Parameters
        ----------
        freqs : np.ndarray
            A 1D array of frequencies (in Hz) to be plotted on the x-axis.

        vswr : np.ndarray
            A 1D array of VSWR values corresponding to each frequency.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "VSWR".

        title : str, optional
            Title of the plot. Default is "VSWR vs Frequency".

        label : str, optional
            Label used for the plot legend. Default is "VSWR".

        Returns
        -------
        None
            This method does not return any value. It displays the VSWR plot.
        """
        plt.figure()
        plt.plot(freqs, vswr, label=label)
        min_idx = np.nanargmin(vswr)
        plt.scatter(
            freqs[min_idx],
            vswr[min_idx],
            color="red",
            label=f"Min VSWR = {vswr[min_idx]:.2f} found at {freq_formatter(freqs[min_idx])}",
        )
        tar_idx = len(freqs) // 2
        plt.scatter(
            freqs[tar_idx],
            vswr[tar_idx],
            color="black",
            label=f"VSWR is {vswr[tar_idx]:.2f} at target frequency {freq_formatter(freqs[tar_idx])}",
        )

        plt.gca().xaxis.set_major_formatter(EngFormatter(unit="Hz"))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()

    @staticmethod
    def plot_impedance(
        freqs: np.ndarray,
        z11: np.ndarray,
        x_label: str = "Frequency",
        y_label: str = "Z11",
        title: str = "Z11 vs Frequency",
    ):
        """
        Plot the input impedance (Z11) as a function of frequency.

        This method generates a 2D plot of the complex input impedance Z11 versus
        frequency. The real and imaginary components of Z11 are plotted
        to analyze impedance behavior across the frequency range.

        Parameters
        ----------
        freqs : np.ndarray
            A 1D array of frequencies (in Hz) to be plotted on the x-axis.

        z11 : np.ndarray
            A 1D array of complex input impedance values corresponding to each
            frequency.

        x_label : str, optional
            Label for the x-axis. Default is "Frequency".

        y_label : str, optional
            Label for the y-axis. Default is "Z11".

        title : str, optional
            Title of the plot. Default is "Z11 vs Frequency".

        Returns
        -------
        None
            This method does not return any value. It displays the impedance plot.
        """
        z11_real = np.real(z11)
        z11_imag = np.imag(z11)
        plt.figure()
        plt.plot(freqs, z11_real, label=f"Real Z11")
        plt.plot(freqs, z11_imag, label=f"Imag Z11")

        tar_idx = len(freqs) // 2
        plt.scatter(
            freqs[tar_idx],
            z11_real[tar_idx],
            color="black",
            label=f"Z11 is {z11_real[tar_idx]:.0f}Ω at target frequency {freq_formatter(freqs[tar_idx])}",
        )
        plt.gca().xaxis.set_major_formatter(EngFormatter(unit="Hz"))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()

    @staticmethod
    def plot_smith_chart(freqs, s11, label="", charac_imp=50):
        """
        Plot S11 data on a Smith chart.

        This method generates a Smith chart representation of the complex reflection
        coefficient (S11) over a specified frequency range.

        Parameters
        ----------
        freqs : np.ndarray
            A 1D array of frequencies (in Hz) corresponding to the S11 data.

        s11 : np.ndarray
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
            label=f"S11 is {s11[tar_idx]:.2f} at target frequency {freq_formatter(freqs[tar_idx])}",
        )
        plt.title("Smith Chart - S11")
        plt.legend(loc="lower right", bbox_to_anchor=(1.5, 1.0))

    @staticmethod
    def plot_2d_rad_pattern(nf2ff, freq, output_path):
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

        output_path : Path
            Path to the directory where the simulation result is saved.

        Returns
        -------
        None
            This method does not return any value. It generates the 2D
            radiation pattern plot.
        """
        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency and not array for radiation pattern calculation"
            )

        theta = np.arange(-180.0, 181.0, 2.0)
        print("Calculating 2D Radiation Pattern.........")
        nf2ff_res_phi0 = nf2ff.CalcNF2FF(
            output_path,
            freq,
            theta,
            0,
            read_cached=False,
            outfile="nf2ff_xz.h5",
            verbose=0,
        )

        plt.figure()
        ax = plt.subplot(121, polar=True)

        efield = np.squeeze(nf2ff_res_phi0.E_norm)
        efield_norm = efield / np.max(efield)
        efield_norm_dB = 20 * np.log10(efield_norm)

        ax.plot(
            np.deg2rad(theta),
            efield_norm_dB,
            linewidth=2,
            label="xz-plane",
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
            read_cached=False,
            outfile="nf2ff_xy.h5",
        )

        ax = plt.subplot(122, polar=True)

        efield = np.squeeze(nf2ff_res_theta90.E_norm)
        efield_norm = efield / np.max(efield)
        efield_norm_dB = 20 * np.log10(efield_norm)

        ax.plot(
            np.deg2rad(phi),
            efield_norm_dB,
            linewidth=2,
            label="xy-plane",
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
    def plot_2d_directivity(nf2ff, freq, output_path):
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

        output_path : Path
            Path to the directory where the simulation result is saved.

        Returns
        -------
        None
            This method does not return any value. It plots the 2D directivity.
        """
        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency and not array for directivity calculation"
            )

        theta = np.arange(-180.0, 181.0, 2.0)
        print("Calculating Farfield Directivity.........")
        nf2ff_res_phi0 = nf2ff.CalcNF2FF(
            output_path,
            freq,
            theta,
            0,
            read_cached=False,
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

        ax.plot(
            np.deg2rad(theta),
            directivity_dbi,
            linewidth=2,
            label="xz-plane",
        )

        # ---- HPBW calculation ----
        peak_idx = np.argmax(directivity_dbi)
        peak_theta = theta[peak_idx]

        # Half-power level (−3 dB)
        hpbw_level = -3.0
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
        hpbw_arc = np.linspace(left_theta, right_theta, 100)
        ax.plot(
            np.deg2rad(hpbw_arc),
            directivity_dbi[left_idx[-1]] * np.ones_like(hpbw_arc),
            "r",
            linewidth=2,
        )
        main_lobe_mag = np.round(np.max(directivity_dbi), 2)
        ax.text(
            1.0,
            0.1,
            f"HPBW (3dB) = {hpbw}°\nMain Lobe Direction = {peak_theta}°\nMain Lobe Magnitude ={main_lobe_mag}dBi",
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
        ax.grid(True)
        ax.set_xlabel("theta (deg)")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend()

    @staticmethod
    def compute_nf2ff_3d(nf2ff, freq: float, output_path: Path) -> object:
        """
        Compute the 3D far-field radiation pattern.

        The resulting far-field data can be used for visualization, directivity analysis, and
        post-processing of antenna performance.

        Parameters
        ----------
        nf2ff : object
            The near-field to far-field (NF2FF) object containing the simulation data.

        freq : float
            Frequency (in Hz) at which the 3D far-field radiation pattern is evaluated.

        output_path : Path
            Path to the directory where the simulation result is saved.

        Returns
        -------
        nf2ff_3d_result : object
            Returns nf2ff 3D result object.
        """
        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency and not array for 3D far-field calculation"
            )

        theta = np.arange(0, 181, 2)  # elevation
        phi = np.arange(0, 361, 2)  # azimuth

        print("Calculating 3D Pattern.........")
        nf2ff_3d_result = nf2ff.CalcNF2FF(
            output_path,
            freq,
            theta,
            phi,
            read_cached=False,
            outfile="nf2ff_3d.h5",
            verbose=0,
        )
        return nf2ff_3d_result

    @staticmethod
    def plot_3d_directivity(nf2ff_3d_result, freq, output_path):
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

        if not np.isscalar(freq):
            raise TypeError(
                "Please specify only one frequency and not array for directivity calculation"
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
    def plot_3d_gain(nf2ff_3d, freq, input_power, output_path):
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

        output_path : Path
            Path to the directory where the simulation result is saved.

        Returns
        -------
        None
            This method does not return any value. It generates and saves a 3D plot of
            the gain pattern.
        """

        e_field = np.squeeze(nf2ff_3d.E_norm)
        e_field /= np.max(e_field)  # normalize

        plots_3d_path = output_path / "3D_plots"
        plots_3d_path.mkdir(parents=True, exist_ok=True)

        max_directivity = nf2ff_3d.Dmax[0]
        directivity = max_directivity * e_field

        efficiency = nf2ff_3d.Prad[0] / (np.max(input_power))
        gain = efficiency * directivity
        gain_dbi = 10 * np.log10(gain)

        theta = nf2ff_3d.theta
        phi = nf2ff_3d.phi

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
    def plot_3d_power(nf2ff_3d, freq: float, output_path: Path) -> None:
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

        plots_3d_path = output_path / "3D_plots"
        plots_3d_path.mkdir(parents=True, exist_ok=True)

        power = np.squeeze(nf2ff_3d.P_rad)

        power /= np.max(power)  # normalize

        power_db = 10 * np.log10(power)

        theta = nf2ff_3d.theta
        phi = nf2ff_3d.phi

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
    def save_plots(output_path: Path, file_format: str = "png") -> None:
        """
        Save all currently open plots with unique filenames to a specified output path.

        Parameters
        ----------

        output_path : Path
            Path to the directory where the simulation result is saved.

        file_format: str
            File format to save the plot (e.g., 'png', 'jpg', 'pdf').

        Notes
        -----
        This method must be called after ``show_plots`` while all matplotlib plots are open

        Returns
        -------
            None
        """
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
            print(f"Plot {i + 1} saved as {file_path}")

    @staticmethod
    def add_field_dump(
        CSX,
        params,
        output_path: Path,
        dump_type: DumpType = DumpType.efield_time,
    ) -> None:
        """
        Add a field dump to the simulation setup.

        This method configures and attaches a field dump to the openEMS simulation
        for storing electromagnetic field data (e.g., electric or magnetic fields)
        during the simulation run.

        Parameters
        ----------
        CSX : object
            The openEMS CSX object representing the simulation geometry and settings.

        params : object
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
        # TODO Add appropriate dump mode based on openEMS docs
        dump_path = output_path / "field_dump"
        dump_path.mkdir(parents=True, exist_ok=True)
        dump = CSX.AddDump(
            str(dump_path / dump_type.value[1]),
            file_type=0,
            dump_type=dump_type.value[0],
            dump_mode=0,
        )
        start = [params.simulation_box[0] / 2, params.simulation_box[1] / 2, 0]
        stop = [
            -params.simulation_box[0] / 2,
            -params.simulation_box[1] / 2,
            params.substrate_thickness_mm + params.copper_thickness_mm,
        ]
        dump.AddBox(start=start, stop=stop)

    @staticmethod
    def export_touchstone(
        network_params, output_path: Path, charac_imp: float = 50.0, filename="s_param"
    ):
        """
        Export  S-parameters to a Touchstone file.

        This method writes the provided S-parameter data to a Touchstone
        file.

        Parameters
        ----------
        network_params : object
            Computed network parameter result returned from ``compute_network_params`` method.

        output_path : Path
            Path to the directory where simulation result is saved.

        charac_imp : float, optional
            Characteristic impedance used for the Touchstone export. Default is 50.0.

        filename:str, optional
            Name of the file.

        Returns
        -------
        None
            This method does not return any value.

        """
        touchstone_path = output_path / "touchstone"
        touchstone_path.mkdir(parents=True, exist_ok=True)
        ntwk = Network(
            frequency=network_params.freqs, s=network_params.s11, z0=charac_imp
        )
        ntwk.write_touchstone(filename=filename, dir=touchstone_path)

    @staticmethod
    def export_gerber(
        CSX, output_path: Path, options: dict[str, list] = {"ignore": []}
    ):
        """
        Export CSXCAD geometry to Gerber format.


        Parameters
        ----------
        CSX : object
            CSXCAD geometry object.

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
        The gerber export is currently limited and might not be able to export all geometries.
        Layers should be exported separately by specifying ignore options. i.e. to export only top metal layer ignore the ground.

        """
        gerber_path = output_path / "gerber"
        gerber_path.mkdir(parents=True, exist_ok=True)
        export_gerber.export_gerber(
            CSX=CSX,
            output_path=gerber_path,
            options=options,
        )

    @staticmethod
    def show_plots():
        if not plt.get_fignums():
            view_3d = QApplication.instance()
            if view_3d is not None:
                sys.exit(view_3d.exec())
        plt.show()

    @staticmethod
    def print_and_save_params(params, output_path):
        params_path = output_path / "params"
        params_path.mkdir(parents=True, exist_ok=True)
        cls_name = params.__class__.__name__
        print(f"{cls_name}:")
        lines = []
        for field in fields(params):
            value = getattr(params, field.name)
            print(f"  {field.name}: {value}")
            lines.append(f"  {field.name}: {value}\n")
        with open(params_path / "params.txt", "w") as file:
            file.write(f"{cls_name}:\n")
            file.writelines(lines)


def optimize_s11(simulate_fn, x0: dict[str, float], output_path, bounds=None):
    print("-------------------------------------")
    print("Running Optimization")
    print("-------------------------------------")
    x0_values = list(x0.values())
    x0_keys = list(x0.keys())

    def make_callback(tol=1e-4, patience=5):
        prev_x = None
        stall = 0

        def callback(xk):
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

    def optimize_fun(x):
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
    fn_to_sweep,  # the simulation function
    sweep_vals,  # list or array of values to sweep
    output_path,  # base folder to save outputs
    sweep=True,
):
    """
    Sweep a parameter using a simulation function

    simulate_fn(val, output_path) -> returns network_params object
    """
    print("*************************************")
    print("Running Parameter Sweep")
    print("*************************************")

    sweep_values = {key: np.linspace(*val).tolist() for key, val in sweep_vals.items()}
    cartesian_sweep = list(product(*sweep_values.values()))

    for values in cartesian_sweep:
        kv_pairs = list(zip(sweep_values.keys(), values))
        key_value = "_".join(f"{k}_{v}" for k, v in kv_pairs)
        label = ", ".join(f"{k}={v}" for k, v in kv_pairs)
        sweep_path = output_path / "sweep" / key_value
        sweep_path.mkdir(parents=True, exist_ok=True)

        network_params = fn_to_sweep(sweep_path, sweep, values)
        SimUtils.plot_s11(
            network_params.freqs,
            network_params.s11,
            label=label,
        )
    SimUtils.show_plots()


def mm_to_m(mm):
    """
    Converts millimeters to meters.
    :param mm: Value in millimeters
    :return: Value in meters
    """
    return mm / 1000.0


def m_to_mm(m):
    """
    Converts meters to millimeters.
    :param m: Value in meters
    :return: Value in millimeters
    """
    return m * 1000.0
