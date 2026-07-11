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
Calculation utilities for microstrip patch antenna design.

Provides functions for computing patch dimensions, microstrip line
width from target impedance, phase shift lengths, conductance, and
inset depths. All formulas follow standard references (Balanis,
Hammerstad-Jensen).
"""

from typing import NamedTuple

import numpy as np
from openEMS.physical_constants import C0
from scipy.integrate import quad

from .sim_tools import m_to_mm, mm_to_m


def calculate_electrical_length_mm(
    elec_length_deg_rad: float,
    eps_eff: float,
    frequency: float,
    radians: bool = False,
) -> float:
    """
    Compute transmission line length (mm) required to obtain
    a specified phase shift at a given frequency.

    Parameters
    ----------
    elec_length_deg_rad : float
        Phase shift in degrees (default) or radians.
    eps_eff : float
        Effective dielectric constant.
    frequency : float
        Frequency in Hz.
    radians : bool
        If True, elec_length_deg_rad is already in radians.

    Returns
    -------
    float
        Required length in mm.
    """
    theta = elec_length_deg_rad if radians else np.deg2rad(elec_length_deg_rad)

    # Guided phase constant (rad/m)
    beta = 2 * np.pi * frequency * np.sqrt(eps_eff) / C0

    # Length in meters
    length_m = theta / beta

    return length_m * 1e3


def microstrip_width_from_impedance(
    charac_imp: float,
    subs_height: float,
    copper_thickness: float,
    subst_eps_r: float,
    freq_hz: float,
    tolerance: float = 0.01,
) -> tuple[float, float]:
    """
    Calculates microstrip width for a target impedance including dispersion
    and thickness using Hammerstad-Jensen equations.

    Parameters
    ----------

    charac_imp: float
        The required characteristic impedance of the microstrip line
    subs_height: float
        The height of the substrate
    copper_thickness: float
        The thickness of the copper trace
    subst_eps_r: float
        Dielectric constant of the substrate material.
    freq_hz: float
        Frequency in Hz
    tolerance: float
        Maximum deviation allowed for calculated characteristic impedance

    Returns
    -------
    width: float
        Width of trace for given characteristic impedance in mm.
    er_eff: float
        Effective dielectric constant at the given frequency.

    Notes
    -----
    For the formulas used, refer to:
    `<https://qucs.github.io/docs/technical/technical.pdf>`_.
    `<https://ieeexplore.ieee.org/document/1124303>`_.
    """

    def get_Z_at_freq(
        w: float,
        h: float,
        t: float,
        er: float,
        f_hz: float,
    ) -> tuple[float, float]:
        """Calculate microstrip characteristic impedance at a given
        frequency using Hammerstad-Jensen equations."""
        # Constants
        eta0 = 376.730313668
        mue0 = 4 * np.pi * 1e-7
        h_m = mm_to_m(h)

        if t == 0:
            t = 1e-6

        # 1. Thickness Correction
        u = w / h
        tau = t / h
        # coth(x) = 1/tanh(x)
        coth_term = 1.0 / np.tanh(np.sqrt(6.517 * u))
        delta_u1 = (tau / np.pi) * np.log(1 + (4 * np.e) / (tau * (coth_term**2)))
        delta_ur = 0.5 * delta_u1 * (1 + 1.0 / np.cosh(np.sqrt(er - 1)))

        # Effective width ratio to be used in all static formulas
        ur = u + delta_ur

        # 2. Static Impedance (Air-filled Z01)
        f_ur = 6 + (2 * np.pi - 6) * np.exp(-((30.666 / ur) ** 0.7528))
        Z01 = (eta0 / (2 * np.pi)) * np.log((f_ur / ur) + np.sqrt(1 + (2 / ur) ** 2))

        # 3. Static Effective Permittivity
        a_u = (
            1
            + (1 / 49) * np.log((ur**4 + (ur / 52) ** 2) / (ur**4 + 0.432))
            + (1 / 18.7) * np.log(1 + (ur / 18.1) ** 3)
        )
        b_er = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
        er_eff_0 = (er + 1) / 2 + ((er - 1) / 2) * (1 + 10 / ur) ** (-a_u * b_er)

        # 4. Static Impedance
        Z0 = Z01 / np.sqrt(er_eff_0)

        # 5. Frequency Dispersion
        if f_hz <= 0:
            return Z0

        # Transition frequency (fp) in Hz
        fp = Z0 / (2 * mue0 * h_m)

        # Dispersion factor G
        G = (np.pi**2 / 12) * ((er - 1) / er_eff_0) * np.sqrt(2 * np.pi * Z0 / eta0)

        # Frequency-dependent effective permittivity
        er_eff_f = er - (er - er_eff_0) / (1 + G * (f_hz / fp) ** 2)

        # Frequency-dependent impedance
        Z_f = Z0 * ((er_eff_f - 1) / (er_eff_0 - 1)) * np.sqrt(er_eff_0 / er_eff_f)

        return Z_f, er_eff_f

    # --- Convergence check ---
    Z_thinnest, _ = get_Z_at_freq(
        0.001, subs_height, copper_thickness, subst_eps_r, freq_hz
    )
    Z_widest, _ = get_Z_at_freq(
        100, subs_height, copper_thickness, subst_eps_r, freq_hz
    )
    Z_min = min(Z_thinnest, Z_widest)
    Z_max = max(Z_thinnest, Z_widest)
    if not (Z_min <= charac_imp <= Z_max):
        raise ValueError(
            f"Target impedance {charac_imp} Ω not realizable on this substrate"
            f" (range [{Z_min:.1f}, {Z_max:.1f}] Ω)"
        )

    # --- Binary Search for Width ---
    low, high = 0.001, 100
    mid_w = 0.0
    for _ in range(100):
        mid_w = (low + high) / 2
        Z_curr, _ = get_Z_at_freq(
            mid_w, subs_height, copper_thickness, subst_eps_r, freq_hz
        )

        if abs(Z_curr - charac_imp) < tolerance:
            break
        if Z_curr > charac_imp:
            low = mid_w
        else:
            high = mid_w
    _, er_eff = get_Z_at_freq(
        mid_w, subs_height, copper_thickness, subst_eps_r, freq_hz
    )

    return mid_w, er_eff


def conductance_G1(patch_width: float, frequency: float) -> float:
    """
    Computes the conductance of patch antenna.

    Parameters
    ----------

    patch_width: float
        width of patch antenna in meters
    frequency: float
        frequency in Hz

    Returns
    -------
    conductance_G1: float
        conductance of patch antenna

    Notes
    -----
    Refer to Balanis (3rd ed.) 14-12 formula for radiation conductance of one slot.
    """

    lambda0 = C0 / frequency
    k0 = 2 * np.pi / lambda0

    def integrand(theta: float) -> float:
        """Integrand for conductance calculation using the cavity model."""
        ct = np.cos(theta)
        if abs(ct) < 1e-5:
            return 0.0
        term = np.sin((k0 * patch_width / 2) * ct) / ct
        return (term**2) * (np.sin(theta) ** 3)

    integral, _ = quad(integrand, 0, np.pi, limit=1000)
    return integral / (120 * np.pi**2)


def inset_depth(
    charac_imp: float,
    patch_width: float,
    patch_length: float,
    frequency: float,
) -> tuple[float, float]:
    """
    Compute the inset feed depth of an inset fed patch antenna
    and probe position of probe fed patch antenna.

    Parameters
    ----------
    charac_imp: float
        characteristic impedance of the antenna.
    patch_width: float
        patch antenna width in meters.
    patch_length: float
        patch antenna length in meters.
    frequency: float
        frequency in Hz.

    Returns
    -------
    inset_length: float
        inset length in meters
    probe_pos: float
        probe position in meters. note that probe position is in Y direction and X is 0.

    Notes
    -----
    Refer to Balanis formula 14-20a

    """

    G1 = conductance_G1(patch_width, frequency)
    R_edge = 1.0 / (2.0 * G1)

    if charac_imp > R_edge:
        raise ValueError("characteristic impedance must be <= edge resistance")

    inset_length = (patch_length / np.pi) * np.arccos(np.sqrt(charac_imp / R_edge))
    probe_pos = inset_length
    return inset_length, probe_pos


class PatchDims(NamedTuple):
    """
    Named tuple holding all rectangular patch antenna dimensions.

    Attributes
    ----------
    patch_width_mm : float
        Width of the patch in the x-direction in mm.
    patch_length_mm : float
        Length of the patch in the y-direction in mm.
    inset_length_mm : float
        Inset depth of the patch in mm.
    inset_width_mm : float
        Inset gap width of the patch in mm.
    probe_pos_mm : float
        Feed position for probe-fed patch in the y-direction in mm.
    """

    patch_width_mm: float
    patch_length_mm: float
    inset_length_mm: float
    inset_width_mm: float
    probe_pos_mm: float


def patch_dims(
    frequency_hz: float,
    subst_eps_r: float,
    subst_height: float,
    charac_imp: float,
    copper_thickness: float,
) -> PatchDims:
    """
    Calculate rectangular microstrip patch dimensions.

    Parameters
    ----------
    frequency_hz: float
        resonant frequency of antenna in Hz.
    subst_eps_r: float
        Dielectric constant of the substrate.
    subst_height: float
        Thickness of the substrate.
    charac_imp: float
        characteristic impedance of the feed line.
    copper_thickness: float
         Thickness of the copper trace.

    Returns
    -------
    PatchDims
        a named tuple with the following attributes.
            patch_width_mm: float
                Width of the patch antenna in direction of x in mm.
            patch_length_mm: float
                Length of the patch antenna in direction of y in mm
            inset_length_mm: float
                Inset length of the patch in direction of y in mm
            inset_width_mm: float
                Inset width of the patch in the direction of x in mm
            probe_pos_mm: float
                The feed position of probe fed patch antenna in direction of y in mm
    """

    patch_width = C0 / (2 * frequency_hz) * np.sqrt(2 / (subst_eps_r + 1))
    patch_width_mm = m_to_mm(patch_width)

    eff_eps_r = (subst_eps_r + 1) / 2 + (subst_eps_r - 1) / 2 * (
        1 / np.sqrt(1 + 12 * subst_height / patch_width)
    )

    delta_length = (
        0.412
        * subst_height
        * ((eff_eps_r + 0.3) * (patch_width / subst_height + 0.264))
        / ((eff_eps_r - 0.258) * (patch_width / subst_height + 0.8))
    )

    patch_length = (C0 / (2 * frequency_hz * np.sqrt(eff_eps_r))) - 2 * delta_length
    patch_length_mm = m_to_mm(patch_length)

    inset_length, probe_pos = inset_depth(
        charac_imp, patch_width, patch_length, frequency_hz
    )

    inset_length_mm = m_to_mm(inset_length)
    probe_pos_mm = m_to_mm(probe_pos)

    inset_width_mm, _ = microstrip_width_from_impedance(
        charac_imp,
        m_to_mm(subst_height),
        copper_thickness,
        subst_eps_r,
        frequency_hz,
    )

    return PatchDims(
        patch_width_mm,
        patch_length_mm,
        inset_length_mm,
        inset_width_mm,
        probe_pos_mm,
    )
