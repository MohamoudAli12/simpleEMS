from openEMS.physical_constants import C0
import numpy as np
from scipy.integrate import quad
from typing import NamedTuple
from .sim_utils import m_to_mm, mm_to_m
from openEMS.physical_constants import C0

def phase_shift_length(
    phase_shift: float, dielectric: float, frequency: float
) -> float:
    """
    Compute the length (in mm) for a signal to undergo a given phase
    shift.  When computing this value for a transmission line not
    surrounded by a homogenous medium (e.g. a microstrip trace), make
    sure to use the effective dielectric.

    :param phase_shift: Phase shift in degrees.
    :param dielectric: Dielectric or effective dielectric constant.
    :param frequency: Signal frequency.
    """
    rad = phase_shift * np.pi / 180
    vac_lambda = 2 * np.pi * frequency / C0/1e3
    return rad / (np.sqrt(dielectric) * vac_lambda)

def microstrip_width_from_impedance(
    charac_imp, subs_height, copper_thickness, subs_epr, freq_hz, tolerance=0.01
):
    """
    Calculates microstrip width for a target impedance including dispersion
    and thickness using Hammerstad-Jensen equations.
    """

    def get_Z_at_freq(w, h, t, er, f_hz):
        # Constants
        eta0 = 376.730313668
        mue0 = 4 * np.pi * 1e-7
        h_m = mm_to_m(h)  # height in meters for SI units

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

        return Z_f

    # --- Binary Search for Width ---
    low, high = 0.0001 * subs_height, 100 * subs_height
    for _ in range(100):
        mid_w = (low + high) / 2
        Z_curr = get_Z_at_freq(mid_w, subs_height, copper_thickness, subs_epr, freq_hz)

        if abs(Z_curr - charac_imp) < tolerance:
            break
        if Z_curr > charac_imp:  # Needs to be wider to lower Z
            low = mid_w
        else:
            high = mid_w

    return mid_w


def conductance_G1(patch_width, frequency):
    """
    Balanis (3rd ed.) 14-12 radiation conductance of one slot.
    """

    lambda0 = C0 / frequency
    k0 = 2 * np.pi / lambda0

    def integrand(theta):
        ct = np.cos(theta)
        if abs(ct) < 1e-5:
            return 0.0
        term = np.sin((k0 * patch_width / 2) * ct) / ct
        return (term**2) * (np.sin(theta) ** 3)

    integral, _ = quad(integrand, 0, np.pi, limit=1000)
    return integral / (120 * np.pi**2)


def inset_depth(charac_imp, patch_width, patch_length, frequency):
    """
    Compute inset feed depth using Balanis formulation.
    """

    G1 = conductance_G1(patch_width, frequency)
    R_edge = 1.0 / (2.0 * G1)

    if charac_imp > R_edge:
        raise ValueError("characteristic impedance must be <= edge resistance")

    inset_length = (patch_length / np.pi) * np.arccos(np.sqrt(charac_imp / R_edge))
    probe_pos = inset_length
    return inset_length, probe_pos


def patch_dims(
    frequency_hz: float,
    subs_eps_r: float,
    subs_height: float,
    charac_imp: float,
    copper_thickness: float,
):
    """
    Calculate rectangular microstrip patch dimensions.

    Returns:
        patch_width      : Width of the patch (W)
        patch_length     : Length of the patch (L)
        inset_length     : Inset length of the patch in direction of y
        inset_width      : Inset width of the patch in the direction of x
        probe_pos        : The feed position of probe fed patch antenna
    """
    PatchDims = NamedTuple(
        "PatchDims",
        [
            ("patch_width_mm", float),
            ("patch_length_mm", float),
            ("inset_length_mm", float),
            ("inset_width_mm", float),
            ("probe_pos_mm", float),
        ],
    )

    patch_width = C0 / (2 * frequency_hz) * np.sqrt(2 / (subs_eps_r + 1))
    patch_width_mm = m_to_mm(patch_width)

    eff_eps_r = (subs_eps_r + 1) / 2 + (subs_eps_r - 1) / 2 * (
        1 / np.sqrt(1 + 12 * subs_height / patch_width)
    )

    delta_length = (
        0.412
        * subs_height
        * ((eff_eps_r + 0.3) * (patch_width / subs_height + 0.264))
        / ((eff_eps_r - 0.258) * (patch_width / subs_height + 0.8))
    )

    patch_length = (C0 / (2 * frequency_hz * np.sqrt(eff_eps_r))) - 2 * delta_length
    patch_length_mm = m_to_mm(patch_length)

    inset_length, probe_pos = inset_depth(
        charac_imp, patch_width, patch_length, frequency_hz
    )

    inset_length_mm = m_to_mm(inset_length)
    probe_pos_mm = m_to_mm(probe_pos)

    inset_width_mm = microstrip_width_from_impedance(
        charac_imp, m_to_mm(subs_height), copper_thickness, subs_eps_r, frequency_hz
    )

    return PatchDims(
        patch_width_mm, patch_length_mm, inset_length_mm, inset_width_mm, probe_pos_mm
    )
