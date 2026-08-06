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
Materials, physical constants, and region tags for the FEM backend.

Defines the :class:`Dielectric` material, a role guesser for solids named after
what they are, and the region tags that identify each part of the mesh.

The mesh and the problem file refer to each part of the structure by the same
integer tag, so the numbering lives here and both modules read it from here::

    100 + i      dielectric volume i
    200          air / vacuum volume
    210          PML shell
    300          PEC surfaces (all perfect conductors, merged)
    350 + i      lossy conductor (surface-impedance boundary) i
    400 + k      port surface k (1-indexed)
    500          outer absorbing boundary (Silver-Muller)
    600          symmetry plane
"""

from __future__ import annotations

from dataclasses import dataclass

EPS0 = 8.8541878128e-12  # F/m
MU0 = 1.25663706212e-6  # H/m
C0 = 299792458.0  # m/s
ETA0 = 376.730313668  # ohm, free-space wave impedance
SIGMA_CU = 5.8e7  # S/m

# ----------------------------
# Physical-group region IDs
# ----------------------------
# These integers are the sole contract between the Gmsh mesh and the GetDP .pro:
# fem_geometry tags mesh entities with them and fem_formulation references the
# same numbers, so both modules import them from here to stay in sync.
AIR = 200
PML = 210  # single PML shell (position-based damping handles x/y/z and corners)
PEC = 300
LOSSY_CONDUCTOR = 350  # lossy conductor (surface-impedance boundary)
ABC = 500  # outer Silver-Muller / SigmaInf surface
SYM = 600  # symmetry plane (PEC electric wall, or PMC natural)

_DIELECTRIC_BASE = 100
_PORT_BASE = 400


@dataclass
class Dielectric:
    """
    A linear isotropic dielectric.

    Parameters
    ----------
    eps_r : float
        Relative permittivity. Default ``1.0``.
    tan_d : float
        Loss tangent. Default ``0.0``.
    mu_r : float
        Relative permeability. Default ``1.0``.
    """

    eps_r: float = 1.0
    tan_d: float = 0.0
    mu_r: float = 1.0

    def eps_complex(self) -> complex:
        """
        Return the complex relative permittivity.

        Returns
        -------
        complex
            ``eps_r * (1 - 1j * tan_d)``, in the engineering sign convention.
        """
        # convenience helper only: fem_formulation writes its own permittivity
        # into the problem file, in the same (engineering) sign convention
        return self.eps_r * (1.0 - 1j * self.tan_d)


# Substring -> role, applied to lowercased solid names for auto-detection.
# Order matters: the first matching substring wins, so "port" is checked before
# the conductor hints. Otherwise a solid named e.g. "port_feed_1" would match
# "feed" and be classified PEC, silently losing the excitation.
_NAME_ROLE_HINTS: tuple[tuple[str, str], ...] = (
    ("port", "port"),
    ("substrate", "dielectric"),
    ("dielectric", "dielectric"),
    ("diel", "dielectric"),
    ("ground", "pec"),
    ("gnd", "pec"),
    ("patch", "pec"),
    ("trace", "pec"),
    ("feed", "pec"),
    ("line", "pec"),
    ("metal", "pec"),
    ("cond", "pec"),
)


def guess_role(name: str) -> str | None:
    """
    Guess what a solid is from its name.

    Parameters
    ----------
    name : str
        The solid's name, e.g. ``"substrate"`` or ``"port_1"``.

    Returns
    -------
    str | None
        ``"dielectric"``, ``"pec"``, ``"port"``, or ``None`` if the name gives
        no clue.
    """
    low = name.lower()
    for key, role in _NAME_ROLE_HINTS:
        if key in low:
            return role
    return None


def dielectric_region(index: int) -> int:
    """
    Region tag identifying a dielectric volume.

    Parameters
    ----------
    index : int
        Zero-based index of the dielectric.

    Returns
    -------
    int
        The region tag.
    """
    return _DIELECTRIC_BASE + index


def port_region(number: int) -> int:
    """
    Region tag identifying a port surface.

    Parameters
    ----------
    number : int
        One-based port number, matching the S-parameter numbering.

    Returns
    -------
    int
        The region tag.
    """
    return _PORT_BASE + number
