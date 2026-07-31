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
Material and region helpers for the FEM (GetDP) backend.

Defines the :class:`Dielectric` material, physical constants, a name-based
role guesser, the Hammerstad-Jensen microstrip impedance model, and the
canonical integer physical-group IDs shared between the Gmsh mesh and the
generated GetDP ``.pro`` problem file.

The coupling between Gmsh and GetDP is purely by integer tag:
``Physical Volume(ID)`` in the mesh is read back as ``Region[{ID}]`` in the
problem file. Centralising the numbering here keeps the geometry module and
the ``.pro`` generator from ever disagreeing::

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
        Return the complex relative permittivity in the engineering convention.

        Uses the ``exp(+j w t)`` sign (passive loss -> negative imaginary part).
        Convenience helper only: the GetDP problem file writes its own permittivity
        in the ``exp(-i w t)`` convention (positive imaginary part) directly in
        :mod:`~simpleEMS.fem_formulation`.

        Returns
        -------
        complex
            ``eps_r * (1 - 1j * tan_d)``.
        """
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
    Best-guess electromagnetic role from a solid name.

    Parameters
    ----------
    name : str
        The solid / CAD product name.

    Returns
    -------
    str | None
        ``"dielectric"``, ``"pec"``, ``"port"``, or ``None`` if no hint matches.
    """
    low = name.lower()
    for key, role in _NAME_ROLE_HINTS:
        if key in low:
            return role
    return None


def dielectric_region(index: int) -> int:
    """
    Region tag for the ``index``-th dielectric volume (0-indexed).

    Parameters
    ----------
    index : int
        Zero-based dielectric index.

    Returns
    -------
    int
        The physical-group integer tag.
    """
    return _DIELECTRIC_BASE + index


def port_region(number: int) -> int:
    """
    Region tag for port ``number`` (1-indexed, matching S-parameter numbering).

    Parameters
    ----------
    number : int
        One-based port number.

    Returns
    -------
    int
        The physical-group integer tag.
    """
    return _PORT_BASE + number
