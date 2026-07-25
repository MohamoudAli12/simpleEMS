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

__all__ = ["FEMOptions"]

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


@dataclass
class FEMOptions:
    """
    Global FEM solver/mesh options, forwarded to the GetDP problem.

    Bundles the ``Problem``-level knobs so the simpleEMS entry can carry them
    without threading many arguments. ``None``/defaults reproduce today's
    behaviour.

    Parameters
    ----------
    boundary : str
        Outer truncation: ``"silver_muller"`` (default) or ``"pml"``.
    symmetry : tuple | None
        Optional mirror-symmetry plane ``(axis, kind, at)`` with ``axis`` in
        ``x/y/z``, ``kind`` in ``pec/pmc``, ``at`` the plane coordinate (m) or
        ``None`` for the structure centre. Halves the mesh.
    fe_order : int
        Nedelec edge-element order: ``1`` (default) or ``2``.
    air_pad_frac : float
        Air padding as a fraction of the longest wavelength. Default ``0.2``.
    elems_per_wavelength : float
        Target coarse mesh density in open air. Default ``8.0``.
    mesh_fine_scale : float
        Multiplier on the near-conductor element size. Default ``1.0``.
    min_layers : int
        Element layers through the dielectric thickness. Default ``3``.
    port_type : str
        ``"lumped"`` (impedance z0, default) or ``"wave"`` (matched to the line
        characteristic impedance) for all ports.
    """

    boundary: str = "silver_muller"
    symmetry: tuple | None = None
    fe_order: int = 1
    air_pad_frac: float = 0.2
    elems_per_wavelength: float = 8.0
    mesh_fine_scale: float = 1.0
    min_layers: int = 3
    port_type: str = "lumped"

    def __post_init__(self) -> None:
        """
        Validate the option choices.

        Raises
        ------
        ValueError
            If ``boundary``, ``fe_order``, or ``port_type`` is not one of the
            supported choices.
        """
        if self.boundary not in ("silver_muller", "pml"):
            raise ValueError(
                f"boundary must be 'silver_muller' or 'pml', got {self.boundary!r}"
            )
        if self.fe_order not in (1, 2):
            raise ValueError(f"fe_order must be 1 or 2, got {self.fe_order}")
        if self.port_type not in ("lumped", "wave"):
            raise ValueError(
                f"port_type must be 'lumped' or 'wave', got {self.port_type!r}"
            )


# Substring -> role, applied to lowercased solid names for auto-detection.
_NAME_ROLE_HINTS: tuple[tuple[str, str], ...] = (
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
    ("port", "port"),
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
