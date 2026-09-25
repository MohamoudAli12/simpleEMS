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

""" """

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from CSXCAD import ContinuousStructure
from CSXCAD.CSPrimitives import CSPrimitives
from openEMS.physical_constants import EPS0
from openEMS.ports import CPWPort, LumpedPort, Port

from .fdtd_mesh import Mesh
from .sim_params import SimParams
from .sim_tools import SimTools, SimSetup


__all__ = ["GenericParams", "GenericStructure"]

COPPER_COLOR = "#B87333"
SUBSTRATE_COLOR = "#0F8A00"


def circle_points(
    cx: float, cy: float, radius: float, num_faces: int = 60
) -> list[list[float]]:
    """Generate polygon points approximating a circle.

    Parameters
    ----------
    cx, cy : float
        Centre of the circle.
    radius : float
        Circle radius.
    num_faces : int, optional
        Number of polygon vertices. Default ``60``.

    Returns
    -------
    list[list[float]]
        ``[x_coords, y_coords]``, each a list of ``num_faces`` values.
    """
    angles = np.linspace(0, 2 * np.pi, num_faces, endpoint=False)
    x_coords = (cx + radius * np.cos(angles)).tolist()
    y_coords = (cy + radius * np.sin(angles)).tolist()
    return [x_coords, y_coords]


@dataclass(kw_only=True)
class GenericParams(SimParams):
    """Generic parameters for arbitrary structures built with primitives.

    Unlike the structure-specific subclasses (InsetFedPatchParams,
    MicrostripLineParams, etc.), this class lets the user provide
    substrate dimensions and frequency parameters directly, making it
    suitable for any geometry assembled from the primitives in
    :class:`GenericStructure`.

    Parameters
    ----------
    min_freq : float
        Minimum frequency of the simulation range in Hz.
    max_freq : float
        Maximum frequency of the simulation range in Hz.
    target_freq : float
        Frequency the structure is designed for, in Hz. It sizes the mesh.
    substrate_width_mm : float
        Substrate width in mm.
    substrate_length_mm : float
        Substrate length in mm.
    """

    min_freq: float
    max_freq: float
    target_freq: float
    substrate_width_mm: float
    substrate_length_mm: float

    @property
    def freq_range(self) -> tuple[float, float]:
        """
        Return the simulation frequency range.

        Returns
        -------
        tuple[float, float]
            ``(min_freq, max_freq)`` in Hz.
        """
        return self.min_freq, self.max_freq

    @property
    def main_freq(self) -> float:
        """
        Return the primary frequency of interest.

        Returns
        -------
        float
            ``target_freq`` in Hz.
        """
        return self.target_freq


class CPWLumpedPort(Port):
    """A coplanar waveguide port fed by one lumped port per slot.

    openEMS's own :class:`~openEMS.ports.CPWPort` draws every part of itself --
    the probes, the excitation, the terminations -- as a sheet on a single
    ``z`` plane, which models a coplanar waveguide whose conductors have no
    thickness. Give it a line with real copper on it and the sheet lands on
    one face of that copper, the current probe's loop straddles a conductor it
    cannot see the extent of, and the impedance it reports drifts.

    This port instead fills each slot: two lumped ports, one per slot, each
    box spanning the gap, the length of the port along the run, and the full
    conductor thickness in ``z``. Both are oriented signal pour -> ground
    pour, so their excitation vectors point in opposite directions along the
    slot axis and the mode driven is the odd one the line actually carries.
    Each slot is terminated in ``2 * impedance``; in parallel they present
    ``impedance`` to the line.

    The two are reported as one port. :class:`~openEMS.ports.Port` already
    sums every file in ``U_filenames`` and ``I_filenames``, which is right for
    the currents -- the slots conduct in parallel and their currents add --
    but not for the voltages, which are one trace-to-ground potential measured
    twice, so :func:`ReadUIData` halves their sum. Everything downstream
    (``uf_inc``, ``uf_ref``, ``Z_ref``, the powers) is then the inherited
    :func:`~openEMS.ports.Port.CalcPort` arithmetic on a plain one-port at
    ``impedance``, which is what makes this object usable anywhere an openEMS
    port is: :func:`SimTools.compute_sim_data` needs no special case for it.

    Build one with :func:`GenericStructure.create_cpw_lumped_port` rather than
    directly.

    Parameters
    ----------
    CSX : ContinuousStructure
        The structure the slot ports were added to.
    port_nr : int
        Port number (1-indexed). Shared by both slots -- they are one port.
    start, stop : list[float]
        ``[x, y, z]`` opposite corners of the signal trace the port sits on,
        as passed to :func:`GenericStructure.create_cpw_lumped_port`.
    excite : float
        Excitation amplitude the slots were given.
    slots : list[LumpedPort]
        The two lumped ports, one per slot.
    impedance : float
        Reference impedance of the pair, in ohms. Each slot carries twice it.

    See Also
    --------
    openEMS.ports.Port, openEMS.ports.LumpedPort, openEMS.ports.CPWPort
    """

    def __init__(
        self,
        CSX: ContinuousStructure,
        port_nr: int,
        start: list[float],
        stop: list[float],
        excite: float,
        slots: list[LumpedPort],
        impedance: float,
    ) -> None:
        super().__init__(CSX, port_nr, start, stop, excite)
        self.slots = list(slots)
        # The resistance actually in the model is per slot; the pair in
        # parallel is what the S-parameters are referenced to.
        self.R = 2 * impedance
        self.Z_ref = impedance
        self.U_filenames = [fn for slot in self.slots for fn in slot.U_filenames]
        self.I_filenames = [fn for slot in self.slots for fn in slot.I_filenames]
        # Carried over so SetEnabled() finds both slots' excitations, which is
        # how param_sweep and the optimisers drive one port at a time.
        self.port_props = [prop for slot in self.slots for prop in slot.port_props]

    def ReadUIData(
        self, sim_path: str, freq: NDArray, signal_type: str = "pulse"
    ) -> None:
        """Read both slots' probe files and combine them into one port.

        Extends :func:`~openEMS.ports.Port.ReadUIData`, which sums every file
        it is given. The currents are left summed: the two slots carry the
        return current in parallel. The voltages are halved, because each slot
        probe measures the same trace-to-ground potential and summing them
        counts it twice.

        Parameters
        ----------
        sim_path : str
            Directory the simulation wrote its probe files to.
        freq : NDArray
            Frequencies to evaluate the probes at, in Hz.
        signal_type : str, optional
            Passed through to openEMS's DFT. Default ``"pulse"``.

        Returns
        -------
        None
        """
        super().ReadUIData(sim_path, freq, signal_type)
        self.uf_tot = 0.5 * self.uf_tot
        self.ut_tot = 0.5 * self.ut_tot

    def CalcPort(  # noqa: N802 -- openEMS's own port interface
        self,
        sim_path: str,
        freq: NDArray,
        ref_impedance: float | None = None,
        ref_plane_shift: float | None = None,
        signal_type: str = "pulse",
    ) -> None:
        """Compute the port's incident and reflected waves.

        Parameters
        ----------
        sim_path : str
            Directory the simulation wrote its probe files to.
        freq : NDArray
            Frequencies to evaluate the port at, in Hz.
        ref_impedance : float | None, optional
            Impedance to reference the waves to. ``None`` (default) keeps the
            ``impedance`` the port was built with.
        ref_plane_shift : float | None, optional
            Not supported; see Raises.
        signal_type : str, optional
            Passed through to openEMS's DFT. Default ``"pulse"``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``ref_plane_shift`` is given. De-embedding needs the
            propagation constant, and a lumped feed measures the voltage and
            current at one plane only -- there is no second plane to get a
            phase difference from. Place the port where the reference plane
            belongs instead, or use :func:`GenericStructure.create_cpw_port`,
            whose three probe planes do measure ``beta``.
        """
        if ref_plane_shift is not None:
            raise ValueError(
                "a lumped-fed CPW port measures at one plane and so has no "
                "propagation constant to de-embed with: put the port's own "
                "start where the reference plane belongs, or use "
                "create_cpw_port(), whose probes span three planes"
            )
        super().CalcPort(sim_path, freq, ref_impedance, None, signal_type)


class GenericStructure(SimTools):
    """Arbitrary structure assembled from CSXCAD primitives.

    Each ``create_*`` method adds one primitive to the CSXCAD structure and
    returns it, so a caller can transform it further or register its edges
    with the FDTD grid. Shared quantities -- the trace elevation, the copper
    thickness, the port impedance, the substrate footprint -- default to the
    matching value on ``params`` and only need to be passed when a primitive
    differs from it.

    Parameters
    ----------
    params : GenericParams
        Substrate, frequency and mesh parameters.
    sim : SimSetup
        Named tuple containing the CSXCAD geometry and openEMS FDTD engine.
    """

    def __init__(
        self,
        params: GenericParams,
        sim: SimSetup,
    ) -> None:
        self.params = params
        self.sim = sim
        self.CSX = sim.CSX
        self.FDTD = sim.FDTD
        self._requested_mesh_lines: list[list[float]] = [[], [], []]

    def _request_mesh_lines(self, dim: int, positions: list[float]) -> None:
        """Ask :func:`create_mesh` for a mesh line at each of ``positions``
        along dimension ``dim`` (``0`` x, ``1`` y, ``2`` z)."""
        self._requested_mesh_lines[dim].extend(float(pos) for pos in positions)

    @staticmethod
    def _apply_rotation(
        prim: CSPrimitives, cx: float, cy: float, rotation: float
    ) -> CSPrimitives:
        """Rotate ``prim`` by ``rotation`` degrees CCW about ``(cx, cy)``."""
        if rotation != 0:
            prim.AddTransform("Translate", [-cx, -cy, 0])
            prim.AddTransform("RotateAxis", "z", rotation)
            prim.AddTransform("Translate", [cx, cy, 0])
        return prim

    @staticmethod
    def _rotate_point(
        x: float, y: float, cx: float, cy: float, rotation: float
    ) -> tuple[float, float]:
        """Return ``(x, y)`` rotated ``rotation`` degrees CCW about ``(cx, cy)``."""
        theta = np.radians(rotation)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        return (
            cx + (x - cx) * cos_t - (y - cy) * sin_t,
            cy + (x - cx) * sin_t + (y - cy) * cos_t,
        )

    def _elevation(self, z_elevation_mm: float | None) -> float:
        """Resolve a trace elevation, defaulting to the substrate's top face."""
        if z_elevation_mm is None:
            return self.params.substrate_thickness_mm
        return z_elevation_mm

    def _thickness(self, copper_thickness_mm: float | None) -> float:
        """Resolve a copper thickness, defaulting to the parameter value."""
        if copper_thickness_mm is None:
            return self.params.copper_thickness_mm
        return copper_thickness_mm

    def create_via(
        self,
        name: str = "via",
        *,
        position: tuple[float, float],
        via_diameter_mm: float,
        z_bottom_mm: float,
        z_top_mm: float,
        antipad_diameter_mm: float | None = None,
        antipad_layers: list[float] | None = None,
        antipad_thickness_mm: float | None = None,
        antipad_eps_r: float | None = None,
        priority: int = 6,
        antipad_priority: int = 4,
    ) -> CSPrimitives:
        """Create a plated through-hole via as a solid conductor.

        The barrel is a solid metal cylinder rather than a plated shell around a
        drilled hole.  Copper's skin depth is a fraction of a micron at these
        frequencies, so a plated wall and a solid barrel are the same conductor
        seen from outside, while the wall thickness would otherwise be a feature
        orders of magnitude finer than the requested mesh resolution -- and the
        FDTD timestep follows the smallest cell in the grid.

        No pad or annular ring is drawn; the barrel lands directly on whatever
        copper is already at its ends.  Clearances are placed only at the
        elevations named in ``antipad_layers``, and each is a ring of dielectric
        -- the substrate by default -- with the same thickness as the copper
        layer it cuts through; a zero-thickness sheet has no volume for the grid
        to resolve and would not reliably clear the plane.

        The ring starts at the barrel wall and runs outwards to
        ``antipad_diameter_mm``, so it never overlaps the via.  A solid disc
        would work in FDTD, where the barrel's higher priority wins inside the
        hole, but the STEP export carries no priorities: the FEM path sees only
        the geometry, and a dielectric disc intersecting the metal barrel is an
        ambiguous solid for Gmsh to mesh.  An annulus is unambiguous for both.

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the barrel is added to, e.g.
            ``"short_via"``. Clearance discs go on ``"<name>_antipad"``.
            Default ``"via"``.
        position : tuple[float, float]
            ``(x, y)`` centre of the via.
        via_diameter_mm : float
            Finished outer diameter of the conductor -- the drill diameter plus
            twice the plating thickness, as a fab would state it.
        z_bottom_mm, z_top_mm : float
            Extent of the barrel along z.
        antipad_diameter_mm : float | None
            Outer diameter of the clearance rings; their inner diameter is
            ``via_diameter_mm``.  Required when ``antipad_layers`` is given, and
            must exceed ``via_diameter_mm``.  Default ``None``.
        antipad_layers : list[float] | None
            Elevations at which to place a clearance ring, each the **bottom**
            face of the copper layer being cleared.  Default ``None``, no
            antipads.
        antipad_thickness_mm : float | None, optional
            Thickness of each clearance ring along z.  Defaults to
            ``params.copper_thickness_mm``, i.e. the thickness of the plane it
            cuts.
        antipad_eps_r : float | None, optional
            Relative permittivity filling the clearance rings.  Defaults to
            ``params.substrate_eps_r`` (with its loss), which suits a plane
            buried in the stack-up.  Pass ``1.0`` for a plane on an outer face,
            where the clearance opens to air.
        priority : int, optional
            CSXCAD priority of the barrel; where primitives overlap, the higher
            priority wins. Default ``6``.
        antipad_priority : int, optional
            CSXCAD priority of the clearance rings. Keep it above the plane
            being cleared so the ring wins over the copper it clears.
            Default ``4``.

        Returns
        -------
        CSPrimitives
            The via barrel primitive.

        Raises
        ------
        ValueError
            If ``antipad_layers`` is given without an ``antipad_diameter_mm``, or the
            antipad is no larger than the via.
        """
        if antipad_layers is None:
            antipad_layers = []
        if antipad_layers:
            if antipad_diameter_mm is None:
                raise ValueError("antipad_layers given without an antipad_diameter_mm")
            if antipad_diameter_mm <= via_diameter_mm:
                raise ValueError(
                    f"antipad_diameter_mm must exceed via_diameter_mm, got "
                    f"{antipad_diameter_mm} <= {via_diameter_mm}"
                )

        x, y = position

        via = self.CSX.AddMetal(name)
        via.SetColor(COPPER_COLOR, 255)
        barrel = via.AddCylinder(
            priority=priority,
            start=[x, y, z_bottom_mm],
            stop=[x, y, z_top_mm],
            radius=via_diameter_mm / 2,
        )

        antipad_thickness_mm = self._thickness(antipad_thickness_mm)
        if antipad_eps_r is None:
            antipad_eps_r = self.params.substrate_eps_r
            antipad_kappa = self.params.substrate_kappa
        else:
            antipad_kappa = 0.0
        if antipad_layers:
            # CSXCAD describes a shell by its mid-wall radius and wall width, so
            # a ring spanning the barrel wall out to antipad_diameter_mm sits at
            # the mean of the two radii and is half their difference thick.
            antipad_wall_mm = (antipad_diameter_mm - via_diameter_mm) / 2
            antipad_radius_mm = (antipad_diameter_mm + via_diameter_mm) / 4
            antipad = self.CSX.AddMaterial(
                f"{name}_antipad", epsilon=antipad_eps_r, kappa=antipad_kappa
            )
            antipad.SetColor(SUBSTRATE_COLOR, 50)
            for z_layer in antipad_layers:
                antipad.AddCylindricalShell(
                    priority=antipad_priority,
                    start=[x, y, z_layer],
                    stop=[x, y, z_layer + antipad_thickness_mm],
                    radius=antipad_radius_mm,
                    shell_width=antipad_wall_mm,
                )

        return barrel

    def create_taper(
        self,
        name: str = "taper",
        *,
        position: tuple[float, float],
        width1_mm: float,
        width2_mm: float,
        length_mm: float,
        z_elevation_mm: float | None = None,
        copper_thickness_mm: float | None = None,
        rotation: float = 0,
        priority: int = 6,
    ) -> CSPrimitives:
        """Create a trapezoidal taper trace.

        ``position`` is the midpoint of the input edge, the one ``width1_mm``
        wide, so the taper sits wholly beyond it and butts flush against the
        end of the incoming arm.  In the unrotated frame the taper runs
        ``+y``: it is ``width1_mm`` wide at ``y = cy`` and ``width2_mm`` wide at
        ``y = cy + length_mm``, which is where the next primitive starts.

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the taper is added to. Default
            ``"taper"``.
        position : tuple[float, float]
            ``(x, y)`` midpoint of the input edge, in the unrotated frame.
        width1_mm, width2_mm : float
            Trace width at the input (``position``) and output ends.
        length_mm : float
            Taper length along y, in the unrotated frame.
        z_elevation_mm : float | None, optional
            Z position of the trace's bottom face. Defaults to the substrate's
            top face.
        copper_thickness_mm : float | None, optional
            Trace thickness along z. Defaults to ``params.copper_thickness_mm``.
        rotation : float, optional
            Rotation in degrees, counter-clockwise about ``position``. Default
            ``0``.
        priority : int, optional
            CSXCAD priority of the primitive; where primitives overlap, the
            higher priority wins. Default ``6``.

        Returns
        -------
        CSPrimitives
            The taper primitive.
        """
        cx, cy = position
        z_elevation_mm = self._elevation(z_elevation_mm)
        copper_thickness_mm = self._thickness(copper_thickness_mm)
        y0 = cy
        y1 = cy + length_mm

        points = [
            [
                cx - width1_mm / 2,
                cx + width1_mm / 2,
                cx + width2_mm / 2,
                cx - width2_mm / 2,
            ],
            [y0, y0, y1, y1],
        ]

        taper = self.CSX.AddMetal(name)
        taper.SetColor(COPPER_COLOR, 255)
        taper_prim = taper.AddLinPoly(
            points, "z", z_elevation_mm, copper_thickness_mm, priority=priority
        )
        return self._apply_rotation(taper_prim, cx, cy, rotation)

    def create_miter(
        self,
        name: str = "miter",
        *,
        position: tuple[float, float],
        width_mm: float,
        miter_distance_mm: float,
        z_elevation_mm: float | None = None,
        copper_thickness_mm: float | None = None,
        rotation: float = 0,
        turn: str = "left",
        priority: int = 6,
    ) -> CSPrimitives:
        """Create a 90-degree mitered bend.

        The primitive is the square corner block that joins two ``width_mm``-wide
        arms; the arms themselves are separate (see ``create_microstrip``).
        ``position`` is the midpoint of the input edge, so the block sits wholly
        beyond it and butts flush against the end of the incoming arm rather than
        straddling it.  In the unrotated frame the arm arrives travelling ``+y``,
        the block spans ``y = cy .. cy + width_mm``, the outgoing arm leaves
        through the ``-x`` edge with its centre line at ``position +
        (-width_mm/2, +width_mm/2)``, and the outer corner is
        ``(+hw, cy + width_mm)``.  ``rotation`` is degrees CCW about
        ``position``, which keeps the join flush at any angle.

        ``turn`` is the handedness of the bend as seen by a wave travelling along
        the incoming arm: ``"left"`` exits to the arm's left (``-x`` at
        ``rotation=0``), ``"right"`` to its right (``+x``).  The two are mirror
        images, so ``rotation`` alone cannot swap them; being defined relative to
        the direction of travel, ``turn`` keeps its meaning at any ``rotation``.

        ``miter_distance_mm`` is the cutback from the outer corner along each edge,
        in the same units as ``width_mm``, and must lie in ``[0, width_mm]``.  It maps
        to the usual miter percentage as ``M = miter_distance_mm / (2*width_mm)`` —
        ``M = 0`` is an unmitered square, ``M = 0.5`` cuts the full diagonal
        (a triangle).

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the miter is added to. Default
            ``"miter"``.
        position : tuple[float, float]
            ``(x, y)`` midpoint of the input edge, in the unrotated frame.
        width_mm : float
            Width of the arms joined by the miter.
        miter_distance_mm : float
            Cutback from the outer corner, in ``[0, width_mm]``.
        z_elevation_mm : float | None, optional
            Z position of the miter's bottom face. Defaults to the substrate's
            top face.
        copper_thickness_mm : float | None, optional
            Miter thickness along z. Defaults to ``params.copper_thickness_mm``.
        rotation : float, optional
            Rotation in degrees, counter-clockwise about ``position``. Default
            ``0``.
        turn : str, optional
            Bend handedness, ``"left"`` or ``"right"``. Default ``"left"``.
        priority : int, optional
            CSXCAD priority of the primitive; where primitives overlap, the
            higher priority wins. Default ``6``.

        Returns
        -------
        CSPrimitives
            The miter primitive.

        Raises
        ------
        ValueError
            If ``miter_distance_mm`` lies outside ``[0, width_mm]``, or ``turn`` is
            neither ``"left"`` nor ``"right"``.
        """
        cx, cy = position
        z_elevation_mm = self._elevation(z_elevation_mm)
        copper_thickness_mm = self._thickness(copper_thickness_mm)
        hw = width_mm / 2
        d = miter_distance_mm

        if not 0 <= d <= width_mm:
            raise ValueError(
                f"miter_distance_mm must lie within [0, {width_mm}], "
                f"got {miter_distance_mm}"
            )
        if turn not in ("left", "right"):
            raise ValueError(f"turn must be 'left' or 'right', got {turn!r}")

        # Square corner block with its outer (+x, far) corner replaced by a
        # chamfer running from (+hw - d, y1) to (+hw, y1 - d).  ``cy`` is the
        # input edge, so the block reaches a full ``width_mm`` beyond ``position``.
        y0 = cy
        y1 = cy + width_mm
        verts = [
            (cx - hw, y0),
            (cx - hw, y1),
            (cx + hw - d, y1),
            (cx + hw, y1 - d),
            (cx + hw, y0),
        ]
        if turn == "right":
            # A right turn is the mirror image about the input edge's normal.
            # Reverse as well, so the winding matches the left-hand version.
            verts = [(2 * cx - vx, vy) for vx, vy in reversed(verts)]
        # d == 0 (unmitered) and d == width_mm (full diagonal cut) collapse the
        # chamfer onto a corner; drop the coincident vertex so CSXCAD gets a clean
        # polygon.
        verts = [
            v for i, v in enumerate(verts) if i == 0 or not np.allclose(v, verts[i - 1])
        ]

        points = [[v[0] for v in verts], [v[1] for v in verts]]

        miter = self.CSX.AddMetal(name)
        miter.SetColor(COPPER_COLOR, 255)
        prim = miter.AddLinPoly(
            points, "z", z_elevation_mm, copper_thickness_mm, priority=priority
        )
        return self._apply_rotation(prim, cx, cy, rotation)

    def create_curved_bend(
        self,
        name: str = "curved_bend",
        *,
        position: tuple[float, float],
        width_mm: float,
        bend_radius_mm: float,
        z_elevation_mm: float | None = None,
        copper_thickness_mm: float | None = None,
        rotation: float = 0,
        turn: str = "left",
        bend_angle: float = 90,
        num_segments: int = 60,
        priority: int = 6,
    ) -> CSPrimitives:
        """Create a curved (circular-arc) bend.

        An alternative to :func:`create_miter` that joins two ``width_mm``-wide
        arms with a constant-width band swept along a circular arc instead of a
        mitered corner block.  The primitive is an annular sector: the outer
        edge is an arc of radius ``bend_radius_mm + width_mm/2`` and the inner edge an
        arc of radius ``bend_radius_mm - width_mm/2``, both centred on the bend's
        centre of curvature; ``bend_radius_mm`` itself is the radius of the arm's
        centreline. ``position`` is the midpoint of the input edge, so the band
        butts flush against the end of the incoming arm rather than straddling
        it. In the unrotated frame the arm arrives travelling ``+y``,
        the centre of curvature sits at ``(cx - bend_radius_mm, cy)``, and the arc
        sweeps ``bend_angle`` degrees from there. ``rotation`` is degrees CCW
        about ``position``, which keeps the join flush at any angle.

        ``turn`` is the handedness of the bend as seen by a wave travelling
        along the incoming arm, with the same meaning as in
        :func:`create_miter`: ``"left"`` curves towards the arm's left (``-x``
        at ``rotation=0``), ``"right"`` towards its right (``+x``). The two are
        mirror images, so ``rotation`` alone cannot swap them.

        The outgoing arm's input edge -- where the next primitive should
        start -- is centred at
        ``(cx - bend_radius_mm + bend_radius_mm*cos(a), cy + bend_radius_mm*sin(a))``
        for ``turn="left"``, or the mirror image about ``cx`` for
        ``turn="right"``, with ``a = radians(bend_angle)`` before ``rotation``
        is applied.

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the bend is added to. Default
            ``"curved_bend"``.
        position : tuple[float, float]
            ``(x, y)`` midpoint of the input edge, in the unrotated frame.
        width_mm : float
            Width of the arms joined by the bend.
        bend_radius_mm : float
            Radius of the arm's centreline through the bend. Must exceed
            ``width_mm / 2`` so the inner edge does not cross the centre of
            curvature.
        z_elevation_mm : float | None, optional
            Z position of the bend's bottom face. Defaults to the substrate's
            top face.
        copper_thickness_mm : float | None, optional
            Bend thickness along z. Defaults to ``params.copper_thickness_mm``.
        rotation : float, optional
            Rotation in degrees, counter-clockwise about ``position``. Default
            ``0``.
        turn : str, optional
            Bend handedness, ``"left"`` or ``"right"``. Default ``"left"``.
        bend_angle : float, optional
            Angle swept by the arc, in degrees, in ``(0, 360]``. Default ``90``.
        num_segments : int, optional
            Number of polygon segments approximating each arc. Default ``60``.
        priority : int, optional
            CSXCAD priority of the primitive; where primitives overlap, the
            higher priority wins. Default ``6``.

        Returns
        -------
        CSPrimitives
            The bend primitive.

        Raises
        ------
        ValueError
            If ``bend_radius_mm`` does not exceed ``width_mm / 2``, ``turn`` is
            neither ``"left"`` nor ``"right"``, or ``bend_angle`` lies outside
            ``(0, 360]``.
        """
        cx, cy = position
        z_elevation_mm = self._elevation(z_elevation_mm)
        copper_thickness_mm = self._thickness(copper_thickness_mm)

        if bend_radius_mm <= width_mm / 2:
            raise ValueError(
                f"bend_radius_mm must exceed width_mm / 2 ({width_mm / 2}), "
                f"got {bend_radius_mm}"
            )
        if turn not in ("left", "right"):
            raise ValueError(f"turn must be 'left' or 'right', got {turn!r}")
        if not 0 < bend_angle <= 360:
            raise ValueError(f"bend_angle must lie within (0, 360], got {bend_angle}")

        r_in = bend_radius_mm - width_mm / 2
        r_out = bend_radius_mm + width_mm / 2
        ox = cx - bend_radius_mm
        oy = cy

        angles = np.linspace(0, np.radians(bend_angle), num_segments)

        x_inner = ox + r_in * np.cos(angles)
        y_inner = oy + r_in * np.sin(angles)
        x_outer = ox + r_out * np.cos(angles[::-1])
        y_outer = oy + r_out * np.sin(angles[::-1])

        x_coords = np.concatenate([x_inner, x_outer]).tolist()
        y_coords = np.concatenate([y_inner, y_outer]).tolist()

        if turn == "right":
            # A right turn is the mirror image about the input edge's normal.
            # Reverse as well, so the winding matches the left-hand version.
            x_coords = [2 * cx - x for x in reversed(x_coords)]
            y_coords = list(reversed(y_coords))

        points = [x_coords, y_coords]

        bend = self.CSX.AddMetal(name)
        bend.SetColor(COPPER_COLOR, 255)
        prim = bend.AddLinPoly(
            points, "z", z_elevation_mm, copper_thickness_mm, priority=priority
        )
        return self._apply_rotation(prim, cx, cy, rotation)

    def create_radial_stub(
        self,
        name: str = "radial_stub",
        *,
        position: tuple[float, float],
        inner_radius_mm: float,
        outer_radius_mm: float,
        angle_start: float,
        angle_end: float,
        z_elevation_mm: float | None = None,
        copper_thickness_mm: float | None = None,
        rotation: float = 0,
        num_segments: int = 60,
        priority: int = 6,
    ) -> CSPrimitives:
        """Create a radial (fan-shaped) stub.

        ``position`` is the midpoint of the stub's neck (its flat inner edge),
        so the copper starts exactly there.  ``angle_start``/``angle_end`` are
        in degrees, measured from ``+x``, and set the fan opening; the fan
        points along their bisector, and ``rotation`` turns the whole fan a
        further amount CCW about ``position``.  With ``half_span`` half the
        opening, the neck is ``2*inner_radius_mm*sin(half_span)`` wide and the
        stub reaches ``outer_radius_mm - inner_radius_mm*cos(half_span)`` radially
        beyond ``position``.

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the stub is added to. Default
            ``"radial_stub"``.
        position : tuple[float, float]
            ``(x, y)`` midpoint of the stub's neck, in the unrotated frame.
        inner_radius_mm, outer_radius_mm : float
            Radii of the fan's inner (neck) and outer edges.
        angle_start, angle_end : float
            Fan opening bounds, in degrees, measured from ``+x``.
        z_elevation_mm : float | None, optional
            Z position of the stub's bottom face. Defaults to the substrate's
            top face.
        copper_thickness_mm : float | None, optional
            Stub thickness along z. Defaults to ``params.copper_thickness_mm``.
        rotation : float, optional
            Rotation in degrees, counter-clockwise about ``position``. Default
            ``0``.
        num_segments : int, optional
            Number of polygon segments approximating the outer arc. Default
            ``60``.
        priority : int, optional
            CSXCAD priority of the primitive; where primitives overlap, the
            higher priority wins. Default ``6``.

        Returns
        -------
        CSPrimitives
            The radial stub primitive.
        """
        cx, cy = position
        z_elevation_mm = self._elevation(z_elevation_mm)
        copper_thickness_mm = self._thickness(copper_thickness_mm)
        angle_start = np.radians(angle_start)
        angle_end = np.radians(angle_end)
        half_span = (angle_end - angle_start) / 2
        bisector = (angle_start + angle_end) / 2

        # ``position`` marks the neck midpoint.  The fan's apex lies
        # inner_radius_mm*cos(half_span) behind it along the bisector, which puts the
        # flat inner chord exactly on ``position``.  Rotation still pivots about
        # ``position``, so the neck stays put whatever the rotation.
        neck_offset = inner_radius_mm * np.cos(half_span)
        ax = cx - neck_offset * np.cos(bisector)
        ay = cy - neck_offset * np.sin(bisector)

        angles = np.linspace(angle_start, angle_end, num_segments)

        x_inner = [
            ax + inner_radius_mm * np.cos(angle_start),
            ax + inner_radius_mm * np.cos(angle_end),
        ]
        y_inner = [
            ay + inner_radius_mm * np.sin(angle_start),
            ay + inner_radius_mm * np.sin(angle_end),
        ]
        x_outer = ax + outer_radius_mm * np.cos(angles[::-1])
        y_outer = ay + outer_radius_mm * np.sin(angles[::-1])

        x_coords = np.concatenate([x_inner, x_outer]).tolist()
        y_coords = np.concatenate([y_inner, y_outer]).tolist()

        points = [x_coords, y_coords]

        stub = self.CSX.AddMetal(name)
        stub.SetColor(COPPER_COLOR, 255)
        radial_prim = stub.AddLinPoly(
            points, "z", z_elevation_mm, copper_thickness_mm, priority=priority
        )
        return self._apply_rotation(radial_prim, cx, cy, rotation)

    def create_microstrip(
        self,
        name: str = "microstrip",
        *,
        position: tuple[float, float],
        width_mm: float,
        length_mm: float,
        z_elevation_mm: float | None = None,
        copper_thickness_mm: float | None = None,
        rotation: float = 0,
        priority: int = 6,
    ) -> CSPrimitives:
        """Create a straight rectangular microstrip trace.

        ``position`` is the midpoint of the input edge, so the trace starts
        exactly there and butts flush against whatever precedes it.  In the
        unrotated frame it runs ``+y``, spanning ``x = cx +/- width_mm/2`` and
        ``y = cy .. cy + length_mm``; the output edge -- where the next
        primitive starts -- is the midpoint of ``y = cy + length_mm`` before
        ``rotation`` is applied.

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the trace is added to, e.g. ``"feed"``
            -- one per trace, so each piece is its own part in the viewer and
            in the STEP and Gerber exports. Default ``"microstrip"``.
        position : tuple[float, float]
            ``(x, y)`` midpoint of the input edge, in the unrotated frame.
        width_mm : float
            Trace width, across the direction of travel.
        length_mm : float
            Trace length along y, in the unrotated frame.
        z_elevation_mm : float | None, optional
            Z position of the trace's bottom face. Defaults to the substrate's
            top face.
        copper_thickness_mm : float | None, optional
            Trace thickness along z. Defaults to ``params.copper_thickness_mm``.
        rotation : float, optional
            Rotation in degrees, counter-clockwise about ``position``. Default
            ``0``.
        priority : int, optional
            CSXCAD priority of the primitive; where primitives overlap, the
            higher priority wins. Default ``6``.

        Returns
        -------
        CSPrimitives
            The trace primitive.
        """
        cx, cy = position
        z_elevation_mm = self._elevation(z_elevation_mm)
        copper_thickness_mm = self._thickness(copper_thickness_mm)

        trace = self.CSX.AddMetal(name)
        trace.SetColor(COPPER_COLOR, 255)
        prim = trace.AddBox(
            priority=priority,
            start=[cx - width_mm / 2, cy, z_elevation_mm],
            stop=[
                cx + width_mm / 2,
                cy + length_mm,
                z_elevation_mm + copper_thickness_mm,
            ],
        )
        return self._apply_rotation(prim, cx, cy, rotation)

    def create_cpw(
        self,
        name: str = "cpw",
        *,
        position: tuple[float, float],
        trace_width_mm: float,
        gap_mm: float,
        ground_width_mm: float,
        length_mm: float,
        z_elevation_mm: float | None = None,
        copper_thickness_mm: float | None = None,
        rotation: float = 0,
        priority: int = 6,
    ) -> tuple[CSPrimitives, CSPrimitives, CSPrimitives]:
        """Create a straight coplanar waveguide (CPW) segment.

        Three parallel copper pours sharing ``z_elevation_mm``/``copper_thickness_mm``:
        a central signal trace of width ``trace_width_mm``, flanked on both sides
        by a slot of width ``gap_mm`` and then a ground pour of width
        ``ground_width_mm``. Nothing is drawn underneath -- this is the plain,
        ungrounded line, whose return path is the two coplanar pours alone.

        A ground plane does not belong under it on its own. Pours that are not
        tied to that plane are free to drift apart in potential, and the
        parallel-plate mode between them and it starts to carry power; a plane
        comes with the stitching vias that hold the pours to it, which is
        :func:`create_gcpw`, the conductor-backed line.

        Feed a CPW segment with :func:`create_cpw_port`, not a lumped port:
        the mode lives in the two slots, and a lumped port can only drive one
        gap at a single point.

        ``position`` is the midpoint of the signal trace's input edge, so the
        segment butts flush against the end of the incoming line.  In the
        unrotated frame it runs ``+y``, with the trace centred on ``cx`` and
        the two pours flanking it in ``x``; the output edge -- where the next
        primitive starts -- is ``(cx, cy + length_mm)`` before ``rotation`` is
        applied.

        The segment also asks :func:`create_mesh` for four mesh lines across
        each slot -- its two edges plus the interior -- which the automatic
        mesher would otherwise leave unresolved: a slot is thinner than the
        resolution at which it stops subdividing an interval, so it would get
        one line at its midpoint, which is both too coarse for the region the
        mode lives in and makes every box :func:`create_cpw_port` hands
        openEMS a box with no extent. A
        ``rotation`` that leaves the slots off-axis cannot be expressed as an
        axis-aligned line, so nothing is requested for it and
        :func:`create_cpw_port` raises on that grid.

        Parameters
        ----------
        name : str, optional
            Base name for the segment's properties: the trace goes on
            ``"<name>_trace"`` and the pours on ``"<name>_ground_1"`` and
            ``"<name>_ground_2"``. Default ``"cpw"``.
        position : tuple[float, float]
            ``(x, y)`` midpoint of the trace's input edge, in the unrotated
            frame.
        trace_width_mm : float
            Width of the central signal trace.
        gap_mm : float
            Width of the slot separating the signal trace from each ground
            pour.
        ground_width_mm : float
            Width of each of the two ground pours.
        length_mm : float
            Segment length along y, in the unrotated frame.
        z_elevation_mm : float | None, optional
            Z position of the conductors' bottom face. Defaults to the
            substrate's top face.
        copper_thickness_mm : float | None, optional
            Conductor thickness along z. Defaults to
            ``params.copper_thickness_mm``.
        rotation : float, optional
            Rotation in degrees, counter-clockwise about ``position``. Default
            ``0``.
        priority : int, optional
            CSXCAD priority of the trace and both pours; where primitives
            overlap, the higher priority wins. Default ``6``.

        Returns
        -------
        tuple[CSPrimitives, CSPrimitives, CSPrimitives]
            The signal trace and the two ground pours, in the order
            ``(trace, ground_lo, ground_hi)`` (``lo``/``hi`` by x in the
            unrotated frame).
        """
        cx, cy = position
        z_elevation_mm = self._elevation(z_elevation_mm)
        copper_thickness_mm = self._thickness(copper_thickness_mm)
        y0 = cy
        y1 = cy + length_mm
        z0 = z_elevation_mm
        z1 = z_elevation_mm + copper_thickness_mm

        x_trace0 = cx - trace_width_mm / 2
        x_trace1 = cx + trace_width_mm / 2
        x_ground_lo0 = x_trace0 - gap_mm - ground_width_mm
        x_ground_lo1 = x_trace0 - gap_mm
        x_ground_hi0 = x_trace1 + gap_mm
        x_ground_hi1 = x_trace1 + gap_mm + ground_width_mm

        trace = self.CSX.AddMetal(f"{name}_trace")
        trace.SetColor(COPPER_COLOR, 255)
        trace_prim = trace.AddBox(
            priority=priority, start=[x_trace0, y0, z0], stop=[x_trace1, y1, z1]
        )

        ground_lo = self.CSX.AddMetal(f"{name}_ground_1")
        ground_lo.SetColor(COPPER_COLOR, 255)
        ground_lo_prim = ground_lo.AddBox(
            priority=priority,
            start=[x_ground_lo0, y0, z0],
            stop=[x_ground_lo1, y1, z1],
        )

        ground_hi = self.CSX.AddMetal(f"{name}_ground_2")
        ground_hi.SetColor(COPPER_COLOR, 255)
        ground_hi_prim = ground_hi.AddBox(
            priority=priority,
            start=[x_ground_hi0, y0, z0],
            stop=[x_ground_hi1, y1, z1],
        )

        for prim in (trace_prim, ground_lo_prim, ground_hi_prim):
            self._apply_rotation(prim, cx, cy, rotation)

        slot_offsets = []
        for slot_lower, slot_upper in (
            (-trace_width_mm / 2 - gap_mm, -trace_width_mm / 2),
            (trace_width_mm / 2, trace_width_mm / 2 + gap_mm),
        ):
            slot_offsets.extend(np.linspace(slot_lower, slot_upper, 3))
        if rotation % 180 == 0:
            self._request_mesh_lines(0, [cx + offset for offset in slot_offsets])
        elif rotation % 90 == 0:
            self._request_mesh_lines(
                1,
                [
                    self._rotate_point(cx + offset, cy, cx, cy, rotation)[1]
                    for offset in slot_offsets
                ],
            )

        return trace_prim, ground_lo_prim, ground_hi_prim

    def create_gcpw(
        self,
        name: str = "gcpw",
        *,
        position: tuple[float, float],
        trace_width_mm: float,
        gap_mm: float,
        ground_width_mm: float,
        length_mm: float,
        via_diameter_mm: float,
        via_pitch_mm: float,
        via_z_bottom_mm: float,
        via_z_top_mm: float,
        z_elevation_mm: float | None = None,
        copper_thickness_mm: float | None = None,
        rotation: float = 0,
        priority: int = 6,
    ) -> tuple[CSPrimitives, CSPrimitives, CSPrimitives]:
        """Create a straight grounded (conductor-backed) CPW segment.

        Everything :func:`create_cpw` draws -- a central signal trace flanked
        by a slot and a ground pour on each side -- plus a row of stitching
        vias (via :func:`create_via`) centred along each pour, evenly spaced by
        ``via_pitch_mm`` with one via inset by its radius from each end of the
        segment, so its disc lands fully inside the ground pour rather than
        straddling the edge.

        The vias are what make the line conductor-backed: they tie the coplanar
        pours to the plane underneath, which this method does **not** draw --
        add it with :func:`create_ground`, whose footprint is the whole board
        rather than this one segment. Keep ``via_pitch_mm`` well inside
        lambda/10, or the two pours drift apart in potential between stitches
        and the parallel-plate mode between them and the backside ground starts
        to carry power.

        Feed the segment with :func:`create_cpw_port`, not a lumped port, for
        the reason given in :func:`create_cpw`; the slot mesh lines that port
        needs are requested here too, by the same call that draws the pours.

        ``position`` is the midpoint of the signal trace's input edge, so the
        segment butts flush against the end of the incoming line.  In the
        unrotated frame it runs ``+y``, with the trace centred on ``cx`` and
        the two pours flanking it in ``x``; the output edge -- where the next
        primitive starts -- is ``(cx, cy + length_mm)`` before ``rotation`` is
        applied.

        Parameters
        ----------
        name : str, optional
            Base name for the segment's properties: the trace goes on
            ``"<name>_trace"``, the pours on ``"<name>_ground_1"`` and
            ``"<name>_ground_2"``, and each stitching via on
            ``"<name>_via_<row>_<side>"``. Default ``"gcpw"``.
        position : tuple[float, float]
            ``(x, y)`` midpoint of the trace's input edge, in the unrotated
            frame.
        trace_width_mm : float
            Width of the central signal trace.
        gap_mm : float
            Width of the slot separating the signal trace from each ground
            pour.
        ground_width_mm : float
            Width of each of the two ground pours.
        length_mm : float
            Segment length along y, in the unrotated frame.
        via_diameter_mm : float
            Diameter of the stitching vias placed along each ground pour. Must
            be less than both ``length_mm`` and ``ground_width_mm``.
        via_pitch_mm : float
            Centre-to-centre spacing along the segment between stitching vias.
        via_z_bottom_mm, via_z_top_mm : float
            Z extent of the stitching vias -- typically the backside ground
            plane's bottom face and the top of the copper.
        z_elevation_mm : float | None, optional
            Z position of the conductors' bottom face. Defaults to the
            substrate's top face.
        copper_thickness_mm : float | None, optional
            Conductor thickness along z. Defaults to
            ``params.copper_thickness_mm``.
        rotation : float, optional
            Rotation in degrees, counter-clockwise about ``position``. Default
            ``0``.
        priority : int, optional
            CSXCAD priority of the trace, the pours and the stitching vias;
            where primitives overlap, the higher priority wins. Default
            ``6``.

        Returns
        -------
        tuple[CSPrimitives, CSPrimitives, CSPrimitives]
            The signal trace and the two ground pours, in the order
            ``(trace, ground_lo, ground_hi)`` (``lo``/``hi`` by x in the
            unrotated frame). The stitching vias are reachable by name.

        Raises
        ------
        ValueError
            If ``via_diameter_mm`` is not less than both ``length_mm`` and
            ``ground_width_mm``.
        """
        # Checked before anything is drawn, so a rejected call leaves no
        # half-built segment behind in the structure.
        if via_diameter_mm >= length_mm:
            raise ValueError(
                f"via_diameter_mm must be less than length_mm, got "
                f"{via_diameter_mm} >= {length_mm}"
            )
        if via_diameter_mm >= ground_width_mm:
            raise ValueError(
                f"via_diameter_mm must be less than ground_width_mm, got "
                f"{via_diameter_mm} >= {ground_width_mm}"
            )

        cx, cy = position
        prims = self.create_cpw(
            name,
            position=position,
            trace_width_mm=trace_width_mm,
            gap_mm=gap_mm,
            ground_width_mm=ground_width_mm,
            length_mm=length_mm,
            z_elevation_mm=z_elevation_mm,
            copper_thickness_mm=copper_thickness_mm,
            rotation=rotation,
            priority=priority,
        )

        # Inset by the via radius so the end vias' discs land fully inside
        # the ground pours' y-extent instead of straddling the segment edge.
        y0_via = cy + via_diameter_mm / 2
        y1_via = cy + length_mm - via_diameter_mm / 2
        n_vias = max(2, round((y1_via - y0_via) / via_pitch_mm) + 1)
        via_y = np.linspace(y0_via, y1_via, n_vias)
        x_via_lo = cx - trace_width_mm / 2 - gap_mm - ground_width_mm / 2
        x_via_hi = cx + trace_width_mm / 2 + gap_mm + ground_width_mm / 2

        for n, vy in enumerate(via_y, start=1):
            for side, vx in enumerate((x_via_lo, x_via_hi), start=1):
                self.create_via(
                    f"{name}_via_{n}_{side}",
                    position=self._rotate_point(vx, vy, cx, cy, rotation),
                    via_diameter_mm=via_diameter_mm,
                    z_bottom_mm=via_z_bottom_mm,
                    z_top_mm=via_z_top_mm,
                    priority=priority,
                )

        return prims

    def create_substrate(
        self,
        name: str = "substrate",
        *,
        start: list[float] | None = None,
        stop: list[float] | None = None,
        eps_r: float | None = None,
        tand: float | None = None,
        main_freq: float | None = None,
        priority: int = 0,
    ) -> CSPrimitives:
        """Create a dielectric substrate from two corner points.

        Also asks :func:`create_mesh` for ``params.substrate_cells`` mesh lines
        through the thickness, since the field between a trace and the ground
        plane under it varies across exactly that region and the automatic
        mesher leaves a thin substrate with a single line in it.

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the substrate is added to. Default
            ``"substrate"``.
        start, stop : list[float] | None, optional
            ``[x, y, z]`` opposite corners of the substrate box. Both default
            to the footprint in ``params``: the substrate width and length
            centred on the origin, from ``z = 0`` to the substrate thickness.
        eps_r : float | None, optional
            Relative permittivity. Defaults to ``params.substrate_eps_r``.
        tand : float | None, optional
            Loss tangent. Defaults to ``params.substrate_tand``.
        main_freq : float | None, optional
            Frequency in Hz used to convert ``tand`` into an equivalent
            conductivity (``kappa``). Defaults to ``params.main_freq``.
        priority : int, optional
            CSXCAD priority of the primitive; where primitives overlap, the
            higher priority wins. Default ``0``.

        Returns
        -------
        CSPrimitives
            The substrate box primitive.
        """
        params = self.params
        if start is None:
            start = [
                -params.substrate_width_mm / 2,
                -params.substrate_length_mm / 2,
                0,
            ]
        if stop is None:
            stop = [
                params.substrate_width_mm / 2,
                params.substrate_length_mm / 2,
                params.substrate_thickness_mm,
            ]

        overridden = (eps_r, tand, main_freq) != (None, None, None)
        eps_r = params.substrate_eps_r if eps_r is None else eps_r
        tand = params.substrate_tand if tand is None else tand
        main_freq = params.main_freq if main_freq is None else main_freq
        # params already carries the kappa for its own eps_r/tand/main_freq.
        if overridden:
            kappa = tand * 2 * np.pi * main_freq * EPS0 * eps_r
        else:
            kappa = params.substrate_kappa

        substrate = self.CSX.AddMaterial(name, epsilon=eps_r, kappa=kappa)
        substrate.SetColor(SUBSTRATE_COLOR, 100)

        z_lower, z_upper = sorted((start[2], stop[2]))
        self._request_mesh_lines(
            2, list(np.linspace(z_lower, z_upper, params.substrate_cells)[1:-1])
        )

        return substrate.AddBox(priority=priority, start=start, stop=stop)

    def create_ground(
        self,
        name: str = "ground",
        *,
        start: list[float] | None = None,
        stop: list[float] | None = None,
        priority: int = 2,
    ) -> CSPrimitives:
        """Create a copper ground plane from two corner points.

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the ground plane is added to. Default
            ``"ground"``.
        start, stop : list[float] | None, optional
            ``[x, y, z]`` opposite corners of the ground box. Both default to
            the footprint in ``params``: the substrate width and length centred
            on the origin, one copper thickness below ``z = 0``.
        priority : int, optional
            CSXCAD priority of the primitive; where primitives overlap, the
            higher priority wins. Default ``2``.

        Returns
        -------
        CSPrimitives
            The ground plane primitive.
        """
        params = self.params
        if start is None:
            start = [
                -params.substrate_width_mm / 2,
                -params.substrate_length_mm / 2,
                -params.copper_thickness_mm,
            ]
        if stop is None:
            stop = [
                params.substrate_width_mm / 2,
                params.substrate_length_mm / 2,
                0,
            ]

        ground = self.CSX.AddMetal(name)
        ground.SetColor(COPPER_COLOR, 255)
        return ground.AddBox(priority=priority, start=start, stop=stop)

    def create_lumped_port(
        self,
        port_nr: int,
        start: list[float],
        stop: list[float],
        direction: str = "z",
        impedance: float | None = None,
        excite: float = 0,
        priority: int = 6,
        edges2grid: str = "y",
    ) -> LumpedPort:
        """Create a lumped port for S-parameter extraction.

        Parameters
        ----------
        port_nr : int
            Port number (1-indexed).
        start, stop : list[float]
            ``[x, y, z]`` coordinates defining the port box.  For a
            microstrip z-port, this typically spans from just below the
            ground plane to just above the trace.
        direction : str
            Excitation direction (``"x"``, ``"y"``, or ``"z"``).
        impedance : float | None, optional
            Port reference impedance in ohms. Defaults to
            ``params.charac_imp``.
        excite : float
            Excitation amplitude.  ``0`` = matched load, nonzero = driven.
        priority : int
            CSXCAD mesh priority for the port primitives.
        edges2grid : str
            Which transverse edges to snap to the mesh grid.  Typically
            the direction the line runs along, so a mesh line lands on the
            port plane (``"y"`` for a line running along y).

        Returns
        -------
        LumpedPort
            The created port object (needed for post-simulation
            ``CalcPort``).
        """
        if impedance is None:
            impedance = self.params.charac_imp

        return self.FDTD.AddLumpedPort(
            port_nr,
            impedance,
            start,
            stop,
            direction,
            excite=excite,
            priority=priority,
            edges2grid=edges2grid,
        )

    def create_cpw_port(
        self,
        name: str = "cpw_port",
        *,
        port_nr: int,
        start: list[float],
        stop: list[float],
        gap_mm: float,
        prop_dir: str = "y",
        excite: float = 0,
        impedance: float | None = None,
        feed_shift_mm: float | None = None,
        meas_plane_shift_mm: float | None = None,
        priority: int = 7,
    ) -> CPWPort:
        """Create a coplanar waveguide port on a CPW segment.

        The CPW mode is odd: the signal trace swings against both ground
        pours at once, and the voltage that defines it is the field across
        the two slots together. A lumped port cannot represent that -- it
        drives a single gap at a single point, so it excites the wrong field
        and reports the wrong impedance. openEMS's CPW port instead places
        voltage probes across both slots at three planes along the run and
        current probes between them, which gives the propagation constant and
        the line's own characteristic impedance as well as a clean
        single-mode excitation.

        Like :func:`create_lumped_port`, and unlike the rotatable primitives,
        the port is given as two opposite corners of the signal trace it sits
        on -- the same pair openEMS itself takes. The corners span the trace
        only, never the slots or the pours; ``gap_mm`` adds those. They share one
        ``z``, because every part of the port is a sheet. ``prop_dir`` says
        which of the two in-plane axes the run follows, leaving the other one
        to span the trace width, and the direction along it is the sign of
        ``stop - start``: a port that feeds a segment starts where the segment
        starts, and a port that terminates one starts at its far end and runs
        back into it, with ``stop`` behind ``start``.

        ``start`` is the reference plane -- the plane the termination sits on
        and the S-parameters are referenced to -- so it is the corner worth
        placing deliberately. Both it and the shared ``z`` are snapped onto
        the mesh, since a sheet the grid has no line on covers no cells at
        all: openEMS drops it, excitation included.

        Two ordering rules come from openEMS, not from here. The port reads
        the grid to place its probes on it, so **the mesh must already exist**
        -- build the geometry, call :func:`create_mesh`, then add CPW ports.
        And the port draws its own zero-thickness trace, on the same
        ``cpw_trace`` property :func:`create_cpw` uses, so it lands inside the
        segment's own copper rather than beside it.

        Parameters
        ----------
        name : str, optional
            Name of the CSXCAD property the trace sheet openEMS draws for the
            port is added to, kept distinct from the segment it sits on so it
            reads as the port's own copper. Default ``"cpw_port"``.
        port_nr : int
            Port number (1-indexed).
        start, stop : list[float]
            ``[x, y, z]`` opposite corners of the port's trace sheet. Both
            ``z`` values must be equal. Along ``prop_dir`` they give the
            reference plane and the end of the run; along the other in-plane
            axis, the two edges of the signal trace.
        gap_mm : float
            Width of the slot on each side of the trace. Not covered by
            ``start``/``stop``: it is how far past each trace edge the probes,
            the excitation and the termination reach.
        prop_dir : str, optional
            In-plane axis the run follows, ``"x"`` or ``"y"``. Default
            ``"y"``. It cannot be inferred, since the corners differ on both
            in-plane axes -- one span is the run, the other the trace width.
        excite : float, optional
            Excitation amplitude. ``0`` (default) is a matched load, nonzero
            drives the port.
        impedance : float | None, optional
            Termination resistance in ohms, placed at the reference plane.
            Defaults to ``params.charac_imp``.
        feed_shift_mm : float | None, optional
            Distance from the reference plane to the excitation plane.
            ``None`` (default) excites at the reference plane.
        meas_plane_shift_mm : float | None, optional
            Distance from the reference plane to the measurement plane.
            ``None`` (default) measures at the middle of the run.
        priority : int, optional
            CSXCAD priority for the port's primitives.

        Returns
        -------
        CPWPort
            The created port object (needed for post-simulation
            ``CalcPort``).

        Raises
        ------
        ValueError
            If ``prop_dir`` is not ``"x"`` or ``"y"``, if ``start`` and
            ``stop`` disagree in z or agree along ``prop_dir``, if the mesh has
            not been generated yet, or if either slot holds fewer than two mesh
            lines -- which would collapse every probe, excitation and
            termination box the port spans across it onto a single line.
        """
        prop_dim = {"x": 0, "y": 1}.get(prop_dir, -1)
        if prop_dim < 0:
            raise ValueError(
                "a CPW port propagates along an in-plane axis, so prop_dir "
                f'must be "x" or "y", got {prop_dir!r}'
            )

        start = [float(v) for v in start]
        stop = [float(v) for v in stop]

        if start[2] != stop[2]:
            raise ValueError(
                "every part of a CPW port is a sheet, so start and stop must "
                f"share one z, got {start[2]} and {stop[2]}"
            )
        if start[prop_dim] == stop[prop_dim]:
            raise ValueError(
                "openEMS reads the propagation direction from the sign of "
                f"stop - start, so they must differ along {prop_dir}, got "
                f"{start[prop_dim]} twice"
            )

        if impedance is None:
            impedance = self.params.charac_imp

        grid = self.CSX.GetGrid()
        if min(len(grid.GetLines(dim)) for dim in range(3)) < 5:
            raise ValueError(
                "a CPW port reads the grid to place its probes on it: build "
                "the geometry and call create_mesh() before adding the port"
            )

        z_lines = np.asarray(grid.GetLines(2))
        start[2] = stop[2] = float(z_lines[np.argmin(np.abs(z_lines - start[2]))])

        prop_lines = np.asarray(grid.GetLines(prop_dim))
        start[prop_dim] = float(
            prop_lines[np.argmin(np.abs(prop_lines - start[prop_dim]))]
        )

        slot_dim = 1 - prop_dim
        slot_lines = np.asarray(grid.GetLines(slot_dim))
        trace_lo, trace_hi = sorted((start[slot_dim], stop[slot_dim]))
        for lower, upper in (
            (trace_lo - gap_mm, trace_lo),
            (trace_hi, trace_hi + gap_mm),
        ):
            inside = slot_lines[(slot_lines >= lower) & (slot_lines <= upper)]
            if len(inside) < 2:
                raise ValueError(
                    f"the CPW slot from {lower:.4g} to {upper:.4g} along "
                    f"{'xyz'[slot_dim]} holds {len(inside)} mesh line(s), and "
                    "every probe this port spans across it would collapse to a "
                    "point: a gap of "
                    f"{gap_mm:.4g} needs at least two lines in it, but the "
                    "nearest cell is "
                    f"{np.min(np.diff(slot_lines)):.4g} wide. Build the slot "
                    "with create_cpw (which registers its edges with the "
                    "mesher) and keep its rotation a multiple of 90 degrees, "
                    "or widen the gap."
                )

        shifts: dict[str, float] = {}
        if feed_shift_mm is not None:
            shifts["FeedShift"] = feed_shift_mm
        if meas_plane_shift_mm is not None:
            shifts["MeasPlaneShift"] = meas_plane_shift_mm

        port_metal = self.CSX.AddMetal(name)
        port_metal.SetColor(COPPER_COLOR, 255)

        return self.FDTD.AddCPWPort(
            port_nr,
            port_metal,
            start,
            stop,
            prop_dir,
            # openEMS takes the E-field direction across the slots here and
            # derives the sheet normal from it and prop_dir.
            "xyz"[slot_dim],
            gap_mm,
            excite=excite,
            priority=priority,
            Feed_R=impedance,
            **shifts,
        )

    def create_cpw_lumped_port(
        self,
        name: str = "cpw_lumped_port",
        *,
        port_nr: int,
        start: list[float],
        stop: list[float],
        gap_mm: float,
        prop_dir: str = "y",
        excite: float = 0,
        impedance: float | None = None,
        priority: int = 7,
    ) -> CPWLumpedPort:
        """Create a coplanar waveguide port from one lumped port per slot.

        The alternative to :func:`create_cpw_port`, and the one to reach for
        when the line's conductors have thickness. openEMS's CPW port is drawn
        entirely on one ``z`` plane, so on a line with real copper on it the
        probes sit on a face of that copper rather than across the slot, and
        the current probe's loop has to guess where in ``z`` the conductor
        ends. This port instead gives each slot a lumped port whose box fills
        the slot -- the gap across, the port length along the run, and the
        whole conductor thickness in ``z``.

        Both slots are driven signal pour -> ground pour, which puts their
        excitation vectors in opposition along the slot axis and so excites
        the odd mode the line carries rather than the even parallel-plate one
        a single lumped port would. Each is terminated in ``2 * impedance``,
        which is ``impedance`` in parallel, and the two are returned as one
        :class:`CPWLumpedPort` that behaves like any other openEMS port.

        Corners follow :func:`create_cpw_port`: ``start`` and ``stop`` are two
        opposite corners of the signal trace, never of the slots or the pours,
        and ``gap_mm`` is how far past each trace edge the boxes reach. The
        one difference is ``z``, which here must *differ* between the two --
        that span is the conductor thickness the port fills. ``start`` is the
        reference plane, so a port feeding a line starts where the line starts
        and a port terminating one starts at its far end, with ``stop`` behind
        ``start``.

        **Call this before** :func:`create_mesh`, the opposite of
        :func:`create_cpw_port`. openEMS's CPW port reads the grid to place
        its probes on it, so it has to come after; a lumped port never looks
        at the grid, and the mesher should see its boxes and the lines it asks
        for -- above all the one on the port's far edge, which is what fixes
        the port's length instead of letting it snap to whatever the grid
        happened to offer.

        Parameters
        ----------
        name : str, optional
            Base name for the two slot ports, which take the suffixes
            ``"_lo_"`` and ``"_hi_"`` (by position along the slot axis) as the
            prefix on their probe file names. Default
            ``"cpw_lumped_port"``.
        port_nr : int
            Port number (1-indexed), shared by both slots -- they are one
            port, and it is the returned object that carries the number.
        start, stop : list[float]
            ``[x, y, z]`` opposite corners of the port's signal trace. Along
            ``prop_dir`` they give the reference plane and the end of the run;
            along the other in-plane axis, the two edges of the trace; in
            ``z``, the bottom and top of the conductor.
        gap_mm : float
            Width of the slot on each side of the trace. Not covered by
            ``start``/``stop``: it is how far past each trace edge the slot
            boxes reach.
        prop_dir : str, optional
            In-plane axis the run follows, ``"x"`` or ``"y"``. Default
            ``"y"``. It cannot be inferred, since the corners differ on both
            in-plane axes -- one span is the run, the other the trace width.
        excite : float, optional
            Excitation amplitude. ``0`` (default) is a matched load, nonzero
            drives the port.
        impedance : float | None, optional
            Reference impedance of the pair, in ohms. Each slot is terminated
            in twice it. Defaults to ``params.charac_imp``.
        priority : int, optional
            CSXCAD priority for the port's primitives.

        Returns
        -------
        CPWLumpedPort
            The created port (needed for post-simulation ``CalcPort``).

        Raises
        ------
        ValueError
            If ``prop_dir`` is not ``"x"`` or ``"y"``; if ``start`` and
            ``stop`` agree along ``prop_dir``, along the trace-width axis, or
            in ``z``; or if the mesh has already been generated.

        See Also
        --------
        create_cpw : the segment this port feeds.
        create_cpw_port : openEMS's own CPW port, for a zero-thickness line.
        """

        prop_dim = {"x": 0, "y": 1}.get(prop_dir, -1)
        if prop_dim < 0:
            raise ValueError(
                "a CPW port propagates along an in-plane axis, so prop_dir "
                f'must be "x" or "y", got {prop_dir!r}'
            )
        slot_dim = 1 - prop_dim

        start = [float(v) for v in start]
        stop = [float(v) for v in stop]

        if start[prop_dim] == stop[prop_dim]:
            raise ValueError(
                "the propagation direction is read from the sign of "
                f"stop - start, so they must differ along {prop_dir}, got "
                f"{start[prop_dim]} twice"
            )
        if start[slot_dim] == stop[slot_dim]:
            raise ValueError(
                "start and stop span the signal trace across "
                f"{'xyz'[slot_dim]}, so they must differ along it, got "
                f"{start[slot_dim]} twice"
            )
        if start[2] == stop[2]:
            raise ValueError(
                "a lumped-fed CPW port fills the slot rather than covering it "
                "with a sheet, so start and stop must differ in z -- that "
                f"span is the conductor thickness -- got {start[2]} twice. A "
                "line whose conductors really have no thickness wants "
                "create_cpw_port() instead."
            )

        if impedance is None:
            impedance = self.params.charac_imp

        grid = self.CSX.GetGrid()
        if min(len(grid.GetLines(dim)) for dim in range(3)) >= 5:
            raise ValueError(
                "the mesher places lines on this port's boxes and on the "
                "edges it registers, so add it before create_mesh(), not "
                "after (the opposite of create_cpw_port, which reads the "
                "finished grid)"
            )

        trace_lo, trace_hi = sorted((start[slot_dim], stop[slot_dim]))
        slot_dir = "xyz"[slot_dim]

        slots = []
        for suffix, trace_edge, ground_edge in (
            ("lo", trace_lo, trace_lo - gap_mm),
            ("hi", trace_hi, trace_hi + gap_mm),
        ):
            box_start = list(start)
            box_stop = list(stop)
            box_start[slot_dim] = trace_edge
            box_stop[slot_dim] = ground_edge
            slots.append(
                self.FDTD.AddLumpedPort(
                    port_nr,
                    2 * impedance,
                    box_start,
                    box_stop,
                    slot_dir,
                    excite=excite,
                    priority=priority,
                    # Both slots are one port and share port_nr, so the prefix
                    # is what keeps their probe files apart.
                    PortNamePrefix=f"{name}_{suffix}_",
                )
            )

        self._request_mesh_lines(
            slot_dim, [trace_lo - gap_mm, trace_lo, trace_hi, trace_hi + gap_mm]
        )
        self._request_mesh_lines(prop_dim, [start[prop_dim], stop[prop_dim]])
        self._request_mesh_lines(2, [start[2], stop[2]])

        return CPWLumpedPort(self.CSX, port_nr, start, stop, excite, slots, impedance)

    def create_mesh(self, smooth_ratio: float = 1.5) -> None:
        """Generate the FDTD mesh for everything added so far.

        Scans the primitives already in the CSXCAD structure and builds the
        grid around them (see :class:`simpleEMS.fdtd_mesh.Mesh`), so call this
        after the geometry and the ports are in place.

        Any mesh line a primitive asked for while it was being built -- a
        CPW's slot edges are the only case today -- is passed through as a
        fixed line, so it lands exactly where the primitive needs it rather
        than wherever the automatic passes would have put it.

        Parameters
        ----------
        smooth_ratio : float, optional
            Maximum ratio between adjacent mesh cells. Default ``1.5``.

        Returns
        -------
        None
        """
        Mesh(
            self.CSX,
            self.params,
            smooth_ratio,
            requested_lines=self._requested_mesh_lines,
        )
