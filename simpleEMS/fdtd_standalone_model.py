# simpleEMS
# Copyright (C) 2026 Mohamoud Ali
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
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
Load a ``structure.xml``, run the simulation, and return CSX structure and network
parameters.
"""

from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from numpy.typing import NDArray

from CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS
from openEMS.ports import LumpedPort, Port

from .console import console
from .sim_tools import SimData, SimSetup, SimTools

__all__ = ["add_fdtd_setup", "simulate_model"]


def get_freq_range(structure_xml_path: str | Path) -> tuple[float, float]:
    """
    Extract the excitation centre frequency and cutoff from
    ``structure.xml``.

    Parameters
    ----------
    structure_xml_path : str | Path
        Path to the ``structure.xml`` file.

    Returns
    -------
    tuple[float, float]
        ``(f0, fc)`` – centre frequency and cutoff frequency in Hz.

    Raises
    ------
    ValueError
        If the model carries no openEMS ``<FDTD>`` setup (see
        :func:`add_fdtd_setup`), or if its excitation is not a Gaussian
        pulse and therefore defines no frequency band.
    """
    root = ET.parse(structure_xml_path).getroot()
    if root.tag != "openEMS":
        raise ValueError(
            f"{structure_xml_path}: root element is <{root.tag}>, so this model "
            f"carries no openEMS FDTD setup (excitation, boundary conditions, "
            f"run limits) and cannot be simulated.\n\n"
            "Add the FDTD setup first with simpleEMS.add_fdtd_setup():\n\n"
            "    from simpleEMS import add_fdtd_setup, simulate_model\n\n"
            "    xml = add_fdtd_setup('structure.xml', freq_range=(1e9, 2e9))\n"
            "    sim_data, sim, charac_imp = simulate_model(xml, output_path)\n"
        )

    exc = root.find("./FDTD/Excitation")
    if exc is None:
        raise ValueError(
            f"{structure_xml_path}: <FDTD> has no <Excitation> element.\n\n"
            "Add the FDTD setup first with simpleEMS.add_fdtd_setup():\n\n"
            "    from simpleEMS import add_fdtd_setup, simulate_model\n\n"
            "    xml = add_fdtd_setup('structure.xml', freq_range=(1e9, 2e9))\n"
            "    sim_data, sim, charac_imp = simulate_model(xml, output_path)\n"
        )

    f0, fc = exc.get("f0"), exc.get("fc")
    if f0 is None or fc is None:
        raise ValueError(
            f'{structure_xml_path}: <Excitation Type="{exc.get("Type")}"> is not '
            "a Gaussian pulse, so no post-processing band can be derived from it. "
            "Pass an explicit freqs= array to simulate_model(), or rebuild with "
            "add_fdtd_setup(..., excitation='gauss')."
        )
    return float(f0), float(fc)


def add_fdtd_setup(
    structure_xml_path: str | Path,
    freq_range: tuple[float, float],
    output_xml_path: str | Path | None = None,
    excitation: str = "gauss",
    FDTD_boundary: list[str] | None = None,
    FDTD_timestep: int = 90000000,
    FDTD_end_criteria: float = 1e-4,
    overwrite: bool = False,
) -> Path:
    """
    Attach an openEMS FDTD setup to a geometry-only ``structure.xml``.

    A file written by ``CSX.Write2XML()`` (root ``<ContinuousStructure>``)
    holds the mesh, materials, metals, lumped-port properties and probes,
    but none of the solver settings :func:`simulate_model` needs -- the
    excitation waveform, the boundary conditions and the run limits all
    live in the ``<FDTD>`` section that only ``openEMS.Write2XML()``
    writes. This function supplies those settings and re-writes the model
    as a full ``<openEMS>`` document, ready for :func:`simulate_model`.

    An existing ``<openEMS>`` file is also accepted, in which case its
    ``<FDTD>`` section is replaced -- use that to re-band or re-terminate a
    model without rebuilding its geometry.

    Parameters
    ----------
    structure_xml_path : str | Path
        Path to the model, root ``<ContinuousStructure>`` or ``<openEMS>``.
    freq_range : tuple[float, float]
        ``(fmin, fmax)`` in Hz. The Gaussian excitation is centred at
        ``(fmin + fmax) / 2`` with cutoff ``(fmax - fmin) / 2``, matching
        :func:`~simpleEMS.sim_tools.setup_simulation`.
    output_xml_path : str | Path, optional
        Where to write the result. Defaults to ``<stem>_fdtd.xml`` in
        ``cwd / "Sim_Path"``.
    excitation : str
        ``"gauss"`` (default), ``"sinus"``, ``"dirac"`` or ``"step"``.
        Only ``"gauss"`` lets :func:`simulate_model` derive its
        post-processing band from the file; the others require an explicit
        ``freqs=`` argument there.
    FDTD_boundary : list[str], optional
        Six openEMS boundary-condition strings, in the order
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``. Defaults to
        ``["PML_8"] * 6``.
    FDTD_timestep : int
        Maximum FDTD time-step count. Default ``90000000``.
    FDTD_end_criteria : float
        Energy decay stopping criterion. Default ``1e-4``.
    overwrite : bool
        Allow writing over an existing ``output_xml_path``. Default
        ``False``.

    Returns
    -------
    Path
        Path to the written ``<openEMS>`` XML.

    Raises
    ------
    ValueError
        If the source root element is neither ``<ContinuousStructure>``
        nor ``<openEMS>``, if ``freq_range`` is not a positive increasing
        pair, if ``excitation`` is unknown, or if the loaded structure
        carries no mesh lines on some axis (openEMS cannot run an unmeshed
        model).
    RuntimeError
        If the loaded structure carries no lumped ports
        (``port_resist_<N>`` properties).
    FileExistsError
        If ``output_xml_path`` exists and ``overwrite`` is ``False``.
    """
    src = Path(structure_xml_path)
    dst = (
        Path(output_xml_path)
        if output_xml_path
        else Path.cwd() / "Sim_Path" / f"{src.stem}_fdtd.xml"
    )
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"{dst} already exists; pass overwrite=True to replace it"
        )

    fmin, fmax = (float(f) for f in freq_range)
    if not 0 <= fmin < fmax:
        raise ValueError(f"freq_range must be 0 <= fmin < fmax, got {freq_range!r}")

    root_tag = ET.parse(src).getroot().tag
    if root_tag == "openEMS":
        # Strip the old <FDTD> section by round-tripping through the CSX alone.
        loader = openEMS()
        loader.ReadFromXML(str(src))
        with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
            loader.GetCSX().Write2XML(tmp.name)
            CSX = ContinuousStructure()
            CSX.ReadFromXML(tmp.name)
    elif root_tag == "ContinuousStructure":
        CSX = ContinuousStructure()
        CSX.ReadFromXML(str(src))
    else:
        raise ValueError(
            f"{src}: unexpected root element <{root_tag}>; expected "
            "<ContinuousStructure> (CSX.Write2XML) or <openEMS> (FDTD.Write2XML)"
        )

    grid = CSX.GetGrid()
    unmeshed = [ax for n, ax in enumerate("xyz") if len(grid.GetLines(n)) < 2]
    if unmeshed:
        raise ValueError(
            f"{src}: no mesh lines on the {', '.join(unmeshed)} axis/axes. "
            "The model must already carry its RectilinearGrid -- add_fdtd_setup "
            "supplies solver settings only, it does not mesh the geometry."
        )

    names = {CSX.GetProperty(i).GetName() for i in range(CSX.GetQtyProperties())}
    if not any(re.fullmatch(r"port_resist_\d+", name) for name in names):
        raise RuntimeError(
            f"{src}: no port_resist_<N> properties found, so the model has no "
            "lumped ports and no network parameters can be computed from it. "
            "add_fdtd_setup supplies solver settings only -- ports must already "
            "exist in the geometry (openEMS.ports.LumpedPort writes them as "
            "port_resist_<N>/port_excite_<N>/port_ut_<N>/port_it_<N>)."
        )

    FDTD = openEMS(NrTS=FDTD_timestep, EndCriteria=FDTD_end_criteria)
    FDTD.SetCSX(CSX)
    if FDTD_boundary is None:
        FDTD_boundary = ["PML_8"] * 6
    FDTD.SetBoundaryCond(FDTD_boundary)

    f0, fc = 0.5 * (fmin + fmax), 0.5 * (fmax - fmin)
    if excitation == "gauss":
        FDTD.SetGaussExcite(f0, fc)
    elif excitation == "sinus":
        FDTD.SetSinusExcite(f0)
    elif excitation == "dirac":
        FDTD.SetDiracExcite(fmax)
    elif excitation == "step":
        FDTD.SetStepExcite(fmax)
    else:
        raise ValueError(
            f"excitation must be 'gauss', 'sinus', 'dirac' or 'step', "
            f"got {excitation!r}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    FDTD.Write2XML(str(dst))
    console.print(f"[success]FDTD setup written to {dst}[/success]")
    return dst


def reconstruct_ports(csx: ContinuousStructure) -> tuple[list[LumpedPort], float]:
    """
    Build ``LumpedPort`` objects from the CSXCAD properties of a
    loaded ``structure.xml``.

    The port naming convention must follow the pattern created by
    ``openEMS.ports.LumpedPort``:

    - ``port_resist_<N>`` – :class:`CSPropLumpedElement`
    - ``port_excite_<N>`` – :class:`CSPropExcitation`
    - ``port_ut_<N>``      – voltage probe (:class:`CSPropProbeBox`,
      ``ProbeType == 0``)
    - ``port_it_<N>``      – current probe (:class:`CSPropProbeBox`,
      ``ProbeType == 1``)

    Parameters
    ----------
    csx : ContinuousStructure
        The CSXCAD geometry container obtained via
        ``FDTD.GetCSX()`` after loading.

    Returns
    -------
    tuple[list[LumpedPort], float]
        ``(ports, R)`` -- one ``LumpedPort`` per port number found in the
        structure (sorted by port number), and ``R``, the resistance (ohms)
        of the last port processed in that order, used by the caller as the
        reference/characteristic impedance. ``R`` is ``0.0`` if no ports
        were found.
    """
    lumped_elements: dict[int, object] = {}
    excitations: dict[int, object] = {}
    probes_by_port: dict[int, list] = {}

    for i in range(csx.GetQtyProperties()):
        property = csx.GetProperty(i)
        property_name = property.GetName()

        match = re.search(r"_(\d+)$", property_name)
        if not match:
            continue
        port_nr = int(match.group(1))

        type_string = property.GetTypeString()
        if type_string == "LumpedElement":
            lumped_elements[port_nr] = property
        elif type_string == "Excitation":
            excitations[port_nr] = property
        elif type_string == "ProbeBox":
            probes_by_port.setdefault(port_nr, []).append(property)

    ports: list[LumpedPort] = []
    R = 0.0
    for port_nr in sorted(lumped_elements):
        le = lumped_elements[port_nr]
        exc = excitations.get(port_nr)
        port_probes = probes_by_port.get(port_nr, [])

        R = le.GetResistance()
        exc_ny = le.GetDirection()

        prims = le.GetAllPrimitives()
        if not prims:
            continue
        bb = prims[0].GetBoundBox()
        start = bb[0]
        stop = bb[1]

        direction = np.sign(stop[exc_ny] - start[exc_ny])
        excite = 1 if (exc is not None and exc.GetEnabled()) else 0

        U_filenames = []
        I_filenames = []
        for probe in port_probes:
            if probe.GetProbeType() == 0:
                U_filenames.append(probe.GetName())
            elif probe.GetProbeType() == 1:
                I_filenames.append(probe.GetName())

        port = LumpedPort.__new__(LumpedPort)
        Port.__init__(
            port,
            csx,
            port_nr=port_nr,
            start=start,
            stop=stop,
            excite=excite,
            U_filenames=U_filenames,
            I_filenames=I_filenames,
        )
        port.R = R
        port.Z_ref = R
        port.exc_ny = exc_ny
        port.direction = direction

        ports.append(port)

    return ports, R


def simulate_model(
    structure_xml_path: str | Path,
    output_path: str | Path | None = None,
    num_points: int = 1000,
    run: bool = True,
    freqs: NDArray | None = None,
) -> tuple[SimData, SimSetup, float]:
    """
    Load a ``structure.xml``, run the simulation and compute network
    parameters.

    Parameters
    ----------
    structure_xml_path : str | Path
        Path to the ``structure.xml`` file exported by CSXCAD.
    output_path : str | Path, optional
        Directory where simulation results will be written / already
        exist. Defaults to ``cwd / "Sim_Path"``.
    num_points : int
        Number of frequency points for post-processing. Default ``1000``.
        Ignored when ``freqs`` is given.
    run : bool
        Whether to execute the FDTD solver. Default ``True``. Set
        to ``False`` to only post-process existing results.
    freqs : NDArray, optional
        Explicit post-processing frequency points (Hz). By default the band
        is derived from the model's own Gaussian excitation
        (:func:`get_freq_range`); pass this to post-process over a narrower
        band, or when the model uses a non-Gaussian excitation that defines
        no band of its own.

    Returns
    -------
    tuple[SimData, SimSetup, float]
        A tuple containing:

        - **sim_data** (*SimData*) -- Named tuple with ``freqs``, ``s11``,
          ``s21``, ``z11``, ``vswr``, and ``input_power``.
        - **sim** (*SimSetup*) -- Named tuple with ``CSX``, ``FDTD``, and ``freqs``.
        - **charac_imp** (*float*) -- Characteristic (reference) impedance in
          ohms, extracted from the loaded model's lumped-port resistance.

    Raises
    ------
    RuntimeError
        If no ports are found in the loaded structure.
    """
    output_path = Path(output_path) if output_path else Path.cwd() / "Sim_Path"
    output_path.mkdir(parents=True, exist_ok=True)

    if freqs is None:
        f0, fc = get_freq_range(structure_xml_path)
        freqs = np.linspace(f0 - fc, f0 + fc, num_points)
    else:
        freqs = np.asarray(freqs, dtype=float)

    FDTD = openEMS()
    FDTD.ReadFromXML(structure_xml_path)
    _CSX = FDTD.GetCSX()

    with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
        _CSX.Write2XML(tmp.name)
        CSX = ContinuousStructure()
        CSX.ReadFromXML(tmp.name)

    sim = SimSetup(CSX=CSX, FDTD=FDTD, freqs=freqs)
    SimTools.write_and_show_structure(sim, output_path)
    ports, charac_imp = reconstruct_ports(CSX)

    if not ports:
        raise RuntimeError(f"No ports found in {structure_xml_path}")

    if run:
        FDTD.Run(str(output_path))

    port_arg = ports if len(ports) > 1 else ports[0]
    sim_data = SimTools.compute_sim_data(sim, port_arg, output_path)

    return (
        sim_data,
        sim,
        charac_imp,
    )
