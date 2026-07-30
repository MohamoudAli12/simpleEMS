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

from CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS
from openEMS.ports import LumpedPort, Port

from .sim_tools import SimData, SimSetup, SimTools

__all__ = ["simulate_model"]


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
    """
    tree = ET.parse(structure_xml_path)
    root = tree.getroot()
    if root.tag != "openEMS":
        raise ValueError(
            "This model is missing openEMS FDTD parameters and cannot be simulated."
            "model should be generated using FDTD.Write2XML() method"
        )

    exc = tree.find(".//FDTD/Excitation")
    if exc is None:
        raise ValueError("No <FDTD><Excitation> element found in XML")
    f0 = float(exc.get("f0"))
    fc = float(exc.get("fc"))
    return f0, fc


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
    output_path: str | Path,
    num_points: int = 1000,
    run: bool = True,
) -> tuple[SimData, SimSetup, float]:
    """
    Load a ``structure.xml``, run the simulation and compute network
    parameters.

    Parameters
    ----------
    structure_xml_path : str | Path
        Path to the ``structure.xml`` file exported by CSXCAD.
    output_path : str | Path
        Directory where simulation results will be written / already
        exist.
    num_points : int
        Number of frequency points for post-processing. Default ``1000``.
    run : bool
        Whether to execute the FDTD solver. Default ``True``. Set
        to ``False`` to only post-process existing results.

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
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    f0, fc = get_freq_range(structure_xml_path)
    freqs = np.linspace(f0 - fc, f0 + fc, num_points)

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
