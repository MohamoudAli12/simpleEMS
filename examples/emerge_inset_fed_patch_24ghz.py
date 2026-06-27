#!/usr/bin/env python3
"""Inset-fed patch antenna at 60 GHz — EMerge FEM simulation from STEP import."""

import gmsh
import emerge as em
from emerge.plot import plot_sp, smith
import numpy as np
from pathlib import Path

from simpleEMS.emerge_fem import emerge_load_step, emerge_assign_materials, detect_ports

mm = 0.001

model = em.Simulation("InsetFedPatch_24GHz")

step_path = Path(__file__).parent / "InsetFedPatch_24GHz" / "step" / "structure.step"
vols, step = emerge_load_step(step_path, label="Patch")
emerge_assign_materials(vols, er=3.48, tand=0.0037)

air = step.enclose(4 * mm)
ports = detect_ports(vols)
p1 = ports[0]
port_plate = em.geo.Plate(
    p1.origin,
    p1.v1,
    p1.v2,
)

model.mw.set_frequency_range(22e9, 27e9, 7)
model.mw.set_resolution(0.20)
model.commit_geometry()
model.mesher.set_face_size(port_plate, 0.01 * mm)
model.generate_mesh()
model.view(plot_mesh=True, volume_mesh=True)

# Use Plate directly for LumpedPort (like demo4)
model.mw.bc.LumpedPort(
    port_plate,
    p1.port_num,
    width=p1.width,
    height=p1.height,
    direction=p1.direction,
    Z0=50,
)

model.mw.bc.AbsorbingBoundary(air.boundary())

data = model.mw.run_sweep()
print(data)
freqs = data.scalar.grid.freq
freq_dense = np.linspace(22e9, 25e9, 1001)
S11 = data.scalar.grid.model_S(1, 1, freq_dense)
plot_sp(freq_dense, S11)
smith(S11, f=freq_dense)
