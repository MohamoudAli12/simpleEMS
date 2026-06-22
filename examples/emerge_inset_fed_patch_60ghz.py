#!/usr/bin/env python3
"""Inset-fed patch antenna at 60 GHz — EMerge FEM simulation from STEP import."""

import gmsh
import emerge as em
from emerge.plot import plot_sp, smith
import numpy as np
from pathlib import Path

mm = 0.001

model = em.Simulation("InsetFedPatch_60GHz")

step_path = Path(__file__).parent / "InsetFedPatch_60GHz" / "step" / "structure.step"
step = em.geo.STEPItems("Patch", str(step_path), unit=0.001)
vols = step.dictionary
print(vols)

substrate = vols["Patch_substrate"]
substrate.set_material(em.Material(er=3.0, tand=0.001, color="#0F8B00", opacity=0.9))

pec_names = ["Patch_ground", "Patch_patch_inset", "Patch_feed"]
for name in pec_names:
    if name in vols:
        vols[name].set_material(em.lib.PEC)

air = step.enclose(5 * mm)

# Get port dimensions from port_resist volume (zero-thickness, used only for positioning)
port_resist = vols["Patch_port_resist_1"]
dim, tag = port_resist.dimtags[0]
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
port_w = xmax - xmin
port_h = zmax - zmin

# port_resist is a zero-thickness surface — no need to subtract from air

# Create Plate for the lumped port (like demo4)
port_plate = em.geo.Plate(
    np.array([xmin, ymin, zmin]), np.array([port_w, 0, 0]), np.array([0, 0, port_h])
)

model.mw.set_frequency_range(57e9, 63e9, 201)
model.mw.set_resolution(0.25)
model.commit_geometry()
model.mesher.set_face_size(port_plate, 0.02 * mm)
model.generate_mesh()

# Use Plate directly for LumpedPort (like demo4)
model.mw.bc.LumpedPort(
    port_plate, 1, width=port_w, height=port_h, direction=em.ZAX, Z0=50
)

model.mw.bc.AbsorbingBoundary(air.boundary())

data = model.mw.run_sweep()
model.display.populate()
freqs = data.scalar.grid.freq
S11 = data.scalar.grid.S(1, 1)
plot_sp(freqs, S11)
smith(S11, f=freqs)
model.display.show()
