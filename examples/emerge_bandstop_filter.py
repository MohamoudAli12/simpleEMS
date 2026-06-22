#!/usr/bin/env python3
"""Band-stop quarter-wave stub filter — EMerge FEM simulation from STEP import."""

import gmsh
import emerge as em
from emerge.plot import plot_sp, smith
from pathlib import Path

mm = 0.001

model = em.Simulation("BandStopFilter_1_5GHz")

step_path = (
    Path(__file__).parent
    / "BandStopQuarterWaveFilter_1_5GHz"
    / "step"
    / "structure.step"
)
step = em.geo.STEPItems("Filter", str(step_path), unit=0.001)
vols = step.dictionary
print(vols)

substrate = vols["Filter_substrate"]
substrate.set_material(
    em.Material(er=3.48, tand=0.001074, color="#0F8B00", opacity=0.9)
)

pec_names = [
    "Filter_ground",
    "Filter_series_line_1",
    "Filter_series_line_2",
    "Filter_series_line_3",
    "Filter_series_line_4",
    "Filter_shunt_line_1",
    "Filter_shunt_line_2",
    "Filter_shunt_line_3",
]
for name in pec_names:
    if name in vols:
        vols[name].set_material(em.lib.PEC)


air = step.enclose(50 * mm)


# Compute port face parameters from port_resist volumes (before commit_geometry)
def get_port_face_params(port_vol):
    dim, tag = port_vol.dimtags[0]
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
    center = (xmin, (ymin + ymax) / 2, (zmin + zmax) / 2)
    width = ymax - ymin
    height = zmax - zmin
    return center, width, height


(port1_center, port1_w, port1_h) = get_port_face_params(vols["Filter_port_resist_1"])
(port2_center, port2_w, port2_h) = get_port_face_params(vols["Filter_port_resist_2"])

model.mw.set_frequency_range(0.5e9, 2.5e9, 201)
model.mw.set_resolution(0.25)
model.commit_geometry()
model.generate_mesh()
model.view(plot_mesh=True, volume_mesh=False)

# Select port faces using computed coordinates
port1_face = model.select.face.inplane(
    x=port1_center[0],
    y=port1_center[1],
    z=port1_center[2],
    normal_axis=em.XAX,
    tolerance=1e-6,
)
port2_face = model.select.face.inplane(
    x=port2_center[0],
    y=port2_center[1],
    z=port2_center[2],
    normal_axis=em.XAX,
    tolerance=1e-6,
)

model.mw.bc.LumpedPort(
    port1_face, 1, width=port1_w, height=port1_h, direction=em.ZAX, Z0=50
)
model.mw.bc.LumpedPort(
    port2_face, 2, width=port2_w, height=port2_h, direction=em.ZAX, Z0=50
)

model.mw.bc.AbsorbingBoundary(air.boundary())

data = model.mw.run_sweep()
model.display.populate()
freqs = data.scalar.grid.freq
S11 = data.scalar.grid.S(1, 1)
S21 = data.scalar.grid.S(2, 1)
plot_sp(freqs, S11)
plot_sp(freqs, S21)
smith(S11, f=freqs)
model.display.show()
