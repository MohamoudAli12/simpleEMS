from dataclasses import dataclass, field
import warnings
import numpy as np
from .calc import microstrip_width_from_impedance, patch_dims, phase_shift_length
from .sim_params import SimParams
from .sim_tools import SimTools, mm_to_m

# ----------------------------
# Public APIS
# ----------------------------
__all__ = [
    "InsetPatchParams",
    "InsetFedPatchAntenna",
    "ProbePatchParams",
    "ProbeFedPatchAntenna",
]


@dataclass
class InsetPatchParams(SimParams):
    ang_length: int = 90
    patch_length_mm: float = field(init=False)
    patch_width_mm: float = field(init=False)
    inset_length_mm: float = field(init=False)
    inset_width_mm: float = field(init=False)
    feed_width_mm: float = field(init=False)
    feed_length_mm: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.feed_length_mm = np.round(
            phase_shift_length(
                self.ang_length, self.substrate_eps_r, self.resonant_freq
            ),
            self.fp_precision,
        )
        self.feed_width_mm = np.round(
            microstrip_width_from_impedance(
                self.charac_imp,
                self.substrate_thickness_mm,
                self.copper_thickness_mm,
                self.substrate_eps_r,
                self.resonant_freq,
            ),
            self.fp_precision,
        )
        self.patch_dims = patch_dims(
            self.resonant_freq,
            self.substrate_eps_r,
            mm_to_m(self.substrate_thickness_mm),
            self.charac_imp,
            self.copper_thickness_mm,
        )

        self.patch_width_mm = np.round(
            self.patch_dims.patch_width_mm, self.fp_precision
        )
        self.patch_length_mm = np.round(
            self.patch_dims.patch_length_mm, self.fp_precision
        )
        self.inset_width_mm = np.round(
            self.patch_dims.inset_width_mm / 2, self.fp_precision
        )

        self.inset_length_mm = np.round(
            self.patch_dims.inset_length_mm, self.fp_precision
        )
        self.substrate_width_mm = np.round(
            self.patch_width_mm + 2 * self.lambda0, self.fp_precision
        )
        self.substrate_length_mm = np.round(
            self.patch_length_mm + 2 * self.lambda0, self.fp_precision
        )

        self.simulation_box = np.round(
            np.array(
                [
                    self.substrate_width_mm + self.lambda0,
                    self.substrate_length_mm + self.lambda0,
                    self.lambda0 * 2,
                ]
            ),
            self.fp_precision,
        )


class PatchAntenna(SimTools):
    def __init__(self, params, CSX, FDTD):
        self.params = params
        self.CSX = CSX
        self.FDTD = FDTD

    def create_substrate(self):
        substrate = self.CSX.AddMaterial(
            "substrate",
            epsilon=self.params.substrate_eps_r,
            kappa=self.params.substrate_kappa,
        )
        substrate.SetColor("#0F8A00", 100)
        substrate_start = [
            -self.params.substrate_width_mm / 2,
            -self.params.substrate_length_mm / 2,
            0,
        ]
        substrate_stop = [
            self.params.substrate_width_mm / 2,
            self.params.substrate_length_mm / 2,
            self.params.substrate_thickness_mm,
        ]
        substrate.AddBox(priority=0, start=substrate_start, stop=substrate_stop)

    def create_ground(self):
        ground = self.CSX.AddMetal("ground")
        ground.SetColor("#B87333", 255)
        ground_start = [
            -self.params.substrate_width_mm / 2,
            -self.params.substrate_length_mm / 2,
            0,
        ]
        ground_stop = [
            self.params.substrate_width_mm / 2,
            self.params.substrate_length_mm / 2,
            -self.params.copper_thickness_mm,
        ]
        ground.AddBox(priority=2, start=ground_start, stop=ground_stop)


class InsetFedPatchAntenna(PatchAntenna):
    def create_patch_with_inset(self):
        patch_inset = self.CSX.AddMetal("patch_inset")
        patch_inset.SetColor("#B87333", 255)
        if self.params.inset_width_mm / 2 < 0.089:
            warnings.warn(
                f"Inset Width {self.params.inset_width_mm / 2} is too small, check minimum trace spacing with your PCB manufacturer",
                UserWarning,
            )
        pp = [
            [
                -self.params.patch_width_mm / 2,
                -self.params.feed_width_mm / 2 - self.params.inset_width_mm / 2,
                -self.params.feed_width_mm / 2 - self.params.inset_width_mm / 2,
                self.params.feed_width_mm / 2 + self.params.inset_width_mm / 2,
                self.params.feed_width_mm / 2 + self.params.inset_width_mm / 2,
                self.params.patch_width_mm / 2,
                self.params.patch_width_mm / 2,
                -self.params.patch_width_mm / 2,
            ],
            [
                -self.params.patch_length_mm / 2,
                -self.params.patch_length_mm / 2,
                -self.params.patch_length_mm / 2 + self.params.inset_length_mm,
                -self.params.patch_length_mm / 2 + self.params.inset_length_mm,
                -self.params.patch_length_mm / 2,
                -self.params.patch_length_mm / 2,
                self.params.patch_length_mm / 2,
                self.params.patch_length_mm / 2,
            ],
        ]

        patch_inset.AddLinPoly(
            pp, "z", self.params.substrate_thickness_mm, self.params.copper_thickness_mm
        )
        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=patch_inset,
            metal_edge_res=self.params.metal_mesh_resolution,
        )

    def create_feed(self):
        feed = self.CSX.AddMetal("feed")
        feed.SetColor("#B87333", 255)
        if self.params.feed_width_mm < 0.1:
            warnings.warn(
                f"Trace Width {self.params.feed_width_mm} is too small, check minimum trace width with your PCB manufacturer",
                UserWarning,
            )
        feed_start = [
            self.params.feed_width_mm / 2,
            -self.params.patch_length_mm / 2 + self.params.inset_length_mm,
            self.params.substrate_thickness_mm,
        ]
        feed_stop = [
            -self.params.feed_width_mm / 2,
            -self.params.patch_length_mm / 2 - self.params.feed_length_mm,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        feed.AddBox(priority=6, start=feed_start, stop=feed_stop)
        self.FDTD.AddEdges2Grid(
            dirs="xy", properties=feed, metal_edge_res=self.params.metal_mesh_resolution
        )

    def create_port(self):
        port_position = -self.params.patch_length_mm / 2 - self.params.feed_length_mm
        port_start = [self.params.feed_width_mm / 2, port_position, 0]
        port_stop = [
            -self.params.feed_width_mm / 2,
            port_position,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        port = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_start,
            port_stop,
            "z",
            1.0,
            priority=6,
            edges2grid="y",
        )
        return port

    def create_mesh(self):
        mesh = self.CSX.GetGrid()
        mesh.SetDeltaUnit(self.params.unit)

        mesh.AddLine(
            "x", [-self.params.simulation_box[0] / 2, self.params.simulation_box[0] / 2]
        )
        mesh.AddLine(
            "y", [-self.params.simulation_box[1] / 2, self.params.simulation_box[1] / 2]
        )
        mesh.AddLine(
            "z",
            [-self.params.simulation_box[2] / 3, self.params.simulation_box[2] * 2 / 3],
        )
        # Add mesh lines for substrate
        mesh.AddLine(
            "x",
            [-self.params.substrate_width_mm / 2, self.params.substrate_width_mm / 2],
        )
        mesh.AddLine(
            "y",
            [-self.params.substrate_length_mm / 2, self.params.substrate_length_mm / 2],
        )
        # Add mesh lines for patch and feed
        mesh.AddLine("x", -self.params.patch_width_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("x", self.params.patch_width_mm / 2 + self.params.thirds_rule)
        mesh.AddLine("y", -self.params.patch_length_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("y", self.params.patch_length_mm / 2 + self.params.thirds_rule)

        mesh.AddLine("x", -self.params.feed_width_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("x", self.params.feed_width_mm / 2 + self.params.thirds_rule)
        mesh.AddLine("y", -self.params.patch_length_mm / 2 - self.params.feed_length_mm)
        mesh.AddLine(
            "y", -self.params.patch_length_mm / 2 + self.params.inset_length_mm
        )

        mesh.AddLine(
            "z",
            np.linspace(
                -self.params.copper_thickness_mm / 2,
                self.params.substrate_thickness_mm
                + self.params.copper_thickness_mm / 2,
                self.params.substrate_cells,
            ),
        )

        mesh.SmoothMeshLines("all", self.params.mesh_resolution, 1.5)

    def automesh(self):
        mesh = self.CSX.GetGrid()
        mesh.SetDeltaUnit(self.params.unit)

        primitives_mesh_setup = {}
        properties_mesh_setup = {}
        global_mesh_setup = {
            # Either provide start/stop OR f0/fc (unit = drawing units)
            "start_frequency": self.params.resonant_freq,
            "stop_frequency": self.params.corner_freq,
            # alternative: 'f0': 2e9, 'fc': 1e9,
            "drawing_unit": self.params.unit,  # geometry unit (meters per drawing unit); 1e-6 => um units
            "mesh_resolution": "medium",  # one of: 'low'|'medium'|'high'|'very_high'
            # Optional knobs
            # Heuristics/toggles
            "smooth_metal_edge": "one_third_two_thirds",  # useful for thin metal layers, Options: False, 'one_third_two_thirds', 'extra_lines'
            "use_circle_detection": False,  # detect circles for better angular resolution
            "handle_closely_placed_edges": True,  # if True, then mesher will try to handle close placed edges by merging them
        }
        CSX = enhance_csx_for_auto_mesh(self.CSX, primitives_mesh_setup={})
        FDTD = enhance_FDTD_for_auto_mesh(self.FDTD, primitives_mesh_setup={})
        GenerateMesh(
            CSX, global_mesh_setup, primitives_mesh_setup, properties_mesh_setup
        )


@dataclass
class ProbePatchParams(SimParams):
    patch_length_mm: float = field(init=False)
    patch_width_mm: float = field(init=False)
    probe_pos: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.patch_dims = patch_dims(
            self.resonant_freq,
            self.substrate_eps_r,
            self.substrate_thickness_mm * self.unit,
            self.charac_imp,
            self.copper_thickness_mm,
        )
        self.patch_width_mm = np.round(
            self.patch_dims.patch_width_mm, self.fp_precision
        )
        self.patch_length_mm = np.round(
            self.patch_dims.patch_length_mm, self.fp_precision
        )
        self.probe_pos = np.round((self.patch_dims.probe_pos_mm), self.fp_precision)
        self.substrate_width_mm = np.round(
            self.patch_width_mm + 2 * self.lambda0, self.fp_precision
        )
        self.substrate_length_mm = np.round(
            self.patch_length_mm + 2 * self.lambda0, self.fp_precision
        )

        self.simulation_box = np.round(
            np.array(
                [
                    self.substrate_width_mm + self.lambda0,
                    self.substrate_length_mm + self.lambda0,
                    self.lambda0 * 2,
                ]
            ),
            self.fp_precision,
        )


class ProbeFedPatchAntenna(PatchAntenna):
    def create_probe_fed_patch(self):
        patch_probe = self.CSX.AddMetal("patch_probe")
        patch_probe.SetColor("#B87333", 255)
        patch_start = [
            self.params.patch_width_mm / 2,
            self.params.patch_length_mm / 2,
            self.params.substrate_thickness_mm,
        ]
        patch_stop = [
            -self.params.patch_width_mm / 2,
            -self.params.patch_length_mm / 2,
            self.params.substrate_thickness_mm + self.params.copper_thickness_mm,
        ]
        patch_probe.AddBox(priority=6, start=patch_start, stop=patch_stop)
        self.FDTD.AddEdges2Grid(
            dirs="xy",
            properties=patch_probe,
            metal_edge_res=self.params.metal_mesh_resolution,
        )

    def create_port(self):
        port_start = [1, self.params.probe_pos, 0]
        port_stop = [0, self.params.probe_pos, self.params.substrate_thickness_mm]
        port = self.FDTD.AddLumpedPort(
            1,
            self.params.charac_imp,
            port_start,
            port_stop,
            "z",
            1.0,
            priority=6,
            edges2grid="xy",
        )
        return port

    def create_mesh(self):
        mesh = self.CSX.GetGrid()
        mesh.SetDeltaUnit(self.params.unit)

        mesh.AddLine("y", self.params.probe_pos)
        mesh.AddLine(
            "x",
            np.linspace(
                0,
                1,
                6,
            ),
        )
        mesh.AddLine(
            "x", [-self.params.simulation_box[0] / 2, self.params.simulation_box[0] / 2]
        )
        mesh.AddLine(
            "y", [-self.params.simulation_box[1] / 2, self.params.simulation_box[1] / 2]
        )
        mesh.AddLine(
            "z",
            [-self.params.simulation_box[2] / 3, self.params.simulation_box[2] * 2 / 3],
        )
        mesh.AddLine("x", -self.params.patch_width_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("x", self.params.patch_width_mm / 2 + self.params.thirds_rule)
        mesh.AddLine("y", -self.params.patch_length_mm / 2 - self.params.thirds_rule)
        mesh.AddLine("y", self.params.patch_length_mm / 2 + self.params.thirds_rule)
        mesh.AddLine(
            "z",
            np.linspace(
                -self.params.copper_thickness_mm / 2,
                self.params.substrate_thickness_mm
                + self.params.copper_thickness_mm / 2,
                self.params.substrate_cells,
            ),
        )

        mesh.SmoothMeshLines("all", self.params.mesh_resolution, 1.5)
