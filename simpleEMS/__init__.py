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
simpleEMS — A Python library built on openEMS for simplified antenna
and RF structure design, simulation, optimisation, and visualisation.
"""

import importlib


_NAMES_BY_MODULE = {
    "microstrip_line": ["MicrostripLine", "MicrostripLineParams"],
    "patch_antenna": [
        "InsetFedPatchAntenna",
        "InsetFedPatchParams",
        "ProbeFedPatchAntenna",
        "ProbeFedPatchParams",
    ],
    "quarterwave_stub_filter": [
        "QuarterWaveFilterParams",
        "BandStopQuarterWaveFilter",
        "BandPassQuarterWaveFilter",
    ],
    "standalone_model": ["simulate_model"],
    "fem_backend": [
        "simulate_step_fem",
        "Problem",
        "SolidSpec",
        "PortSpec",
    ],
    "fem_materials": ["FemOptions"],
    "sim_tools": [
        "DumpType",
        "SimTools",
        "optimize_s11",
        "optimize_s21",
        "optimize_s_params",
        "param_sweep",
        "setup_simulation",
    ],
}

_NAME_TO_MODULE = {}

for _module_name, _names in _NAMES_BY_MODULE.items():
    for _name in _names:
        _NAME_TO_MODULE[_name] = _module_name


def __getattr__(name: str) -> object:
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is not None:
        module = importlib.import_module(f".{module_name}", __package__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InsetFedPatchAntenna",
    "ProbeFedPatchAntenna",
    "MicrostripLine",
    "InsetFedPatchParams",
    "ProbeFedPatchParams",
    "MicrostripLineParams",
    "QuarterWaveFilterParams",
    "BandStopQuarterWaveFilter",
    "BandPassQuarterWaveFilter",
    "DumpType",
    "SimTools",
    "setup_simulation",
    "optimize_s11",
    "optimize_s21",
    "optimize_s_params",
    "param_sweep",
    "simulate_model",
    "simulate_step_fem",
    "Problem",
    "SolidSpec",
    "PortSpec",
    "FemOptions",
]

__version__ = "0.1.0"
__author__ = "Mohamoud Ali"
__license__ = "AGPL-3.0-or-later"
