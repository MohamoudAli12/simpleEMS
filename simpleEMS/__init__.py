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

from .microstrip_line import (
    MicrostripLine,
    MicrostripLineParams,
)
from .patch_antenna import (
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    ProbeFedPatchAntenna,
    ProbeFedPatchParams,
)
from .quarterwave_stub_filter import (
    QuarterWaveFilterParams,
    BandStopQuarterWaveFilter,
    BandPassQuarterWaveFilter,
)
from .model import simulate_model
from .sim_tools import (
    DumpType,
    SimTools,
    optimize_s11,
    optimize_s21,
    optimize_s_params,
    param_sweep,
    setup_simulation,
)

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
]

__version__ = "0.1.0"
__author__ = "Mohamoud Ali"
__license__ = "AGPL-3.0-or-later"
