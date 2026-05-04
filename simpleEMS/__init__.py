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

from .patch_antenna import (
    InsetFedPatchAntenna,
    ProbeFedPatchAntenna,
    InsetFedPatchParams,
    ProbeFedPatchParams,
)

from .sim_tools import (
    DumpType,
    SimTools,
    setup_simulation,
    optimize_s11,
    optimize_s_params,
    param_sweep,
)
from .microstrip_line import (
    MicrostripLineParams,
    MicrostripLine,
)

__all__ = [
    "InsetFedPatchAntenna",
    "ProbeFedPatchAntenna",
    "MicrostripLine",
    "InsetFedPatchParams",
    "ProbeFedPatchParams",
    "MicrostripLineParams",
    "DumpType",
    "SimTools",
    "setup_simulation",
    "optimize_s11",
    "optimize_s_params",
    "param_sweep",
]

__version__ = "0.1.0"
__author__ = "Mohamoud Ali"
__license__ = "AGPL-3.0-or-later"
