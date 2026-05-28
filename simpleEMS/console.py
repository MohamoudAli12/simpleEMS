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
Console output configuration using the Rich library.

Defines a colour theme and a shared Console instance used across
all simpleEMS modules for consistent terminal output.
"""

from rich.console import Console
from rich.theme import Theme

color_theme = Theme(
    {
        "black": "black",
        "red": "red",
        "green": "green",
        "yellow": "yellow",
        "blue": "blue",
        "magenta": "magenta",
        "cyan": "cyan",
        "white": "white",
        "bright_red": "bright_red",
        "bright_green": "bright_green",
        "bright_yellow": "bright_yellow",
        "bright_blue": "bright_blue",
        "bright_magenta": "bright_magenta",
        "bright_cyan": "bright_cyan",
        "class_name": "yellow",
        "warning": "bright_red",
        "field": "cyan",
        "value": "green",
        "info": "green",
    }
)

console = Console(theme=color_theme)
