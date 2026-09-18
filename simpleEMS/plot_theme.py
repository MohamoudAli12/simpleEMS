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
Dark and light themes for simpleEMS plots, in matplotlib and PyVista.

The dark theme is applied the first time this module is imported, which happens
whenever :mod:`simpleEMS.sim_tools` is imported. :func:`use_light_theme` and
:func:`use_dark_theme` switch between the two for every figure and plotter created
afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mplcursors
from cycler import cycler
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget
    from pyvista.plotting.themes import Theme
    from pyvistaqt import BackgroundPlotter

# Only the two theme switches are public. themed_cursor and
# style_background_plotter are internal helpers for sim_tools.
__all__ = ["use_dark_theme", "use_light_theme"]

SURFACE = "#1a1a19"
PRIMARY_TEXT = "#ffffff"
SECONDARY_TEXT = "#c3c2b7"
MUTED_TEXT = "#898781"
SPINE = "#52514e"
GRIDLINE = "#383835"
HIGHLIGHT = "#e66767"

# Fixed order: blue, orange, aqua, yellow, magenta, green, violet, red. Adjacent
# pairs stay distinguishable under colour-vision deficiency on the dark surface.
SERIES_COLORS = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

PYVISTA_THEME_NAME = "simpleems_dark"

_BASE_RCPARAMS: dict[str, object] = {
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
}

_DARK_RCPARAMS: dict[str, object] = {
    "figure.facecolor": SURFACE,
    "figure.edgecolor": SURFACE,
    "savefig.facecolor": "auto",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": SECONDARY_TEXT,
    "axes.titlecolor": PRIMARY_TEXT,
    "axes.prop_cycle": cycler(color=SERIES_COLORS),
    "text.color": PRIMARY_TEXT,
    "xtick.color": MUTED_TEXT,
    "ytick.color": MUTED_TEXT,
    "xtick.labelcolor": SECONDARY_TEXT,
    "ytick.labelcolor": SECONDARY_TEXT,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "patch.edgecolor": SPINE,
    "legend.facecolor": SURFACE,
    "legend.edgecolor": SPINE,
    "legend.framealpha": 0.9,
    "legend.labelcolor": SECONDARY_TEXT,
    "figure.hooks": ["simpleEMS.plot_theme:_style_figure_window"],
}


def use_dark_theme() -> None:
    """
    Apply the simpleEMS dark theme to matplotlib and PyVista.

    Affects every figure and plotter created afterwards; windows that are
    already open keep their look.

    Returns
    -------
    None
    """
    import pyvista as pv

    plt.style.use("default")
    plt.rcParams.update({**_BASE_RCPARAMS, **_DARK_RCPARAMS})
    pv.set_plot_theme(_dark_pyvista_theme())


def use_light_theme() -> None:
    """
    Restore the default light looks of matplotlib and PyVista.

    Keeps the simpleEMS layout and resolution settings. Affects every figure and
    plotter created afterwards; windows that are already open keep their look.

    Returns
    -------
    None
    """
    import pyvista as pv

    plt.style.use("default")
    plt.rcParams.update(_BASE_RCPARAMS)
    pv.set_plot_theme("document")


def _dark_pyvista_theme() -> Theme:
    """Build the PyVista theme matching the matplotlib dark palette."""
    import pyvista as pv

    theme = pv.themes.DarkTheme()
    theme.name = PYVISTA_THEME_NAME
    theme.title = "simpleEMS"
    theme.background = SURFACE
    theme.font.color = SECONDARY_TEXT
    theme.color = SERIES_COLORS[0]
    theme.edge_color = SPINE
    theme.outline_color = SECONDARY_TEXT
    theme.floor_color = GRIDLINE
    theme.nan_color = MUTED_TEXT
    theme.axes.x_color = HIGHLIGHT
    theme.axes.y_color = SERIES_COLORS[2]
    theme.axes.z_color = SERIES_COLORS[0]
    theme.silhouette.color = SURFACE
    return theme


def _apply_dark_qt_palette(window: QWidget) -> None:
    """Give a Qt window, its toolbars and its drop-down menus the dark palette."""
    from matplotlib.backends.qt_compat import QtGui, QtWidgets

    role = QtGui.QPalette.ColorRole
    palette = window.palette()
    for color_role, color in {
        role.Window: SURFACE,
        role.Base: SURFACE,
        role.Button: SURFACE,
        role.ToolTipBase: SURFACE,
        role.WindowText: SECONDARY_TEXT,
        role.ButtonText: SECONDARY_TEXT,
        role.Text: PRIMARY_TEXT,
        role.ToolTipText: PRIMARY_TEXT,
        role.Highlight: SERIES_COLORS[0],
        role.HighlightedText: PRIMARY_TEXT,
    }.items():
        palette.setColor(color_role, QtGui.QColor(color))
    window.setPalette(palette)
    # Drop-down menus are separate top-level windows and don't inherit the palette.
    for menu in window.findChildren(QtWidgets.QMenu):
        menu.setPalette(palette)


def _style_figure_window(figure: Figure) -> None:
    """
    Darken a matplotlib Qt figure window; a no-op on non-Qt backends.

    Registered through ``rcParams["figure.hooks"]``, so pyplot calls it for every
    new figure while the dark theme is active.
    """
    manager = figure.canvas.manager
    window = getattr(manager, "window", None)
    if window is None or not hasattr(window, "setPalette"):
        return
    _apply_dark_qt_palette(window)
    # The toolbar picks its icon colour once, at construction; re-render the icons.
    toolbar = getattr(manager, "toolbar", None)
    if toolbar is not None:
        for _text, _tooltip, image_file, callback in toolbar.toolitems:
            if callback in toolbar._actions:
                toolbar._actions[callback].setIcon(toolbar._icon(image_file + ".png"))


def style_background_plotter(plotter: BackgroundPlotter) -> None:
    """
    Give a pyvistaqt ``BackgroundPlotter`` window the dark Qt palette.

    Does nothing unless the simpleEMS dark theme is the active PyVista theme, or
    when the plotter has no Qt window.

    Parameters
    ----------
    plotter : pyvistaqt.BackgroundPlotter
        The plotter whose menus and toolbars to restyle.

    Returns
    -------
    None
    """
    import pyvista as pv

    window = getattr(plotter, "app_window", None)
    if window is None or pv.global_theme.name != PYVISTA_THEME_NAME:
        return
    _apply_dark_qt_palette(window)


def themed_cursor(artists: Iterable[Artist]) -> mplcursors.Cursor:
    """
    Create a multi-selection :func:`mplcursors.cursor` styled for the active theme.

    The tooltip takes its box and text colours from the current rcParams, and each
    selection gets a dot ringed in the surface colour at the picked point.

    Parameters
    ----------
    artists : Iterable[Artist]
        The artists (typically the lines returned by ``plt.plot``) to attach to.

    Returns
    -------
    mplcursors.Cursor
        The cursor; connect an ``"add"`` callback to set the tooltip text.
    """
    text_color = plt.rcParams["text.color"]
    surface_color = plt.rcParams["axes.facecolor"]
    markers_by_annotation: dict[Artist, Line2D] = {}

    def mark_selection(selection: mplcursors.Selection) -> None:
        artist = selection.artist
        if not isinstance(artist, Line2D) or artist.axes is None:
            return
        axes = artist.axes
        marker = Line2D(
            [selection.target[0]],
            [selection.target[1]],
            transform=axes.transData,
            linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor=artist.get_color(),
            markeredgecolor=surface_color,
            markeredgewidth=2,
            zorder=artist.get_zorder() + 1,
        )
        axes.add_artist(marker)  # add_artist, not add_line: leaves autoscaling alone
        # Draw onto the canvas buffer so mplcursors blits it with the tooltip.
        # Appending to selection.extras would force a full figure redraw per click.
        marker.draw(axes.figure.canvas.get_renderer())
        markers_by_annotation[selection.annotation] = marker

    def unmark_selection(selection: mplcursors.Selection) -> None:
        marker = markers_by_annotation.pop(selection.annotation, None)
        if marker is not None:
            marker.remove()

    cursor = mplcursors.cursor(
        artists,
        multiple=True,
        annotation_kwargs={
            "color": text_color,
            "fontsize": 9,
            "bbox": {
                "boxstyle": "round,pad=0.5",
                "facecolor": surface_color,
                "edgecolor": text_color,
                "linewidth": 0.8,
                "alpha": 0.95,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "connectionstyle": "arc3",
                "shrinkB": 0,
                "color": text_color,
            },
        },
    )
    cursor.connect("add", mark_selection)
    cursor.connect("remove", unmark_selection)
    return cursor


# Dark is the default. Python runs this once per process, on the first import of
# this module, so a later `import simpleEMS.sim_tools` never undoes a user's
# `use_light_theme()` call.
use_dark_theme()
