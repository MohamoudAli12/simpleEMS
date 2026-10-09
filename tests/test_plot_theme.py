"""Tests for the matplotlib and PyVista themes in ``simpleEMS.plot_theme``.

Everything runs on the Agg backend (``conftest.py``). The Qt palette code runs
against stand-in windows that hold a real ``QPalette``, so no window ever
opens.
"""

import importlib
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pytest
import pyvista as pv
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import to_rgb

import simpleEMS.plot_theme as plot_theme


@pytest.fixture(autouse=True)
def _restore_dark_theme():
    """Leave the package default (dark) active for whatever test runs next."""
    yield
    plot_theme.use_dark_theme()


def select_point(line, index):
    """Click ``line`` at data point ``index`` through a themed cursor."""
    figure, axes = line.figure, line.axes
    cursor = plot_theme.themed_cursor([line])
    figure.canvas.draw()
    x_pixel, y_pixel = axes.transData.transform(line.get_xydata()[index])
    event = MouseEvent("button_press_event", figure.canvas, x_pixel, y_pixel, button=1)
    pick = mplcursors.compute_pick(line, event)
    return cursor, cursor.add_selection(pick, figure, axes)


def added_markers(line):
    return [other for other in line.axes.lines if other is not line]


class TestMatplotlibThemes:
    def test_dark_theme_sets_the_surface_and_series_colours(self):
        plot_theme.use_dark_theme()

        assert plt.rcParams["axes.facecolor"] == plot_theme.SURFACE
        assert plt.rcParams["figure.facecolor"] == plot_theme.SURFACE
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        assert cycle_colors == plot_theme.SERIES_COLORS

    def test_dark_theme_registers_the_figure_window_hook(self):
        plot_theme.use_dark_theme()

        assert plt.rcParams["figure.hooks"] == [
            "simpleEMS.plot_theme:_style_figure_window"
        ]

    def test_light_theme_restores_defaults_but_keeps_layout_and_dpi(self):
        plot_theme.use_light_theme()

        assert plt.rcParams["axes.facecolor"] == "white"
        assert plt.rcParams["figure.hooks"] == []
        assert plt.rcParams["savefig.dpi"] == 300
        assert plt.rcParams["figure.constrained_layout.use"] is True

    def test_switching_themes_leaves_the_backend_alone(self):
        backend = matplotlib.get_backend()

        plot_theme.use_light_theme()
        plot_theme.use_dark_theme()

        assert matplotlib.get_backend() == backend


class TestPyvistaThemes:
    def test_dark_theme_activates_the_simpleems_pyvista_theme(self):
        plot_theme.use_dark_theme()

        assert pv.global_theme.name == plot_theme.PYVISTA_THEME_NAME
        assert pv.global_theme.background == plot_theme.SURFACE

    def test_light_theme_restores_the_document_theme(self):
        plot_theme.use_light_theme()

        assert pv.global_theme.name == "document"

    @pytest.mark.parametrize(
        "switch_theme",
        [plot_theme.use_dark_theme, plot_theme.use_light_theme],
        ids=["dark", "light"],
    )
    def test_style_background_plotter_skips_a_plotter_without_a_window(
        self, switch_theme
    ):
        switch_theme()

        plot_theme.style_background_plotter(SimpleNamespace())


class TestDefaultTheme:
    @pytest.mark.needs_csxcad
    def test_importing_sim_tools_after_use_light_theme_keeps_it_light(self):
        plot_theme.use_light_theme()

        importlib.import_module("simpleEMS.sim_tools")

        assert plt.rcParams["axes.facecolor"] == "white"


class TestThemedCursor:
    @pytest.fixture
    def line(self):
        plt.figure()
        (line,) = plt.plot(np.linspace(0, 1, 50), np.linspace(0, 1, 50) ** 2)
        return line

    def test_tooltip_box_uses_the_active_theme_colours(self, line):
        _cursor, selection = select_point(line, 25)

        box = selection.annotation.get_bbox_patch()
        assert box.get_facecolor()[:3] == pytest.approx(to_rgb(plot_theme.SURFACE))
        assert selection.annotation.get_color() == plot_theme.PRIMARY_TEXT

    def test_selection_adds_a_ringed_marker_in_the_line_colour(self, line):
        _cursor, selection = select_point(line, 25)

        (marker,) = added_markers(line)
        assert marker.get_markerfacecolor() == line.get_color()
        assert marker.get_markeredgecolor() == plot_theme.SURFACE
        assert marker.get_xydata()[0] == pytest.approx(selection.target)

    def test_marker_keeps_selection_extras_empty_for_the_blit_path(self, line):
        _cursor, selection = select_point(line, 25)

        assert selection.extras == []

    def test_removing_a_selection_removes_its_marker(self, line):
        cursor, selection = select_point(line, 25)

        cursor.remove_selection(selection)

        assert added_markers(line) == []


class TestFigureWindowHook:
    def test_hook_is_a_no_op_on_the_agg_backend(self):
        figure = plt.figure()

        plot_theme._style_figure_window(figure)


class FakeMenu:
    def __init__(self):
        self.palette = None

    def setPalette(self, palette):  # noqa: N802 - matches Qt
        self.palette = palette


class FakeWindow:
    """Just enough of a Qt window for the palette code: a real ``QPalette``."""

    def __init__(self):
        from matplotlib.backends.qt_compat import QtGui

        self._palette = QtGui.QPalette()
        self.applied = None
        self.menus = [FakeMenu(), FakeMenu()]

    def palette(self):
        return self._palette

    def setPalette(self, palette):  # noqa: N802 - matches Qt
        self.applied = palette

    def findChildren(self, widget_type):  # noqa: N802 - matches Qt
        return self.menus


def palette_color(palette, role_name):
    from matplotlib.backends.qt_compat import QtGui

    return palette.color(getattr(QtGui.QPalette.ColorRole, role_name)).name()


class TestDarkQtPalette:
    @pytest.mark.parametrize(
        ("role_name", "color"),
        [
            ("Window", plot_theme.SURFACE),
            ("Base", plot_theme.SURFACE),
            ("WindowText", plot_theme.SECONDARY_TEXT),
            ("Text", plot_theme.PRIMARY_TEXT),
            ("Highlight", plot_theme.SERIES_COLORS[0]),
        ],
        ids=["window", "base", "window-text", "text", "highlight"],
    )
    def test_sets_the_dark_colours(self, role_name, color):
        window = FakeWindow()

        plot_theme._apply_dark_qt_palette(window)

        assert palette_color(window.applied, role_name) == color.lower()

    def test_menus_get_the_palette_too(self):
        """Drop-down menus are their own windows and do not inherit it."""
        window = FakeWindow()

        plot_theme._apply_dark_qt_palette(window)

        assert all(menu.palette is window.applied for menu in window.menus)


class TestStyleFigureWindow:
    class FakeAction:
        def __init__(self):
            self.icon = None

        def setIcon(self, icon):  # noqa: N802 - matches Qt
            self.icon = icon

    @pytest.fixture
    def figure(self):
        home = self.FakeAction()
        toolbar = SimpleNamespace(
            toolitems=[
                ("Home", "Reset", "home", "home"),
                (None, None, None, None),
            ],
            _actions={"home": home},
            _icon=lambda file_name: f"icon:{file_name}",
        )
        manager = SimpleNamespace(window=FakeWindow(), toolbar=toolbar)
        return SimpleNamespace(canvas=SimpleNamespace(manager=manager)), home

    def test_darkens_a_qt_window(self, figure):
        fake_figure, _home = figure

        plot_theme._style_figure_window(fake_figure)

        window = fake_figure.canvas.manager.window
        assert palette_color(window.applied, "Window") == plot_theme.SURFACE.lower()

    def test_re_renders_the_toolbar_icons(self, figure):
        fake_figure, home = figure

        plot_theme._style_figure_window(fake_figure)

        assert home.icon == "icon:home.png"

    def test_a_window_without_a_toolbar_is_still_darkened(self, figure):
        fake_figure, _home = figure
        fake_figure.canvas.manager.toolbar = None

        plot_theme._style_figure_window(fake_figure)

        assert fake_figure.canvas.manager.window.applied is not None


class TestStyleBackgroundPlotter:
    def test_darkens_the_window_under_the_dark_theme(self):
        plot_theme.use_dark_theme()
        plotter = SimpleNamespace(app_window=FakeWindow())

        plot_theme.style_background_plotter(plotter)

        assert plotter.app_window.applied is not None

    def test_leaves_the_window_alone_under_the_light_theme(self):
        plot_theme.use_light_theme()
        plotter = SimpleNamespace(app_window=FakeWindow())

        plot_theme.style_background_plotter(plotter)

        assert plotter.app_window.applied is None


class TestThemedCursorOnOtherArtists:
    def test_a_selection_off_a_line_adds_no_marker(self):
        """Only a Line2D gets the ringed marker; a scatter is left as is."""
        figure, axes = plt.subplots()
        scatter = axes.scatter([0.2, 0.5, 0.8], [0.2, 0.5, 0.8])
        cursor = plot_theme.themed_cursor([scatter])
        figure.canvas.draw()
        x_pixel, y_pixel = axes.transData.transform((0.5, 0.5))
        event = MouseEvent(
            "button_press_event", figure.canvas, x_pixel, y_pixel, button=1
        )

        cursor.add_selection(mplcursors.compute_pick(scatter, event), figure, axes)

        assert len(axes.lines) == 0
