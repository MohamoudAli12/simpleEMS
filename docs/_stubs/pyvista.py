"""Docs-build stand-in for PyVista: just enough for simpleEMS.plot_theme's import."""

from types import SimpleNamespace


class _Theme:
    def __init__(self) -> None:
        self.name = "document"
        self.font = SimpleNamespace()
        self.axes = SimpleNamespace()
        self.silhouette = SimpleNamespace()


themes = SimpleNamespace(DarkTheme=_Theme)
global_theme = _Theme()


def set_plot_theme(theme: object) -> None:
    pass
