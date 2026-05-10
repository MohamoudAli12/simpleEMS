# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "simpleEMS"
copyright = "%Y, Mohamoud Ali"
author = "Mohamoud Ali"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import shutil
from pathlib import Path


def setup(app):
    def copy_assets(app):
        docs_dir = Path(__file__).parent.parent.resolve()
        image_src = (docs_dir / "../images").resolve()
        image_dst = (docs_dir / "build/html/images").resolve()

        changlog_src = docs_dir.parent / "CHANGELOG.md"
        changelog_dst = (docs_dir / "source").resolve()

        if image_src.exists():
            shutil.copytree(image_src, image_dst, dirs_exist_ok=True)

        if changlog_src.exists():
            shutil.copy2(changlog_src, changelog_dst)

    app.connect("builder-inited", copy_assets)


extensions = [
    "sphinx.ext.autodoc",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "myst_parser",
]

sourcautodoc_member_order = "bysource"

automodapi_toctreedirnm = "api/build"

templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 3,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MohamoudAli12",
            "icon": "fab fa-github",
        }
    ],
}
