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
from pathlib import Path
import shutil

def setup(app):
    def copy_images(app):
        docs_dir = Path(__file__).parent.parent.resolve()
        src = (docs_dir / "../images").resolve()
        dst = (docs_dir / "build/html/images").resolve()

        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    app.connect("builder-inited", copy_images)


extensions = [
    "sphinx.ext.autodoc",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "myst_parser",
]
autodoc_member_order = "bysource"

automodapi_toctreedirnm = "api/build"

templates_path = ["_templates"]
html_static_path = ["_static"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

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
