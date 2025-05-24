import os
import sys

project = "Verry"
copyright = "2025, R. Iwanami"
author = "R. Iwanami"

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx_design",
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_theme_options = {
    "logo": {"text": "Verry"},
    "navigation_depth": 1,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}

favicons = [
    "favicon.ico",
    "logo.svg",
    {"rel": "apple-touch-icon", "href": "apple-touch-icon.png"},
]

autosummary_generate = True
autodoc_typehints = "none"

napoleon_preprocess_types = False
napoleon_attr_annotations = False
napoleon_use_ivar = True

mathjax3_config = {
    "loader": {"load": ["[tex]/mathtools"]},
    "tex": {"packages": {"[+]": ["mathtools"]}},
}

extlinks = {"doi": ("https://dx.doi.org/%s", "doi:%s")}

sys.path.insert(0, os.path.abspath("../"))
