import os
import sys
import inspect
import importlib
from datetime import datetime
from typing import Any, Optional

# -- Path setup --------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------------
# -- Project information -----------------------------------------------------
project = "Keras Reservoir Computing"
author = "Daniel Estevez"
copyright = f"{datetime.now().year}, {author}"

# Try importing package for version
try:
    import keras_reservoir_computing as krc  # noqa: F401
    version = release = "0.1.0"
except Exception:
    version = release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinx_design",
]

# Napoleon for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_rtype = False

# MyST configuration (Markdown)
myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "substitution",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Avoid importing the package at autosummary stage when possible
autosectionlabel_prefix_document = True
autosummary_generate = True
autosummary_ignore_module_all = False
autodoc_typehints = "description"
autodoc_default_options = {
    "members": False,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "imported-members": True,
}
# Avoid importing heavy dependencies when building docs
autodoc_mock_imports = [
    "tensorflow", "keras", "networkx", "pandas", "scipy", "rich",
    "matplotlib", "natsort", "ipykernel", "ipywidgets", "ipympl",
    "netCDF4", "xarray", "optuna", "plotly", "kaleido",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "use_edit_page_button": True,
    "github_url": "https://github.com/ZentropyUH/ESN",
    "header_links_before_dropdown": 6,
}
html_context = {
    "github_user": "ZentropyUH",
    "github_repo": "ESN",
    "github_version": "main",
    "doc_path": "docs",
}
html_title = "Keras Reservoir Computing"
html_static_path = ["_static"]
templates_path = ["_templates"]

def _linkcode_resolve(domain: str, info: dict[str, Any]) -> Optional[str]:
    if domain != "py":
        return None
    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname:
        return None
    try:
        submod = importlib.import_module(modname)
    except Exception:
        return None

    # Find object and its source file/lines
    obj = submod
    for part in fullname.split(".") if fullname else []:
        try:
            obj = getattr(obj, part)
        except Exception:
            obj = submod
            break
    try:
        fn = inspect.getsourcefile(obj) or inspect.getfile(obj)
        fn = os.path.relpath(fn, ROOT)
        source, lineno = inspect.getsourcelines(obj)
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    except Exception:
        # Fallback to module file
        try:
            fn = inspect.getsourcefile(submod) or inspect.getfile(submod)
            fn = os.path.relpath(fn, ROOT)
            linespec = ""
        except Exception:
            return None

    return f"https://github.com/ZentropyUH/ESN/blob/main/{fn}{linespec}"

linkcode_resolve = _linkcode_resolve

# Misc
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"