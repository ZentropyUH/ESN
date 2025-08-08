import os
import sys
import inspect
import importlib
import types
from datetime import datetime
from typing import Any, Optional

# -- Path setup --------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# Absolute fallback for CI/containers
if "/workspace/src" not in sys.path and os.path.isdir("/workspace/src"):
    sys.path.insert(0, "/workspace/src")

# ---------------------------------------------------------------------------
# Pre-mock heavy/optional dependencies using lightweight ModuleType stubs
# ---------------------------------------------------------------------------

def _ensure_module(fullname: str) -> types.ModuleType:
    if fullname in sys.modules:
        return sys.modules[fullname]  # type: ignore[return-value]
    mod = types.ModuleType(fullname)
    sys.modules[fullname] = mod
    return mod

# tensorflow stub
_tf = _ensure_module("tensorflow")
_tf.raw_ops = types.SimpleNamespace(ControlTrigger=lambda: None)  # type: ignore[attr-defined]

# Common symbols used in annotations
class _Tensor:  # noqa: D401
    """Tensor placeholder for type annotations."""
    pass

_tf.Tensor = _Tensor

class _TensorShape:
    def __init__(self, dims):
        self.dims = dims
    def __repr__(self):
        return f"TensorShape({self.dims})"

_tf.TensorShape = _TensorShape

# Basic ops used during import-time helpers (no-ops)
_tf.constant = lambda *a, **k: None
_tf.shape = lambda x: (0,)
_tf.reduce_min = lambda xs: 0
_tf.transpose = lambda x, perm=None: x
_tf.concat = lambda tensors, axis=-1: tensors[0] if isinstance(tensors, (list, tuple)) else tensors

# tf.function decorator stub
def _tf_function(*dargs, **dkwargs):
    def _wrap(func):
        return func
    # Support bare @tf.function
    if dargs and callable(dargs[0]) and not dkwargs:
        return dargs[0]
    return _wrap

_tf.function = _tf_function

# Logger stub
class _Logger:
    def __init__(self):
        self.level = 0
    def setLevel(self, level):  # noqa: N802
        self.level = level

_tf._logger = _Logger()
_tf.get_logger = lambda: _tf._logger

# random.Generator stub
class _Generator:
    def normal(self, shape=None):
        return None

class _RandomNS:
    Generator = _Generator
    @staticmethod
    def from_seed(seed: int):  # noqa: D401
        return _Generator()
    @staticmethod
    def from_non_deterministic_state():  # noqa: D401
        return _Generator()

_tf.random = _RandomNS()

# Submodules
_tf_keras = _ensure_module("tensorflow.keras")
setattr(_tf, "keras", _tf_keras)
_tf_keras_layers = _ensure_module("tensorflow.keras.layers")
_tf_keras_utils = _ensure_module("tensorflow.keras.utils")
_tf_keras_models = _ensure_module("tensorflow.keras.models")
_tf_keras_activations = _ensure_module("tensorflow.keras.activations")
_tf_keras_callbacks = _ensure_module("tensorflow.keras.callbacks")

class _Callback:
    def __init__(self, *a, **k):
        pass

_tf_keras_callbacks.Callback = _Callback

# Minimal layer/model stubs
class _Layer:  # noqa: D401
    """Minimal Keras Layer stub for docs import."""
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name", self.__class__.__name__)
        self.dtype = kwargs.get("dtype", "float32")
        self._inbound_nodes = []
    def __call__(self, *args, **kwargs):  # behave like a layer
        return args[0] if args else None
    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class _InputLayer(_Layer):
    def __init__(self, batch_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.batch_shape = batch_shape or (None,)

class _Model:
    def __init__(self, *args, **kwargs):
        self.inputs = []
        self.layers = []
        self.output = None
        self.output_names = []
        self.dtype = kwargs.get("dtype", "float32")
    def __call__(self, *args, **kwargs):
        return args[0] if args else None
    def predict(self, *args, **kwargs):
        return None
    def save(self, *args, **kwargs):
        return None
    def get_layer(self, name):
        return _Layer(name=name)
    def set_weights(self, *a, **k):
        pass
    def get_weights(self):
        return []

class _Input:
    def __init__(self, *args, **kwargs):
        self.name = "input"
        self._keras_history = ( _InputLayer(batch_shape=kwargs.get("shape", (None,))), 0, 0)

# Activations serialize
_tf_keras_activations.serialize = lambda x: str(x)

# Bind stubs
_tf_keras_layers.Layer = _Layer
_tf_keras_layers.Input = _Input
_tf_keras_layers.InputLayer = _InputLayer
_tf_keras_layers.Concatenate = _Layer
_tf_keras_layers.Dense = _Layer
_tf_keras_layers.RNN = _Layer
_tf_keras_utils.register_keras_serializable = lambda *a, **k: (lambda obj: obj)
_tf_keras_models.Model = _Model
setattr(_tf_keras, "layers", _tf_keras_layers)
setattr(_tf_keras, "utils", _tf_keras_utils)
setattr(_tf_keras, "Model", _Model)
setattr(_tf_keras, "activations", _tf_keras_activations)
setattr(_tf_keras, "callbacks", _tf_keras_callbacks)

# Minimal top-level keras stub for internal import used in loaders
_keras = _ensure_module("keras")
_keras_src = _ensure_module("keras.src")
_keras_src_saving = _ensure_module("keras.src.saving")

def _deserialize_keras_object(config):
    class _Obj:
        def __init__(self, **kwargs):
            pass
    return _Obj

setattr(_keras_src_saving, "deserialize_keras_object", _deserialize_keras_object)

# Other heavy libs mocked via empty modules
for name in [
    "networkx", "pandas", "scipy", "rich", "matplotlib", "natsort",
    "ipykernel", "ipywidgets", "ipympl", "netCDF4", "xarray", "optuna", "plotly", "kaleido",
]:
    _ensure_module(name)

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

autosummary_generate = True
# Avoid importing the package at autosummary stage when possible
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
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
    "tensorflow": ("https://www.tensorflow.org/api_docs/python/", None),
    "keras": ("https://keras.io/api/", None),
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