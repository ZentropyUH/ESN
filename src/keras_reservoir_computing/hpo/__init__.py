# =============================================================
# krc/hpo/__init__.py
# =============================================================
"""Top-level API for the hyper-parameter optimisation (HPO) module.

This sub-package exposes two public symbols only:
    * :func:`run_hpo` - convenience wrapper executing an Optuna study.
    * :data:`LOSSES` - registry mapping string names to built-in loss functions.

Everything else is considered *private* implementation detail.
"""
from .main import run_hpo
from ._losses import LOSSES


__all__ = ["run_hpo", "LOSSES"]


