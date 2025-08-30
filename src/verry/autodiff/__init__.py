"""
#################################################
Automatic differentiation (:mod:`verry.autodiff`)
#################################################

.. currentmodule:: verry.autodiff

This module provides forward-mode automatic differentiation.

Differential operators
----------------------

.. autosummary::
    :toctree: generated/

    deriv
    grad
    jacobian

Number systems containing infinitesimals
----------------------------------------

.. autosummary::
    :toctree: generated/

    Dual
    Jet
    DynDual
    DynJet
    IntervalDual
    IntervalJet

"""

from .autodiff import deriv, grad, jacobian
from .dual import Dual, DynDual, DynJet, IntervalDual, IntervalJet, Jet

__all__ = [
    "deriv",
    "grad",
    "jacobian",
    "Dual",
    "DynDual",
    "DynJet",
    "IntervalDual",
    "IntervalJet",
    "Jet",
]
