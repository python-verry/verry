"""
#################################################
Automatic differentiation (:mod:`verry.autodiff`)
#################################################

.. currentmodule:: verry.autodiff

This module provides forward-mode automatic differentiation.

.. autosummary::
    :toctree: generated/

    deriv
    grad
    jacobian

"""

from .autodiff import deriv, grad, jacobian

__all__ = ["deriv", "grad", "jacobian"]
