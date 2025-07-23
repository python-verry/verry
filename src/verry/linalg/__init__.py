"""
##############################################
Numerical linear algebra (:mod:`verry.linalg`)
##############################################

.. currentmodule:: verry.linalg

This module provides rigorous numerical linear algebra.

Matrices
========

.. autosummary::
    :toctree: generated/

    IntervalMatrix
    FloatIntervalMatrix

Operations
==========

.. autosummary::
    :toctree: generated/

    approx_inv
    approx_norm
    approx_qr
    approx_solve
    inv
    norm
    solve

Miscellaneous
=============

.. autosummary::
    :toctree: generated/

    LinAlgError

"""

from .floatintervalmatrix import FloatIntervalMatrix
from .intervalmatrix import (
    IntervalMatrix,
    LinAlgError,
    approx_inv,
    approx_norm,
    approx_qr,
    approx_solve,
    inv,
    norm,
    solve,
)

__all__ = [
    "FloatIntervalMatrix",
    "IntervalMatrix",
    "LinAlgError",
    "approx_inv",
    "approx_norm",
    "approx_qr",
    "approx_solve",
    "inv",
    "norm",
    "solve",
]
