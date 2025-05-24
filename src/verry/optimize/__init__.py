"""
#####################################################
Optimization and root finding (:mod:`verry.optimize`)
#####################################################

.. currentmodule:: verry.optimize

This module provides solvers for root finding.

Global optimization
===================

.. autosummary::
    :toctree: generated/

    branchbound
    branchbound_scalar

Root finding
============

.. autosummary::
    :toctree: generated/

    allroot
    allroot_scalar
    krawczyk
    krawczyk_scalar

Miscellaneous
=============

.. autosummary::
    :toctree: generated/

    AllRootResult
    AllRootScalarResult

"""

from .optimize import branchbound, branchbound_scalar
from .rootfinding import (
    AllRootResult,
    AllRootScalarResult,
    allroot,
    allroot_scalar,
    krawczyk,
    krawczyk_scalar,
)

__all__ = [
    "branchbound",
    "branchbound_scalar",
    "AllRootResult",
    "AllRootScalarResult",
    "allroot",
    "allroot_scalar",
    "krawczyk",
    "krawczyk_scalar",
]
