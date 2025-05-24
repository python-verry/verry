"""
####################################################
Quadrature and solving ODEs (:mod:`verry.integrate`)
####################################################

.. currentmodule:: verry.integrate

This module provides verified routines for quadrature and solving ordinary differential
equations.

Quadrature functions
====================

.. autosummary::
    :toctree: generated/

    cumulative_simpson
    cumulative_trapezoid

Solving initial value problems for ODEs
=======================================

Solvers
-------

.. autosummary::
    :toctree: generated/

    C0Solver
    C1Solver

Integrators
-----------

.. autosummary::
    :toctree: generated/

    Integrator
    eilo
    kashi

Trackers
--------

.. autosummary::
    :toctree: generated/

    Tracker
    affinetracker
    directtracker
    doubletontracker
    qrtracker

Miscellaneous
-------------

.. autosummary::
    :toctree: generated/

    AbortSolving
    C0SolverCallbackArg
    C0SolverResultContent
    C1SolverCallbackArg
    C1SolverResultContent
    OdeSolution
    SolverResult
    seriessolution
    variationaleq

"""

from .integrator import Integrator, eilo, kashi
from .quad import cumulative_simpson, cumulative_trapezoid
from .solver import (
    AbortSolving,
    C0Solver,
    C0SolverCallbackArg,
    C0SolverResultContent,
    C1Solver,
    C1SolverCallbackArg,
    C1SolverResultContent,
    OdeSolution,
    SolverResult,
)
from .tracker import Tracker, affinetracker, directtracker, doubletontracker, qrtracker
from .utility import seriessolution, variationaleq

__all__ = [
    "Integrator",
    "eilo",
    "kashi",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "AbortSolving",
    "C0Solver",
    "C0SolverCallbackArg",
    "C0SolverResultContent",
    "C1Solver",
    "C1SolverCallbackArg",
    "C1SolverResultContent",
    "OdeSolution",
    "SolverResult",
    "Tracker",
    "affinetracker",
    "directtracker",
    "doubletontracker",
    "qrtracker",
    "seriessolution",
    "variationaleq",
]
