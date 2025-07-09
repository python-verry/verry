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
    IntegratorFactory
    eilo
    kashi

Trackers
--------

.. autosummary::
    :toctree: generated/

    Tracker
    TrackerFactory
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

from .integrator import Integrator, IntegratorFactory, eilo, kashi
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
from .tracker import (
    Tracker,
    TrackerFactory,
    affinetracker,
    directtracker,
    doubletontracker,
    qrtracker,
)
from .utility import seriessol, variationaleq

__all__ = [
    "Integrator",
    "IntegratorFactory",
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
    "TrackerFactory",
    "affinetracker",
    "directtracker",
    "doubletontracker",
    "qrtracker",
    "seriessol",
    "variationaleq",
]
