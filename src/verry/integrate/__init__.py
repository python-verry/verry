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
    affine
    direct
    doubleton
    qr

VarEqSolvers
------------

.. autosummary::
    :toctree: generated/

    VarEqSolver
    VarEqSolverFactory
    brute
    lognorm

Miscellaneous
-------------

.. autosummary::
    :toctree: generated/

    AbortSolving
    C0SolverCallbackArg
    C0SolverResultContent
    C1SolverCallbackArg
    C1SolverResultContent
    ODESolution
    SolverResult

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
    ODESolution,
    SolverResult,
)
from .tracker import (
    Tracker,
    TrackerFactory,
    affine,
    direct,
    doubleton,
    qr,
)
from .vareqsolver import VarEqSolver, VarEqSolverFactory, brute, lognorm

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
    "ODESolution",
    "SolverResult",
    "Tracker",
    "TrackerFactory",
    "affine",
    "direct",
    "doubleton",
    "qr",
    "VarEqSolver",
    "VarEqSolverFactory",
    "brute",
    "lognorm",
]
