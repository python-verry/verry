"""
###########################################
Interval arithmetic (:mod:`verry.interval`)
###########################################

.. currentmodule:: verry.interval

This module provides basic interval arithmetic.

Intervals
=========

.. autosummary::
    :toctree: generated/

    Interval
    FloatInterval

Miscellaneous
=============

.. autosummary::
    :toctree: generated/

    Converter
    Operator
    RoundingMode

"""

from .floatinterval import FloatInterval
from .interval import (
    Converter,
    Interval,
    Operator,
    RoundingMode,
)

__all__ = [
    "FloatInterval",
    "Converter",
    "Interval",
    "Operator",
    "RoundingMode",
]
