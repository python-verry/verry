"""
##############################################
Mathematical functions (:mod:`verry.function`)
##############################################

.. currentmodule:: verry.function

This module provides mathematical functions.

Constant functions
==================

.. autosummary::
    :toctree: generated/

    e
    ln2
    pi

Power, exponents, and logarithmic functions
===========================================

.. autosummary::
    :toctree: generated/

    exp
    log
    pow
    sqrt

"""

import math
from typing import Any, overload

import mpmath
import mpmath.ctx_mp_python

from verry.affineform import AffineForm
from verry.autodiff.autodiff import _defderiv, _primitive
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries

_e: Any = None
_exp: Any = None
_ln2: Any = None
_log: Any = None
_pi: Any = None
_pow: Any = None
_sqrt: Any = None


@overload
def e[T: Interval | AffineForm | IntervalSeries](x: T, /) -> T: ...


@overload
def e(x: float | int, /) -> float: ...


@overload
def e(x: Any, /) -> Any: ...


@_primitive
def e(x, /):
    """Napier's constant.

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> print(format(e(1.0), ".6f"))
    2.718282
    >>> print(e(FI()))
    [inf=2.71828, sup=2.71829]
    """
    if fun := getattr(type(x), "_verry_overload_", None):
        if (res := fun(x, _e, x)) is not NotImplemented:
            return res

        raise TypeError

    match x:
        case mpmath.ctx_mp_python.mpnumeric():
            return mpmath.e

        case float() | int():
            return math.e

        case _:
            raise TypeError


@overload
def exp[T: Interval](x: T, /) -> T: ...


@overload
def exp(x: float | int, /) -> float: ...


@overload
def exp(x: Any, /) -> Any: ...


@_primitive
def exp(x, /):
    """Exponential.

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> print(format(exp(2), ".6f"))
    7.389056
    >>> print(exp(FI(2)))
    [inf=7.38905, sup=7.38906]
    """
    if fun := getattr(type(x), "_verry_overload_", None):
        if (res := fun(x, _exp, x)) is not NotImplemented:
            return res

        raise TypeError

    match x:
        case mpmath.ctx_mp_python.mpnumeric():
            return mpmath.exp(x)

        case float() | int():
            return math.exp(x)

        case _:
            raise TypeError


@overload
def ln2[T: Interval | AffineForm | IntervalSeries](x: T, /) -> T: ...


@overload
def ln2(x: float | int, /) -> float: ...


@overload
def ln2(x: Any, /) -> Any: ...


@_primitive
def ln2(x, /):
    """Natural logarithm of 2.

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> print(format(ln2(1.0), ".6f"))
    0.693147
    >>> print(ln2(FI()))
    [inf=0.693147, sup=0.693148]
    """
    if fun := getattr(type(x), "_verry_overload_", None):
        if (res := fun(x, _ln2, x)) is not NotImplemented:
            return res

        raise TypeError

    match x:
        case mpmath.ctx_mp_python.mpnumeric():
            return mpmath.ln2

        case float() | int():
            return float.fromhex("0x1.62e42fefa39efp-1")

        case _:
            raise TypeError


@overload
def log[T: Interval](x: T, /) -> T: ...


@overload
def log(x: float | int, /) -> float: ...


@overload
def log(x: Any, /) -> Any: ...


@_primitive
def log(x, /):
    """Natural logarithm.

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> print(format(log(5), ".6f"))
    1.609438
    >>> print(log(FI(5)))
    [inf=1.60943, sup=1.60944]
    """
    if fun := getattr(type(x), "_verry_overload_", None):
        if (res := fun(x, _log, x)) is not NotImplemented:
            return res

        raise TypeError

    match x:
        case mpmath.ctx_mp_python.mpnumeric():
            return mpmath.log(x)

        case float() | int():
            return math.log(x)

        case _:
            raise TypeError


@overload
def pi[T: Interval | AffineForm | IntervalSeries](x: T, /) -> T: ...


@overload
def pi(x: float | int, /) -> float: ...


@overload
def pi(x: Any, /) -> Any: ...


@_primitive
def pi(x, /):
    """Pi.

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> print(format(pi(1.0), ".6f"))
    3.141593
    >>> print(pi(FI()))
    [inf=3.14159, sup=3.14160]
    """
    if fun := getattr(type(x), "_verry_overload_", None):
        if (res := fun(x, _pi, x)) is not NotImplemented:
            return res

        raise TypeError

    match x:
        case mpmath.ctx_mp_python.mpnumeric():
            return mpmath.pi

        case float() | int():
            return math.pi

        case _:
            raise TypeError


@overload
def pow[T: Interval](x: T | float | int, y: T, /) -> T: ...


@overload
def pow[T: Interval](x: T, y: float | int, /) -> T: ...


@overload
def pow(x: float | int, y: float | int, /) -> float: ...


@overload
def pow(x: Any, y: Any, /) -> Any: ...


@_primitive
def pow(x, y, /):
    """`x` raised to the power `y`.

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> print(format(pow(3.25, 1.25), ".6f"))
    4.363693
    >>> print(pow(FI("3.25"), FI("1.25")))
    [inf=4.36369, sup=4.36370]
    """
    linearized = (x, y)

    if type(x) is not type(y) and issubclass(type(y), type(x)):
        linearized = (y, x)

    for z in linearized:
        if fun := getattr(type(z), "_verry_overload_", None):
            if (res := fun(z, _pow, x, y)) is not NotImplemented:
                return res

    mpnumeric = mpmath.ctx_mp_python.mpnumeric

    match x, y:
        case (mpnumeric(), _) | (_, mpnumeric()):
            return mpmath.power(x, y)

        case (float() | int(), float() | int()):
            return math.pow(x, y)

        case _:
            raise TypeError


@overload
def sqrt[T: Interval](x: T, /) -> T: ...


@overload
def sqrt(x: float | int, /) -> float: ...


@overload
def sqrt(x: Any, /) -> Any: ...


@_primitive
def sqrt(x, /):
    """Square root.

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> print(format(sqrt(2.0), ".6f"))
    1.414214
    >>> print(sqrt(FI(2)))
    [inf=1.41421, sup=1.41422]
    """
    if fun := getattr(type(x), "_verry_overload_", None):
        if (res := fun(x, _sqrt, x)) is not NotImplemented:
            return res

        raise TypeError

    match x:
        case mpmath.ctx_mp_python.mpnumeric():
            return mpmath.sqrt(x)

        case float() | int():
            return math.sqrt(x)

        case _:
            raise TypeError


_e = e
_exp = exp
_ln2 = ln2
_log = log
_pi = pi
_pow = pow
_sqrt = sqrt

_defderiv(e, lambda x: x * 0)
_defderiv(exp, exp)
_defderiv(ln2, lambda x: x * 0)
_defderiv(log, lambda x: 1 / x)
_defderiv(pi, lambda x: x * 0)
_defderiv(pow, lambda x, y: y * pow(x, y) / x, argnum=0)
_defderiv(pow, lambda x, y: log(x) * pow(x, y), argnum=1)
_defderiv(sqrt, lambda x: 1 / (2 * sqrt(x)))
