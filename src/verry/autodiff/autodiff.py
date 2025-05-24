import functools
from collections.abc import Callable
from typing import Any

from verry.autodiff._dual import Dual, Vector


def deriv[T, **P](fun: Callable[P, T]) -> Callable[P, T]:
    """Return a function that evaluates the derivative of the univariate scalar-valued
    function.

    Parameters
    ----------
    fun : Callable
        Diffarentiated function.

    Returns
    -------
    Callable
        Derivative of `fun`.

    Warnings
    --------
    `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    Examples
    --------
    This example computes a value of the derivative.

    >>> from verry import FloatInterval as FI
    >>> from verry import function as vrf
    >>> f = lambda x: x**2 + vrf.sqrt(x + 3)
    >>> df = deriv(f)
    >>> c0 = df(1.2)
    >>> print(format(c0, ".6g"))
    2.64398

    To obtain the value that accounts for rounding errors, pass an interval to `df`.

    >>> c1 = df(FI("1.2"))
    >>> print(format(c1, ".6g"))
    [2.64397, 2.64398]

    Also, the second-order derivative can be obtained in the same way.

    >>> ddf = deriv(df)
    >>> print(format(ddf(FI("1.2")), ".6g"))
    [1.97095, 1.97096]
    """

    def result(*args, **kwargs):
        tmp: Any = fun(*Dual.variable(*args), **kwargs)  # type: ignore
        return tmp.imag[0]

    return result


def grad[T, **P](fun: Callable[P, T]) -> Callable[P, tuple[T, ...]]:
    """Return a function that evaluates the gradient of the multivariate scalar-valued
    function.

    Parameters
    ----------
    fun : Callable
        Diffarentiated function.

    Returns
    -------
    Callable
        Gradient of `fun`.

    Warnings
    --------
    `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    Examples
    --------
    This example computes a value of the gradient.

    >>> from verry import FloatInterval as FI
    >>> from verry import function as vrf
    >>> f = lambda x, y: vrf.sqrt(x * y + 3)
    >>> df = grad(f)
    >>> c0 = df(0.5, 1.0)
    >>> print(format(c0[0], ".6g"), format(c0[1], ".6g"))
    0.267261 0.133631

    To obtain the value that accounts for rounding errors, pass intervals to `df`.

    >>> c1 = df(FI("0.5"), FI("1"))
    >>> print(format(c1[0], ".6g"))
    [0.267261, 0.267262]
    """

    def result(*args, **kwargs):
        tmp: Any = fun(*Dual.variable(*args), **kwargs)  # type: ignore
        return tuple(tmp.imag)

    return result


def jacobian[T: tuple, **P](fun: Callable[P, T]) -> Callable[P, tuple[T, ...]]:
    """Return a function that evaluates the Jacobian matrix of the multivariate
    vector-valued function.

    Parameters
    ----------
    fun : Callable
        Diffarentiated function.

    Returns
    -------
    Callable
        Fr\u00e9chet derivative of `fun`.

    Warnings
    --------
    `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).
    """

    def result(*args, **kwargs):
        tmp: Any = fun(*Dual.variable(*args), **kwargs)  # type: ignore
        return tuple(tuple(x.imag) for x in tmp)

    return result


def _defderiv[**P](
    fun: Callable[P, Any], deriv: Callable[P, Any], *, argnum: int = 0
) -> None:
    if "_verry_is_primitive" not in fun.__dict__:
        raise ValueError

    fun.__dict__["_verry_derivs"][argnum] = deriv


def _primitive[T, **P](fun: Callable[P, T]) -> Callable[P, T]:
    derivs: dict[int, Callable] = {}

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        max_level = max((x.level for x in args if isinstance(x, Dual)), default=0)

        if max_level == 0:
            return fun(*args, **kwargs)

        real: list = []
        imag: list[tuple[int, Vector]] = []

        for i, x in enumerate(args):
            if isinstance(x, Dual) and x.level == max_level:
                real.append(x.real)
                imag.append((i, x.imag))
            else:
                real.append(x)

        tmp = [derivs[i](*real, **kwargs) * x for i, x in imag]
        return Dual(wrapper(*real, **kwargs), sum(tmp[1:], start=tmp[0]))

    wrapper.__dict__["_verry_is_primitive"] = True
    wrapper.__dict__["_verry_derivs"] = derivs
    return wrapper  # type: ignore
