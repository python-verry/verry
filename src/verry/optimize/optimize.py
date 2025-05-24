import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from verry.autodiff.autodiff import deriv, grad
from verry.interval.interval import Interval
from verry.linalg.intervalmatrix import IntervalMatrix


def branchbound[T: Interval](
    fun: Callable,
    domain: IntervalMatrix[T],
    fprime: Callable | bool = True,
    xtol: Any = float("inf"),
    ytol: Any = 0.0,
    max_iter: int = 16,
) -> tuple[T, list[IntervalMatrix[T]]]:
    """Find the minimum of the multivariate scalar-valued function by repeatedly
    dividing the interval vector.

    Parameters
    ----------
    fun : Callable
        Function to be optimized.
    domain : IntervalMatrix
        Interval vector in which to search for the minimum value.
    fprime : Callable | bool, optional
        Gradient of `fun` (the default is ``grad(fun)``). If `fprime` is ``False``,
        The differentiability of `fun` is not assumed.
    xtol : default=inf
        Absolute tolerance.
    ytol : default=0.0
        Relative tolerance.
    max_iter : int, default=16
        Maximum number of iterations.

    Returns
    -------
    r0 : Interval
        Interval containing the minimum value.
    r1 : list[IntervalMatrix]
        Interval vectors at which `fun` may take a minimum value.

    Warnings
    --------
    `fun` must be a :math:`C^1`-function on `domain` if `fprime` is not ``False``.
    Futhermore, `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    See Also
    --------
    verry.autodiff.grad

    Examples
    --------
    >>> from verry.linalg import FloatIntervalMatrix as FIM
    >>> y, x = branchbound(lambda x, y: x**2 + y, FIM(inf=[-2, -1], sup=[1, 2]))
    >>> print(format(y, ".3f"))
    [-1.000, -0.999]
    """
    if not isinstance(domain, IntervalMatrix):
        raise TypeError

    intvlmat = type(domain)
    intvl = domain.interval
    ZERO = intvl.operator.ZERO
    INFINITY = intvl.operator.INFINITY

    if fprime is True:
        fprime = grad(fun)

    if isinstance(xtol, (float, int)):
        xtol = intvl.converter.fromfloat(float(xtol), strict=False)

    if not isinstance(xtol, intvl.endtype):
        raise TypeError

    if isinstance(ytol, (float, int)):
        ytol = intvl.converter.fromfloat(float(ytol), strict=False)

    if not isinstance(ytol, intvl.endtype):
        raise TypeError

    if max_iter <= 0:
        raise ValueError

    branches = [domain]
    bound = inf = sup = INFINITY
    argmins: list[IntervalMatrix[T]] = []

    for i in range(max_iter):
        if not branches:
            break

        next_branches: list[IntervalMatrix[T]] = []

        for x in branches:
            y = fun(*x)

            if y.inf > bound:
                continue

            x0 = x.mid()
            y0 = fun(*intvlmat(x0, intvl=intvl))

            if fprime:
                c = intvlmat(fprime(*x), intvl=intvl)

                if np.all((c.inf > ZERO) & (x.inf != domain.inf)):
                    continue

                if np.all((c.sup < ZERO) & (x.sup != domain.sup)):
                    continue

                y &= y0 + c @ (x - x0)

                if y.inf > bound:
                    continue

            bound = min(bound, y0.sup)
            is_sufficient = (x.diam() <= xtol) & (y.diam() <= abs(y.mid()) * ytol)

            if np.all(is_sufficient) or i == max_iter - 1:
                argmins.append(x)
                inf = min(inf, y.inf)
                sup = min(sup, y.sup)
                continue

            tmp = ((intvl(y.inf, y.mid()), intvl(y.mid(), y.sup)) for y in x)
            next_branches.extend(
                intvlmat(x, intvl=intvl) for x in itertools.product(*tmp)
            )

        branches = next_branches

    return intvl(inf, sup), argmins


def branchbound_scalar[T: Interval](
    fun: Callable,
    domain: T,
    fprime: Callable | bool = True,
    xtol: Any = float("inf"),
    ytol: Any = 0.0,
    max_iter: int = 16,
) -> tuple[T, list[T]]:
    """Find the minimum of the univariate scalar-valued function by repeatedly
    dividing the interval.

    Parameters
    ----------
    fun : Callable
        Function to be optimized.
    domain : Interval
        Interval in which to search for the minimum value.
    fprime : Callable | bool, optional
        Derivative of `fun` (the default is ``deriv(fun)``). If `fprime` is ``False``,
        The differentiability of `fun` is not assumed.
    xtol : default=inf
        Absolute tolerance.
    ytol : default=0.0
        Relative tolerance.
    max_iter : int, default=16
        Maximum number of iterations.

    Returns
    -------
    r0 : Interval
        Interval containing the minimum value.
    r1 : list[Interval]
        Intervals at which `fun` may take a minimum value.

    Warnings
    --------
    `fun` must be a :math:`C^1`-function on `domain` if `fprime` is not ``False``.
    Futhermore, `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    See Also
    --------
    verry.autodiff.deriv

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> y, x = branchbound_scalar(lambda x: x**3 - 2 * x + 3, FI(0, 2))
    >>> print(format(y, ".3f"))
    [1.911, 1.912]
    >>> print(format(x[0], ".3f"))
    [0.816, 0.817]
    """
    if not isinstance(domain, Interval):
        raise TypeError

    intvl: type[T] = type(domain)
    ZERO = domain.operator.ZERO
    INFINITY = domain.operator.INFINITY

    if fprime is True:
        fprime = deriv(fun)

    if isinstance(xtol, (float, int)):
        xtol = domain.converter.fromfloat(float(xtol), strict=False)

    if not isinstance(xtol, intvl.endtype):
        raise TypeError

    if isinstance(ytol, (float, int)):
        ytol = domain.converter.fromfloat(float(ytol), strict=False)

    if not isinstance(ytol, intvl.endtype):
        raise TypeError

    if max_iter <= 0:
        raise ValueError

    branches = [domain]
    bound = inf = sup = INFINITY
    argmins: list[T] = []

    for i in range(max_iter):
        if not branches:
            break

        next_branches: list[T] = []

        for x in branches:
            y = fun(x)

            if y.inf > bound:
                continue

            x0 = x.mid()
            y0 = fun(intvl(x0))

            if fprime:
                c = fprime(x)

                if c.inf > ZERO and x.inf != domain.inf:
                    continue

                if c.sup < ZERO and x.sup != domain.sup:
                    continue

                y &= y0 + c * (x - x0)

                if y.inf > bound:
                    continue

            bound = min(bound, y0.sup)
            is_sufficient = x.diam() <= xtol and y.diam() <= abs(y.mid()) * ytol

            if is_sufficient or i == max_iter - 1:
                argmins.append(x)
                inf = min(inf, y.inf)
                sup = min(sup, y.sup)
                continue

            next_branches.extend((intvl(x.inf, x0), intvl(x0, x.sup)))

        branches = next_branches

    return intvl(inf, sup), argmins
