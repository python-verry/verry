import itertools
from typing import Callable

from verry.autodiff.autodiff import deriv
from verry.interval.interval import Interval


def cumulative_trapezoid[T: Interval](fun: Callable, a: T, b: T, n: int) -> T:
    """Cumulatively integrate `fun` using the composite trapezoidal rule.

    Parameters
    ----------
    fun : Callable
        Integrand. `fun` must be an univariate scalar-valued function.
    a : Interval
        Lower limit of integration.
    b : Interval
        Upper limit of integration.
    n : int
        Number of divisions.

    Warnings
    --------
    `fun` must be a :math:`C^2`-function on some open interval that contains `a` and
    `b`. Furthermore, `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> from verry import function as vrf
    >>> s = cumulative_trapezoid(lambda x: x**2 + 2 * vrf.sqrt(x), FI(1), FI(2), 300)
    >>> print(s)
    [4.771236, 4.771237]
    """
    if not (isinstance(a, Interval) and type(a) is type(b)):
        raise TypeError

    if a.sup >= b.inf:
        raise ValueError

    intvl = type(a)
    h = (b - a) / n

    approx = fun(a) + fun(b)
    approx += 2 * sum(fun(a + h * i) for i in range(1, n))
    approx *= h / 2

    mesh = (x | y for x, y in itertools.pairwise(a + h * i for i in range(n + 1)))
    dfun = deriv(deriv(fun))
    error = -intvl.hull(*(dfun(x) for x in mesh)) * (b - a) ** 3 / (12 * n**2)

    return approx + error


def cumulative_simpson[T: Interval](fun: Callable, a: T, b: T, n: int) -> T:
    """Cumulatively integrate `fun` using the composite Simpson's 1/3 rule.

    Parameters
    ----------
    fun : Callable
        Integrand. `fun` must be an univariate scalar-valued function.
    a : Interval
        Lower limit of integration.
    b : Interval
        Upper limit of integration.
    n : int
        Number of divisions.

    Warnings
    --------
    `fun` must be a :math:`C^4`-function on some open interval that contains `a` and
    `b`. Furthermore, `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> from verry import function as vrf
    >>> s = cumulative_simpson(lambda x: x**2 + 2 * vrf.sqrt(x), FI(1), FI(2), 10)
    >>> print(s)
    [4.771236, 4.771237]
    """
    if not (isinstance(a, Interval) and type(a) is type(b)):
        raise TypeError

    if a.sup >= b.inf:
        raise ValueError

    intvl = type(a)
    HALF = intvl(1) / 2
    h = (b - a) / n

    approx = fun(a) + 4 * fun(a + h * (n - HALF)) + fun(b)
    approx += 2 * sum(fun(a + h * i) for i in range(1, n))
    approx += 4 * sum(fun(a + h * (i - HALF)) for i in range(1, n))
    approx *= h / 6

    mesh = (x | y for x, y in itertools.pairwise(a + h * i for i in range(n + 1)))
    dfun = deriv(deriv(deriv(deriv(fun))))
    error = -intvl.hull(*(dfun(x) for x in mesh)) * (b - a) ** 5 / (2880 * n**4)

    return approx + error
