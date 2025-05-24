"""
#############################################
Interval series (:mod:`verry.intervalseries`)
#############################################

.. currentmodule:: verry.intervalseries

This module provides operations on power series with interval coefficients.

Interval series
===============

.. autosummary::
    :toctree: generated/

    IntervalSeries

Context
=======

.. autosummary::
    :toctree: generated/

    Context
    getcontext
    localcontext
    setcontext

"""

import contextlib
import contextvars
from collections.abc import Sequence
from typing import Any, Literal, Self, overload

from verry import function as vrf
from verry.interval.interval import Interval
from verry.typing import Scalar, SignedComparable


class Context[T: Interval]:
    """Create a new context."""

    __slots__ = ("rounding", "deg", "domain")
    rounding: Literal["TYPE1", "TYPE2"]
    deg: int
    domain: T | None

    def __init__(
        self,
        rounding: Literal["TYPE1", "TYPE2"] = "TYPE1",
        deg: int = 15,
        domain: T | None = None,
    ):
        self.rounding = rounding
        self.deg = deg
        self.domain = domain

        if rounding == "TYPE2" and domain is None:
            raise ValueError

    def copy(self) -> Self:
        return self.__class__(self.rounding, self.deg, self.domain)

    def round(self, series: "IntervalSeries") -> None:
        match self.rounding:
            case "TYPE1":
                series.round_type1(self.deg)

            case "TYPE2":
                if not isinstance(self.domain, series.interval):
                    raise TypeError

                series.round_type2(self.deg, self.domain)

    def __repr__(self):
        rounding = self.rounding
        deg = self.deg

        match rounding:
            case "TYPE1":
                return f"{type(self).__name__}({rounding=!r}, {deg=})"

            case "TYPE2":
                domain = self.domain
                return f"{type(self).__name__}({rounding=!r}, {deg=}, {domain=!r})"


_var: contextvars.ContextVar[Context] = contextvars.ContextVar("intervalseries")


def getcontext() -> Context:
    """Return the current context for the active thread."""
    if ctx := _var.get(None):
        return ctx

    ctx = Context()
    _var.set(ctx)
    return ctx


def setcontext(ctx: Context) -> None:
    """Set the current context for the active thread to `ctx`."""
    _var.set(ctx)


@contextlib.contextmanager
def localcontext(
    ctx: Context | None = None,
    *,
    rounding: Literal["TYPE1", "TYPE2"] | None = None,
    deg: int | None = None,
    domain: Interval | None = None,
):
    """Return a context manager that will set the current context for the active thread
    to a copy of `ctx` on entry to the with-statement and restore the previous context
    when exiting the with-statement."""
    if ctx is None:
        ctx = getcontext()

    if rounding is None:
        rounding = ctx.rounding

    if deg is None:
        deg = ctx.deg

    if domain is None:
        domain = ctx.domain

    ctx = Context(rounding, deg, domain)
    token = _var.set(ctx)

    try:
        yield ctx
    finally:
        _var.reset(token)


class IntervalSeries[T1: Interval, T2: SignedComparable = Any](Scalar):
    """Interval series.

    Parameters
    ----------
    coeffs
        Sequence of coefficients with respect to the monomial basis.
    intvl : type[Interval] | None
        Type of each coefficient.
    """

    __slots__ = ("coeffs", "_intvl")
    coeffs: list[T1]
    _intvl: type[T1]

    def __init__(self, coeffs: Sequence[T1 | T2 | int | float | str], intvl: type[T1]):
        if not issubclass(intvl, Interval):
            raise TypeError

        self.coeffs = [intvl.ensure(x) for x in coeffs]
        self._intvl = intvl

    @property
    def interval(self) -> type[T1]:
        return self._intvl

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.coeffs})"

    def compose(self, arg: Self) -> Self:
        """Return the composition.

        See Also
        --------
        eval, __call__

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> f = IntervalSeries([4, 3, 1], intvl=FI)
        >>> g = IntervalSeries([2, -1, 5], intvl=FI)
        >>> h = f.compose(g)
        >>> print([coeff.mid() for coeff in h.coeffs])
        [14.0, -7.0, 36.0, -10.0, 25.0]
        """
        result = self.__class__([self.coeffs[-1]], intvl=self._intvl)

        for x in reversed(self.coeffs[:-1]):
            result *= arg
            result += x

        getcontext().round(result)
        return result

    def copy(self) -> Self:
        """Return a copy of the series."""
        return self.__class__([x.copy() for x in self.coeffs], intvl=self._intvl)

    def eval(self, arg: T1 | T2 | int | float) -> T1:
        """Return an interval containing the image of `arg`.

        See Also
        --------
        compose, __call__

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> f = IntervalSeries([8, 2, -1], intvl=FI)
        >>> y1 = f.eval(2)
        >>> print(format(y1, ".2f"))
        [8.00, 8.00]
        >>> y2 = f.eval(FI(-1, 1))
        >>> FI(5, 9).issubset(y2)
        True
        """
        result = self.coeffs[-1].copy()

        for x in reversed(self.coeffs[:-1]):
            result *= arg
            result += x

        return result

    def integrate(self) -> Self:
        """Integrate the series in a term by term.

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> x = IntervalSeries([0, 1], intvl=FI)
        >>> y = 3 * x**2 + 4 * x + 5
        >>> z = y.integrate()
        >>> z == x**3 + 2 * x**2 + 5 * x
        True
        """
        coeffs = [self._intvl(), *(x.copy() for x in self.coeffs)]

        for i in range(2, len(coeffs)):
            coeffs[i] = coeffs[i] / i

        return self.__class__(coeffs, intvl=self._intvl)

    def reciprocal(self) -> Self:
        """Return the reciprocal of the series.

        Raises
        ------
        ZeroDivisionError
            If the constant term contains zero.
        """
        ZERO = self._intvl.operator.ZERO

        if ZERO in (x0 := self.coeffs[0]):
            raise ZeroDivisionError

        context = getcontext()
        dx = self.copy()
        dx.coeffs[0] = self._intvl()

        result = self.__class__([0], intvl=self._intvl)
        coeff = 1 / x0
        dxpow = self.__class__([1], intvl=self._intvl)

        for _ in range(1, context.deg):
            result += coeff * dxpow
            coeff /= -x0
            dxpow *= dx

        match context.rounding:
            case "TYPE1":
                result += coeff * dxpow
                return result

            case "TYPE2":
                if not isinstance(context.domain, self._intvl):
                    raise TypeError

                coeff = -pow(-1 / self.eval(x0 | context.domain), context.deg + 1)
                result += coeff * dxpow
                return result

    def round_type1(self, deg: int) -> None:
        """Round the series in Type-I PSA. This method modifies the series in-place.

        Parameters
        ----------
        deg
            Truncation degree.

        See Also
        --------
        round_type2

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> f = IntervalSeries([1, 2, -3], intvl=FI)
        >>> g = IntervalSeries([1, -1, 1], intvl=FI)
        >>> h1 = f * g
        >>> h2 = h1.copy()
        >>> h2.round_type1(deg=2)
        >>> h1(1).issubset(h2(1))
        False
        """
        if (n := deg + 1) < len(self.coeffs):
            del self.coeffs[n:]

    def round_type2(self, deg: int, domain: T1) -> None:
        """Round the series in Type-II PSA. This method modifies the series in-place.

        Parameters
        ----------
        deg
            Truncation degree.

        See Also
        --------
        round_type1

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> f = IntervalSeries([1, 2, -3], intvl=FI)
        >>> g = IntervalSeries([1, -1, 1], intvl=FI)
        >>> h1 = f * g
        >>> h2 = h1.copy()
        >>> h2.round_type2(deg=2, domain=FI(0, 2))
        >>> h1(1).issubset(h2(1))
        True
        """
        if not isinstance(domain, self._intvl):
            raise TypeError

        if deg >= len(self.coeffs) - 1:
            return

        tmp = self.coeffs[-1].copy()

        for x in reversed(self.coeffs[deg:-1]):
            tmp *= domain
            tmp += x

        del self.coeffs[deg:]
        self.coeffs.append(tmp)

    def _verry_overload_(self, fun, *args, **kwargs):
        if fun is vrf.e:
            return self.__class__([vrf.e(self._intvl())], intvl=self._intvl)

        if fun is vrf.ln2:
            return self.__class__([vrf.ln2(self._intvl())], intvl=self._intvl)

        if fun is vrf.pi:
            return self.__class__([vrf.pi(self._intvl())], intvl=self._intvl)

        return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return other._intvl is self._intvl and other.coeffs == self.coeffs

    @overload
    def __call__(self, arg: Self) -> Self: ...

    @overload
    def __call__(self, arg: T1 | T2 | int | float) -> T1: ...

    def __call__(self, arg):
        """Return ``compose(arg)`` or ``eval(arg)`` depending on the type of `arg`.

        See Also
        --------
        compose, eval
        """
        match arg:
            case self._intvl.endtype() | self._intvl() | int() | float():
                return self.eval(arg)

            case self.__class__():
                return self.compose(arg)

            case _:
                raise TypeError

    def __add__(self, rhs: Self | T1 | T2 | int | float) -> Self:
        match rhs:
            case self._intvl.endtype() | self._intvl() | int() | float():
                result = self.copy()
                result.coeffs[0] += rhs
                return result

            case self.__class__():
                n, m = len(self.coeffs), len(rhs.coeffs)
                coeffs = [self.coeffs[i] + rhs.coeffs[i] for i in range(min(n, m))]

                if n < m:
                    for i in range(n, m):
                        coeffs.append(rhs.coeffs[i])
                else:
                    for i in range(m, n):
                        coeffs.append(self.coeffs[i])

                return self.__class__(coeffs, intvl=self._intvl)

            case _:
                return NotImplemented

    def __sub__(self, rhs: Self | T1 | T2 | int | float) -> Self:
        match rhs:
            case self._intvl.endtype() | self._intvl() | int() | float():
                result = self.copy()
                result.coeffs[0] -= rhs
                return result

            case self.__class__():
                n, m = len(self.coeffs), len(rhs.coeffs)
                coeffs = [self.coeffs[i] - rhs.coeffs[i] for i in range(min(n, m))]

                if n < m:
                    for i in range(n, m):
                        coeffs.append(-rhs.coeffs[i])
                else:
                    for i in range(m, n):
                        coeffs.append(self.coeffs[i])

                return self.__class__(coeffs, intvl=self._intvl)

            case _:
                return NotImplemented

    def __mul__(self, rhs: Self | T1 | T2 | int | float) -> Self:
        match rhs:
            case self._intvl.endtype() | self._intvl() | int() | float():
                return self.__class__([x * rhs for x in self.coeffs], intvl=self._intvl)

            case self.__class__():
                n, m = len(self.coeffs), len(rhs.coeffs)
                result = self.__class__([0], intvl=self._intvl)
                result.coeffs = [self._intvl()] * (n + m - 1)

                for i in range(n):
                    for j in range(m):
                        result.coeffs[i + j] += self.coeffs[i] * rhs.coeffs[j]

                getcontext().round(result)
                return result

            case _:
                return NotImplemented

    def __truediv__(self, rhs: Self | T1 | T2 | int | float) -> Self:
        match rhs:
            case self._intvl.endtype() | self._intvl() | int() | float():
                return self.__class__([x / rhs for x in self.coeffs], intvl=self._intvl)

            case self.__class__():
                return self.__mul__(rhs.reciprocal())

            case _:
                return NotImplemented

    def __pow__(self, rhs: int) -> Self:
        if not isinstance(rhs, int):
            return NotImplemented

        if rhs < 0:
            return self.__pow__(-rhs).reciprocal()

        result = self.__class__([1], intvl=self._intvl)
        tmp = self.copy()

        while rhs != 0:
            if rhs % 2 != 0:
                result *= tmp

            rhs //= 2
            tmp *= tmp

        return result

    def __radd__(self, lhs: Self | T1 | T2 | int | float) -> Self:
        return self.__add__(lhs)

    def __rsub__(self, lhs: Self | T1 | T2 | int | float) -> Self:
        return self.__neg__().__add__(lhs)

    def __rmul__(self, lhs: Self | T1 | T2 | int | float) -> Self:
        return self.__mul__(lhs)

    def __rtruediv__(self, lhs: Self | T1 | T2 | int | float) -> Self:
        return self.reciprocal().__mul__(lhs)

    def __neg__(self) -> Self:
        return self.__class__([-x for x in self.coeffs], intvl=self._intvl)

    def __pos__(self) -> Self:
        return self.copy()

    def __copy__(self) -> Self:
        return self.copy()
