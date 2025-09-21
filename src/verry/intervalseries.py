"""
#############################################
Interval series (:mod:`verry.intervalseries`)
#############################################

.. currentmodule:: verry.intervalseries

This module provides operations on power series with interval coefficients.

.. autoclass:: IntervalSeries
    :show-inheritance:
    :members:
    :inherited-members:
    :special-members: __call__
    :member-order: groupwise

"""

from collections.abc import Iterable
from typing import Self, overload

from verry import function as vrf
from verry.interval.interval import Interval
from verry.typing import ComparableScalar, Scalar


class IntervalSeries[T: ComparableScalar](Scalar):
    """Interval polynomial function.

    Parameters
    ----------
    domain : Interval
        Domain of the interval polynomial function. Note that `domain` must contain
        zero.
    coeffs : Iterable[Interval]
        Coefficients with respect to the monomial basis.

    Attributes
    ----------
    domain : Interval
        Domain of the interval polynomial function.
    coeffs : list[Interval]
        Coefficients with respect to the monomial basis.
    """

    __slots__ = ("domain", "coeffs")
    domain: Interval[T]
    coeffs: list[Interval[T]]

    def __init__(self, domain: Interval[T], coeffs: Iterable[Interval[T]]):
        if not isinstance(domain, Interval):
            raise TypeError

        if 0 not in domain:
            raise ValueError("domain must contain zero")

        self.domain = domain
        self.coeffs = list(coeffs)

    @property
    def interval(self) -> type[Interval[T]]:
        return type(self.domain)

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    def __repr__(self) -> str:
        return f"{type(self).__name__}(domain={self.domain!r}, coeffs={self.coeffs!r})"

    def __str__(self) -> str:
        return f"{type(self).__name__}(domain={self.domain}, coeffs={self.coeffs})"

    def compose(self, arg: Self) -> Self:
        """Return the composition.

        See Also
        --------
        eval, __call__

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> dom = FI(0, 1)
        >>> f = IntervalSeries(dom, [FI(4), FI(3), FI(1)])
        >>> g = IntervalSeries(dom, [FI(2), FI(-1), FI(5)])
        >>> h = f.compose(g)
        >>> print([x.mid() for x in h.coeffs[:-1]])
        [14.0, -7.0]
        """
        result = self.__class__(self.domain, (self.coeffs[-1],))

        for x in reversed(self.coeffs[:-1]):
            result *= arg
            result += x

        result = result.round(min(self.order, arg.order))
        return result

    def copy(self) -> Self:
        """Return a copy of the series."""
        return self.__class__(self.domain.copy(), (x.copy() for x in self.coeffs))

    def eval(self, arg: Interval[T] | T | int | float) -> Interval[T]:
        """Return an interval containing the image of `arg`.

        See Also
        --------
        compose, __call__

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> dom = FI(-2, 2)
        >>> f = IntervalSeries(dom, [FI(8), FI(2), FI(-1)])
        >>> u = f.eval(2)
        >>> print(format(u, ".2f"))
        [inf=8.00, sup=8.00]
        >>> v = f.eval(FI(-1, 1))
        >>> FI(5, 9).issubset(v)
        True
        """
        result = self.coeffs[-1]

        for x in reversed(self.coeffs[:-1]):
            result *= arg
            result += x

        return result

    def integrate(self) -> Self:
        """Integrate the series in a term by term.

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> dom = FI(0, 1)
        >>> x = IntervalSeries(dom, [FI(0), FI(1), FI(0), FI(0)])
        >>> y = 3 * x**2 + 4 * x + 5
        >>> z = y.integrate().round(3)
        >>> z == x**3 + 2 * x**2 + 5 * x
        True
        """
        tail = (x / (k + 1) for k, x in enumerate(self.coeffs))
        return self.__class__(self.domain, (self.interval(), *tail))

    def reciprocal(self) -> Self:
        """Return the reciprocal of the series.

        Raises
        ------
        ZeroDivisionError
            If the constant term contains zero.

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> dom = FI(0, 1)
        >>> x = IntervalSeries(dom, [FI(1), FI(1), FI(0), FI(0)])
        >>> y = x.reciprocal()
        >>> z = x * y
        >>> print([x.sup for x in z.coeffs[:-1]])
        [1.0, 0.0, 0.0]
        """
        if 0 in self.coeffs[0]:
            raise ZeroDivisionError

        dx = self.__class__(self.domain, (self.interval(), *self.coeffs[1:]))
        dp = self.__class__(self.domain, (self.interval(1),))
        result = self.__class__(self.domain, (1 / self.coeffs[0],))

        for k in range(1, self.order):
            dp *= dx
            result += -dp / pow(-self.coeffs[0], k + 1)

        dp *= dx
        result += -dp / pow(-self.eval(self.domain), self.order + 1)
        return result

    def round(self, order: int) -> Self:
        """Round the series.

        Parameters
        ----------
        order : int

        Examples
        --------
        >>> from verry import FloatInterval as FI
        >>> dom = FI(0, 1)
        >>> x = IntervalSeries(dom, [FI(2), FI(1), FI(-3)])
        >>> y = IntervalSeries(dom, [FI(5), FI(2), FI(1)])
        >>> z = x * y
        >>> w = z.round(1)
        >>> z(0.5).issubset(w(0.5))
        True
        """
        if order < 0:
            raise ValueError("order must be greater than or equal to zero")

        if order == 0:
            return self.__class__(self.domain, (self.eval(self.domain),))

        if order >= self.order:
            return self

        result = self.__class__(self.domain, self.coeffs[:order])
        tail = self.coeffs[-1]

        for x in reversed(self.coeffs[order:-1]):
            tail *= self.domain
            tail += x

        result.coeffs.append(tail)
        return result

    def _verry_overload_(self, fun, *args, **kwargs):
        if fun is vrf.e:
            return self.__class__(self.domain, (vrf.e(self.interval()),))

        if fun is vrf.ln2:
            return self.__class__(self.domain, (vrf.ln2(self.interval()),))

        if fun is vrf.pi:
            return self.__class__(self.domain, (vrf.pi(self.interval()),))

        return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return other.domain == self.domain and other.coeffs == self.coeffs

    @overload
    def __call__(self, arg: Self) -> Self: ...

    @overload
    def __call__(self, arg: Interval[T] | T | float | int) -> Interval[T]: ...

    def __call__(self, arg):
        """Return ``compose(arg)`` or ``eval(arg)`` depending on the type of `arg`.

        See Also
        --------
        compose, eval
        """
        match arg:
            case self.interval() | self.interval.endtype() | int() | float():
                return self.eval(arg)

            case self.__class__():
                return self.compose(arg)

            case _:
                raise TypeError

    def __add__(self, rhs: Self | Interval[T] | T | int | float) -> Self:
        if isinstance(rhs, self.interval | self.interval.endtype | int | float):
            return self.__class__(self.domain, (self.coeffs[0] + rhs, *self.coeffs[1:]))

        if not isinstance(rhs, type(self)):
            return NotImplemented

        domain = self.domain if self.domain is rhs.domain else self.domain & rhs.domain

        if len(rhs.coeffs) == 1:
            coeffs = (self.coeffs[0] + rhs.coeffs[0], *self.coeffs[1:])
            return self.__class__(domain, coeffs)

        if len(self.coeffs) == 1:
            coeffs = (self.coeffs[0] + rhs.coeffs[0], *rhs.coeffs[1:])
            return self.__class__(domain, coeffs)

        coeffs = [x + y for x, y in zip(self.coeffs, rhs.coeffs)]

        if len(self.coeffs) > len(rhs.coeffs):
            coeffs.extend(self.coeffs[len(rhs.coeffs) :])
        else:
            coeffs.extend(rhs.coeffs[len(self.coeffs) :])

        return self.__class__(domain, coeffs).round(self.order)

    def __sub__(self, rhs: Self | Interval[T] | T | int | float) -> Self:
        if isinstance(rhs, self.interval | self.interval.endtype | int | float):
            return self.__class__(self.domain, (self.coeffs[0] - rhs, *self.coeffs[1:]))

        if not isinstance(rhs, type(self)):
            return NotImplemented

        domain = self.domain if self.domain is rhs.domain else self.domain & rhs.domain

        if len(rhs.coeffs) == 1:
            coeffs = (self.coeffs[0] - rhs.coeffs[0], *self.coeffs[1:])
            return self.__class__(domain, coeffs)

        if len(self.coeffs) == 1:
            coeffs = (self.coeffs[0] - rhs.coeffs[0], *(-x for x in rhs.coeffs[1:]))
            return self.__class__(domain, coeffs)

        coeffs = [x - y for x, y in zip(self.coeffs, rhs.coeffs)]

        if len(self.coeffs) > len(rhs.coeffs):
            coeffs.extend(self.coeffs[len(rhs.coeffs) :])
        else:
            coeffs.extend(-x for x in rhs.coeffs[len(self.coeffs) :])

        return self.__class__(domain, coeffs).round(self.order)

    def __mul__(self, rhs: Self | Interval[T] | T | float | int) -> Self:
        if isinstance(rhs, self.interval | self.interval.endtype | int | float):
            return self.__class__(self.domain, (x * rhs for x in self.coeffs))

        if not isinstance(rhs, type(self)):
            return NotImplemented

        domain = self.domain if self.domain is rhs.domain else self.domain & rhs.domain

        if len(rhs.coeffs) == 1:
            return self.__class__(domain, (x * rhs.coeffs[0] for x in self.coeffs))

        if len(self.coeffs) == 1:
            return self.__class__(domain, (self.coeffs[0] * x for x in rhs.coeffs))

        coeffs = [self.interval()] * (len(self.coeffs) + len(rhs.coeffs) - 1)

        for i in range(len(self.coeffs)):
            for j in range(len(rhs.coeffs)):
                coeffs[i + j] += self.coeffs[i] * rhs.coeffs[j]

        return self.__class__(domain, coeffs).round(min(self.order, rhs.order))

    def __truediv__(self, rhs: Self | Interval[T] | T | float | int) -> Self:
        if isinstance(rhs, self.interval | self.interval.endtype | int | float):
            return self.__class__(self.domain, (x / rhs for x in self.coeffs))

        if not isinstance(rhs, type(self)):
            return NotImplemented

        return self.__mul__(rhs.reciprocal())

    def __pow__(self, rhs: int) -> Self:
        if not isinstance(rhs, int):
            return NotImplemented

        if rhs < 0:
            return self.__pow__(-rhs).reciprocal()

        result = self.__class__(self.domain, (self.interval(1),))
        tmp = self.copy()

        while rhs != 0:
            if rhs % 2 != 0:
                result *= tmp

            rhs //= 2
            tmp *= tmp

        return result

    def __radd__(self, lhs: Self | Interval[T] | T | float | int) -> Self:
        return self.__add__(lhs)

    def __rsub__(self, lhs: Self | Interval[T] | T | float | int) -> Self:
        return self.__neg__().__add__(lhs)

    def __rmul__(self, lhs: Self | Interval[T] | T | float | int) -> Self:
        return self.__mul__(lhs)

    def __rtruediv__(self, lhs: Self | Interval[T] | T | float | int) -> Self:
        return self.reciprocal().__mul__(lhs)

    def __neg__(self) -> Self:
        return self.__class__(self.domain, (-x for x in self.coeffs))

    def __pos__(self) -> Self:
        return self.__class__(self.domain, (+x for x in self.coeffs))

    def __copy__(self) -> Self:
        return self.copy()
