"""
###########################################
Affine arithmetic (:mod:`verry.affineform`)
###########################################

.. currentmodule:: verry.affineform

This module provides affine arithmetic.

Affine form
===========

.. autosummary::
    :toctree: generated/

    AffineForm
    summarize
    summarized

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
from typing import Literal, Self, final

from verry import function as vrf
from verry.interval.interval import Interval
from verry.typing import ComparableScalar, Scalar


class Context:
    """Create a new context.

    Context can be regarded as a collection of noise symbols. All instances of
    :class:`AffineForm` belong to the context in which they were initialized, and all
    operations must be performed within the context.

    Parameters
    ----------
    rounding : Literal["BRUTE", "FAST"], default="BRUTE"
        Rounding mode. If `rounding` is ``"FAST"``, no new noise symbols are appended by
        addition, subtraction, or constant multiplication. Thus, the operations are
        relatively fast. Instead, the radius of the resulting interval may be increased.
    """

    __slots__ = ("rounding", "_count")
    rounding: Literal["BRUTE", "FAST"]
    _count: int

    def __init__(self, rounding: Literal["BRUTE", "FAST"] = "BRUTE"):
        self.rounding = rounding
        self._count = 0

    def copy(self) -> Self:
        result = self.__class__(self.rounding)
        result._count = self._count
        return result

    def create(self) -> int:
        result = self._count
        self._count += 1
        return result

    def __str__(self):
        return f"{type(self).__name__}({self.rounding!r})"

    def __copy__(self) -> Self:
        return self.copy()


_var: contextvars.ContextVar[Context] = contextvars.ContextVar("affineform")


def getcontext() -> Context:
    """Return the current context for the active thread."""
    if context := _var.get(None):
        return context

    context = Context()
    _var.set(context)
    return context


def setcontext(ctx: Context) -> None:
    """Set the current context for the active thread to `ctx`."""
    _var.set(ctx)


@contextlib.contextmanager
def localcontext(
    ctx: Context | None = None, /, *, rounding: Literal["FAST", "BRUTE"] | None = None
):
    """Return a context manager."""
    if ctx is None:
        ctx = getcontext()

    if rounding is None:
        rounding = ctx.rounding

    ctx = Context(rounding)
    token = _var.set(ctx)

    try:
        yield ctx
    finally:
        _var.reset(token)


@final
class AffineForm[T: ComparableScalar](Scalar):
    """Affine form.

    Parameters
    ----------
    value : Interval
        Interval transformed into a new affine form that is independent of all existing
        affine forms.

    Examples
    --------
    >>> from verry import FloatInterval
    >>> x0 = FloatInterval("-0.1", "0.1")
    >>> y0 = (x0 + 1)**2 - 2 * x0
    >>> print(format(y0, ".6f"))
    [inf=0.609999, sup=1.410001]
    >>> x1 = AffineForm(x0)
    >>> y1 = (x1 + 1)**2 - 2 * x1
    >>> print(format(y1.range(), ".6f"))
    [inf=0.989999, sup=1.010001]
    """

    __slots__ = ("_mid", "_coeffs", "_excess", "_intvl", "_context")
    _mid: T
    _coeffs: dict[int, T]
    _excess: T
    _intvl: type[Interval[T]]

    def __init__(self, value: Interval[T]):
        if not isinstance(value, Interval):
            raise TypeError

        self._intvl = type(value)
        self._excess = value.operator.ZERO

        if value.inf == value.sup:
            self._mid = value.inf
            self._coeffs = {}
            return

        self._mid = value.mid()
        self._coeffs = dict([(getcontext().create(), value.rad())])

    @property
    def interval(self) -> type[Interval[T]]:
        return self._intvl

    def copy(self) -> Self:
        """Return a shallow copy of the affine form."""
        result = self.__class__(self._intvl())
        result._mid = self._mid
        result._coeffs = self._coeffs.copy()
        result._excess = self._excess
        return result

    def mid(self) -> T:
        return self._mid

    def rad(self) -> T:
        """Return an upper bound of the radius."""
        result = self._excess

        for value in self._coeffs.values():
            result = self._intvl.operator.cadd(result, abs(value))

        return result

    def range(self) -> Interval[T]:
        """Return the range of the affine form.

        Example
        -------
        >>> from verry import FloatInterval
        >>> x = AffineForm(FloatInterval("0.1"))
        >>> x.range() == x.mid() + x.rad() * x.interval(-1, 1)
        True
        """
        tmp = self.rad()
        return self._mid + self._intvl(-tmp, tmp)

    def reciprocal(self) -> Self:
        """Return the reciprocal of the affine form.

        Raises
        ------
        ZeroDivisionError
            If the range contains zero.
        """
        ZERO = self._intvl.operator.ZERO
        cadd = self._intvl.operator.cadd
        cmul = self._intvl.operator.cmul

        if ZERO in (range := self.range()):
            raise ZeroDivisionError

        result = self.__class__(self._intvl())
        tmp = self._intvl(range.inf)
        inv = -1 / (tmp * range.sup)
        am = (tmp + range.sup) / 2
        gm = vrf.sqrt(tmp * range.sup)

        if range.inf > ZERO:
            tmp = (inv * (self._mid - (am + gm))).midrad()
            result._mid = tmp[0]
            error = cadd((inv * (am - gm)).mag(), tmp[1])
        else:
            tmp = (inv * (self._mid - (am - gm))).midrad()
            result._mid = tmp[0]
            error = cadd((inv * (am + gm)).mag(), tmp[1])

        for key, coeff in self._coeffs.items():
            tmp = (inv * coeff).midrad()
            result._coeffs[key] = tmp[0]
            error = cadd(error, tmp[1])

        if self._excess != ZERO:
            error = cadd(error, cmul(self._excess, inv.mag()))

        result._coeffs[getcontext().create()] = error
        return result

    def _verry_overload_(self, fun, *args, **kwargs):
        match fun:
            case vrf.e:
                return self.__class__(vrf.e(self._intvl()))

            case vrf.ln2:
                return self.__class__(vrf.ln2(self._intvl()))

            case vrf.pi:
                return self.__class__(vrf.pi(self._intvl()))

        return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return (
            other._intvl is self._intvl
            and other._mid == self._mid
            and other._coeffs == self._coeffs
            and other._excess == self._excess
        )

    def __len__(self) -> int:
        """Return the number of noise symbols."""
        return len(self._coeffs)

    def __add__(self, rhs: Self | Interval[T] | T | float | int) -> Self:
        ZERO = self._intvl.operator.ZERO
        cadd = self._intvl.operator.cadd
        ctx = getcontext()

        match rhs:
            case self._intvl.endtype() | int() | float():
                result = self.__class__(self._intvl())
                tmp = (self._intvl(self._mid) + rhs).midrad()
                result._mid = tmp[0]
                result._excess = cadd(self._excess, tmp[1])
                result._coeffs = self._coeffs.copy()

                if ctx.rounding == "BRUTE":
                    result._coeffs[ctx.create()] = result._excess
                    result._excess = ZERO

                return result

            case self._intvl():
                return self.__add__(self.__class__(rhs))

            case self.__class__():
                result = self.__class__(self._intvl())
                tmp = (self._intvl(self._mid) + rhs._mid).midrad()
                result._mid = tmp[0]
                result._excess = cadd(self._excess, tmp[1])
                result._coeffs = self._coeffs.copy()

                for [key, x] in self._coeffs.items():
                    if (y := rhs._coeffs.get(key)) is None:
                        result._coeffs[key] = x
                        continue

                    tmp = (self._intvl(x) + y).midrad()
                    result._coeffs[key] = tmp[0]
                    result._excess = cadd(result._excess, tmp[1])

                for [key, y] in rhs._coeffs.items():
                    if key not in self._coeffs:
                        result._coeffs[key] = y

                if ctx.rounding == "BRUTE":
                    result._coeffs[ctx.create()] = result._excess
                    result._excess = ZERO

                return result

            case _:
                return NotImplemented

    def __sub__(self, rhs: Self | Interval[T] | T | float | int) -> Self:
        ZERO = self._intvl.operator.ZERO
        cadd = self._intvl.operator.cadd
        ctx = getcontext()

        match rhs:
            case self._intvl.endtype() | float() | int():
                result = self.__class__(self._intvl())
                tmp = (self._intvl(self._mid) - rhs).midrad()
                result._mid = tmp[0]
                result._excess = cadd(self._excess, tmp[1])
                result._coeffs = self._coeffs.copy()

                if ctx.rounding == "BRUTE":
                    result._coeffs[ctx.create()] = result._excess
                    result._excess = ZERO

                return result

            case self._intvl():
                return self.__sub__(self.__class__(rhs))

            case self.__class__():
                result = self.__class__(self._intvl())
                tmp = (self._intvl(self._mid) - rhs._mid).midrad()
                result._mid = tmp[0]
                result._excess = cadd(self._excess, tmp[1])
                result._coeffs = self._coeffs.copy()

                for [key, x] in self._coeffs.items():
                    if (y := rhs._coeffs.get(key)) is None:
                        result._coeffs[key] = x
                        continue

                    tmp = (self._intvl(x) - y).midrad()
                    result._coeffs[key] = tmp[0]
                    result._excess = cadd(result._excess, tmp[1])

                for [key, y] in rhs._coeffs.items():
                    if key not in self._coeffs:
                        result._coeffs[key] = -y

                if ctx.rounding == "BRUTE":
                    result._coeffs[ctx.create()] = result._excess
                    result._excess = ZERO

                return result

            case _:
                return NotImplemented

    def __mul__(self, rhs: Self | Interval[T] | T | float | int) -> Self:
        ZERO = self._intvl.operator.ZERO
        cadd = self._intvl.operator.cadd
        cmul = self._intvl.operator.cmul
        ctx = getcontext()

        match rhs:
            case self._intvl.endtype() | float() | int():
                rhs = self._intvl(rhs)
                result = self.__class__(self._intvl())
                tmp = (self._mid * rhs).midrad()
                result._mid = tmp[0]
                error = tmp[1]

                for key, x in self._coeffs.items():
                    tmp = (x * rhs).midrad()
                    result._coeffs[key] = tmp[0]
                    error = cadd(error, tmp[1])

                if self._excess != ZERO:
                    error = cadd(error, cmul(self._excess, rhs.mag()))

                result._excess = error

                if ctx.rounding == "BRUTE":
                    result._coeffs[ctx.create()] = result._excess
                    result._excess = ZERO

                return result

            case self._intvl():
                return self.__mul__(self.__class__(rhs))

            case self.__class__():
                result = self.__class__(self._intvl())
                tmp = self._intvl(self._mid) * rhs._mid
                result._mid = tmp.mid()
                error = cadd(cmul(self.rad(), rhs.rad()), tmp.rad())

                for key, x in self._coeffs.items():
                    if (y := rhs._coeffs.get(key)) is None:
                        tmp = (self._intvl(x) * rhs._mid).midrad()
                        result._coeffs[key] = tmp[0]
                        error = cadd(error, tmp[1])
                        continue

                    tmp = self._intvl(x) * rhs._mid + self._intvl(self._mid) * y
                    tmp = tmp.midrad()
                    result._coeffs[key] = tmp[0]
                    error = cadd(error, tmp[1])

                for key, y in rhs._coeffs.items():
                    if self._coeffs.get(key) is None:
                        tmp = (self._intvl(self._mid) * y).midrad()
                        result._coeffs[key] = tmp[0]
                        error = cadd(error, tmp[1])

                if rhs._excess != ZERO:
                    error = cadd(error, cmul(abs(self._mid), rhs._excess))

                if self._excess != ZERO:
                    error = cadd(error, cmul(self._excess, abs(rhs._mid)))

                result._coeffs[ctx.create()] = error
                return result

            case _:
                return NotImplemented

    def __truediv__(self, rhs: Self | Interval[T] | T | float | int) -> Self:
        match rhs:
            case self._intvl.endtype() | float() | int():
                return self.__mul__(1 / self._intvl(rhs))

            case self._intvl():
                return self.__mul__(self.__class__(rhs).reciprocal())

            case self.__class__():
                return self.__mul__(rhs.reciprocal())

            case _:
                return NotImplemented

    def __pow__(self, rhs: int) -> Self:
        if not isinstance(rhs, int):
            return NotImplemented

        if rhs < 0:
            return self.__pow__(-rhs).reciprocal()

        result = self.__class__(self._intvl(self._intvl.operator.ONE))
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
        result = self.__class__(self._intvl())
        result._mid = -self._mid
        result._coeffs = {key: -value for key, value in self._coeffs.items()}
        result._excess = self._excess
        return result

    def __pos__(self) -> Self:
        return self.copy()

    def __copy__(self) -> Self:
        return self.copy()


def summarized[T: AffineForm](vars: Sequence[T], n: int, m: int = 0) -> tuple[T, ...]:
    """Reduce the number of noise symbols while keeping correlation between `vars`.

    Parameters
    ----------
    vars : Sequence[AffineForm]
        Affine forms to which the summarization is applied.
    n : int
        Number of noise symbols after summarization. `n` must be greater than or equal
        to ``len(vars)``.
    m : int, default=0
        Threshold of summarization. If the number of noise symbols is less than `m`,
        summarization is not applied.

    Returns
    -------
    tuple[AffineForm, ...]

    Notes
    -----
    This is an implementation of [#Kas12]_.

    References
    ----------
    .. [#Kas12] M. Kashiwagi, "An algorithm to reduce the number of dummy variables in
        affine arithmetic," in *Proc. 15th GAMM-IMACS International Symposium on
        Scientific Computing Computer Arithmetic and Verified Numerical Computations
        (SCAN 2012)*, Novosibirsk, Russia, Sep. 23--29, 2012, pp. 70--71. [Online].
        Available: http://conf.nsc.ru/scan2012/scan2012_27
    """
    result = tuple(var.copy() for var in vars)
    summarize(result, n, m)
    return result


def summarize(vars: Sequence[AffineForm], n: int, m: int = 0) -> None:
    """In-place version of :func:`summarized`."""
    if not 2 <= len(vars) <= n:
        raise ValueError

    ctx = getcontext()
    keys: set[int] = set()

    for var in vars:
        if var._intvl is not vars[0]._intvl:
            raise ValueError

        keys |= var._coeffs.keys()

    if len(keys) < min(n, m):
        return

    ZERO = vars[0]._intvl.operator.ZERO
    cadd = vars[0]._intvl.operator.cadd
    tmp: list[tuple] = []

    for key in keys:
        a0 = a1 = ZERO

        for var in vars:
            if (a := var._coeffs.get(key)) is not None:
                if abs(a) > a0:
                    a0 = abs(a)
                elif abs(a) > a1:
                    a1 = abs(a)

        penalty = a0 * a1 / (a0 + a1) if a0 != ZERO else ZERO
        tmp.append((key, penalty))

    tmp.sort(key=lambda tmp: tmp[1], reverse=True)
    del tmp[: n - len(vars)]

    for var in vars:
        rad = ZERO
        coeffs = var._coeffs

        for key, _ in tmp:
            if (a := coeffs.get(key)) is not None:
                rad = cadd(rad, abs(a))
                del coeffs[key]

        if var._excess != ZERO:
            rad = cadd(rad, var._excess)
            var._excess = ZERO

        coeffs[ctx.create()] = rad
