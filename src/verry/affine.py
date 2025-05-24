"""
#######################################
Affine arithmetic (:mod:`verry.affine`)
#######################################

.. currentmodule:: verry.affine

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
from typing import Any, Literal, Never, Self

from verry import function as vrf
from verry.interval.interval import Interval
from verry.typing import ComparableScalar, Scalar


class Context:
    """Create a new context.

    Context can be regarded as a collection of noise symbols. Each instance of
    :class:`AffineForm` belongs to one context, and affine forms belonging to
    different contexts are considered independent.

    Parameters
    ----------
    rounding : Literal["BRUTE", "FAST"], default="BRUTE"
        Rounding mode. If `rounding` is ``"FAST"``, no new noise symbols are appended
        by addition, subtraction, or constant multiplication. Thus, the operations are
        relatively fast. Instead, the radius of the resulting interval may be increased.
    """

    __slots__ = ("_rounding", "_count")
    _rounding: Literal["BRUTE", "FAST"]
    _count: int

    def __init__(self, rounding: Literal["BRUTE", "FAST"] = "BRUTE"):
        self._rounding = rounding
        self._count = 0

    @property
    def rounding(self) -> Literal["BRUTE", "FAST"]:
        return self._rounding

    def copy(self) -> Self:
        return self.__class__(self._rounding)

    def create_noisesymbol(self) -> int:
        result = self._count
        self._count += 1
        return result

    def __str__(self):
        return f"{type(self).__name__}({self._rounding!r})"

    def __copy__(self) -> Self:
        return self.copy()


_var: contextvars.ContextVar[Context] = contextvars.ContextVar("affine")


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
    ctx: Context | None = None, *, rounding: Literal["FAST", "BRUTE"] | None = None
):
    """Return a context manager that will set the current context for the active thread
    to a copy of `ctx` on entry to the with-statement and restore the previous context
    when exiting the with-statement."""
    if ctx is None:
        ctx = getcontext()

    if rounding is None:
        rounding = ctx._rounding

    ctx = Context(rounding)
    token = _var.set(ctx)

    try:
        yield ctx
    finally:
        _var.reset(token)


class AffineForm[T1: Interval, T2: ComparableScalar = Any](Scalar):
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
    [0.609999, 1.410001]
    >>> x1 = AffineForm(x0)
    >>> y1 = (x1 + 1)**2 - 2 * x1
    >>> print(format(y1.range(), ".6f"))
    [0.989999, 1.010001]
    """

    __slots__ = ("_mid", "_coeffs", "_excess", "_intvl", "_context")
    _mid: T2
    _coeffs: dict[int, T2]
    _excess: T2
    _intvl: type[T1]
    _context: Context

    def __init__(self, value: T1, **kwargs: Never):
        if kwargs.get("_skipinit", False):
            return

        if not isinstance(value, Interval):
            raise TypeError

        self._intvl = type(value)
        self._coeffs = {}
        self._excess = value.operator.ZERO
        self._context = getcontext()

        key = self._context.create_noisesymbol()
        self._mid = value.mid()
        self._coeffs[key] = value.rad()

    @property
    def context(self) -> Context:
        return self._context

    @property
    def interval(self) -> type[T1]:
        return self._intvl

    @classmethod
    def zero(cls, intvl: type[T1], ctx: Context | None = None) -> Self:
        """Return an affine form consisting only of the constant term 0.

        Parameters
        ----------
        intvl : type[Interval]
            Interval type.
        ctx : Context, optional
            Context to which the affine form belongs. If no context is specified, the
            current context is used.
        """
        if not issubclass(intvl, Interval):
            raise TypeError

        if ctx is None:
            ctx = getcontext()

        result = cls(None, _skipinit=True)  # type: ignore
        result._mid = intvl.operator.ZERO
        result._coeffs = {}
        result._excess = intvl.operator.ZERO
        result._intvl = intvl
        result._context = ctx
        return result

    def copy(self) -> Self:
        """Return a shallow copy of the affine form."""
        result = self.zero(self._intvl, self._context)
        result._mid = self._mid
        result._coeffs = self._coeffs.copy()
        result._excess = self._excess
        return result

    def mid(self) -> T2:
        return self._mid

    def rad(self) -> T2:
        """Return an upper bound of the radius."""
        result = self._excess

        for value in self._coeffs.values():
            result = self._intvl.operator.cadd(result, abs(value))

        return result

    def range(self) -> T1:
        """Return the range of the affine form.

        Example
        -------
        >>> from verry import FloatInterval
        >>> x = AffineForm(FloatInterval("0.1"))
        >>> x.range() == x.mid() + x.rad() * x.interval(-1, 1)
        True
        """
        rad = self.rad()
        return self._mid + self._intvl(-rad, rad)

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

        context = getcontext()
        self._ensurecontext(context)

        result = self.zero(self._intvl, context)
        tmp = self._intvl(range.inf)
        inv = -1 / (tmp * range.sup)
        am = (tmp + range.sup) / 2
        gm = vrf.sqrt(tmp * range.sup)

        if range.inf > ZERO:
            tmp = inv * (self._mid - (am + gm))
            result._mid = tmp.mid()
            error = cadd((inv * (am - gm)).mag(), tmp.rad())
        else:
            tmp = inv * (self._mid - (am - gm))
            result._mid = tmp.mid()
            error = cadd((inv * (am + gm)).mag(), tmp.rad())

        for key, coeff in self._coeffs.items():
            tmp = inv * coeff
            result._coeffs[key] = tmp.mid()
            error = cadd(error, tmp.rad())

        if context._rounding == "FAST":
            error = cadd(error, cmul(self._excess, inv.mag()))
            result._excess = ZERO

        key = context.create_noisesymbol()
        result._coeffs[key] = error
        return result

    def _ensurecontext(self, ctx: Context) -> None:
        if self._context is ctx:
            return

        coeffs = {}

        for value in self._coeffs.values():
            key = ctx.create_noisesymbol()
            coeffs[key] = value

        if self._context._rounding == "FAST" and ctx._rounding == "BRUTE":
            key = ctx.create_noisesymbol()
            coeffs[key] = self._excess
            self._excess = self._intvl.operator.ZERO

        self._coeffs.clear()
        self._coeffs = coeffs
        self._context = ctx

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
            and other._context is self._context
        )

    def __len__(self) -> int:
        """Return the number of noise symbols."""
        return len(self._coeffs)

    def __add__(self, rhs: Self | T1 | T2 | float | int) -> Self:
        ZERO = self._intvl.operator.ZERO
        cadd = self._intvl.operator.cadd

        context = getcontext()
        self._ensurecontext(context)

        match rhs:
            case self._intvl.endtype() | float() | int():
                result = self.zero(self._intvl, context)
                tmp = self._intvl(self._mid) + rhs
                result._mid = tmp.mid()
                result._coeffs = self._coeffs.copy()
                error = tmp.rad()

                match context._rounding:
                    case "BRUTE":
                        key = context.create_noisesymbol()
                        result._coeffs[key] = error
                        return result

                    case "FAST":
                        error = cadd(error, self._excess)
                        result._excess = error
                        return result

            case self.__class__():
                rhs._ensurecontext(context)
                result = self.zero(self._intvl, context)
                tmp = self._intvl(self._mid) + rhs._mid
                result._mid = tmp.mid()
                error = tmp.rad()

                for key in set(self._coeffs) | set(rhs._coeffs):
                    x = self._intvl(self._coeffs.get(key, ZERO))
                    y = self._intvl(rhs._coeffs.get(key, ZERO))
                    tmp = x + y
                    result._coeffs[key] = tmp.mid()
                    error = cadd(error, tmp.rad())

                match context._rounding:
                    case "BRUTE":
                        key = context.create_noisesymbol()
                        result._coeffs[key] = error
                        return result

                    case "FAST":
                        error = cadd(error, cadd(self._excess, rhs._excess))
                        result._excess = error
                        return result

            case self._intvl():
                return self.__add__(self.__class__(rhs))

            case _:
                return NotImplemented

    def __sub__(self, rhs: Self | T1 | T2 | float | int) -> Self:
        ZERO = self._intvl.operator.ZERO
        cadd = self._intvl.operator.cadd

        context = getcontext()
        self._ensurecontext(context)

        match rhs:
            case self._intvl.endtype() | float() | int():
                result = self.zero(self._intvl, context)
                tmp = self._intvl(self._mid) - rhs
                result._mid = tmp.mid()
                result._coeffs = self._coeffs.copy()
                error = tmp.rad()

                match context._rounding:
                    case "BRUTE":
                        key = context.create_noisesymbol()
                        result._coeffs[key] = error
                        return result

                    case "FAST":
                        error = cadd(error, self._excess)
                        result._excess = error
                        return result

            case self.__class__():
                rhs._ensurecontext(context)
                result = self.zero(self._intvl, context)
                tmp = self._intvl(self._mid) - rhs._mid
                result._mid = tmp.mid()
                error = tmp.rad()

                for key in set(self._coeffs) | set(rhs._coeffs):
                    x = self._intvl(self._coeffs.get(key, ZERO))
                    y = self._intvl(rhs._coeffs.get(key, ZERO))
                    tmp = x - y
                    result._coeffs[key] = tmp.mid()
                    error = cadd(error, tmp.rad())

                match context._rounding:
                    case "BRUTE":
                        key = context.create_noisesymbol()
                        result._coeffs[key] = error
                        return result

                    case "FAST":
                        error = cadd(error, cadd(self._excess, rhs._excess))
                        result._excess = error
                        return result

            case self._intvl():
                return self.__sub__(self.__class__(rhs))

            case _:
                return NotImplemented

    def __mul__(self, rhs: Self | T1 | T2 | float | int) -> Self:
        ZERO = self._intvl.operator.ZERO
        cadd = self._intvl.operator.cadd
        cmul = self._intvl.operator.cmul

        context = getcontext()
        self._ensurecontext(context)

        match rhs:
            case self._intvl.endtype() | float() | int():
                rhs = self._intvl(rhs)
                result = self.zero(self._intvl, context)
                tmp = self._mid * rhs
                result._mid = tmp.mid()
                error = tmp.rad()

                for key, coeff in self._coeffs.items():
                    tmp = coeff * rhs
                    result._coeffs[key] = tmp.mid()
                    error = cadd(error, tmp.rad())

                match context._rounding:
                    case "BRUTE":
                        key = context.create_noisesymbol()
                        result._coeffs[key] = error
                        return result

                    case "FAST":
                        error = cadd(error, cmul(self._excess, rhs.mag()))
                        result._excess = error
                        return result

            case self.__class__():
                rhs._ensurecontext(context)
                result = self.zero(self._intvl, context)
                tmp = self._intvl(self._mid) * rhs._mid
                result._mid = tmp.mid()
                error = cadd(cmul(self.rad(), rhs.rad()), tmp.rad())

                for key in set(self._coeffs) | set(rhs._coeffs):
                    x = self._intvl(self._coeffs.get(key, ZERO))
                    y = self._intvl(rhs._coeffs.get(key, ZERO))
                    tmp = x * rhs._mid + self._mid * y
                    result._coeffs[key] = tmp.mid()
                    error = cadd(error, tmp.rad())

                if context._rounding == "FAST":
                    error = cadd(error, cmul(abs(self._mid), rhs._excess))
                    error = cadd(error, cmul(self._excess, abs(rhs._mid)))
                    result._excess = ZERO

                key = context.create_noisesymbol()
                result._coeffs[key] = error
                return result

            case self._intvl():
                return self.__mul__(self.__class__(rhs))

            case _:
                return NotImplemented

    def __truediv__(self, rhs: Self | T1 | T2 | float | int) -> Self:
        match rhs:
            case self._intvl.endtype() | float() | int():
                return self.__mul__(1 / self._intvl(rhs))

            case self.__class__():
                return self.__mul__(rhs.reciprocal())

            case self._intvl():
                return self.__mul__(self.__class__(rhs).reciprocal())

            case _:
                return NotImplemented

    def __pow__(self, rhs: int) -> Self:
        if not isinstance(rhs, int):
            return NotImplemented

        if rhs < 0:
            return self.__pow__(-rhs).reciprocal()

        result = self.__class__(self._intvl(1))
        tmp = self.copy()

        while rhs != 0:
            if rhs % 2 != 0:
                result *= tmp

            rhs //= 2
            tmp *= tmp

        return result

    def __radd__(self, lhs: Self | T1 | T2 | float | int) -> Self:
        return self.__add__(lhs)

    def __rsub__(self, lhs: Self | T1 | T2 | float | int) -> Self:
        return self.__neg__().__add__(lhs)

    def __rmul__(self, lhs: Self | T1 | T2 | float | int) -> Self:
        return self.__mul__(lhs)

    def __rtruediv__(self, lhs: Self | T1 | T2 | float | int) -> Self:
        return self.reciprocal().__mul__(lhs)

    def __neg__(self) -> Self:
        result = self.zero(self._intvl, self._context)
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
    This is an implementation of [#Ka12]_.

    References
    ----------
    .. [#Ka12] M. Kashiwagi, "An algorithm to reduce the number of dummy variables in
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

    context = getcontext()
    keys: set[int] = set()

    for var in vars:
        if var._intvl is not vars[0]._intvl:
            raise ValueError

        var._ensurecontext(context)
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

        if context.rounding == "FAST":
            rad = cadd(rad, var._excess)
            var._excess = ZERO

        key = context.create_noisesymbol()
        coeffs[key] = rad
