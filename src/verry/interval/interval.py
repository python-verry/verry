import enum
from abc import ABC, abstractmethod
from typing import Final, Literal, Self

from verry import function as vrf
from verry.misc.formatspec import FormatSpec
from verry.typing import Scalar, SignedComparable


class RoundingMode(enum.Enum):
    """Rounding mode specifier.

    Attributes
    ----------
    ROUND_CEILING
    ROUND_FAST
    ROUND_FLOOR
    """

    ROUND_CEILING = enum.auto()
    ROUND_FAST = enum.auto()
    ROUND_FLOOR = enum.auto()

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}>"


ROUND_CEILING: Final = RoundingMode.ROUND_CEILING
ROUND_FAST: Final = RoundingMode.ROUND_FAST
ROUND_FLOOR: Final = RoundingMode.ROUND_FLOOR


class Converter[T: SignedComparable](ABC):
    """Provides numeric/numeric and numeric/string conversions."""

    __slots__ = ()

    @abstractmethod
    def fromfloat(self, value: float, strict: bool = True) -> T:
        """Convert the float to a number.

        Raises
        ------
        ValueError
            If `strict` is ``True`` and `value` cannot be converted without loss.
        """
        raise NotImplementedError

    @abstractmethod
    def fromstr(self, value: str, rounding: RoundingMode) -> T:
        """Convert the string to a number with rounding taken into account.

        Raises
        ------
        ValueError
            If `value` does not represent a number.
        """
        raise NotImplementedError

    def tostr(
        self,
        value: T,
        rounding: Literal[RoundingMode.ROUND_CEILING, RoundingMode.ROUND_FLOOR],
    ) -> str:
        """Convert the number to a string with rounding taken into account."""
        raise NotImplementedError

    def format(
        self,
        value: T,
        spec: FormatSpec,
        rounding: Literal[RoundingMode.ROUND_CEILING, RoundingMode.ROUND_FLOOR],
    ) -> str:
        raise NotImplementedError

    def fromint(self, value: int, rounding: RoundingMode) -> T:
        """Convert the integer to a number with rounding taken into account.

        This method is defined as ``fromstr(str(value), rounding)`` if not overloaded.
        """
        return self.fromstr(str(value), rounding)

    def repr(self, value: T) -> str:
        raise NotImplementedError


class Operator[T: SignedComparable](ABC):
    """Provides arithmetic with directed rounding and basic constants.

    Notes
    -----
    Classes that inherit from this must define the class constants `ZERO`, `ONE`, and
    `INFINITY`.
    """

    __slots__ = ()
    ZERO: T
    ONE: T
    INFINITY: T

    @abstractmethod
    def cadd(self, lhs: T, rhs: T) -> T:
        """Add and round towards positive infinity."""
        raise NotImplementedError

    def fadd(self, lhs: T, rhs: T) -> T:
        """Add and round towards negative infinity.

        This method is defined as ``-cadd(-lhs, -rhs)`` if not overloaded.
        """
        return -self.cadd(-lhs, -rhs)

    def csub(self, lhs: T, rhs: T) -> T:
        """Subtract and round towards positive infinity.

        This method is defined as ``cadd(lhs, -rhs)`` if not overloaded.
        """
        return self.cadd(lhs, -rhs)

    def fsub(self, lhs: T, rhs: T) -> T:
        """Subtract and round towards negative infinity.

        This method is defined as ``-cadd(-lhs, rhs)`` if not overloaded.
        """
        return -self.cadd(-lhs, rhs)

    @abstractmethod
    def cmul(self, lhs: T, rhs: T) -> T:
        """Multiply and round towards positive infinity."""
        raise NotImplementedError

    def fmul(self, lhs: T, rhs: T) -> T:
        """Multiply and round towards negative infinity.

        This method is defined as ``-cmul(-lhs, rhs)`` if not overloaded.
        """
        return -self.cmul(-lhs, rhs)

    @abstractmethod
    def cdiv(self, lhs: T, rhs: T) -> T:
        """Divide and round towards positive infinity."""
        raise NotImplementedError

    def fdiv(self, lhs: T, rhs: T) -> T:
        """Divide and round towards negative infinity.

        This method is defined as ``-cdiv(-lhs, rhs)`` if not overloaded.
        """
        return -self.cdiv(-lhs, rhs)

    @abstractmethod
    def csqr(self, value: T) -> T:
        """Calcurate the square root and round towards positive infinity."""
        raise NotImplementedError

    @abstractmethod
    def fsqr(self, value: T) -> T:
        """Calcurate the square root and round towards negative infinity."""
        raise NotImplementedError


class Interval[T: SignedComparable](Scalar, ABC):
    """Abstract base class for inf-sup type intervals.

    Parameters
    ----------
    inf : endtype | float | int | str | None, optional
        Infimum of the interval.
    sup : endtype | float | int | str | None, optional
        Supremum of the interval.

    Attributes
    ----------
    inf : endtype
        Infimum of the interval.
    sup : endtype
        Supremum of the interval.
    converter : Converter
    endtype : type[endtype]
    operator : Operator

    Notes
    -----
    Classes that inherit from this must define class constants `converter`, `operator`,
    and `endtype`, where `endtype` is a type of endpoints.
    """

    __slots__ = ("inf", "sup")
    inf: T
    sup: T
    converter: Converter[T]
    endtype: type[T]
    operator: Operator[T]

    def __init__(
        self,
        inf: T | float | int | str | None = None,
        sup: T | float | int | str | None = None,
    ):
        if inf is None:
            if sup is None:
                self.inf = self.sup = self.operator.ZERO
                return

            inf = sup

        match inf:
            case self.endtype():
                self.inf = inf

            case str():
                self.inf = self.converter.fromstr(inf, ROUND_FLOOR)

            case int():
                self.inf = self.converter.fromint(inf, ROUND_FLOOR)

            case float():
                self.inf = self.converter.fromfloat(inf)

            case _:
                raise TypeError

        if sup is None:
            self.sup = self.inf
            return

        match sup:
            case self.endtype():
                self.sup = sup

            case str():
                self.sup = self.converter.fromstr(sup, ROUND_CEILING)

            case int():
                self.sup = self.converter.fromint(sup, ROUND_CEILING)

            case float():
                self.sup = self.converter.fromfloat(sup)

            case _:
                raise TypeError

        if self.inf > self.sup:
            raise ValueError

    @classmethod
    def ensure(cls, value: Self | T | float | int | str) -> Self:
        """Convert `value` to an interval and return its copy."""
        return value.copy() if isinstance(value, cls) else cls(value)  # type: ignore

    def copy(self) -> Self:
        """Return a shallow copy of the interval."""
        return self.__class__(self.inf, self.sup)

    def diam(self) -> T:
        """Return an upper bound of the diameter.

        Warning
        -------
        ``x.diam()`` might not be finite even if `x` is bounded.

        See Also
        --------
        isbounded

        Examples
        --------
        >>> from verry import FloatInterval
        >>> x = FloatInterval(-1, 1)
        >>> x.diam()
        2.0
        >>> y = 1e+308 * x
        >>> y.isbounded()
        True
        >>> y.diam()  # results in a positive infinity due to overflow.
        inf
        """
        return self.operator.csub(self.sup, self.inf)

    def hull(self, *args: Self | T | float | int) -> Self:
        """Return an interval hull."""

        result = self.__class__(self.inf, self.sup)

        for arg in args:
            result |= arg

        return result

    def interiorcontains(self, other: Self | T | float | int) -> bool:
        """Return ``True`` if the interior of the interval contains `other`."""
        match other:
            case self.endtype():
                return self.inf < other < self.sup

            case self.__class__():
                return self.inf < other.inf and other.sup < self.sup

            case int():
                return self.interiorcontains(self.__class__(other))

            case float():
                return self.interiorcontains(self.converter.fromfloat(other))

        raise TypeError

    def isbounded(self) -> bool:
        """Return ``True`` if both `inf` and `sup` are finite."""
        return self.mag() < self.operator.INFINITY

    def isdisjoint(self, other: Self | T | float | int) -> bool:
        """Return ``True`` if the interval has no elements in common with `other`."""
        match other:
            case self.endtype():
                return self.inf > other or self.sup < other

            case self.__class__():
                return self.inf > other.sup or self.sup < other.inf

            case int():
                return self.isdisjoint(self.__class__(other))

            case float():
                return self.isdisjoint(self.converter.fromfloat(other))

        raise TypeError

    def issubset(self, other: Self) -> bool:
        """Test whether every element in the interval is in `other`."""
        if type(self) is not type(other):
            return False

        return self.inf >= other.inf and self.sup <= other.sup

    def issuperset(self, other: Self) -> bool:
        """Test whether every element in `other` is in the interval."""
        if type(self) is not type(other):
            return False

        return self.inf <= other.inf and self.sup >= other.sup

    def mag(self) -> T:
        """Return a magnitude of the interval.

        The magnitude of `x` is defined as ``max(abs(x.inf), abs(x.sup))``.
        """
        return max(abs(self.inf), abs(self.sup))

    @abstractmethod
    def mid(self) -> T:
        """Return an approximation of the midpoint.

        ``x.mid() in x`` is guaranteed to be ``True`` for any `x`.
        """
        raise NotImplementedError

    def mig(self) -> T:
        """Return a mignitude of the interval.

        The mignitude of `x` is defined as ``min(abs(x.inf), abs(x.sup))``.
        """
        return min(abs(self.inf), abs(self.sup))

    def rad(self) -> T:
        """Return an upper bound of the radius."""
        return (self - self.mid()).mag()

    def width(self) -> T:
        """Alias of :meth:`diam`."""
        return self.diam()

    def _verry_overload_(self, fun, *args, **kwargs):
        match fun:
            case vrf.sqrt:
                return self.__sqrt()

        return NotImplemented

    def __repr__(self) -> str:
        try:
            inf = self.converter.repr(self.inf)
            sup = self.converter.repr(self.sup)
            return f"{type(self).__name__}(inf={inf}, sup={sup})"
        except NotImplementedError:
            return super().__repr__()

    def __str__(self) -> str:
        try:
            inf = self.converter.tostr(self.inf, ROUND_FLOOR)
            sup = self.converter.tostr(self.sup, ROUND_CEILING)
            return f"[{inf}, {sup}]"
        except NotImplementedError:
            return self.__repr__()

    def __format__(self, format_spec: str) -> str:
        spec = FormatSpec(format_spec)

        try:
            tmp = spec.replace(fill="\u0020", align=None, zfill=False, width=None)
            inf = self.converter.format(self.inf, tmp.copy(), ROUND_FLOOR)
            sup = self.converter.format(self.sup, tmp, ROUND_CEILING)
        except NotImplementedError:
            raise ValueError

        spec = FormatSpec(fill=spec.fill, align=spec.align, width=spec.width)
        return spec.format(f"[{inf}, {sup}]")

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return other.inf == self.inf and other.sup == self.sup

    def __contains__(self, item) -> bool:
        match item:
            case self.endtype():
                INFINITY = self.operator.INFINITY
                return abs(item) < INFINITY and self.inf <= item <= self.sup

            case int():
                return self.__class__(item).issubset(self)

            case float():
                return self.__contains__(self.converter.fromfloat(item))

        raise TypeError

    def __add__(self, rhs: Self | T | float | int) -> Self:
        cadd = self.operator.cadd
        fadd = self.operator.fadd

        match rhs:
            case self.endtype():
                return self.__class__(fadd(self.inf, rhs), cadd(self.sup, rhs))

            case self.__class__():
                return self.__class__(fadd(self.inf, rhs.inf), cadd(self.sup, rhs.sup))

            case int():
                return self.__add__(self.__class__(rhs))

            case float():
                return self.__add__(self.converter.fromfloat(rhs))

        return NotImplemented

    def __sub__(self, rhs: Self | T | float | int) -> Self:
        csub = self.operator.csub
        fsub = self.operator.fsub

        match rhs:
            case self.endtype():
                return self.__class__(fsub(self.inf, rhs), csub(self.sup, rhs))

            case self.__class__():
                return self.__class__(fsub(self.inf, rhs.sup), csub(self.sup, rhs.inf))

            case int():
                return self.__sub__(self.__class__(rhs))

            case float():
                return self.__sub__(self.converter.fromfloat(rhs))

        return NotImplemented

    def __mul__(self, rhs: Self | T | float | int) -> Self:
        ZERO = self.operator.ZERO
        cmul = self.operator.cmul
        fmul = self.operator.fmul

        match rhs:
            case self.endtype():
                inf = min(fmul(self.inf, rhs), fmul(self.sup, rhs))
                sup = max(cmul(self.inf, rhs), cmul(self.sup, rhs))
                return self.__class__(inf, sup)

            case self.__class__():
                if self.sup <= ZERO:
                    if rhs.sup <= ZERO:
                        inf = fmul(self.sup, rhs.sup)
                        sup = cmul(self.inf, rhs.inf)
                    elif rhs.inf >= ZERO:
                        inf = fmul(self.inf, rhs.sup)
                        sup = cmul(self.sup, rhs.inf)
                    else:
                        inf = fmul(self.inf, rhs.sup)
                        sup = cmul(self.inf, rhs.inf)

                    return self.__class__(inf, sup)

                if self.inf >= ZERO:
                    if rhs.sup <= ZERO:
                        inf = fmul(self.sup, rhs.inf)
                        sup = cmul(self.inf, rhs.sup)
                    elif rhs.inf >= ZERO:
                        inf = fmul(self.inf, rhs.inf)
                        sup = cmul(self.sup, rhs.sup)
                    else:
                        inf = fmul(self.sup, rhs.inf)
                        sup = cmul(self.sup, rhs.sup)

                    return self.__class__(inf, sup)

                if rhs.sup <= ZERO:
                    inf = fmul(self.sup, rhs.inf)
                    sup = cmul(self.inf, rhs.inf)
                elif rhs.inf >= ZERO:
                    inf = fmul(self.inf, rhs.sup)
                    sup = cmul(self.sup, rhs.sup)
                else:
                    inf = min(fmul(self.inf, rhs.sup), fmul(self.sup, rhs.inf))
                    sup = max(cmul(self.inf, rhs.inf), cmul(self.sup, rhs.sup))

                return self.__class__(inf, sup)

            case int():
                return self.__mul__(self.__class__(rhs))

            case float():
                return self.__mul__(self.converter.fromfloat(rhs))

        return NotImplemented

    def __truediv__(self, rhs: Self | T | float | int) -> Self:
        ZERO = self.operator.ZERO
        cdiv = self.operator.cdiv
        fdiv = self.operator.fdiv

        match rhs:
            case self.endtype():
                if rhs == ZERO:
                    raise ZeroDivisionError

                inf = min(fdiv(self.inf, rhs), fdiv(self.sup, rhs))
                sup = max(cdiv(self.inf, rhs), cdiv(self.sup, rhs))
                return self.__class__(inf, sup)

            case self.__class__():
                if rhs.inf == rhs.sup == ZERO:
                    raise ZeroDivisionError

                if rhs.inf <= ZERO <= rhs.sup:
                    INFINITY = self.operator.INFINITY

                    if self.inf == self.sup == ZERO:
                        return self.__class__()

                    if rhs.inf != ZERO and rhs.sup != ZERO:
                        return self.__class__(-INFINITY, INFINITY)

                    if self.sup < ZERO:
                        if rhs.sup == ZERO:
                            return self.__class__(fdiv(self.sup, rhs.inf), INFINITY)

                        return self.__class__(-INFINITY, cdiv(self.sup, rhs.sup))

                    if self.sup == ZERO:
                        if rhs.sup == ZERO:
                            return self.__class__(ZERO, INFINITY)

                        return self.__class__(-INFINITY, ZERO)

                    if self.inf < ZERO:
                        return self.__class__(-INFINITY, INFINITY)

                    if self.inf == ZERO:
                        if rhs.sup == ZERO:
                            return self.__class__(-INFINITY, ZERO)

                        return self.__class__(ZERO, INFINITY)

                    if rhs.sup == ZERO:
                        return self.__class__(-INFINITY, cdiv(self.inf, rhs.sup))

                    return self.__class__(fdiv(self.inf, rhs.inf), INFINITY)

                if rhs.sup < ZERO:
                    if self.sup < ZERO:
                        inf = fdiv(self.sup, rhs.inf)
                        sup = cdiv(self.inf, rhs.sup)
                    elif self.inf > ZERO:
                        inf = fdiv(self.sup, rhs.sup)
                        sup = cdiv(self.inf, rhs.inf)
                    else:
                        inf = fdiv(self.sup, rhs.sup)
                        sup = cdiv(self.inf, rhs.sup)

                    return self.__class__(inf, sup)

                if self.sup < ZERO:
                    inf = fdiv(self.inf, rhs.inf)
                    sup = cdiv(self.sup, rhs.sup)
                elif self.inf > ZERO:
                    inf = fdiv(self.inf, rhs.sup)
                    sup = cdiv(self.sup, rhs.inf)
                else:
                    inf = fdiv(self.inf, rhs.inf)
                    sup = cdiv(self.sup, rhs.inf)

                return self.__class__(inf, sup)

            case int():
                return self.__truediv__(self.__class__(rhs))

            case float():
                return self.__truediv__(self.converter.fromfloat(rhs))

        return NotImplemented

    def __pow__(self, rhs: int) -> Self:
        ZERO = self.operator.ZERO
        ONE = self.operator.ONE

        if not isinstance(rhs, int):
            return NotImplemented

        if rhs < 0:
            return self.__pow__(-rhs).__rtruediv__(ONE)

        result = self.__class__(ONE)
        tmp = self.copy()
        is_even = rhs % 2 == 0

        while rhs != 0:
            if rhs % 2 != 0:
                result *= tmp

            rhs //= 2
            tmp *= tmp

        if is_even and self.inf <= ZERO <= self.sup:
            result.inf = ZERO

        return result

    def __and__(self, rhs: Self | T | float | int) -> Self:
        match rhs:
            case self.endtype():
                if not self.inf <= rhs <= self.sup:
                    raise ValueError

                return self.__class__(rhs)

            case self.__class__():
                inf = max(self.inf, rhs.inf)
                sup = min(self.sup, rhs.sup)

                if inf > sup:
                    raise ValueError

                return self.__class__(inf, sup)

            case int():
                return self.__and__(self.__class__(rhs))

            case float():
                return self.__and__(self.converter.fromfloat(rhs))

        return NotImplemented

    def __or__(self, rhs: Self | T | float | int) -> Self:
        match rhs:
            case self.endtype():
                return self.__class__(min(self.inf, rhs), max(self.sup, rhs))

            case self.__class__():
                return self.__class__(min(self.inf, rhs.inf), max(self.sup, rhs.sup))

            case int():
                return self.__or__(self.__class__(rhs))

            case float():
                return self.__or__(self.converter.fromfloat(rhs))

        return NotImplemented

    def __radd__(self, lhs: Self | T | float | int) -> Self:
        return self.__add__(lhs)

    def __rsub__(self, lhs: Self | T | float | int) -> Self:
        return self.__neg__().__add__(lhs)

    def __rmul__(self, lhs: Self | T | float | int) -> Self:
        return self.__mul__(lhs)

    def __rtruediv__(self, lhs: Self | T | float | int) -> Self:
        match lhs:
            case self.endtype() | float() | int():
                return self.__class__(lhs).__truediv__(self)

            case self.__class__():
                return lhs.__truediv__(self)

        return NotImplemented

    def __rand__(self, lhs: Self | T | float | int) -> Self:
        return self.__and__(lhs)

    def __ror__(self, lhs: Self | T | float | int) -> Self:
        return self.__or__(lhs)

    def __neg__(self) -> Self:
        return self.__class__(-self.sup, -self.inf)

    def __pos__(self) -> Self:
        return self.__class__(self.inf, self.sup)

    def __abs__(self) -> Self:
        ZERO = self.operator.ZERO

        if self.inf > ZERO:
            return self.__class__(self.inf, self.sup)

        if self.sup < ZERO:
            return self.__class__(-self.sup, -self.inf)

        return self.__class__(ZERO, max(-self.inf, self.sup))

    def __copy__(self) -> Self:
        return self.copy()

    def __sqrt(self):
        if self.inf < self.operator.ZERO:
            raise ValueError("math domain error")

        inf = self.operator.fsqr(self.inf)
        sup = self.operator.csqr(self.sup)
        return self.__class__(inf, sup)
