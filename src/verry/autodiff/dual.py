from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Final, Self, final

from verry.interval.interval import Interval
from verry.typing import ComparableScalar, Scalar


class Dual[T1: Scalar, T2 = Any](Scalar, ABC):
    r"""Abstract base class for dual numbers.

    Parameters
    ----------
    real : T
    imag : Iterable[T]

    Attributes
    ----------
    real : T
    imag : list[T]

    Warnings
    --------
    Users cannot define classes derived from this.

    See Also
    --------
    DynDual, IntervalDual

    Notes
    -----
    Instances of this class behave like elements of the dual number ring

    .. math::

        T[x_1,x_2,\dotsc,x_n]/(x_ix_j\mid i,j\in\{1,2,\dotsc,n\}),

    where :math:`n` is the length of `imag`.
    """

    __slots__ = ("real", "imag", "_priority")
    __IS_SEALED: Final = True
    real: T1
    imag: list[T1]
    _priority: int

    def __init__(self, real: T1, imag: Iterable[T1]):
        self.real = real
        self.imag = list(imag)
        self._priority = (real._priority + 1) if isinstance(real, Dual) else 0

        if len(self.imag) == 0:
            raise ValueError

    @classmethod
    def constant(cls, value: T1, n: int) -> Self:
        ZERO = value * 0
        return cls(value, (ZERO,) * n)

    @classmethod
    def variable(cls, *args: T1) -> tuple[Self, ...]:
        result: list[Self] = []
        ZERO = args[0] * 0
        ONE = ZERO + 1

        for argnum, arg in enumerate(args):
            imag = (ONE if i == argnum else ZERO for i in range(len(args)))
            result.append(cls(arg, imag))

        return tuple(result)

    @property
    def priority(self) -> int:
        return self._priority

    @abstractmethod
    def _is_acceptable(self, value: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(real={self.real!r}, imag={self.imag!r})"

    def __str__(self) -> str:
        imag = (", ").join(str(x) for x in self.imag)
        return f"{type(self).__name__}(real={self.real}, imag=[{imag}])"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return other.real == self.real and other.imag == self.imag  # type: ignore

    def __add__(self, rhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(rhs):
            return NotImplemented

        if not isinstance(rhs, Dual) or self._priority > rhs._priority:
            return self.__class__(self.real + rhs, self.imag)  # type: ignore

        if self._priority < rhs._priority:
            return NotImplemented

        if type(self) is not type(rhs):
            raise TypeError

        imag = (x + y for x, y in zip(self.imag, rhs.imag))
        return self.__class__(self.real + rhs.real, imag)

    def __sub__(self, rhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(rhs):
            return NotImplemented

        if not isinstance(rhs, Dual) or self._priority > rhs._priority:
            return self.__class__(self.real - rhs, self.imag)  # type: ignore

        if self._priority < rhs._priority:
            return NotImplemented

        if type(self) is not type(rhs):
            raise TypeError

        imag = (x - y for x, y in zip(self.imag, rhs.imag))
        return self.__class__(self.real - rhs.real, imag)

    def __mul__(self, rhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(rhs):
            return NotImplemented

        if not isinstance(rhs, Dual) or self._priority > rhs._priority:
            return self.__class__(self.real * rhs, (x * rhs for x in self.imag))  # type: ignore

        if self._priority < rhs._priority:
            return NotImplemented

        if type(self) is not type(rhs):
            raise TypeError

        imag = (self.real * y + x * rhs.real for x, y in zip(self.imag, rhs.imag))
        return self.__class__(self.real * rhs.real, imag)

    def __truediv__(self, rhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(rhs):
            return NotImplemented

        if not isinstance(rhs, Dual) or self._priority > rhs._priority:
            return self.__class__(self.real / rhs, (x / rhs for x in self.imag))  # type: ignore

        if self._priority < rhs._priority:
            return NotImplemented

        s = rhs.real**2
        imag = ((x * rhs.real - self.real * y) / s for x, y in zip(self.imag, rhs.imag))
        return self.__class__(self.real / rhs.real, imag)

    def __pow__(self, rhs: int) -> Self:
        imag = (rhs * self.real ** (rhs - 1) * x for x in self.imag)
        return self.__class__(self.real**rhs, imag)

    def __neg__(self) -> Self:
        return self.__class__(-self.real, (-x for x in self.imag))

    def __pos__(self) -> Self:
        return self.__class__(+self.real, (+x for x in self.imag))

    def __radd__(self, lhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(lhs):
            return NotImplemented

        if not isinstance(lhs, Dual) or self._priority > lhs._priority:
            return self.__class__(lhs + self.real, self.imag)  # type: ignore

        if self._priority < lhs._priority:
            return NotImplemented

        if type(self) is not type(lhs):
            raise TypeError

        imag = (x + y for x, y in zip(lhs.imag, self.imag))
        return self.__class__(lhs.real + self.real, imag)

    def __rsub__(self, lhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(lhs):
            return NotImplemented

        if not isinstance(lhs, Dual) or self._priority > lhs._priority:
            return self.__class__(lhs - self.real, (-x for x in self.imag))  # type: ignore

        if self._priority < lhs._priority:
            return NotImplemented

        if type(self) is not type(lhs):
            raise TypeError

        imag = (x - y for x, y in zip(lhs.imag, self.imag))
        return self.__class__(lhs.real - self.real, imag)

    def __rmul__(self, lhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(lhs):
            return NotImplemented

        if not isinstance(lhs, Dual) or self._priority > lhs._priority:
            return self.__class__(lhs * self.real, (lhs * x for x in self.imag))  # type: ignore

        if self._priority < lhs._priority:
            return NotImplemented

        if type(self) is not type(lhs):
            raise TypeError

        imag = (lhs.real * y + x * self.real for x, y in zip(lhs.imag, self.imag))
        return self.__class__(lhs.real * self.real, imag)

    def __rtruediv__(self, lhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(lhs):
            return NotImplemented

        if not isinstance(lhs, Dual) or self._priority > lhs._priority:
            s = self.real**2
            imag = (-lhs * x / s for x in self.imag)  # type: ignore
            return self.__class__(lhs / self.real, imag)  # type: ignore

        if self._priority < lhs._priority:
            return NotImplemented

        if type(self) is not type(lhs):
            raise TypeError

        s = self.real**2
        imag = ((x * self.real - lhs.real * y) / s for x, y in zip(lhs.imag, self.imag))
        return self.__class__(lhs.real / self.real, imag)  # type: ignore

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.__IS_SEALED:
            raise RuntimeError("subclassing is forbidden")


class Jet[T1: Scalar, T2 = Any](Scalar, ABC):
    r"""Abstract base class for jets.

    Parameters
    ----------
    coeffs : Iterable[T]
        Coefficients of the jet.

    Attributes
    ----------
    coeffs : list[T]
        Coefficients of the jet.

    Warnings
    --------
    Users cannot define classes derived from this.

    See Also
    --------
    DynJet, IntervalJet

    Notes
    -----
    In principle, instances of this class behave like elements of the ring
    :math:`T[x]/(x^n)`, where :math:`n` is the length of `coeffs`. However, this does
    not apply to :math:`n=1`. If :math:`n=1`, the instance is treated as an element of
    the coefficient ring `T`.

    Examples
    --------
    >>> x = DynJet([1.0, 6.0, 7.0, 2.0])
    >>> y = DynJet([3.0, -1.0, 5.0])
    >>> z = x + y
    >>> z
    DynJet([4.0, 5.0, 12.0])

    >>> x = DynJet([1.0, 2.0, 3.0])
    >>> y = DynJet([2.0])  # treated as 2.0 since its order is 0
    >>> z = x * y
    >>> z
    DynJet([2.0, 4.0, 6.0])
    """

    __slots__ = ("coeffs",)
    __IS_SEALED: Final = True
    coeffs: list[T1]

    def __init__(self, coeffs: Iterable[T1]):
        self.coeffs = list(coeffs)

        if len(self.coeffs) == 0:
            raise ValueError

    def reciprocal(self) -> Self:
        """Return the reciprocal of the jet.

        Raises
        ------
        ZeroDivisionError
            If the constant term is not invertible.
        """
        result = [1 / self.coeffs[0]]

        for k in range(1, len(self.coeffs)):
            tmp = sum(result[i] * self.coeffs[k - i] for i in range(k))
            result.append(-tmp / self.coeffs[0])

        return self.__class__(result)

    @abstractmethod
    def _is_acceptable(self, value: object) -> bool:
        raise NotImplementedError

    @property
    def order(self) -> int:
        """Order of the jet.

        This is a shorthand for ``len(self.coeffs) - 1``.
        """
        return len(self.coeffs) - 1

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.coeffs!r})"

    def __str__(self) -> str:
        return f"{type(self).__name__}([{(', ').join(str(x) for x in self.coeffs)}])"

    def __call__(self, arg: Self) -> Self:
        if not isinstance(arg, type(self)):
            raise TypeError

        if len(self.coeffs) < len(arg.coeffs):
            raise ValueError

        order = arg.order
        result = arg.__class__((self.coeffs[order],))

        for x in reversed(self.coeffs[:order]):
            result *= arg
            result += x

        return result

    def __add__(self, rhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(rhs):
            return NotImplemented

        if not isinstance(rhs, type(self)):
            return self.__class__((self.coeffs[0] + rhs, *self.coeffs[1:]))  # type: ignore

        if len(rhs.coeffs) == 1:
            return self.__class__((self.coeffs[0] + rhs.coeffs[0], *self.coeffs[1:]))

        if len(self.coeffs) == 1:
            return self.__class__((self.coeffs[0] + rhs.coeffs[0], *rhs.coeffs[1:]))

        return self.__class__(x + y for x, y in zip(self.coeffs, rhs.coeffs))

    def __sub__(self, rhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(rhs):
            return NotImplemented

        if not isinstance(rhs, type(self)):
            return self.__class__((self.coeffs[0] - rhs, *self.coeffs[1:]))  # type: ignore

        if len(rhs.coeffs) == 1:
            return self.__class__((self.coeffs[0] - rhs.coeffs[0], *self.coeffs[1:]))

        if len(self.coeffs) == 1:
            tail = (-x for x in rhs.coeffs[1:])
            return self.__class__((self.coeffs[0] - rhs.coeffs[0], *tail))

        return self.__class__(x - y for x, y in zip(self.coeffs, rhs.coeffs))

    def __mul__(self, rhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(rhs):
            return NotImplemented

        if not isinstance(rhs, type(self)):
            return self.__class__(x * rhs for x in self.coeffs)  # type: ignore

        if len(rhs.coeffs) == 1:
            return self.__class__(x * rhs.coeffs[0] for x in self.coeffs)

        if len(self.coeffs) == 1:
            return self.__class__(self.coeffs[0] * x for x in rhs.coeffs)

        coeffs: list[T1] = []

        for i in range(min(len(self.coeffs), len(rhs.coeffs))):
            tmp = sum(self.coeffs[i - j] * rhs.coeffs[j] for j in range(i + 1))
            coeffs.append(tmp)  # type: ignore

        return self.__class__(coeffs)

    def __truediv__(self, rhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(rhs):
            return NotImplemented

        if not isinstance(rhs, type(self)):
            return self.__class__(x / rhs for x in self.coeffs)  # type: ignore

        if len(rhs.coeffs) == 1:
            return self.__class__(x / rhs.coeffs[0] for x in self.coeffs)

        if len(self.coeffs) == 1:
            reciprocal = [1 / rhs.coeffs[0]]

            for k in range(1, len(rhs.coeffs)):
                tmp = sum(reciprocal[i] * rhs.coeffs[k - i] for i in range(k))
                reciprocal.append(-tmp / rhs.coeffs[0])

            return self.__class__(self.coeffs[0] * x for x in reciprocal)

        reciprocal = [1 / rhs.coeffs[0]]

        for k in range(1, min(len(self.coeffs), len(rhs.coeffs))):
            tmp = sum(reciprocal[i] * rhs.coeffs[k - i] for i in range(k))
            reciprocal.append(-tmp / rhs.coeffs[0])

        coeffs = [self.coeffs[0] * reciprocal[0]]

        for i in range(1, len(reciprocal)):
            tmp = sum(self.coeffs[i - j] * reciprocal[j] for j in range(i + 1))
            coeffs.append(tmp)  # type: ignore

        return self.__class__(coeffs)

    def __pow__(self, rhs: int) -> Self:
        if not isinstance(rhs, int):
            return NotImplemented

        if rhs < 0:
            return self.__pow__(-rhs).reciprocal()

        result = self.__class__((self.coeffs[0] * 0 + 1,))
        tmp = self.__class__(self.coeffs)

        while rhs != 0:
            if rhs % 2 != 0:
                result *= tmp

            rhs //= 2
            tmp *= tmp

        return result

    def __neg__(self) -> Self:
        return self.__class__(-x for x in self.coeffs)

    def __pos__(self) -> Self:
        return self.__class__(+x for x in self.coeffs)

    def __radd__(self, lhs: Self | T1 | T2 | int):
        return self.__add__(lhs)

    def __rsub__(self, lhs: Self | T1 | T2 | int):
        return self.__neg__().__add__(lhs)

    def __rmul__(self, lhs: Self | T1 | T2 | int) -> Self:
        return self.__mul__(lhs)

    def __rtruediv__(self, lhs: Self | T1 | T2 | int) -> Self:
        if not self._is_acceptable(lhs):
            return NotImplemented

        if not isinstance(lhs, type(self)):
            reciprocal = [1 / self.coeffs[0]]

            for k in range(1, len(self.coeffs)):
                tmp = sum(reciprocal[i] * self.coeffs[k - i] for i in range(k))
                reciprocal.append(-tmp / self.coeffs[0])

            return self.__class__(lhs * x for x in reciprocal)  # type: ignore

        if len(self.coeffs) == 1:
            return self.__class__(x / self.coeffs[0] for x in lhs.coeffs)

        if len(lhs.coeffs) == 1:
            reciprocal = [1 / self.coeffs[0]]

            for k in range(1, len(self.coeffs)):
                tmp = sum(reciprocal[i] * self.coeffs[k - i] for i in range(k))
                reciprocal.append(-tmp / self.coeffs[0])

            return self.__class__(lhs.coeffs[0] * x for x in reciprocal)

        reciprocal = [1 / self.coeffs[0]]

        for k in range(1, min(len(lhs.coeffs), len(self.coeffs))):
            tmp = sum(reciprocal[i] * self.coeffs[k - i] for i in range(k))
            reciprocal.append(-tmp / self.coeffs[0])

        coeffs: list[T1] = []

        for i in range(len(reciprocal)):
            tmp = sum(lhs.coeffs[i - j] * reciprocal[j] for j in range(i + 1))
            coeffs.append(tmp)  # type: ignore

        return self.__class__(coeffs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.__IS_SEALED:
            raise RuntimeError("subclassing is forbidden")


Dual._Dual__IS_SEALED = False  # type: ignore
Jet._Jet__IS_SEALED = False  # type: ignore


@final
class IntervalDual[T: ComparableScalar](Dual[Interval[T], T | int | float]):
    """Dual with interval coefficients.

    Parameters
    ----------
    real : Interval[T]
    imag : Iterable[Interval[T]]

    Attributes
    ----------
    real : Interval[T]
    imag : list[Interval[T]]

    Warnings
    --------
    All the elements of `imag` must be of the same type as `real`.
    """

    __slots__ = ("_intvl",)
    _intvl: type[Interval[T]]

    def __init__(self, real: Interval[T], imag: Iterable[Interval[T]]):
        super().__init__(real, imag)
        self._intvl = type(real)

        if not issubclass(self._intvl, Interval):
            raise TypeError("arguments are not intervals")

    def _is_acceptable(self, value: object) -> bool:
        intvl = self._intvl
        return isinstance(value, type(self) | intvl | intvl.endtype | int | float)


@final
class IntervalJet[T: ComparableScalar](Jet[Interval[T], T | int | float]):
    """Jet with interval coefficients.

    Parameters
    ----------
    coeffs : Iterable[Interval[T]]

    Attributes
    ----------
    coeffs : Iterable[Interval[T]]

    Warnings
    --------
    All the elements of `coeffs` must have a same type.
    """

    __slots__ = ("_intvl",)
    _intvl: type[Interval[T]]

    def __init__(self, coeffs: Iterable[Interval[T]]):
        super().__init__(coeffs)
        self._intvl = type(self.coeffs[0])

        if not issubclass(self._intvl, Interval):
            raise TypeError("arguments are not intervals")

    def _is_acceptable(self, value: object) -> bool:
        intvl = self._intvl
        return isinstance(value, type(self) | intvl | intvl.endtype | int | float)


@final
class DynDual[T: Scalar](Dual[T]):
    """Dual for an arbitrary coefficient type.

    Parameters
    ----------
    real : T
    imag : Iterable[T]

    Attributes
    ----------
    real : T
    imag : list[T]

    Warnings
    --------
    All the elements of `imag` must be of the same type as `real`, and must not be
    instances of :class:`DynJet`.
    """

    __slots__ = ()

    def __init__(self, real: T, imag: Iterable[T]):
        if isinstance(real, DynJet):
            raise TypeError("differentiation w.r.t. DynJet is not allowed")

        super().__init__(real, imag)

    def _is_acceptable(self, value: object) -> bool:
        return True


@final
class DynJet[T: Scalar](Jet[T]):
    """Jet for an arbitrary coefficient type.

    Parameters
    ----------
    coeffs : Iterable[T]
        Coefficients of the jet.

    Attributes
    ----------
    coeffs : list[T]
        Coefficients of the jet.

    Warnings
    --------
    All the elements of `coeffs` must have a same type, and they must not be instances
    of :class:`Jet` or :class:`DynDual`.
    """

    __slots__ = ()

    def __init__(self, coeffs: Iterable[T]):
        super().__init__(coeffs)

        if isinstance(self.coeffs[0], Jet):
            raise TypeError("nesting Jet is forbidden")

        if isinstance(self.coeffs[0], DynDual):
            raise TypeError("coefficients of Jet must not be DynDual")

    def _is_acceptable(self, value: object) -> bool:
        return True


Dual._Dual__IS_SEALED = True  # type: ignore
Jet._Jet__IS_SEALED = True  # type: ignore
