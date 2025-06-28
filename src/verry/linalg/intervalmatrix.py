import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from types import EllipsisType
from typing import Any, Literal, Self, cast, overload

import numpy as np
import numpy.typing as npt

from verry import function as vrf
from verry.interval.interval import Interval
from verry.typing import ComparableScalar


class LinAlgError(ValueError):
    """Error raised by :mod:`verry.linalg` functions."""


class flatiter[T: ComparableScalar](Iterator[Interval[T]]):
    __slots__ = ("_iter", "_matrix")
    _iter: Iterator
    _matrix: "IntervalMatrix[T]"

    def __init__(self, a: "IntervalMatrix[T]", /):
        self._iter = iter(itertools.product(*(range(n) for n in a.shape)))
        self._matrix = a

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Interval[T]:
        return self._matrix[*next(self._iter)]


class matiter(Iterator):
    __slots__ = ("_iter", "_matrix")
    _iter: Iterator
    _matrix: "IntervalMatrix"

    def __init__(self, a: "IntervalMatrix", /):
        self._iter = iter(range(len(a)))
        self._matrix = a

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Any:
        return self._matrix[next(self._iter)]


_intervalmatrices: list[type["IntervalMatrix"]] = []


def resolve_intervalmatrix[T: ComparableScalar](
    intvl: type[Interval[T]],
) -> type["IntervalMatrix[T]"]:
    """Return a type of interval matrix whose components are of type `intvl`.

    Parameters
    ----------
    intvl : type[Interval]
        Component type of interval matrices.

    Returns
    -------
    type[IntervalMatrix]

    Raises
    ------
    TypeError
        Raised when the corresponding type of the interval matrix is unknown.
    """
    if not _intervalmatrices:
        from verry.linalg.floatintervalmatrix import FloatIntervalMatrix

        _intervalmatrices.append(FloatIntervalMatrix)

    if tmp := next((x for x in _intervalmatrices if x.interval is intvl), None):
        return cast(type[IntervalMatrix[T]], tmp)

    raise TypeError("could not resolve the corresponding interval matrix")


class IntervalMatrix[S: ComparableScalar](ABC):
    """Abstract base class for inf-sup type interval matrices.

    Parameters
    ----------
    inf : ndarray | Sequence
    sup : ndarray | Sequence, optional

    Attributes
    ----------
    inf : ndarray
        Infimum of the interval matrix.
    sup : ndarray
        Supremum of the interval matrix.
    """

    __slots__ = ("inf", "sup")
    __array_ufunc__ = None
    interval: type[Interval[S]]
    inf: npt.NDArray
    sup: npt.NDArray

    def __init__(
        self,
        inf: npt.NDArray
        | Sequence[Interval[S] | S | float | int | str]
        | Sequence[npt.NDArray | Sequence[Interval[S] | S | float | int | str]],
        sup: npt.NDArray
        | Sequence[Interval[S] | S | float | int | str]
        | Sequence[npt.NDArray | Sequence[Interval[S] | S | float | int | str]]
        | None = None,
        **kwargs,
    ):
        if kwargs.get("_skipcheck"):
            self.inf = inf  # type: ignore
            self.sup = sup  # type: ignore
            return

        if sup is None:
            inf = np.array(inf, np.object_)

            if not 1 <= inf.ndim <= 2:
                raise ValueError

            self.inf = self._emptyarray(inf.shape)  # type: ignore
            self.sup = self._emptyarray(inf.shape)  # type: ignore

            for key in itertools.product(*(range(n) for n in inf.shape)):
                tmp = self.interval.ensure(inf[key])
                self.inf[key] = tmp.inf
                self.sup[key] = tmp.sup

            return

        inf = np.array(inf, np.object_)
        sup = np.array(sup, np.object_)

        if not (1 <= inf.ndim <= 2 and inf.shape == sup.shape):
            raise ValueError

        self.inf = self._emptyarray(inf.shape)  # type: ignore
        self.sup = self._emptyarray(inf.shape)  # type: ignore

        for key in itertools.product(*(range(n) for n in inf.shape)):
            self.inf[key] = self.interval.ensure(inf[key]).inf  # type: ignore
            self.sup[key] = self.interval.ensure(sup[key]).sup  # type: ignore

            if self.inf[key] > self.sup[key]:
                raise ValueError

    @property
    def flat(self) -> flatiter[S]:
        return flatiter(self)

    @property
    def ndim(self) -> int:
        """Shorthand for ``len(self.shape)``.

        Unlike NumPy's ndarray, ``x.ndim`` is always greater than or equal to 1, i.e.,
        there is no 0D matrix.
        """
        return len(self.shape)

    @property
    def shape(self) -> tuple[int] | tuple[int, int]:
        """Tuple of matrix dimensions.

        Examples
        --------
        >>> from verry.linalg import FloatIntervalMatrix as FIM
        >>> x = FIM([[1, 2, 3], [4, 5, 6]])
        >>> y = FIM([1.0, 2.0])
        >>> x.shape
        (2, 3)
        >>> y.shape
        (2,)
        """
        return self.inf.shape  # type: ignore

    @property
    def size(self) -> int:
        return self.inf.size

    @property
    def T(self) -> Self:
        """Shorthand for :meth:`transpose`.

        See Also
        --------
        transpose
        """
        return self.transpose()

    @classmethod
    def empty(cls, shape: int | tuple[int] | tuple[int, int]) -> Self:
        """Return a new interval matrix of given shape, without initializing entries."""
        inf = cls._emptyarray(shape)
        sup = inf.copy()
        return cls(inf, sup, _skipcheck=True)  # type: ignore

    @classmethod
    def eye(cls, n: int, m: int | None = None) -> Self:
        """Return a matrix with ones on the diagonal and zeros elsewhere."""
        if m is None:
            m = n

        result = cls.zeros((n, m))

        for i in range(min(n, m)):
            result.inf[i, i] = cls.interval.operator.ONE
            result.sup[i, i] = cls.interval.operator.ONE

        return result

    @classmethod
    def ones(cls, shape: int | tuple[int] | tuple[int, int]) -> Self:
        """Return a new interval matrix of given shape, filled with ones."""
        result = cls.empty(shape)
        result.inf[...] = cls.interval.operator.ONE
        result.sup[...] = cls.interval.operator.ONE
        return result

    @classmethod
    def zeros(cls, shape: int | tuple[int] | tuple[int, int]) -> Self:
        """Return a new interval matrix of given shape, filled with zeros."""
        result = cls.empty(shape)
        result.inf[...] = cls.interval.operator.ZERO
        result.sup[...] = cls.interval.operator.ZERO
        return result

    @classmethod
    @abstractmethod
    def _emptyarray(cls, shape: int | tuple[int] | tuple[int, int]) -> npt.NDArray:
        raise NotImplementedError

    def copy(self) -> Self:
        """Return a copy of the interval matrix."""
        return self.__class__(self.inf.copy(), self.sup.copy(), _skipcheck=True)  # type: ignore

    def empty_like(self) -> Self:
        """Return a new interval matrix with the same shape as `self`."""
        return self.empty(self.shape)

    def diam(self) -> npt.NDArray:
        """Return component-wise upper bounds of the diameter."""
        result = self._emptyarray(self.shape)

        for key in itertools.product(*(range(n) for n in self.shape)):
            result[key] = self[key].diam()  # type: ignore

        return result

    def flatten(self) -> Self:
        """Return a copy of the interval matrix collapsed into one dimension."""
        return self.__class__(self.inf.flatten(), self.sup.flatten(), _skipcheck=True)  # type: ignore

    def interiorcontains(self, other: Self | npt.NDArray) -> bool:
        """Return ``True`` if the interior of the interval matrix contains `other`."""
        if not isinstance(other, np.ndarray | type(self)):
            raise TypeError

        if self.shape != other.shape:
            raise ValueError

        return all(x.interiorcontains(y) for x, y in zip(self.flat, other.flat))

    def isbounded(self) -> bool:
        """Return ``True`` if both `inf` and `sup` are bounded."""
        return all(x.isbounded() for x in self.flat)

    def isdisjoint(self, other: Self | npt.NDArray) -> bool:
        """Return ``True`` if the interval matrix has no elements in common with
        `other`."""
        if not isinstance(other, np.ndarray | type(self)):
            raise TypeError

        if self.shape != other.shape:
            raise ValueError

        return all(x.isdisjoint(y) for x, y in zip(self.flat, other.flat))

    def issubset(self, other: Self) -> bool:
        """Test whether every element in the interval matrix is in `other`."""
        if type(self) is not type(other):
            return False

        if self.shape != other.shape:
            return False

        return all(x.issubset(y) for x, y in zip(self.flat, other.flat))

    def issuperset(self, other: Self) -> bool:
        """Test whether every element in `other` is in the interval matrix."""
        if type(self) is not type(other):
            return False

        if self.shape != other.shape:
            return False

        return all(x.issuperset(y) for x, y in zip(self.flat, other.flat))

    def mid(self) -> npt.NDArray:
        """Return an approximation of the midpoint.

        As with scalar intervals, ``x.mid() in x`` is guaranteed to be ``True``.
        """
        result = self._emptyarray(self.shape)

        for key in itertools.product(*(range(n) for n in self.shape)):
            result[key] = self[key].mid()  # type: ignore

        return result

    def ones_like(self) -> Self:
        """Return an interval matrix of ones with the same shape as `self`."""
        return self.ones(self.shape)

    def rad(self) -> npt.NDArray:
        """Return component-wise upper bounds of the radius."""
        result = self._emptyarray(self.shape)

        for key in itertools.product(*(range(n) for n in self.shape)):
            result[key] = self[key].rad()  # type: ignore

        return result

    def reshape(self, shape: int | tuple[int] | tuple[int, int]) -> Self:
        """Give a new shape to an interval matrix without changing its data."""
        inf = self.inf.reshape(shape, copy=True)
        sup = self.sup.reshape(shape, copy=True)
        return self.__class__(inf, sup, _skipcheck=True)  # type: ignore

    def transpose(self) -> Self:
        """Return a view of the transposed matrix."""
        return self.__class__(self.inf.T, self.sup.T, _skipcheck=True)  # type: ignore

    def zeros_like(self) -> Self:
        """Return an interval matrix of zeros with the same shape as `self`."""
        return self.zeros(self.shape)

    def _verry_overload_(self, fun, *args, **kwargs):
        if fun is approx_inv:
            return self.__approx_inv()

        if fun is approx_norm:
            return self.__approx_norm(args[1])

        if fun is approx_qr:
            return self.__approx_qr()

        if fun is approx_solve:
            if args[0] is not self:
                return self.__class__(args[0]).__approx_solve(args[1])

            return self.__approx_solve(args[1])

        if fun is inv:
            return self.__inv(args[1])

        if fun is norm:
            return self.__norm(args[1])

        if fun is solve:
            if args[0] is not self:
                return self.__class__(args[0]).__solve(*args[1:])

            return self.__solve(*args[1:])

        return NotImplemented

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return bool(np.all(other.inf == self.inf) and np.all(other.sup == self.sup))

    def __len__(self) -> int:
        return len(self.inf)

    @overload
    def __getitem__(self, key: int) -> Any: ...

    @overload
    def __getitem__(self, key: tuple[int, int]) -> Interval[S]: ...

    @overload
    def __getitem__(
        self,
        key: slice
        | EllipsisType
        | tuple[int, slice]
        | tuple[slice, int]
        | tuple[slice, slice],
    ) -> Self: ...

    def __getitem__(self, key):
        if isinstance(self.inf[key], np.ndarray):
            return self.__class__(self.inf[key], self.sup[key], _skipcheck=True)  # type: ignore

        return self.interval(self.inf[key], self.sup[key])

    @overload
    def __setitem__(
        self, key: tuple[int, int], value: Interval[S] | S | float | int
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: int
        | slice
        | EllipsisType
        | tuple[int, slice]
        | tuple[slice, int]
        | tuple[slice, slice],
        value: Self | Interval[S] | S | npt.NDArray | float | int,
    ) -> None: ...

    def __setitem__(self, key, value):
        match value:
            case self.__class__():
                self.inf[key] = value.inf
                self.sup[key] = value.sup

            case np.ndarray():
                value = self.__class__(value)
                self.inf[key] = value.inf
                self.sup[key] = value.sup

            case self.interval():
                self.inf[key] = value.inf
                self.sup[key] = value.sup

            case self.interval.endtype():
                self.inf[key] = value
                self.sup[key] = value

            case int():
                value = self.interval(value)
                self.inf[key] = value.inf
                self.sup[key] = value.sup

            case float():
                value = self.interval.converter.fromfloat(value)
                self.inf[key] = value
                self.sup[key] = value

            case _:
                raise TypeError

    def __iter__(self) -> matiter:
        return matiter(self)

    def __contains__(
        self, item: Sequence[S | float | int | Sequence[S | float | int]]
    ) -> bool:
        tmp = np.array(item, dtype=np.object_)

        if tmp.shape != self.shape:
            return False

        for key in itertools.product(*(range(n) for n in self.shape)):
            if tmp[key] not in self[key]:  # type: ignore
                return False

        return True

    def __add__(self, rhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        return self.copy().__iadd__(rhs)

    def __sub__(self, rhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        return self.copy().__isub__(rhs)

    def __mul__(self, rhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        return self.copy().__imul__(rhs)

    def __truediv__(
        self, rhs: Self | Interval[S] | S | npt.NDArray | float | int
    ) -> Self:
        return self.copy().__itruediv__(rhs)

    def __matmul__(self, rhs: Self | npt.NDArray) -> Self:
        lhs, lshape = self, self.shape

        if not isinstance(rhs, (np.ndarray, type(self))):
            return NotImplemented

        rshape = rhs.shape

        if len(lshape) == 1:
            if lshape[0] != rshape[0]:
                raise ValueError

            if len(rshape) == 1:
                return sum(lhs[k] * rhs[k] for k in range(lshape[0]))

            result = self.empty((rshape[1],))

            for j in range(rshape[1]):
                result[j] = sum(lhs[k] * rhs[k, j] for k in range(lshape[0]))

            return result

        if lshape[1] != rshape[0]:
            raise ValueError

        if len(rshape) == 1:
            result = self.empty((lshape[0],))

            for i in range(lshape[0]):
                result[i] = sum(lhs[i, k] * rhs[k] for k in range(lshape[1]))

            return result

        result = self.empty((lshape[0], rshape[1]))

        for i in range(lshape[0]):
            for j in range(rshape[1]):
                result[i, j] = sum(lhs[i, k] * rhs[k, j] for k in range(lshape[1]))

        return result

    def __radd__(self, lhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        return self.copy().__iadd__(lhs)

    def __rsub__(self, lhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        return self.__neg__().__iadd__(lhs)

    def __rmul__(self, lhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        return self.copy().__imul__(lhs)

    def __rtruediv__(
        self, lhs: Self | Interval[S] | S | npt.NDArray | float | int
    ) -> Self:
        match lhs:
            case self.interval.endtype() | self.interval() | float() | int():
                for key in itertools.product(*(range(n) for n in self.shape)):
                    self[key] = lhs / self[key]  # type: ignore

                return self

            case np.ndarray() | self.__class__():
                if self.shape != lhs.shape:
                    raise ValueError

            case _:
                return NotImplemented

        result = self.empty(self.shape)

        for key in itertools.product(*(range(n) for n in self.shape)):
            result[key] = lhs[key] / self[key]  # type: ignore

        return result

    def __rmatmul__(self, lhs: Self | npt.NDArray) -> Self:
        rhs, rshape = self, self.shape

        if not isinstance(lhs, (np.ndarray, type(self))):
            return NotImplemented

        lshape = lhs.shape

        if len(lshape) == 1:
            if lshape[0] != rshape[0]:
                raise ValueError

            if len(rshape) == 1:
                return sum(lhs[k] * rhs[k] for k in range(lshape[0]))

            result = self.empty((rshape[1],))

            for j in range(rshape[1]):
                result[j] = sum(lhs[k] * rhs[k, j] for k in range(lshape[0]))

            return result

        if lshape[1] != rshape[0]:
            raise ValueError

        if len(rshape) == 1:
            result = self.empty((lshape[0],))

            for i in range(lshape[0]):
                result[i] = sum(lhs[i, k] * rhs[k] for k in range(lshape[1]))

            return result

        result = self.empty((lshape[0], rshape[1]))

        for i in range(lshape[0]):
            for j in range(rshape[1]):
                result[i, j] = sum(lhs[i, k] * rhs[k, j] for k in range(lshape[1]))

        return result

    def __iadd__(self, rhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        match rhs:
            case self.interval.endtype() | self.interval() | float() | int():
                for key in itertools.product(*(range(n) for n in self.shape)):
                    self[key] += rhs  # type: ignore

                return self

            case np.ndarray() | self.__class__():
                if self.shape != rhs.shape:
                    raise ValueError

            case _:
                return NotImplemented

        for key in itertools.product(*(range(n) for n in self.shape)):
            self[key] += rhs[key]  # type: ignore

        return self

    def __isub__(self, rhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        match rhs:
            case self.interval.endtype() | self.interval() | float() | int():
                for key in itertools.product(*(range(n) for n in self.shape)):
                    self[key] -= rhs  # type: ignore

                return self

            case np.ndarray() | self.__class__():
                if self.shape != rhs.shape:
                    raise ValueError

            case _:
                return NotImplemented

        for key in itertools.product(*(range(n) for n in self.shape)):
            self[key] -= rhs[key]  # type: ignore

        return self

    def __imul__(self, rhs: Self | Interval[S] | S | npt.NDArray | float | int) -> Self:
        match rhs:
            case self.interval.endtype() | self.interval() | float() | int():
                for key in itertools.product(*(range(n) for n in self.shape)):
                    self[key] *= rhs  # type: ignore

                return self

            case np.ndarray() | self.__class__():
                if self.shape != rhs.shape:
                    raise ValueError

            case _:
                return NotImplemented

        for key in itertools.product(*(range(n) for n in self.shape)):
            self[key] *= rhs[key]  # type: ignore

        return self

    def __itruediv__(
        self, rhs: Self | Interval[S] | S | npt.NDArray | float | int
    ) -> Self:
        match rhs:
            case self.interval.endtype() | self.interval() | float() | int():
                for key in itertools.product(*(range(n) for n in self.shape)):
                    self[key] /= rhs  # type: ignore

                return self

            case np.ndarray() | self.__class__():
                if self.shape != rhs.shape:
                    raise ValueError

            case _:
                return NotImplemented

        for key in itertools.product(*(range(n) for n in self.shape)):
            self[key] /= rhs[key]  # type: ignore

        return self

    def __neg__(self) -> Self:
        return self.copy().__imul__(-self.interval.operator.ONE)

    def __pos__(self) -> Self:
        return self.copy()

    def __abs__(self) -> Self:
        result = self.empty(self.shape)

        for key in itertools.product(*(range(n) for n in self.shape)):
            result[key] = abs(self[key])  # type: ignore

        return result

    def __copy__(self) -> Self:
        return self.copy()

    def __approx_inv(self):
        if not (len(self.shape) == 2 and self.shape[0] == self.shape[1]):
            raise LinAlgError("non-square matrix")

        ZERO = self.interval.operator.ZERO
        ONE = self.interval.operator.ONE
        a = self.mid()
        b = np.full_like(a, ZERO)
        n = self.shape[0]

        for i in range(n):
            b[i, i] = ONE

        for k in range(n):
            if (p := np.argmax(abs(a[k:n, k])) + k) != k:
                a[(k, p),] = a[(p, k),]
                b[(k, p),] = b[(p, k),]

            if a[k, k] == ZERO:
                raise LinAlgError("numerically singular matrix")

            for i in range(k + 1, n):
                tmp = a[i, k] / a[k, k]
                a[i, k:] -= tmp * a[k, k:]
                b[i] -= tmp * b[k]

        for i in reversed(range(n)):
            b[i] -= a[i, i + 1 :] @ b[i + 1 :]
            b[i] /= a[i, i]

        return b

    def __approx_norm(self, ord):
        if len(self.shape) == 1:
            match ord:
                case "fro":
                    raise LinAlgError

                case "inf":
                    return max(abs(x.mid()) for x in self)

                case "one":
                    return sum(abs(x.mid()) for x in self)

                case "two":
                    return vrf.sqrt(sum(x.mid() ** 2 for x in self))

                case _:
                    raise ValueError

        match ord:
            case "fro":
                return vrf.sqrt(sum(x.mid() ** 2 for x in self.flat))

            case "inf":
                n = self.shape[0]
                return max(sum(abs(x.mid()) for x in self[i, :]) for i in range(n))

            case "one":
                m = self.shape[1]
                return max(sum(abs(x.mid()) for x in self[:, j]) for j in range(m))

            case "two":
                return NotImplemented

            case _:
                raise ValueError

    def __approx_qr(self):
        ZERO = self.interval.operator.ZERO
        ONE = self.interval.operator.ONE
        a = self.mid()
        n, m = self.shape
        q = np.full((n, n), ZERO, like=a)

        for i in range(n):
            q[i, i] = ONE

        for k in range(m):
            tmp = a[k:, k]
            e = np.full_like(tmp, ZERO)
            e[0] = vrf.sqrt(sum(x**2 for x in tmp)) * (1 if tmp[0] >= ZERO else -1)
            v = tmp + e
            v /= vrf.sqrt(sum(x**2 for x in v))
            h = np.full((n, n), ZERO, like=a)

            for i in range(n):
                h[i, i] = ONE

            for i in range(k, n):
                for j in range(k, n):
                    h[i, j] -= 2 * v[i - k] * v[j - k]

            q = q @ h
            a = h @ a

        return (q, a)

    def __approx_solve(self, b):
        if not (len(self.shape) == 2 and self.shape[0] == self.shape[1]):
            raise LinAlgError("non-square matrix")

        if not (b.ndim == 1 and len(b) == len(self)):
            raise LinAlgError("dimension mismatch")

        ZERO = self.interval.operator.ZERO
        a = self.mid()
        b = b.copy()
        n = len(a)

        for k in range(n):
            if (p := np.argmax(abs(a[k:n, k])) + k) != k:
                a[(k, p),] = a[(p, k),]
                b[(k, p),] = b[(p, k),]

            if a[k, k] == ZERO:
                raise LinAlgError("numerically singular matrix")

            for i in range(k + 1, n):
                tmp = a[i, k] / a[k, k]
                a[i, k:] -= tmp * a[k, k:]
                b[i] -= tmp * b[k]

        for i in reversed(range(n)):
            b[i] -= sum(a[i, i + 1 :] * b[i + 1 :])
            b[i] /= a[i, i]

        return b

    def __inv(self, r):
        if r is None:
            r = approx_inv(self)

        eye = self.eye(len(self))
        result = self.empty_like()

        for i in range(len(self)):
            result[:, i] = solve(self, eye[:, i], r)

        return result

    def __norm(self, ord):
        if len(self.shape) == 1:
            match ord:
                case "fro":
                    raise LinAlgError

                case "inf":
                    result = self.interval()

                    for i in range(self.shape[0]):
                        tmp = abs(self[i])
                        result.inf = max(result.inf, tmp.inf)
                        result.sup = max(result.sup, tmp.sup)

                    return result

                case "one":
                    return sum(abs(self[i]) for i in range(self.shape[0]))

                case "two":
                    return vrf.sqrt(sum(self[i] ** 2 for i in range(self.shape[0])))

                case _:
                    raise ValueError

        match ord:
            case "fro":
                result = self.interval()

                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        result += self[i, j] ** 2

                return vrf.sqrt(result)

            case "inf":
                result = self.interval()

                for i in range(self.shape[0]):
                    tmp = sum(abs(self[i, j]) for j in range(self.shape[1]))
                    result.inf = max(result.inf, tmp.inf)
                    result.sup = max(result.sup, tmp.sup)

                return result

            case "one":
                result = self.interval()

                for j in range(self.shape[1]):
                    tmp = sum(abs(self[i, j]) for i in range(self.shape[0]))
                    result.inf = max(result.inf, tmp.inf)
                    result.sup = max(result.sup, tmp.sup)

                return result

            case "two":
                return NotImplemented

            case _:
                raise TypeError

    def __solve(self, b, r):
        if r is None:
            r = approx_inv(self)

        if not isinstance(b, type(self)):
            b = self.__class__(b)

        tmp = norm(self.eye(len(self)) - r @ self, "inf")

        if tmp.inf >= self.interval.operator.ONE:
            raise LinAlgError("numerically singular matrix")

        mid = r @ b
        rad = norm(r @ (b - self @ mid), "inf") / (1 - tmp)
        return mid + rad * self.interval(-1, 1) * b.ones_like()


def approx_inv(a: IntervalMatrix) -> npt.NDArray:
    """Approximately compute the inverse of an interval matrix.

    Parameters
    ----------
    a : IntervalMatrix
        Matrix to be inverted.

    Returns
    -------
    ndarray

    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.
    """
    if (res := a._verry_overload_(approx_inv, a)) is not NotImplemented:
        return res

    raise RuntimeError


def approx_norm[T: ComparableScalar](
    a: IntervalMatrix[T], ord: Literal["fro", "inf", "one", "two"]
) -> T:
    """Compute a matrix or vector norm approximately.

    Parameters
    ----------
    ord : Literal["fro", "inf", "one", "two"]
        Order of the norm (see NumPy documentation).
    """
    if (res := a._verry_overload_(approx_norm, a, ord)) is not NotImplemented:
        return res

    raise RuntimeError


def approx_qr(a: IntervalMatrix) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the QR factorization of a matrix approximately.

    Parameters
    ----------
    a : IntervalMatrix
        Matrix to be QR factorized.

    Returns
    -------
    q : ndarray
        Approximately orthogonal matrix.
    r : ndarray
        Upper-triangular matrix.
    """
    if (res := a._verry_overload_(approx_qr, a)) is not NotImplemented:
        return res

    raise RuntimeError


@overload
def approx_solve(a: IntervalMatrix, b: IntervalMatrix | npt.NDArray) -> npt.NDArray: ...


@overload
def approx_solve(a: npt.NDArray, b: IntervalMatrix) -> npt.NDArray: ...


def approx_solve(a, b):
    """Solve a linear equation ``a @ x = b`` approximately.

    Parameters
    ----------
    a : IntervalMatrix | ndarray
        Coefficient matrix.
    b : IntervalMatrix | ndarray
        Right-hand side of the equation.

    Returns
    -------
    ndarray

    Raises
    ------
    LinAlgError
        If `a` is numerically singular or not square.
    """
    linearized = (a, b)

    if type(a) is not type(b) and issubclass(type(b), type(a)):
        linearized = (b, a)

    for x in linearized:
        if fun := getattr(type(x), "_verry_overload_", None):
            if (res := fun(x, approx_solve, a, b)) is not NotImplemented:
                return res

    raise RuntimeError


def inv[T: ComparableScalar](a: IntervalMatrix[T], r: npt.NDArray | None = None) -> T:
    """Compute the inverse of an interval matrix.

    Parameters
    ----------
    a : IntervalMatrix
        Matrix to be inverted.
    r : ndarray, optional
        Approximate inverse of `a`.

    Returns
    -------
    IntervalMatrix

    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.
    """
    if (res := a._verry_overload_(inv, a, r)) is not NotImplemented:
        return res

    raise RuntimeError


def norm[T: ComparableScalar](
    a: IntervalMatrix[T], ord: Literal["fro", "inf", "one", "two"]
) -> Interval[T]:
    """Compute a matrix or vector norm.

    Parameters
    ----------
    ord : Literal["fro", "inf", "one", "two"]
        Order of the norm (see NumPy documentation).

    Returns
    -------
    Interval
    """
    if (res := a._verry_overload_(norm, a, ord)) is not NotImplemented:
        return res

    raise RuntimeError


@overload
def solve[T: ComparableScalar](
    a: IntervalMatrix[T], b: IntervalMatrix[T], r: npt.NDArray | None = ...
) -> IntervalMatrix[T]: ...


@overload
def solve[T: ComparableScalar](
    a: IntervalMatrix[T], b: npt.NDArray, r: npt.NDArray | None = ...
) -> IntervalMatrix[T]: ...


@overload
def solve[T: ComparableScalar](
    a: npt.NDArray, b: IntervalMatrix[T], r: npt.NDArray | None = ...
) -> IntervalMatrix[T]: ...


def solve(a, b, r=None):
    """Solve a linear equation ``a @ x = b``.

    Parameters
    ----------
    a : IntervalMatrix | ndarray
        Coefficient matrix.
    b : IntervalMatrix | ndarray
        Right-hand side of the equation.
    r : ndarray, optional
        Approximate inverse of `a`.

    Returns
    -------
    IntervalMatrix

    Raises
    ------
    LinAlgError
        If `a` is numerically singular or not square.
    """
    linearized = (a, b)

    if type(a) is not type(b) and issubclass(type(b), type(a)):
        linearized = (b, a)

    for x in linearized:
        if fun := getattr(type(x), "_verry_overload_", None):
            if (res := fun(x, solve, a, b, r)) is not NotImplemented:
                return res

    raise RuntimeError
