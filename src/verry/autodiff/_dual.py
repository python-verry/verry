from collections.abc import Iterator, Sequence
from typing import Self

from verry.typing import Scalar


class Vector[T: Scalar]:
    __slots__ = ("_coeffs",)
    _coeffs: list[T]

    def __init__(self, coeffs: Self | Sequence[T]):
        self._coeffs = list(coeffs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._coeffs!r})"

    def __len__(self) -> int:
        return len(self._coeffs)

    def __iter__(self) -> Iterator[T]:
        return iter(self._coeffs)

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return other._coeffs == self._coeffs

    def __getitem__(self, key: int) -> T:
        return self._coeffs[key]

    def __setitem__(self, key: int, value: T) -> None:
        self._coeffs[key] = value

    def __add__(self, rhs: Self) -> Self:
        return self.__class__([x + y for x, y in zip(self._coeffs, rhs._coeffs)])

    def __sub__(self, rhs: Self) -> Self:
        return self.__class__([x - y for x, y in zip(self._coeffs, rhs._coeffs)])

    def __mul__(self, rhs) -> Self:
        return self.__class__([x * rhs for x in self._coeffs])

    def __truediv__(self, rhs) -> Self:
        return self.__class__([x / rhs for x in self._coeffs])

    def __neg__(self) -> Self:
        return self.__class__([-x for x in self._coeffs])

    def __pos__(self) -> Self:
        return self.__class__(self._coeffs)

    def __radd__(self, lhs: Self) -> Self:
        return self.__add__(lhs)

    def __rsub__(self, lhs: Self) -> Self:
        return self.__neg__().__add__(lhs)

    def __rmul__(self, lhs) -> Self:
        return self.__mul__(lhs)


class Dual[T: Scalar](Scalar):
    __slots__ = ("real", "imag", "_level")
    real: T
    imag: Vector[T]
    _level: int

    def __init__(self, real: T, imag: Vector[T] | Sequence[T]):
        self.real = real
        self.imag = Vector(imag)
        self._level = (real._level + 1) if isinstance(real, type(self)) else 1

    @property
    def level(self) -> int:
        return self._level

    @classmethod
    def variable(cls, *args: T) -> tuple[Self, ...]:
        result: list[Self] = []

        for argnum, arg in enumerate(args):
            ZERO = arg * 0
            ONE = ZERO + 1
            imag = tuple(ONE if i == argnum else ZERO for i in range(len(args)))
            result.append(cls(arg, imag))

        return tuple(result)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(real={self.real!r}, imag={self.imag._coeffs!r})"

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return other.real == self.real and other.imag == self.imag

    def __add__(self, rhs) -> Self:
        if not isinstance(rhs, type(self)) or rhs._level < self._level:
            return self.__class__(self.real + rhs, self.imag)

        if rhs._level > self._level:
            return self.__class__(self + rhs.real, rhs.imag)  # type: ignore

        imag = self.imag + rhs.imag
        return self.__class__(self.real + rhs.real, imag)

    def __sub__(self, rhs) -> Self:
        if not isinstance(rhs, type(self)) or rhs._level < self._level:
            return self.__class__(self.real - rhs, self.imag)

        if rhs._level > self._level:
            return self.__class__(self - rhs.real, -rhs.imag)  # type: ignore

        imag = self.imag - rhs.imag
        return self.__class__(self.real - rhs.real, imag)

    def __mul__(self, rhs) -> Self:
        if isinstance(rhs, Vector):
            return NotImplemented

        if not isinstance(rhs, type(self)) or rhs._level < self._level:
            return self.__class__(self.real * rhs, self.imag * rhs)

        if rhs._level > self._level:
            return self.__class__(self * rhs.real, self * rhs.imag)  # type: ignore

        imag = self.real * rhs.imag + rhs.real * self.imag
        return self.__class__(self.real * rhs.real, imag)

    def __truediv__(self, rhs) -> Self:
        if not isinstance(rhs, type(self)) or rhs._level < self._level:
            return self.__class__(self.real / rhs, self.imag / rhs)

        if rhs._level > self._level:
            imag = -self * rhs.imag / rhs.real**2
            return self.__class__(self / rhs.real, imag)  # type: ignore

        imag = (self.imag * rhs.real - self.real * rhs.imag) / rhs.real**2
        return self.__class__(self.real / rhs.real, imag)

    def __pow__(self, rhs: int) -> Self:
        imag = rhs * self.real ** (rhs - 1) * self.imag
        return self.__class__(self.real**rhs, imag)

    def __neg__(self) -> Self:
        return self.__class__(-self.real, -self.imag)

    def __pos__(self) -> Self:
        return self.__class__(self.real, self.imag)

    def __radd__(self, lhs) -> Self:
        return self.__add__(lhs)

    def __rsub__(self, lhs) -> Self:
        return self.__neg__().__add__(lhs)

    def __rmul__(self, lhs) -> Self:
        return self.__mul__(lhs)

    def __rtruediv__(self, lhs) -> Self:
        if isinstance(lhs, Vector):
            return NotImplemented

        if isinstance(lhs, type(self)):
            return lhs.__truediv__(self)

        imag = -lhs * self.imag / self.real**2
        return self.__class__(lhs / self.real, imag)
