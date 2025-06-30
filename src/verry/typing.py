"""
############################
Typing (:mod:`verry.typing`)
############################

This module provides type definitions commonly used between modules.

.. autoclass:: Scalar
    :show-inheritance:
    :no-members:

.. autoclass:: ComparableScalar
    :show-inheritance:
    :no-members:

"""

from abc import abstractmethod
from typing import Protocol, Self, SupportsAbs


class Scalar(Protocol):
    """Protocol that ensures scalar-like behavior.

    Objects implementing this protocol must have four arithmetic operations and
    integer power defined, and four arithmetic operations must be compatible with
    integers.
    """

    __slots__ = ()

    @abstractmethod
    def __add__(self, rhs: Self | int) -> Self: ...

    @abstractmethod
    def __sub__(self, rhs: Self | int) -> Self: ...

    @abstractmethod
    def __mul__(self, rhs: Self | int) -> Self: ...

    @abstractmethod
    def __truediv__(self, rhs: Self | int) -> Self: ...

    @abstractmethod
    def __pow__(self, rhs: int) -> Self: ...

    @abstractmethod
    def __radd__(self, lhs: Self | int) -> Self: ...

    @abstractmethod
    def __rsub__(self, lhs: Self | int) -> Self: ...

    @abstractmethod
    def __rmul__(self, lhs: Self | int) -> Self: ...

    @abstractmethod
    def __rtruediv__(self, lhs: Self | int) -> Self: ...

    @abstractmethod
    def __neg__(self) -> Self: ...

    @abstractmethod
    def __pos__(self) -> Self: ...


class ComparableScalar(Scalar, SupportsAbs, Protocol):
    """Protocol for comparable :class:`Scalar`, like a real number."""

    __slots__ = ()

    @abstractmethod
    def __lt__(self, rhs: Self) -> bool: ...

    @abstractmethod
    def __le__(self, rhs: Self) -> bool: ...

    @abstractmethod
    def __gt__(self, rhs: Self) -> bool: ...

    @abstractmethod
    def __ge__(self, rhs: Self) -> bool: ...

    @abstractmethod
    def __neg__(self) -> Self: ...

    @abstractmethod
    def __pos__(self) -> Self: ...
