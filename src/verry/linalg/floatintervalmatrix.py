from collections.abc import Sequence
from typing import Any, Self, overload

import numpy as np
import numpy.typing as npt

from verry.interval.floatinterval import FloatInterval
from verry.linalg.intervalmatrix import (
    IntervalMatrix,
    approx_inv,
    approx_qr,
    approx_solve,
)


class FloatIntervalMatrix(IntervalMatrix[FloatInterval, float]):
    __slots__ = ()

    @overload
    def __init__(
        self,
        inf: IntervalMatrix[FloatInterval]
        | npt.NDArray
        | Sequence[FloatInterval | float | int | str]
        | Sequence[npt.NDArray | Sequence[FloatInterval | float | int | str]],
        /,
        *,
        intvl: Any = None,
    ): ...

    @overload
    def __init__(
        self,
        inf: npt.NDArray
        | Sequence[float | int | str]
        | Sequence[npt.NDArray | Sequence[float | int | str]],
        sup: npt.NDArray
        | Sequence[float | int | str]
        | Sequence[npt.NDArray | Sequence[float | int | str]],
        *,
        intvl: Any = None,
    ): ...

    def __init__(self, inf, sup=None, *, intvl=None, **kwargs):
        super().__init__(inf, sup, intvl=FloatInterval, **kwargs)

    @classmethod
    def empty(
        cls, shape: int | tuple[int] | tuple[int, int], *, intvl: Any = None
    ) -> Self:
        return super().empty(shape, intvl=FloatInterval)

    @classmethod
    def eye(cls, n: int, m: int | None = None, *, intvl: Any = None) -> Self:
        return super().eye(n, m, intvl=FloatInterval)

    @classmethod
    def ones(
        cls, shape: int | tuple[int] | tuple[int, int], *, intvl: Any = None
    ) -> Self:
        return super().ones(shape, intvl=FloatInterval)

    @classmethod
    def zeros(
        cls, shape: int | tuple[int] | tuple[int, int], *, intvl: Any = None
    ) -> Self:
        return super().zeros(shape, intvl=FloatInterval)

    @classmethod
    def _emptyarray(cls, shape):
        return np.empty(shape, np.float64)

    def _verry_overload_(self, fun, *args, **kwargs):
        if fun is approx_inv:
            return np.linalg.inv(self.mid())

        if fun is approx_qr:
            return tuple(np.linalg.qr(self.mid()))

        if fun is approx_solve:
            a = args[0].mid() if isinstance(args[0], IntervalMatrix) else args[0]
            b = args[1].mid() if isinstance(args[1], IntervalMatrix) else args[1]
            return np.linalg.solve(a, b)

        return super()._verry_overload_(fun, *args, **kwargs)
