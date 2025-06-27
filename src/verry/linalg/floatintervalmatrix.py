import numpy as np

from verry.interval.floatinterval import FloatInterval
from verry.linalg.intervalmatrix import (
    IntervalMatrix,
    approx_inv,
    approx_qr,
    approx_solve,
)


class FloatIntervalMatrix(IntervalMatrix[float]):
    """Double-precision inf-sup type interval matrix."""

    __slots__ = ()
    interval = FloatInterval

    @classmethod
    def _emptyarray(cls, shape):
        return np.empty(shape, np.float64)

    def _verry_overload_(self, fun, *args, **kwargs):
        if fun is approx_inv:
            return np.linalg.inv(self.mid())

        if fun is approx_qr:
            return tuple(np.linalg.qr(self.mid()))

        if fun is approx_solve:
            a = args[0].mid() if isinstance(args[0], type(self)) else args[0]
            b = args[1].mid() if isinstance(args[1], type(self)) else args[1]
            return np.linalg.solve(a, b)

        return super()._verry_overload_(fun, *args, **kwargs)
