from collections.abc import Callable, Iterable, Sequence

from verry.autodiff.autodiff import jacobian
from verry.autodiff.dual import IntervalJet, Jet
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries
from verry.linalg.intervalmatrix import IntervalMatrix
from verry.typing import ComparableScalar


def seriessol[T: ComparableScalar](
    fun: Callable, t0: T, y0: IntervalMatrix[T] | Sequence[Interval[T]], order: int
) -> tuple[IntervalJet[T], ...]:
    intvl = type(t0)
    t = IntervalJet([t0])
    series = tuple(IntervalJet([x]) for x in y0)

    for k in range(1, order + 1):
        dydt = fun(t, *series)

        for i in range(len(series)):
            series[i].coeffs.append(dydt[i].coeffs[k - 1] / k)

        t.coeffs.append(intvl(1 if k == 1 else 0))

    return series


class variationaleq[T: ComparableScalar]:
    __slots__ = ("_fun", "_t", "_sol", "_jet")
    _fun: Callable
    _t: Interval[T]
    _sol: tuple[IntervalSeries[T]]
    _jet: tuple[IntervalJet[T]]

    def __init__(self, fun: Callable, t: Interval[T], sol: Iterable[IntervalSeries[T]]):
        self._fun = fun
        self._t = t
        self._sol = tuple(sol)
        self._jet = tuple(IntervalJet(x.coeffs[:-1]) for x in self._sol)

    def __call__(self, t, *v):
        n = len(self._sol)
        t0 = self._t
        dfun = jacobian(lambda *y: self._fun(t, *y))

        if isinstance(t, Jet):
            jac = dfun(*(x(t - t0) for x in self._jet))
        else:
            jac = dfun(*(x(t - t0) for x in self._sol))

        return tuple(sum(jac[i][j] * v[j] for j in range(n)) for i in range(n))
