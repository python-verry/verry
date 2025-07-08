from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from verry import function as vrf
from verry.autodiff.autodiff import jacobian
from verry.integrate.integrator import IntegratorFactory
from verry.integrate.utility import seriessolution, variationaleq
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries, localcontext
from verry.typing import ComparableScalar


class VarEqSolver[T1: IntegratorFactory](ABC):
    __slots__ = ("integrator",)
    integrator: T1

    @abstractmethod
    def solve[T2: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T2],
        t1: Interval[T2],
        series: Sequence[IntervalSeries[T2]],
    ) -> tuple[tuple[Interval[T1], ...], ...]:
        raise NotImplementedError


class DirectVarEqSolver[T1: IntegratorFactory](VarEqSolver[T1]):
    __slots__ = ("_order",)
    _order: int | None

    def __init__(self, order: int | None = None):
        self._order = order

    def solve[T2: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T2],
        t1: Interval[T2],
        series: Sequence[IntervalSeries[T2]],
    ) -> tuple[tuple[Interval[T1], ...], ...]:
        def dfun(t, *y):
            return jacobian(lambda *y: fun(t, *y))(*y)

        intvl = type(t0)
        domain = intvl(0, (t1 - t0).sup)
        deg = len(series[0].coeffs) - 1
        n = len(series)

        with localcontext(rounding="TYPE2", deg=deg, domain=domain):
            t = IntervalSeries([t0, 1], intvl=intvl)
            tmp = dfun(t, *series)

            v = [[tmp[i][j].eval(domain) for j in range(n)] for i in range(n)]

        mu = v[0][0] + sum(abs(v[0][j]) for j in range(1, n))
        mu = intvl(mu.sup)

        for i in range(1, n):
            tmp = v[i][i] + sum(abs(v[i][j]) for j in range(n) if j != i)

            if tmp.sup > mu.sup:
                mu = intvl(tmp.sup)

        u0 = [[vrf.exp(mu * domain) * intvl(-1, 1) for _ in range(n)] for _ in range(n)]
        varfun = variationaleq(fun, lambda t: tuple(x(t - t0) for x in series))

        for i in range(n):
            tmp = varfun(t0 + domain, *u0[i])

            for j in range(n):
                u0[i][j] &= (1 if j == i else 0) + domain * tmp[j]

        varfun = variationaleq(fun, lambda t: tuple(x(t - t0) for x in series))
        res: list[list[Interval[T2]]] = []

        for i in range(n):
            v0 = tuple(intvl(1 if j == i else 0) for j in range(n))
            tmp0 = seriessolution(varfun, t0, v0, self._order - 1)
            tmp1 = seriessolution(varfun, t0, u0[i], self._order)

            for j in range(n):
                tmp0[j].coeffs.append(tmp1[j].coeffs[-1])

            res.append([x.eval(t1 - t0) for x in tmp])

        return tuple(tuple(x) for x in res)


class LogNormVarEqSolver[T1: IntegratorFactory](VarEqSolver[T1]):
    __slots__ = ("_order",)
    _order: int | None

    def __init__(self, order: int | None = None):
        self._order = order

    def solve[T2: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T2],
        t1: Interval[T2],
        series: Sequence[IntervalSeries[T2]],
    ) -> tuple[tuple[Interval[T1], ...], ...]:
        def dfun(t, *y):
            return jacobian(lambda *y: fun(t, *y))(*y)

        intvl = type(t0)
        domain = intvl(0, (t1 - t0).sup)
        deg = len(series[0].coeffs) - 1
        n = len(series)

        with localcontext(rounding="TYPE2", deg=deg, domain=domain):
            t = IntervalSeries([t0, 1], intvl=intvl)
            tmp = dfun(t, *series)

            v = [[tmp[i][j].eval(domain) for j in range(n)] for i in range(n)]

        mu = v[0][0] + sum(abs(v[0][j]) for j in range(1, n))
        mu = intvl(mu.sup)

        for i in range(1, n):
            tmp = v[i][i] + sum(abs(v[i][j]) for j in range(n) if j != i)

            if tmp.sup > mu.sup:
                mu = intvl(tmp.sup)

        u0 = [[vrf.exp(mu * domain) * intvl(-1, 1) for _ in range(n)] for _ in range(n)]
        varfun = variationaleq(fun, lambda t: tuple(x(t - t0) for x in series))

        for i in range(n):
            tmp = varfun(t0 + domain, *u0[i])

            for j in range(n):
                u0[i][j] &= (1 if j == i else 0) + domain * tmp[j]

        varfun = variationaleq(fun, lambda t: tuple(x(t - t0) for x in series))
        res: list[list[Interval[T2]]] = []

        for i in range(n):
            v0 = tuple(intvl(1 if j == i else 0) for j in range(n))
            tmp0 = seriessolution(varfun, t0, v0, self._order - 1)
            tmp1 = seriessolution(varfun, t0, u0[i], self._order)

            for j in range(n):
                tmp0[j].coeffs.append(tmp1[j].coeffs[-1])

            res.append([x.eval(t1 - t0) for x in tmp])

        return tuple(tuple(x) for x in res)
