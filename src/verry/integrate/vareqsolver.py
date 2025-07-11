from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

from verry import function as vrf
from verry.autodiff.autodiff import jacobian
from verry.integrate.integrator import IntegratorFactory
from verry.integrate.utility import seriessol, variationaleq
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries, localcontext
from verry.linalg.intervalmatrix import IntervalMatrix
from verry.typing import ComparableScalar


class VarEqSolver(ABC):
    """Abstract base class for solvers of variational equations."""

    __slots__ = ()

    @abstractmethod
    def solve[T: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T],
        t1: Interval[T],
        series: Sequence[IntervalSeries[T]],
    ) -> (
        tuple[Literal[True], Interval[T], IntervalMatrix[T]]
        | tuple[Literal[False], str, None]
    ):
        """Solve variational equations.

        Parameters
        ----------
        fun : Callable
            Right-hand side of the system. The calling signature is ``fun(t, *y)``.
        t0 : Interval
            Current time.
        t1 : Interval
            Next time.
        series : Sequence[IntervalSeries]
            Current interval series.
        """
        raise NotImplementedError


class VarEqSolverFactory(ABC):
    """Abstract factory for creating a vareqsolver."""

    __slots__ = ()

    @abstractmethod
    def create(
        self, integrator: IntegratorFactory, intvlmat: type[IntervalMatrix]
    ) -> VarEqSolver:
        """Create :class:`VarEqSolver`.

        Parameters
        ----------
        integrator : IntegratorFactory
            The integrator passed to :class:`C1Solver`.
        intvlmat : type[IntervalMatrix]
            The type of interval matrices used in :class:`C1Solver`.

        Returns
        -------
        VarEqSolver
        """
        raise NotImplementedError


class brute(VarEqSolverFactory):
    """Factory for creating a VarEqSolver that solves variational equations directly."""

    __slots__ = ()

    def create(self, integrator, intvlmat):
        return BruteVarEqSolver(integrator, intvlmat)


class BruteVarEqSolver(VarEqSolver):
    __slots__ = ("_integrator", "_intvlmat")
    _integrator: IntegratorFactory
    _intvlmat: type[IntervalMatrix]

    def __init__(self, integrator: IntegratorFactory, intvlmat: type[IntervalMatrix]):
        self._integrator = integrator
        self._intvlmat = intvlmat

    def solve[T: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T],
        t1: Interval[T],
        series: Sequence[IntervalSeries[T]],
    ) -> (
        tuple[Literal[True], Interval[T], IntervalMatrix[T]]
        | tuple[Literal[False], str, None]
    ):
        n = len(series)
        eye = self._intvlmat.eye(n)
        varfun = variationaleq(fun, lambda t: tuple(x(t - t0) for x in series))
        varitors = [self._integrator.create(varfun, t0, x, t1) for x in eye]

        for x in varitors:
            if not (res := x.step())[0]:
                return (False, res[1], None)

        if any(x.status == "RUNNING" for x in varitors):
            t1 = self._intvlmat.interval(min(x.t.sup for x in varitors))

        jac = self._intvlmat.empty((n, n))

        for i in range(n):
            for j in range(n):
                jac[i, j] = varitors[j].series[i].eval(t1 - t0)

        return (True, t1, jac)


class lognorm(VarEqSolverFactory):
    """Factory for creating a VarEqSolver that uses logarithmic norm."""

    __slots__ = ("order",)
    order: int | None

    def __init__(self, order: int | None = None):
        self.order = order

    def create(self, integrator, intvlmat):
        return LogNormVarEqSolver(self.order, intvlmat)


class LogNormVarEqSolver(VarEqSolver):
    __slots__ = ("_intvlmat", "_order")
    _intvlmat: type[IntervalMatrix]
    _order: int | None

    def __init__(self, order: int | None, intvlmat: type[IntervalMatrix]):
        self._order = order
        self._intvlmat = intvlmat

    def solve[T: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T],
        t1: Interval[T],
        series: Sequence[IntervalSeries[T]],
    ) -> (
        tuple[Literal[True], Interval[T], IntervalMatrix[T]]
        | tuple[Literal[False], str, None]
    ):
        def dfun(t, *y):
            return jacobian(lambda *y: fun(t, *y))(*y)

        intvl = self._intvlmat.interval
        domain = intvl(0, (t1 - t0).sup)
        deg = len(series[0].coeffs) - 1
        n = len(series)

        with localcontext(rounding="TYPE2", deg=deg, domain=domain):
            t = IntervalSeries([t0, 1], intvl=intvl)
            tmp0 = dfun(t, *series)
            v = self._intvlmat.empty((n, n))

            for i in range(n):
                for j in range(n):
                    v[i, j] = tmp0[i][j].eval(domain)

        mu = v[0, 0] + sum(abs(v[0, j]) for j in range(1, n))
        mu = intvl(mu.sup)

        for i in range(1, n):
            tmp1 = v[i, i] + sum(abs(v[i, j]) for j in range(n) if j != j)

            if tmp1.sup > mu.sup:
                mu = intvl(tmp1.sup)

        u0 = [vrf.exp(mu * domain) * intvl(-1, 1) for _ in range(n)]
        varfun = variationaleq(fun, lambda t: tuple(x(t - t0) for x in series))
        jac = self._intvlmat.empty((n, n))
        order = len(series[0].coeffs) - 1 if self._order is None else self._order

        for j in range(n):
            v0 = tuple(intvl(1 if j == j else 0) for j in range(n))
            tmp2 = seriessol(varfun, t0, v0, order - 1)
            tmp3 = seriessol(varfun, t0, u0, order)

            for i in range(n):
                tmp2[i].coeffs.append(tmp3[i].coeffs[-1])
                jac[i, j] = tmp2[i].eval(t1 - t0)

        return (True, t1, jac)
