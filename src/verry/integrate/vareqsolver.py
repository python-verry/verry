from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

from verry import function as vrf
from verry.autodiff.autodiff import jacobian
from verry.integrate import _impl
from verry.integrate.integrator import IntegratorFactory
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries
from verry.linalg.intervalmatrix import IntervalMatrix
from verry.typing import ComparableScalar


class VarEqSolver(ABC):
    """Abstract base class for solvers of variational equations.

    This class is usually not instantiated directly, but is created by
    :class:`VarEqSolverFactory`.

    Attributes
    ----------
    t : Interval
        Current time.
    jac : IntervalMatrix
        The value that the solution of variational equations takes at `t`.

    Warnings
    --------
    The values of `t` and `jac` are undefined if :meth:`solve` has not been invoked or
    if the previous execution resulted in failure.
    """

    __slots__ = ()
    t: Interval
    jac: IntervalMatrix

    @abstractmethod
    def solve[T: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T],
        t1: Interval[T],
        series: Sequence[IntervalSeries[T]],
    ) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        r"""Solve variational equations.

        Parameters
        ----------
        fun : Callable
            Right-hand side of the system.
        t0 : Interval
            Current time.
        t1 : Interval
            Next time.
        series : Sequence[IntervalSeries]
            Current interval series.

        Returns
        -------
        r0 : bool
            `r0` is ``True`` if and only if `t` and `jac` are successfully updated.
        r1 : str | None
            `r1` is ``None`` If `r0` is ``True``; otherwise, `r1` describes the reason
            for failure.

        Notes
        -----
        The updated `t` might not be equal to `t1` even if `r0` is ``True``.
        The only assumption that can be made is that `t` is included in ``t0 | t1``.
        """
        raise NotImplementedError


class VarEqSolverFactory(ABC):
    """Abstract factory for creating :class:`VarEqSolver`."""

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
    """Factory for creating :class:`VarEqSolver` that solves variational equations
    directly."""

    __slots__ = ()

    def create(self, integrator, intvlmat):
        return _BruteVarEqSolver(integrator, intvlmat)


class _BruteVarEqSolver(VarEqSolver):
    __slots__ = ("t", "jac", "_integrator", "_intvlmat")
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
    ) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        n = len(series)
        eye = self._intvlmat.eye(n)
        vareq = _impl.variationaleq(fun, t0, series)
        varitors = [self._integrator.create(vareq, t0, x, t1) for x in eye]

        for x in varitors:
            if not (res := x.step())[0]:
                return (False, res[1])

        if any(x.status == "RUNNING" for x in varitors):
            t1 = self._intvlmat.interval(min(x.t.sup for x in varitors))

        jac = self._intvlmat.empty((n, n))

        for j in range(n):
            for i in range(n):
                jac[i, j] = varitors[j].series[i].eval(t1 - t0)

        self.t = t1
        self.jac = jac
        return (True, None)


class lognorm(VarEqSolverFactory):
    """Factory for creating :class:`VarEqSolver` that uses a logarithmic norm."""

    def create(self, integrator, intvlmat):
        return _LogNormVarEqSolver(intvlmat)


class _LogNormVarEqSolver(VarEqSolver):
    __slots__ = ("t", "jac", "_intvlmat")
    _intvlmat: type[IntervalMatrix]

    def __init__(self, intvlmat: type[IntervalMatrix]):
        self._intvlmat = intvlmat

    def solve[T: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T],
        t1: Interval[T],
        series: Sequence[IntervalSeries[T]],
    ) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        def dfun(t, *y):
            return jacobian(lambda *y: fun(t, *y))(*y)

        intvl = self._intvlmat.interval
        dom = (t1 - t0) | 0
        deg = len(series[0].coeffs) - 1
        n = len(series)

        t = IntervalSeries(dom, (t0, intvl(1), *[intvl()] * (deg - 1)))
        tmp0 = dfun(t, *series)
        v = self._intvlmat.empty((n, n))

        for i in range(n):
            for j in range(n):
                v[i, j] = tmp0[i][j].eval(dom)

        mu = v[0, 0] + sum(abs(v[0, j]) for j in range(1, n))
        mu = intvl(mu.sup)

        for i in range(1, n):
            tmp1 = v[i, i] + sum(abs(v[i, j]) for j in range(n) if i != j)

            if tmp1.sup > mu.sup:
                mu = intvl(tmp1.sup)

        u0 = [vrf.exp(mu * dom) * intvl(-1, 1) for _ in range(n)]
        vareq = _impl.variationaleq(fun, t0, series)
        jac = self._intvlmat.empty((n, n))

        for j in range(n):
            v0 = tuple(intvl(1 if j == j else 0) for j in range(n))
            tmp2 = _impl.seriessol(vareq, t0, v0, series[0].order - 1)
            tmp3 = _impl.seriessol(vareq, t0, u0, series[0].order)
            tmp2 = [IntervalSeries(dom, x.coeffs) for x in tmp2]
            tmp3 = [IntervalSeries(dom, x.coeffs) for x in tmp3]

            for i in range(n):
                tmp2[i].coeffs.append(tmp3[i].coeffs[-1])
                jac[i, j] = tmp2[i].eval(t1 - t0)

        self.t = t1
        self.jac = jac
        return (True, None)
