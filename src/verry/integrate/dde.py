import dataclasses
from collections.abc import Callable
from typing import Literal

from verry.integrate.integrator import AdaptiveStepIntegratorFactory
from verry.integrate.solver import C0Solver, OdeSolution, SolverResult
from verry.integrate.tracker import TrackerFactory
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries
from verry.typing import ComparableScalar


@dataclasses.dataclass(frozen=True, slots=True)
class DC0SolverResultContent[T: ComparableScalar]:
    """Information obtained when :class:`C0Solver` successfully worked.

    Attributes
    ----------
    t : Interval
        Boundary time passed to :meth:`C0Solver.solve`.
    y : tuple[Interval, ...], length n
        The value of ``sol(t)``.
    sol : OdeSolution
        Solution of ODEs.
    """

    t: Interval[T]
    y: tuple[Interval[T], ...]
    sol: OdeSolution[T]


class DC0Solver[T1: AdaptiveStepIntegratorFactory, T2: TrackerFactory]:
    __slots__ = ("_n", "_solver")
    _n: int
    _solver: C0Solver[T1, T2]

    @property
    def integrator(self) -> T1:
        return self._solver.integrator

    @integrator.setter
    def integrator(self, value: T1):
        self._solver.integrator = value

    @property
    def tracker(self) -> T2:
        return self._solver.tracker

    @tracker.setter
    def tracker(self, value: T2):
        self._solver.tracker = value

    def __init__(
        self,
        n: int,
        integrator: T1 | Callable[[], T1] | None = None,
        tracker: T2 | Callable[[], T2] | None = None,
    ):
        self._n = n
        integrator.min_step = None
        integrator.max_step = None
        self._solver = C0Solver(integrator, tracker)

    def solve[T: ComparableScalar](
        self,
        fun: Callable,
        tau: Interval[T],
        t0: Interval[T],
        y0: Callable,
        t_bound: Interval[T],
    ) -> (
        SolverResult[None, Literal["ABORTED"]]
        | SolverResult[None, Literal["FAILURE"]]
        | SolverResult[DC0SolverResultContent[T], Literal["SUCCESS"]]
    ):
        def fun0(s, *y):
            t = t0 + tau * s / self._n
            return fun(t, *y, *y0(t - tau))

        intvl = type(t0)
        integrator = self._solver.integrator
        integrator.min_step = intvl.operator.ONE
        integrator.max_step = intvl.operator.ONE

        res = self._solver.solve(fun0, intvl(), y0(t0), intvl(self._n))

        if res.status != "SUCCESS":
            return res

        ts = [t0]
        sol: list[tuple[IntervalSeries[T], ...]] = []

        for s in res.content.sol.ts[1:]:
            ts.append(t0 + tau * s / self._n)

        for x in res.content.sol.series:
            sol.append(tuple(y(s) for y in x))

        t0 += tau

        while t0 < t_bound:
            i = 0

            def fun1(s, *y):
                t = t0 + (tau * s) / self._n
                return fun(t, *y, *(x(s) for x in res.content.sol.series[i]))

            def callback(*args, **kwargs):
                nonlocal i
                i += 1

            y1 = res.content.y
            res = self._solver.solve(fun1, intvl(), y1, intvl(self._n), callback)

            if res.status != "SUCCESS":
                return res

            t0 += tau
