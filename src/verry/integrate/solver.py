import dataclasses
import itertools
from collections.abc import Callable, Sequence
from typing import Literal

from verry.integrate import _impl
from verry.integrate.integrator import IntegratorFactory, kashi
from verry.integrate.tracker import TrackerFactory, doubleton
from verry.integrate.vareqsolver import VarEqSolverFactory, lognorm
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries
from verry.linalg.intervalmatrix import IntervalMatrix, _resolve_intervalmatrix
from verry.typing import ComparableScalar


class AbortSolving(Exception):
    """Raised by a callback function to abort solvers.

    Parameters
    ----------
    message : str, default="aborted"
    """

    message: str

    def __init__(self, message="aborted", *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.message = message


@dataclasses.dataclass(frozen=True, slots=True)
class SolverResult[T1, T2: Literal["ABORTED", "FAILURE", "SUCCESS"]]:
    """Output of ODE solvers.

    Attributes
    ----------
    status : Literal["ABORTED", "FAILURE", "SUCCESS"]
    content
    message : str
        Report from the solver. Typically a reason for a failure.
    """

    status: T2
    content: T1
    message: str


class ODESolution[T: ComparableScalar]:
    """Solution of ODEs.

    Attributes
    ----------
    ts : list, length n+1
        Segment breakpoints.
    series : list[tuple[IntervalSeries, ...]], length n
        Sequence of solutions for each segment.
    """

    __slots__ = ("ts", "series")
    ts: list[T]
    series: list[tuple[IntervalSeries[T], ...]]

    def __init__(
        self, ts: Sequence[T], series: Sequence[tuple[IntervalSeries[T], ...]]
    ):
        self.ts = list(ts)
        self.series = list(series)

    def __call__(self, arg: Interval[T] | T | float | int) -> tuple[Interval[T], ...]:
        """Return the value that the solution takes in `arg`.

        Returns
        -------
        tuple[Interval, ...]
        """
        if isinstance(arg, str):
            raise TypeError

        intvl = self.series[0][0].interval
        arg = intvl.ensure(arg)
        result: list[Interval[T]] | None = None

        for ts, series in zip(itertools.pairwise(self.ts), self.series):
            if (domain := intvl(ts[0], ts[1])).isdisjoint(arg):
                continue

            if result is None:
                result = [x.eval((arg & domain) - ts[0]) for x in series]
            else:
                for i, x in enumerate(series):
                    result[i] |= x.eval((arg & domain) - ts[0])

        if result is None:
            raise ValueError

        return tuple(result)


@dataclasses.dataclass(frozen=True, slots=True)
class C0SolverResultContent[T: ComparableScalar]:
    """Information obtained when :class:`C0Solver` successfully worked.

    Attributes
    ----------
    t : Interval
        Boundary time passed to :meth:`C0Solver.solve`.
    y : tuple[Interval, ...], length n
        The value of ``sol(t)``.
    sol : ODESolution
        Solution of ODEs.
    """

    t: Interval[T]
    y: tuple[Interval[T], ...]
    sol: ODESolution[T]


@dataclasses.dataclass(frozen=True, slots=True)
class C0SolverCallbackArg[T: ComparableScalar]:
    """Argument of callback functions passed to :meth:`C0Solver.solve`.

    Attributes
    ----------
    t : Interval
        Current time.
    y : tuple[Interval, ...]
        Current state.
    t_prev : Interval
        Previous time.
    series : tuple[IntervalSeries, ...]
        Current interval series.
    """

    t: Interval[T]
    y: tuple[Interval[T], ...]
    t_prev: Interval[T]
    series: tuple[IntervalSeries[T], ...]


class C0Solver:
    """ODE solver.

    Parameters
    ----------
    integrator : IntegratorFactory | Callable[[], IntegratorFactory], optional
        The default is :class:`kashi`.
    tracker : TrackerFactory | Callable[[], TrackerFactory], optional
        The default is :class:`doubleton`.
    """

    __slots__ = ("integrator", "tracker")
    integrator: IntegratorFactory
    tracker: TrackerFactory

    def __init__(
        self,
        integrator: IntegratorFactory | Callable[[], IntegratorFactory] | None = None,
        tracker: TrackerFactory | Callable[[], TrackerFactory] | None = None,
    ):
        if integrator is None:
            self.integrator = kashi()
        elif isinstance(integrator, IntegratorFactory):
            self.integrator = integrator
        else:
            self.integrator = integrator()

        if tracker is None:
            self.tracker = doubleton()
        elif isinstance(tracker, TrackerFactory):
            self.tracker = tracker
        else:
            self.tracker = tracker()

    def solve[T: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T],
        y0: IntervalMatrix[T] | Sequence[Interval[T]],
        t_bound: Interval[T],
        callback: Callable[[C0SolverCallbackArg[T]], None] | None = None,
    ) -> (
        SolverResult[None, Literal["ABORTED"]]
        | SolverResult[None, Literal["FAILURE"]]
        | SolverResult[C0SolverResultContent[T], Literal["SUCCESS"]]
    ):
        r"""Solve an initial value problem for a system of ODEs.

        Parameters
        ----------
        fun : Callable
            Right-hand side of the system. The calling signature is ``fun(t, *y)``.
        t0 : Interval
            Initial time.
        y0 : IntervalMatrix | Sequence[Interval]
            Initial state.
        t_bound : Interval
            Boundary time. Its infimum must be greater than the supremum of `t0`.
        callback : Callable[[C0SolverCallbackArg], None], optional
            Callback function called at each step. You can abort the solver by raising
            :class:`AbortSolving`.

        Returns
        -------
        r : SolverResult
            If ``r.status`` is ``"SUCCESS"``, ``r.content`` is
            :class:`C0SolverResultContent`; otherwise, ``r.content`` is ``None``.

        Warnings
        --------
        `fun` must be a :math:`C^\infty`-function on some open interval that
        contains `t0` and `t_bound`. Furthermore, `fun` must neither be a constant nor
        contain conditional branches (cf. :doc:`/userguide/pitfall`).

        Examples
        --------
        This example computes :math:`\cos(x)` and :math:`\sin(x)`.

        >>> from verry import FloatInterval as FI
        >>> from verry import function as vrf
        >>> PI = vrf.pi(FI())
        >>> solver = C0Solver()
        >>> r = solver.solve(lambda t, x, y: (y, -x), FI(0), [FI(1), FI(0)], PI)
        >>> print(r.status)
        SUCCESS
        >>> print(r.content.y[0])
        [inf=-1.00001, sup=-0.999999]

        The next example fails due to the blow-up of the solution.

        >>> from verry.integrate import eilo, affine
        >>> solver = C0Solver(eilo(min_step=1e-4), affine)
        >>> r = solver.solve(lambda t, x: (x**2,), FI(0), [FI(1)], FI(2))
        >>> print(r.status)
        FAILURE
        >>> print(r.message)
        failed to determine a step size
        """
        if not isinstance(t0, Interval):
            raise TypeError

        if not isinstance(y0, IntervalMatrix):
            y0 = _resolve_intervalmatrix(type(t0))(y0)

        intvlmat = type(y0)
        eye = intvlmat.eye(len(y0))

        ts = [t0.inf]
        series = []

        tracker = self.tracker.create(y0)
        itor = self.integrator.create(fun, t0, tracker.hull(), t_bound)

        while itor.is_active():
            if not (res := itor.step())[0]:
                return SolverResult("FAILURE", None, res[1])

            u0 = itor.t_prev
            u1 = itor.t
            varfun = _impl.variationaleq(fun, u0, itor.series)
            dom = 0 | (u1 - u0)
            tmp = _impl.seriessol(fun, u0, tracker.sample(), itor.order - 1)
            tmp = [IntervalSeries(dom, x.coeffs) for x in tmp]
            a1 = intvlmat([x.eval(u1 - u0) for x in tmp])
            jac = eye.empty_like()

            for j in range(len(y0)):
                tmp = _impl.seriessol(varfun, u0, list(eye[j]), itor.order - 1)
                tmp = [IntervalSeries(dom, x.coeffs) for x in tmp]
                a1[j] += itor.series[j].coeffs[-1] * (u1 - u0) ** itor.order

                for i in range(len(y0)):
                    jac[i, j] = tmp[i].eval(u1 - u0)

            tracker.update(jac, a1)
            itor.update(u1, y1 := tracker.hull())

            ts.append(u1.inf)
            series.append(itor.series)

            if callback is not None:
                try:
                    callback(C0SolverCallbackArg(u1, y1, u0, itor.series))
                except AbortSolving as exc:
                    return SolverResult("ABORTED", None, exc.message)

        sol = ODESolution(ts, series)
        content = C0SolverResultContent(itor.t, itor.y, sol)
        return SolverResult("SUCCESS", content, "success")


@dataclasses.dataclass(frozen=True, slots=True)
class C1SolverResultContent[T: ComparableScalar]:
    """Information obtained when :class:`C1Solver` successfully worked.

    Attributes
    ----------
    t : Interval
        Boundary time passed to :meth:`C1Solver.solve`.
    y : tuple[Interval, ...], length n
        The value of ``sol(t)``.
    sol : ODESolution
        Solution of ODEs.
    jac : IntervalMatrix, shape (n, n)
        Jacobian matrix of flow with respect to initial values.
    """

    t: Interval[T]
    y: tuple[Interval[T], ...]
    sol: ODESolution[T]
    jac: IntervalMatrix[T]


@dataclasses.dataclass(frozen=True, slots=True)
class C1SolverCallbackArg[T: ComparableScalar]:
    """Argument of callback functions passed to :meth:`C1Solver.solve`.

    Attributes
    ----------
    t : Interval
        Current time.
    y : tuple[Interval, ...]
        Current state.
    t_prev : Interval
        Previous time.
    series : tuple[IntervalSeries, ...]
        Current interval series.
    jac : IntervalMatrix
        Jacobian matrix of flow in the current step.
    totjac : IntervalMatrix
        Jacobian matrix of flow with respect to initial values.
    """

    t: Interval[T]
    y: tuple[Interval[T], ...]
    t_prev: Interval[T]
    series: tuple[IntervalSeries[T], ...]
    jac: IntervalMatrix[T]
    totjac: IntervalMatrix[T]


class C1Solver:
    """ODE solver using variational equations.

    Parameters
    ----------
    integrator : IntegratorFactory | Callable[[], IntegratorFactory], optional
        The default is :class:`kashi`.
    tracker : TrackerFactory | Callable[[], TrackerFactory], optional
        The default is :class:`doubleton`.
    vareq : VarEqSolverFactory | Callable[[], VarEqSolverFactory], optional
        The default is :class:`lognorm`.

    Notes
    -----
    A class diagram of :class:`C1Solver` is shown below:

    .. mermaid::

        classDiagram
            C1Solver ..> VarEqSolverFactory: use
            C1Solver ..> IntegratorFactory: use
            C1Solver ..> TrackerFactory: use
            IntegratorFactory --> Integrator: create
            TrackerFactory --> Tracker: create
            VarEqSolverFactory --> VarEqSolver: create
            VarEqSolverFactory ..> IntegratorFactory: use

    """

    __slots__ = ("integrator", "tracker", "vareq")
    integrator: IntegratorFactory
    tracker: TrackerFactory
    vareq: VarEqSolverFactory

    def __init__(
        self,
        integrator: IntegratorFactory | Callable[[], IntegratorFactory] | None = None,
        tracker: TrackerFactory | Callable[[], TrackerFactory] | None = None,
        vareq: VarEqSolverFactory | Callable[[], VarEqSolverFactory] | None = None,
    ):
        if integrator is None:
            self.integrator = kashi()
        elif isinstance(integrator, IntegratorFactory):
            self.integrator = integrator
        else:
            self.integrator = integrator()

        if tracker is None:
            self.tracker = doubleton()
        elif isinstance(tracker, TrackerFactory):
            self.tracker = tracker
        else:
            self.tracker = tracker()

        if vareq is None:
            self.vareq = lognorm()
        elif isinstance(vareq, VarEqSolverFactory):
            self.vareq = vareq
        else:
            self.vareq = vareq()

    def solve[T: ComparableScalar](
        self,
        fun: Callable,
        t0: Interval[T],
        y0: IntervalMatrix[T] | Sequence[Interval[T]],
        t_bound: Interval[T],
        callback: Callable[[C1SolverCallbackArg[T]], None] | None = None,
    ) -> (
        SolverResult[None, Literal["ABORTED"]]
        | SolverResult[None, Literal["FAILURE"]]
        | SolverResult[C1SolverResultContent[T], Literal["SUCCESS"]]
    ):
        r"""Solve an initial value problem for a system of ODEs.

        Parameters
        ----------
        fun : Callable
            Right-hand side of the system. The calling signature is ``fun(t, *y)``.
        t0 : Interval
            Initial time.
        y0 : IntervalMatrix | Sequence[Interval]
            Initial state.
        t_bound : Interval
            Boundary time. Its infimum must be greater than the supremum of `t0`.
        callback : Callable[[C1SolverCallbackArg], None], optional
            Callback function called at each step. You can abort the solver by raising
            :class:`AbortSolving`.

        Returns
        -------
        r : SolverResult
            If ``r.status`` is ``"SUCCESS"``, ``r.content`` is
            :class:`C1SolverResultContent`; otherwise, ``r.content`` is ``None``.

        Warnings
        --------
        `fun` must be a :math:`C^\infty`-function on some open interval that
        contains `t0` and `t_bound`. Furthermore, `fun` must neither be a constant nor
        contain conditional branches (cf. :doc:`/userguide/pitfall`).

        Examples
        --------
        This example computes :math:`\cos(x)` and :math:`\sin(x)`.

        >>> from verry import FloatInterval as FI
        >>> from verry import function as vrf
        >>> PI = vrf.pi(FI())
        >>> solver = C1Solver()
        >>> r = solver.solve(lambda t, x, y: (y, -x), FI(0), [FI(1), FI(0)], PI)
        >>> print(r.status)
        SUCCESS
        >>> print(r.content.y[0])
        [inf=-1.00001, sup=-0.999999]

        The next example fails due to the blow-up of the solution.

        >>> from verry.integrate import eilo, affine
        >>> solver = C1Solver(eilo(min_step=1e-4), affine)
        >>> r = solver.solve(lambda t, x: (x**2,), FI(0), [FI(1)], FI(2))
        >>> print(r.status)
        FAILURE
        >>> print(r.message)
        failed to determine a step size
        """
        if not isinstance(t0, Interval):
            raise TypeError

        if not isinstance(y0, IntervalMatrix):
            y0 = _resolve_intervalmatrix(type(t0))(y0)

        intvl = type(t0)
        intvlmat = type(y0)
        ts = [t0.inf]
        series = []
        totjac = intvlmat.eye(len(y0))

        tracker = self.tracker.create(y0)
        miditor = self.integrator.create(fun, t0, tracker.sample(), t_bound)
        itor = self.integrator.create(fun, t0, tracker.hull(), t_bound)
        vareq = self.vareq.create(self.integrator, intvlmat)

        while miditor.is_active() and itor.is_active():
            if not (res := miditor.step())[0]:
                return SolverResult("FAILURE", None, res[1])

            if not (res := itor.step())[0]:
                return SolverResult("FAILURE", None, res[1])

            u0 = itor.t_prev
            u1 = t_bound

            if miditor.status == "RUNNING" or itor.status == "RUNNING":
                u1 = intvl(min(itor.t.sup, miditor.t.sup))

            if not (res := vareq.solve(fun, u0, u1, itor.series))[0]:
                return SolverResult("FAILURE", None, res[1])

            if u1 != vareq.t:
                u1 = vareq.t

            a1 = intvlmat([x.eval(u1 - u0) for x in miditor.series])
            jac = vareq.jac
            tracker.update(jac, a1)
            miditor.update(u1, tracker.sample())
            itor.update(u1, y1 := tracker.hull())

            ts.append(u1.inf)
            series.append(itor.series)
            totjac = jac @ totjac

            if callback is not None:
                try:
                    callback(C1SolverCallbackArg(u1, y1, u0, itor.series, jac, totjac))
                except AbortSolving as exc:
                    return SolverResult("ABORTED", None, exc.message)

        sol = ODESolution(ts, series)
        content = C1SolverResultContent(itor.t, itor.y, sol, totjac)
        return SolverResult("SUCCESS", content, "success")
