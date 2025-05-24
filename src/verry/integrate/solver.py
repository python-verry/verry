import dataclasses
import inspect
import itertools
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeIs, overload

from verry.integrate.integrator import Integrator, kashi
from verry.integrate.tracker import Tracker, doubletontracker
from verry.integrate.utility import seriessolution, variationaleq
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries
from verry.linalg.intervalmatrix import IntervalMatrix


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


class OdeSolution[T: Interval]:
    """Solution of ODEs.

    Attributes
    ----------
    ts : list, length n+1
        Segment breakpoints.
    series : list[tuple[IntervalSeries, ...]], length n
        Sequence of solutions for each segment.
    """

    __slots__ = ("ts", "series")
    ts: list
    series: list[tuple[IntervalSeries[T], ...]]

    def __init__(self, ts: Sequence, series: Sequence[tuple[IntervalSeries[T], ...]]):
        self.ts = list(ts)
        self.series = list(series)

    def __call__(self, arg: Any) -> tuple[T, ...]:
        """Return the value that the solution takes in `arg`."""
        if isinstance(arg, str):
            raise TypeError

        intvl = self.series[0][0].interval
        arg = intvl.ensure(arg)
        result: list[T] | None = None

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


@dataclasses.dataclass(frozen=True, slots=True)
class C0SolverResultContent[T: Interval]:
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

    t: T
    y: tuple[T, ...]
    sol: OdeSolution[T]


@dataclasses.dataclass(frozen=True, slots=True)
class C0SolverCallbackArg[T: Interval]:
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

    t: T
    y: tuple[T, ...]
    t_prev: T
    series: tuple[IntervalSeries[T], ...]


class C0Solver:
    """ODE solver.

    Parameters
    ----------
    integrator : Callable[[], type[Integrator]] | type[Integrator], optional
        The default is :func:`kashi`.
    tracker : type[Tracker], optional
        The default is :func:`doubletontracker`.

    Notes
    -----
    The choice of `tracker` has a significant impact on accuracy and computation time.
    Our recommendations are as follows:

    1. Use :func:`doubletontracker` at first.
    2. Use :func:`affinetracker` if the solver cannot compute solutions, or the accuracy
       of solutions is not sufficient.
    """

    __slots__ = ("_integrator", "_tracker")
    _integrator: type[Integrator]
    _tracker: type[Tracker]

    def __init__(
        self,
        integrator: Callable[[], type[Integrator]] | type[Integrator] | None = None,
        tracker: Callable[[], type[Tracker]] | type[Tracker] | None = None,
    ):
        if integrator is None:
            integrator = kashi()
        elif not _is_integrator(integrator):
            integrator = integrator()

        if tracker is None:
            tracker = doubletontracker()
        elif not _is_tracker(tracker):
            tracker = tracker()

        if not issubclass(integrator, Integrator):
            raise TypeError

        if not issubclass(tracker, Tracker):
            raise TypeError

        self._integrator = integrator
        self._tracker = tracker

    def solve[T: Interval](
        self,
        fun: Callable,
        t0: T,
        y0: IntervalMatrix[T] | Sequence[T],
        t_bound: T,
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
        [-1.000001, -0.999999]

        The next example fails due to the blow-up of the solution.

        >>> from verry.integrate import eilo, affinetracker
        >>> solver = C0Solver(eilo(min_step=1e-4), affinetracker)
        >>> r = solver.solve(lambda t, x: (x**2,), FI(0), [FI(1)], FI(2))
        >>> print(r.status)
        FAILURE
        >>> print(r.message)
        failed to determine a step size
        """
        if not isinstance(t0, Interval):
            raise TypeError

        if not isinstance(y0, IntervalMatrix):
            y0 = IntervalMatrix(y0, intvl=type(t0))

        intvl = type(t0)
        intvlmat = type(y0)
        eye = intvlmat.eye(len(y0), intvl=intvl)

        ts = [t0.inf]
        series = []

        tracker = self._tracker(y0)
        itor = self._integrator(fun, t0, tracker.hull(), t_bound)

        while itor.status == "RUNNING":
            if not (res := itor.step())[0]:
                return SolverResult("FAILURE", None, res[1])

            u0 = itor.t_prev
            u1 = itor.t

            varfun = variationaleq(fun, lambda t: tuple(x(t - u0) for x in itor.series))  # type: ignore
            tmp = seriessolution(fun, u0, tracker.sample(), itor.order - 1)
            a1 = intvlmat([x.eval(u1 - u0) for x in tmp], intvl=intvl)
            jac = eye.empty_like()

            for i in range(len(y0)):
                tmp = seriessolution(varfun, u0, eye[i], itor.order - 1)
                a1[i] += itor.series[i].coeffs[-1] * (u1 - u0) ** itor.order  # type: ignore
                jac[:, i] = intvlmat([x.eval(u1 - u0) for x in tmp], intvl=intvl)

            tracker.update(jac, a1)
            itor.update(u1, y1 := tracker.hull())

            ts.append(u1.sup)
            series.append(itor.series)

            if callback is not None:
                try:
                    callback(C0SolverCallbackArg(u1, y1, u0, itor.series))  # type: ignore
                except AbortSolving as exc:
                    return SolverResult("ABORTED", None, exc.message)

        sol = OdeSolution(ts, series)  # type: ignore
        content = C0SolverResultContent(itor.t, itor.y, sol)
        return SolverResult("SUCCESS", content, "success")


@dataclasses.dataclass(frozen=True, slots=True)
class C1SolverResultContent[T1: Interval, T2: IntervalMatrix[T1] = IntervalMatrix[T1]]:  # type: ignore
    # TODO: This "type: ignore" must be removed when mypy supports PEP 696.
    #       See https://github.com/python/mypy/issues/14851
    """Information obtained when :class:`C1Solver` successfully worked.

    Attributes
    ----------
    t : Interval
        Boundary time passed to :meth:`C1Solver.solve`.
    y : tuple[Interval, ...], length n
        The value of ``sol(t)``.
    sol : OdeSolution
        Solution of ODEs.
    jac : IntervalMatrix, shape (n, n)
        Jacobian matrix of flow with respect to initial values.
    """

    t: T1
    y: tuple[T1, ...]
    sol: OdeSolution[T1]
    jac: T2


@dataclasses.dataclass(frozen=True, slots=True)
class C1SolverCallbackArg[T1: Interval, T2: IntervalMatrix[T1] = IntervalMatrix[T1]]:  # type: ignore
    # TODO: This "type: ignore" must be removed when mypy supports PEP 696.
    #       See https://github.com/python/mypy/issues/14851
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

    t: T1
    y: tuple[T1, ...]
    t_prev: T1
    series: tuple[IntervalSeries[T1], ...]
    jac: T2
    totjac: T2


class C1Solver:
    """ODE solver using variational equations.

    Parameters
    ----------
    integrator : Callable[[], type[Integrator]] | type[Integrator], optional
        The default is :func:`kashi`.
    tracker : type[Tracker], optional
        The default is :func:`doubletontracker`.

    Notes
    -----
    The choice of `tracker` has a significant impact on accuracy and computation time.
    Our recommendations are as follows:

    1. Use :func:`doubletontracker` at first.
    2. Use :func:`affinetracker` if the solver cannot compute solutions, or the
       accuracy of solutions is not sufficient.
    """

    __slots__ = ("_integrator", "_tracker")
    _integrator: type[Integrator]
    _tracker: type[Tracker]

    def __init__(
        self,
        integrator: Callable[[], type[Integrator]] | type[Integrator] | None = None,
        tracker: Callable[[], type[Tracker]] | type[Tracker] | None = None,
    ):
        if integrator is None:
            integrator = kashi()
        elif not _is_integrator(integrator):
            integrator = integrator()

        if tracker is None:
            tracker = doubletontracker()
        elif not _is_tracker(tracker):
            tracker = tracker()

        if not issubclass(integrator, Integrator):
            raise TypeError

        if not issubclass(tracker, Tracker):
            raise TypeError

        self._integrator = integrator
        self._tracker = tracker

    @overload
    def solve[T: Interval](
        self,
        fun: Callable,
        t0: T,
        y0: Sequence[T],
        t_bound: T,
        callback: Callable[[C1SolverCallbackArg[T]], None] | None = ...,
    ) -> (
        SolverResult[None, Literal["ABORTED"]]
        | SolverResult[None, Literal["FAILURE"]]
        | SolverResult[C1SolverResultContent[T], Literal["SUCCESS"]]
    ): ...

    @overload
    def solve[T1: Interval, T2: IntervalMatrix[T1]](  # type: ignore
        # TODO: This "type: ignore" must be removed when mypy supports PEP 696.
        #       See https://github.com/python/mypy/issues/14851
        self,
        fun: Callable,
        t0: T1,
        y0: T2,
        t_bound: T1,
        callback: Callable[[C1SolverCallbackArg[T1, T2]], None] | None = ...,
    ) -> (
        SolverResult[None, Literal["ABORTED"]]
        | SolverResult[None, Literal["FAILURE"]]
        | SolverResult[C1SolverResultContent[T1, T2], Literal["SUCCESS"]]
    ): ...

    def solve(self, fun, t0, y0, t_bound, callback=None):
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
        [-1.000001, -0.999999]

        The next example fails due to the blow-up of the solution.

        >>> from verry.integrate import eilo, affinetracker
        >>> solver = C1Solver(eilo(min_step=1e-4), affinetracker)
        >>> r = solver.solve(lambda t, x: (x**2,), FI(0), [FI(1)], FI(2))
        >>> print(r.status)
        FAILURE
        >>> print(r.message)
        failed to determine a step size
        """
        if not isinstance(t0, Interval):
            raise TypeError

        if not isinstance(y0, IntervalMatrix):
            y0 = IntervalMatrix(y0, intvl=type(t0))

        intvl = type(t0)
        intvlmat = type(y0)
        eye = intvlmat.eye(len(y0), intvl=intvl)

        ts = [t0.inf]
        series = []
        totjac = intvlmat.eye(len(y0), intvl=intvl)

        tracker = self._tracker(y0)
        miditor = self._integrator(fun, t0, tracker.sample(), t_bound)
        itor = self._integrator(fun, t0, tracker.hull(), t_bound)

        while miditor.status == "RUNNING" and itor.status == "RUNNING":
            if not (res := miditor.step())[0]:
                return SolverResult("FAILURE", None, res[1])

            if not (res := itor.step())[0]:
                return SolverResult("FAILURE", None, res[1])

            u0 = itor.t_prev
            u1 = t_bound

            if miditor.status == "RUNNING" or itor.status == "RUNNING":
                u1 = intvl(min(itor.t.sup, miditor.t.sup))

            varfun = variationaleq(fun, lambda t: tuple(x(t - u0) for x in itor.series))
            varitors = [self._integrator(varfun, u0, x, u1) for x in eye]

            for x in varitors:
                if not (res := x.step())[0]:
                    return SolverResult("FAILURE", None, res[1])

            if any(x.status == "RUNNING" for x in varitors):
                u1 = intvl(min(x.t.sup for x in varitors))

            a1 = intvlmat([x.eval(u1 - u0) for x in miditor.series], intvl=intvl)
            jac = totjac.empty_like()

            for i in range(len(y0)):
                tmp = [x.eval(u1 - u0) for x in varitors[i].series]
                jac[:, i] = intvlmat(tmp, intvl=intvl)

            tracker.update(jac, a1)
            miditor.update(u1, tracker.sample())
            itor.update(u1, y1 := tracker.hull())

            ts.append(u1.sup)
            series.append(itor.series)
            totjac = jac @ totjac

            if callback is not None:
                try:
                    callback(C1SolverCallbackArg(u1, y1, u0, itor.series, jac, totjac))
                except AbortSolving as exc:
                    return SolverResult("ABORTED", None, exc.message)

        sol = OdeSolution(ts, series)
        content = C1SolverResultContent(itor.t, itor.y, sol, totjac)
        return SolverResult("SUCCESS", content, "success")


def _is_integrator(x: object) -> TypeIs[type[Integrator]]:
    return inspect.isclass(x) and issubclass(x, Integrator)


def _is_tracker(x: object) -> TypeIs[type[Tracker]]:
    return inspect.isclass(x) and issubclass(x, Tracker)
