from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal

from verry import function as vrf
from verry.integrate.utility import seriessolution
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries, localcontext
from verry.linalg.intervalmatrix import IntervalMatrix
from verry.typing import ComparableScalar


class Integrator[T: ComparableScalar](ABC):
    r"""Abstract base class for ODE integrators.

    This class is usually not instantiated directly, but is created by
    :class:`IntegratorFactory`.

    Attributes
    ----------
    status : Literal["FAILURE", "RUNNING", "SUCCESS"]
        Current status of the integrator.
    t : Interval
        Current time.
    y : tuple[Interval, ...]
        Current state.
    t_bound : Interval
        Boundary time.
    t_prev : Interval | None
        Previous time. ``None`` if no steps were made yet.
    series : tuple[IntervalSeries, ...] | None
        Current interval series. ``None`` if no steps were made yet.
    order : int
        Degree of `series`.

    Warnings
    --------
    `fun` must be a :math:`C^\infty`-function on some open interval that contains `t0`
    and `t_bound`. Furthermore, `fun` must neither be a constant nor contain conditional
    branches (cf. :doc:`/userguide/pitfall`).

    Notes
    -----
    The implementation of :class:`Integrator` must satisfy the following conditions:

    * `t` and `t_prev` are always disjoint.
    * If `status` is ``"RUNNING"``, the graph of the solution intersects the Cartesian
      product of `t` and `y`.
    * If `status` is ``"RUNNING"`` and the last called method is :meth:`step`, the
      infimum and supremum of `t` are equal.
    * If `status` is ``"SUCCESS"``, `t` equals `t_bound`, and `y` contains the image of
      `t_bound` under the solution.

    Here :math:`\phi(t;t_0,y_0)` denotes a time-dependent flow of ODEs. Suppose that
    :math:`(\hat{t}_{\mathrm{prev}},\hat{y}_{\mathrm{prev}})` is an arbitrary element of
    the Cartesian product of :math:`[t_{\mathrm{prev}}]` and
    :math:`[y_{\mathrm{prev}}]`. Let :math:`d` be `order`, and
    :math:`q_{\mathrm{apx}}(s)` be a (:math:`d-1`)-th order Taylor polynomial of
    :math:`\phi(\hat{t}_{\mathrm{prev}}+s;\hat{t}_{\mathrm{prev}},
    \hat{y}_{\mathrm{prev}})`. Also, let :math:`[p(s)]=[p_{\mathrm{apx}}(s)]+[a]s^d` be
    `series`, assuming that the degree of :math:`[p_{\mathrm{apx}}(s)]` is smaller than
    :math:`d`. Then :math:`[p(s)]` must satisfy the following conditions:

    * :math:`q_{\mathrm{apx}}(s)\in[p_{\mathrm{apx}}(s)]` holds coefficient-wise.
    * :math:`\phi(\hat{t}_{\mathrm{prev}}+s;\hat{t}_{\mathrm{prev}},
      \hat{y}_{\mathrm{prev}})-q_{\mathrm{apx}}(s)\in [a]s^d` holds for all
      :math:`s\in(0,t-\hat{t}_{\mathrm{prev}})`.
    """

    order: int
    series: tuple[IntervalSeries[T], ...] | None
    status: Literal["FAILURE", "RUNNING", "SUCCESS"]
    t: Interval[T]
    t_bound: Interval[T]
    t_prev: Interval[T] | None
    y: tuple[Interval[T], ...]

    @abstractmethod
    def step(self) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        """Perform one integration step.

        Returns
        -------
        r0 : bool
            `r0` is ``True`` if and only if `status` is not ``"FAILURE"``.
        r1 : str | None
            `r1` is ``None`` If `r0` is ``True``; otherwise, `r1` describes the reason
            for failure.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self, t: Interval[T], y: IntervalMatrix[T] | Sequence[Interval[T]]
    ) -> None:
        """Receive a state, usually refined by :class:`Tracker`.

        ``self.t`` and ``self.y`` are updated to `t` and `y`.

        Parameters
        ----------
        t : Interval
        y : IntervalMatrix | Sequence[Interval]

        Raises
        ------
        ValueError
            If `status` is ``"FAILURE"``.
        """
        raise NotImplementedError


class IntegratorFactory[T: ComparableScalar](ABC):
    """Abstract factory for creating an integrator."""

    @abstractmethod
    def create(
        self,
        fun: Callable,
        t0: Interval[T],
        y0: IntervalMatrix[T] | Sequence[Interval[T]],
        t_bound: Interval[T],
    ) -> Integrator[T]:
        """Create :class:`Integrator`.

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

        Returns
        -------
        Integrator
        """
        raise NotImplementedError


class eilo[T: ComparableScalar](IntegratorFactory[T]):
    r"""Factory for creating an integrator based on Eijgenraam and Lohner's algorithm.

    See their publications [#Ei81]_\ [#Lo87]_\ [#Lo92]_ for more details on the theory.

    Parameters
    ----------
    order : int, default=15
        Degree of Taylor polynomial. `order` must be greater than or equal to 3.
    rtol : default=1e-10
        Relative tolerance.
    atol : default=1e-10
        Absolute tolerance.
    max_tries : int, default=5
        Maximum number of trials to verify the existence conditions of the solution.
    min_step : default=0.0
        Allowed minimum step size. If adaptive stepping produced a step size that
        is smaller than `min_step`, step size is set to `min_step`.
    max_step : default=inf
        Allowed maximum step size. If adaptive stepping produced a step size that
        is greater than `max_step`, step size is set to `max_step`.

    References
    ----------
    .. [#Ei81] P. Eijgenraam, *The solution of initial value problems using interval
        arithmetic: formulation and analysis of an algorithm*. Amsterdam, Nederland:
        CWI, 1981. [Online]. Available: https://ir.cwi.nl/pub/13004/
    .. [#Lo87] R. J. Lohner, "Enclosing the Solutions of Ordinary Initial and Boundary
        Value Problem," in *Computerarithmetic*, E. Kaucher, U. Kulisch, and Ch.
        Ullrich, Eds. Stuttgart, Germany: B. G. Teubner, 1987, pp. 225--286.
    .. [#Lo92] R. J. Lohner, "Computation of guaranteed enclosures for the solutions of
        ordinary initial and boundary value problems," in *Computational Ordinary
        Differential Equations*, J. R. Cash and I. Gladwell, Eds. Oxford, UK: Clarendon
        Press, 1992, pp. 425--435.
    """

    _order: int
    _rtol: T
    _atol: T | float | int
    _max_tries: int
    _min_step: T | None
    _max_step: T | None

    def __init__(
        self,
        order: int = 15,
        rtol: T | float | int = 1e-10,
        atol: T | float | int = 1e-10,
        max_tries: int = 5,
        min_step: T | None = None,
        max_step: T | None = None,
    ):
        if order < 3:
            raise ValueError

        if max_tries < 1:
            raise ValueError

        self._order = order
        self._rtol = rtol
        self._atol = atol
        self._max_tries = max_tries
        self._min_step = min_step
        self._max_step = max_step

    def create(self, fun, t0, y0, t_bound):
        return _EiLo(
            fun,
            t0,
            y0,
            t_bound,
            self._order,
            self._rtol,
            self._atol,
            self._max_tries,
            self._min_step,
            self._max_step,
        )


class _EiLo[T: ComparableScalar](Integrator[T]):
    status: Literal["FAILURE", "RUNNING", "SUCCESS"]
    t: Interval[T]
    y: tuple[Interval[T], ...]
    t_bound: Interval[T]
    t_prev: Interval[T] | None
    series: tuple[IntervalSeries[T], ...] | None
    order: int
    _fun: Callable
    _rtol: T
    _atol: T
    _min_step: T
    _max_step: T
    _max_tries: int

    def __init__(
        self,
        fun: Callable,
        t0: Interval[T],
        y0: IntervalMatrix[T] | Sequence[Interval[T]],
        t_bound: Interval[T],
        order: int,
        rtol: T | float | int,
        atol: T | float | int,
        max_tries: int,
        min_step: T | None,
        max_step: T | None,
    ):
        if not isinstance(t0, Interval):
            raise TypeError

        if t0.sup >= t_bound.inf:
            raise ValueError

        self.status = "RUNNING"
        self.t = t0
        self.y = tuple(y0)
        self.t_bound = t_bound
        self.t_prev = None
        self.series = None
        self.order = order
        self._fun = fun
        self._max_tries = max_tries

        match rtol:
            case t0.endtype():
                self._rtol = rtol

            case float() | int():
                self._rtol = t0.converter.fromfloat(float(rtol), strict=False)

            case _:
                raise TypeError

        match atol:
            case t0.endtype():
                self._atol = atol

            case float() | int():
                self._atol = t0.converter.fromfloat(float(atol), strict=False)

            case _:
                raise TypeError

        match min_step:
            case t0.endtype():
                self._min_step = min_step

            case None:
                self._min_step = t0.operator.ZERO

            case _:
                raise TypeError

        match max_step:
            case t0.endtype():
                self._max_step = max_step

            case None:
                self._max_step = t0.operator.INFINITY

            case _:
                raise TypeError

        if not t0.operator.ZERO <= self._min_step <= self._max_step:
            raise ValueError

    def step(self) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        ZERO = self.t.operator.ZERO
        intvl = type(self.t)
        cadd = self.t.operator.cadd
        csub = self.t.operator.csub

        p0 = seriessolution(self._fun, self.t, self.y, self.order)
        tmp = ZERO

        for i in (self.order, self.order - 1, self.order - 2):
            tmp = max(tmp, vrf.pow(max(x.coeffs[i].mag() for x in p0), 1 / i))

        if tmp != ZERO:
            tol = self._atol + self._rtol * max(abs(x.mid()) for x in self.y)
            stepsize = vrf.pow(tol, 1 / self.order) / tmp
        elif self.t_prev is not None:
            stepsize = self.t.mid() - self.t_prev.mid()
        else:
            stepsize = csub(self.t_bound.sup, self.t.inf)

        stepsize = min(max(stepsize, self._min_step), self._max_step)

        if stepsize == ZERO:
            self.status = "FAILURE"
            return (False, "failed to determine a step size")

        t_next = intvl(cadd(self.t.sup, stepsize))
        is_verified = False

        for _ in range(self._max_tries):
            if t_next.sup >= self.t_bound.inf:
                self.status = "SUCCESS"
                t_next = self.t_bound

            h = intvl(0, csub(t_next.sup, self.t.inf))
            t1 = self.t + h
            a0 = [y0 + h * dy for y0, dy in zip(self.y, self._fun(t1, *self.y))]

            for __ in range(5):
                a1 = [y0 + h * dy for y0, dy in zip(self.y, self._fun(t1, *a0))]

                if all(y.issubset(x) for x, y in zip(a0, a1)):
                    is_verified = True
                    break

                EPSILON = intvl("0.1")
                a0 = [(1 + EPSILON) * x - EPSILON * x for x in a1]

            if is_verified:
                break

            if stepsize == self._min_step:
                self.status = "FAILURE"
                return (False, "failed to determine a step size")

            self.status = "RUNNING"
            stepsize = max(stepsize / 2, self._min_step)

            if stepsize == ZERO:
                self.status = "FAILURE"
                return (False, "failed to determine a step size")

            t_next = intvl(cadd(self.t.sup, stepsize))

        if not is_verified:
            self.status = "FAILURE"
            return (False, f"failed to verify within {self._max_tries} trials")

        p1 = seriessolution(self._fun, self.t, a1, self.order)

        for i in range(len(p0)):
            p0[i].coeffs[-1] = p1[i].coeffs[-1]

        self.t_prev = self.t
        self.t = t_next
        self.y = tuple(x.eval(self.t - self.t_prev) for x in p0)
        self.series = tuple(p0)
        return (True, None)

    def update(
        self, t: Interval[T], y: IntervalMatrix[T] | Sequence[Interval[T]]
    ) -> None:
        if self.status == "FAILURE":
            raise RuntimeError

        if t != self.t_bound:
            self.status = "RUNNING"

        self.t = t.copy()
        self.y = tuple(x.copy() for x in y)


class kashi[T: ComparableScalar](IntegratorFactory[T]):
    r"""Factory for creating an integrator based on Kashiwagi's algorithm.

    This is an implementation of [#Ka95]_.

    Parameters
    ----------
    order : int, default=15
        Degree of Taylor polynomial. `order` must be greater than or equal to 3.
    rtol : default=1e-10
        Relative tolerance.
    atol : default=1e-10
        Absolute tolerance.
    max_tries : int, default=5
        Maximum number of trials to verify the existence conditions of the solution.
    min_step : default=0.0
        Allowed minimum step size. If adaptive stepping produced a step size that
        is smaller than `min_step`, step size is set to `min_step`.
    max_step : default=inf
        Allowed maximum step size. If adaptive stepping produced a step size that
        is greater than `max_step`, step size is set to `max_step`.

    References
    ----------
    .. [#Ka95] M. Kashiwagi, "Power series arithmetic and its application to numerical
        validation," in *Proc. 1995 Symposium on Nonlinear Theory and its Applications
        (NOLTA '95)*, Las Vegas, NV, USA, Dec. 10--14, 1995, pp. 251--254.
    """

    _order: int
    _rtol: T
    _atol: T | float | int
    _max_tries: int
    _min_step: T | None
    _max_step: T | None

    def __init__(
        self,
        order: int = 15,
        rtol: T | float | int = 1e-10,
        atol: T | float | int = 1e-10,
        max_tries: int = 5,
        min_step: T | None = None,
        max_step: T | None = None,
    ):
        if order < 3:
            raise ValueError

        if max_tries < 1:
            raise ValueError

        self._order = order
        self._rtol = rtol
        self._atol = atol
        self._max_tries = max_tries
        self._min_step = min_step
        self._max_step = max_step

    def create(self, fun, t0, y0, t_bound):
        return _Kashi(
            fun,
            t0,
            y0,
            t_bound,
            self._order,
            self._rtol,
            self._atol,
            self._max_tries,
            self._min_step,
            self._max_step,
        )


class _Kashi[T: ComparableScalar](Integrator[T], ABC):
    status: Literal["FAILURE", "RUNNING", "SUCCESS"]
    t: Interval[T]
    y: tuple[Interval[T], ...]
    t_bound: Interval[T]
    t_prev: Interval[T] | None
    series: tuple[IntervalSeries[T], ...] | None
    order: int
    _fun: Callable
    _rtol: T
    _atol: T
    _min_step: T
    _max_step: T
    _max_tries: int

    def __init__(
        self,
        fun: Callable,
        t0: Interval[T],
        y0: IntervalMatrix[T] | Sequence[Interval[T]],
        t_bound: Interval[T],
        order: int,
        rtol: T | float | int,
        atol: T | float | int,
        max_tries: int,
        min_step: T | None,
        max_step: T | None,
    ):
        if not isinstance(t0, Interval):
            raise TypeError

        if t0.sup >= t_bound.inf:
            raise ValueError

        self.status = "RUNNING"
        self.t = t0
        self.y = tuple(y0)
        self.t_bound = t_bound
        self.t_prev = None
        self.series = None
        self.order = order
        self._fun = fun
        self._max_tries = max_tries

        match rtol:
            case t0.endtype():
                self._rtol = rtol

            case float() | int():
                self._rtol = t0.converter.fromfloat(float(rtol), strict=False)

            case _:
                raise TypeError

        match atol:
            case t0.endtype():
                self._atol = atol

            case float() | int():
                self._atol = t0.converter.fromfloat(float(atol), strict=False)

            case _:
                raise TypeError

        match min_step:
            case t0.endtype():
                self._min_step = min_step

            case None:
                self._min_step = t0.operator.ZERO

            case _:
                raise TypeError

        match max_step:
            case t0.endtype():
                self._max_step = max_step

            case None:
                self._max_step = t0.operator.INFINITY

            case _:
                raise TypeError

        if not t0.operator.ZERO <= self._min_step <= self._max_step:
            raise ValueError

    def step(self) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        ZERO = self.t.operator.ZERO
        intvl = type(self.t)
        cadd = self.t.operator.cadd
        csub = self.t.operator.csub

        p0 = seriessolution(self._fun, self.t, self.y, self.order)
        tmp: Any = ZERO

        for i in (self.order, self.order - 1, self.order - 2):
            tmp = max(tmp, vrf.pow(max(x.coeffs[i].mag() for x in p0), 1 / i))

        if tmp != ZERO:
            tol = self._atol + self._rtol * max(abs(x.mid()) for x in self.y)
            stepsize = vrf.pow(tol, 1 / self.order) / tmp
        elif self.t_prev is not None:
            stepsize = self.t.mid() - self.t_prev.mid()
        else:
            stepsize = csub(self.t_bound.sup, self.t.inf)

        stepsize = min(max(stepsize, self._min_step), self._max_step)

        if stepsize == ZERO:
            self.status = "FAILURE"
            return (False, "failed to determine a step size")

        t_next = intvl(cadd(self.t.sup, stepsize))

        if t_next.sup >= self.t_bound.inf:
            t_next = self.t_bound

        with localcontext() as ctx:
            ctx.rounding = "TYPE2"
            ctx.deg = self.order
            ctx.domain = intvl(0, csub(t_next.sup, self.t.inf))

            t = IntervalSeries([self.t, 1], intvl=intvl)
            p2 = [y0 + dy.integrate() for y0, dy in zip(self.y, self._fun(t, *p0))]

            for i in range(len(p2)):
                ctx.round(p2[i])

            rad = max((x.coeffs[-1] - y.coeffs[-1]).mag() for x, y in zip(p0, p2))
            p1 = [x.copy() for x in p0]

            for i in range(len(p1)):
                p1[i].coeffs[-1] += rad * intvl(-2, 2)

            p2 = [y0 + dy.integrate() for y0, dy in zip(self.y, self._fun(t, *p1))]

            for i in range(len(p2)):
                ctx.round(p2[i])

            tmp = [x.eval(stepsize) for x in p2]
            e0 = self._atol + self._rtol * max(abs(x.mid()) for x in tmp)
            e1 = max(x.diam() for x in tmp)

            if e1 > ZERO:
                stepsize *= vrf.pow(e0 / e1, 1 / self.order)
                stepsize = min(max(stepsize, self._min_step), self._max_step)

                if stepsize == ZERO:
                    self.status = "FAILURE"
                    return (False, "failed to determine a step size")

            t_next = intvl(cadd(self.t.sup, stepsize))

            for _ in range(self._max_tries):
                if t_next.sup >= self.t_bound.inf:
                    self.status = "SUCCESS"
                    t_next = self.t_bound

                ctx.domain = intvl(0, csub(t_next.sup, self.t.inf))
                t = IntervalSeries([self.t, 1], intvl=intvl)
                p2 = [y0 + dy.integrate() for y0, dy in zip(self.y, self._fun(t, *p1))]

                for i in range(len(p2)):
                    ctx.round(p2[i])

                if all(y.coeffs[-1].issubset(x.coeffs[-1]) for x, y in zip(p1, p2)):
                    self.t_prev = self.t
                    self.t = t_next
                    self.y = tuple(x.eval(self.t - self.t_prev) for x in p2)
                    self.series = tuple(p2)
                    return (True, None)

                if stepsize == self._min_step:
                    self.status = "FAILURE"
                    return (False, "failed to determine a step size")

                self.status = "RUNNING"
                stepsize = max(stepsize / 2, self._min_step)

                if stepsize == ZERO:
                    self.status = "FAILURE"
                    return (False, "failed to determine a step size")

                t_next = intvl(cadd(self.t.sup, stepsize))

        self.status = "FAILURE"
        return (False, f"failed to verify within {self._max_tries} trials")

    def update(
        self, t: Interval[T], y: IntervalMatrix[T] | Sequence[Interval[T]]
    ) -> None:
        if self.status == "FAILURE":
            raise RuntimeError

        if t != self.t_bound:
            self.status = "RUNNING"

        self.t = t.copy()
        self.y = tuple(x.copy() for x in y)
