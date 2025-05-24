from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal

from verry import function as vrf
from verry.integrate.utility import seriessolution
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries, localcontext
from verry.linalg.intervalmatrix import IntervalMatrix


class Integrator[T: Interval](ABC):
    r"""Abstract base class for ODE integrators.

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
    t: T
    t_bound: T
    t_prev: T | None
    y: tuple[T, ...]

    @abstractmethod
    def __init__(
        self,
        fun: Callable,
        t0: T,
        y0: IntervalMatrix[T] | Sequence[T],
        t_bound: T,
    ):
        raise NotImplementedError

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
    def update(self, t: T, y: IntervalMatrix[T] | Sequence[T]) -> None:
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


def eilo[T: Interval = Any](
    order: int = 15,
    rtol: Any = 1e-10,
    atol: Any = 1e-10,
    max_tries: int = 5,
    min_step: Any = 0.0,
    max_step: Any = float("inf"),
) -> type[Integrator[T]]:
    r"""Integrator based on Eijgenraam and Lohner's algorithm.

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

    Returns
    -------
    type[Integrator]

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

    if order < 3:
        raise ValueError

    if max_tries < 1:
        raise ValueError

    class Integrator(
        _eilo,
        order=order,
        rtol=rtol,
        atol=atol,
        max_tries=max_tries,
        min_step=min_step,
        max_step=max_step,
    ):
        pass

    return Integrator


class _eilo[T: Interval](Integrator[T], ABC):
    order: ClassVar[int]
    _max_tries: ClassVar[int]
    _cls_rtol: ClassVar[Any]
    _cls_atol: ClassVar[Any]
    _cls_min_step: ClassVar[Any]
    _cls_max_step: ClassVar[Any]

    status: Literal["FAILURE", "RUNNING", "SUCCESS"]
    t: T
    y: tuple[T, ...]
    t_bound: T
    t_prev: T | None
    series: tuple[IntervalSeries[T], ...] | None
    _fun: Callable
    _rtol: Any
    _atol: Any
    _min_step: Any
    _max_step: Any

    def __init__(
        self, fun: Callable, t0: T, y0: IntervalMatrix[T] | Sequence[T], t_bound: T
    ):
        rtol = self._cls_rtol
        atol = self._cls_atol
        min_step = self._cls_min_step
        max_step = self._cls_max_step

        if not isinstance(t0, Interval):
            raise TypeError

        if t0.sup >= t_bound.inf:
            raise ValueError

        if isinstance(rtol, float):
            rtol = t0.converter.fromfloat(rtol, strict=False)

        if isinstance(atol, float):
            atol = t0.converter.fromfloat(atol, strict=False)

        if isinstance(min_step, float):
            min_step = t0.converter.fromfloat(min_step)

        if isinstance(max_step, float):
            max_step = t0.converter.fromfloat(max_step)

        if not t0.operator.ZERO <= min_step <= max_step:
            raise ValueError

        self.status = "RUNNING"
        self.t = t0
        self.y = tuple(y0)
        self.t_bound = t_bound
        self.t_prev = None
        self.series = None
        self._fun = fun
        self._rtol = rtol
        self._atol = atol
        self._min_step = min_step
        self._max_step = max_step

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

    def update(self, t: T, y: IntervalMatrix[T] | Sequence[T]) -> None:
        if self.status == "FAILURE":
            raise RuntimeError

        if t != self.t_bound:
            self.status = "RUNNING"

        self.t = t.copy()
        self.y = tuple(x.copy() for x in y)

    def __init_subclass__(
        cls,
        /,
        order: int,
        rtol: Any,
        atol: Any,
        max_tries: int,
        min_step: Any,
        max_step: Any,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        if order < 3:
            raise ValueError

        if max_tries < 1:
            raise ValueError

        cls.order = order
        cls._max_tries = max_tries
        cls._cls_rtol = rtol
        cls._cls_atol = atol
        cls._cls_min_step = min_step
        cls._cls_max_step = max_step


def kashi[T: Interval = Any](
    order: int = 15,
    rtol: Any = 1e-10,
    atol: Any = 1e-10,
    max_tries: int = 5,
    min_step: Any = 0.0,
    max_step: Any = float("inf"),
) -> type[Integrator[T]]:
    r"""Integrator based on Kashiwagi's algorithm.

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

    Returns
    -------
    type[Integrator]

    References
    ----------
    .. [#Ka95] M. Kashiwagi, "Power series arithmetic and its application to numerical
        validation," in *Proc. 1995 Symposium on Nonlinear Theory and its Applications
        (NOLTA '95)*, Las Vegas, NV, USA, Dec. 10--14, 1995, pp. 251--254.
    """

    if order < 3:
        raise ValueError

    if max_tries < 1:
        raise ValueError

    class Integrator(
        _kashi,
        order=order,
        rtol=rtol,
        atol=atol,
        max_tries=max_tries,
        min_step=min_step,
        max_step=max_step,
    ):
        pass

    return Integrator


class _kashi[T: Interval](Integrator[T], ABC):
    order: ClassVar[int]
    _max_tries: ClassVar[int]
    _cls_rtol: ClassVar[Any]
    _cls_atol: ClassVar[Any]
    _cls_min_step: ClassVar[Any]
    _cls_max_step: ClassVar[Any]

    status: Literal["FAILURE", "RUNNING", "SUCCESS"]
    t: T
    y: tuple[T, ...]
    t_bound: T
    t_prev: T | None
    series: tuple[IntervalSeries[T], ...] | None
    _fun: Callable
    _rtol: Any
    _atol: Any
    _min_step: Any
    _max_step: Any

    def __init__(
        self, fun: Callable, t0: T, y0: IntervalMatrix[T] | Sequence[T], t_bound: T
    ):
        rtol = self._cls_rtol
        atol = self._cls_atol
        min_step = self._cls_min_step
        max_step = self._cls_max_step

        if not isinstance(t0, Interval):
            raise TypeError

        if t0.sup >= t_bound.inf:
            raise ValueError

        if isinstance(rtol, float):
            rtol = t0.converter.fromfloat(rtol, strict=False)

        if isinstance(atol, float):
            atol = t0.converter.fromfloat(atol, strict=False)

        if isinstance(min_step, float):
            min_step = t0.converter.fromfloat(min_step)

        if isinstance(max_step, float):
            max_step = t0.converter.fromfloat(max_step)

        if not t0.operator.ZERO <= min_step <= max_step:
            raise ValueError

        self.status = "RUNNING"
        self.t = t0
        self.y = tuple(y0)
        self.t_bound = t_bound
        self.t_prev = None
        self.series = None
        self._fun = fun
        self._rtol = rtol
        self._atol = atol
        self._min_step = min_step
        self._max_step = max_step

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

    def update(self, t: T, y: IntervalMatrix[T] | Sequence[T]) -> None:
        if self.status == "FAILURE":
            raise RuntimeError

        if t != self.t_bound:
            self.status = "RUNNING"

        self.t = t.copy()
        self.y = tuple(x.copy() for x in y)

    def __init_subclass__(
        cls,
        /,
        order: int,
        rtol: Any,
        atol: Any,
        max_tries: int,
        min_step: Any,
        max_step: Any,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls.order = order
        cls._max_tries = max_tries
        cls._cls_rtol = rtol
        cls._cls_atol = atol
        cls._cls_min_step = min_step
        cls._cls_max_step = max_step
