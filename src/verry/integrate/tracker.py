from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy.typing as npt

from verry.affine import AffineForm, Context, getcontext, setcontext, summarize
from verry.linalg.intervalmatrix import IntervalMatrix, approx_norm, approx_qr, inv


class Tracker[T: IntervalMatrix](ABC):
    """Abstract base class for trackers.

    Each instance of :class:`Tracker` corresponds to a pair of a point and some closed
    convex set containing that point.

    This class is usually not instantiated directly, but is created by
    :class:`TrackerFactory`.
    """

    __slots__ = ()

    @abstractmethod
    def hull(self) -> T:
        """Return an interval hull of the current set.

        Returns
        -------
        IntervalMatrix, shape (n,)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> T:
        """Return the current point.

        Returns
        -------
        IntervalMatrix, shape (n,)
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, a: T, b: T) -> None:
        r"""Update the current pair.

        Parameters
        ----------
        a : IntervalMatrix, shape (n, n)
            The image of :meth:`hull` mapped by :math:`Df(x)`.
        b : IntervalMatrix, shape (n,)
            The value obtained by mapping :meth:`sample` with :math:`f(x)`.

        Notes
        -----
        This method updates a pair :math:`(\hat{c},C)\to(\hat{c}',C')` with the
        following rule:

        .. math::

            C' \supseteq \{A(x-\hat{c})+b\mid A\in[A],\,b\in[b],\,x\in C\}.

        Note that :math:`C'\supseteq f(C)` holds by the mean value theorem.
        """
        raise NotImplementedError


class TrackerFactory[T: IntervalMatrix](ABC):
    """Abstract factory for creating :class:`Tracker`."""

    __slots__ = ()

    @abstractmethod
    def create(self, x0: T) -> Tracker[T]:
        """Create :class:`Tracker`.

        Parameters
        ----------
        x0 : IntervalMatrix, shape (n,)
            Initial values.

        Returns
        -------
        Tracker
        """
        raise NotImplementedError


class affine[T: IntervalMatrix](TrackerFactory[T]):
    """Factory for creating :class:`Tracker` using affine arithmetic.

    This tracker usually produces the most accurate result.
    """

    __slots__ = ("_n", "_m")
    _n: int | None
    _m: int

    def __init__(self, n: int | None = None, m: int = 0):
        self._n = n
        self._m = m

    def create(self, x0):
        return _AffineTracker(x0, self._n, self._m)


class _AffineTracker[T: IntervalMatrix](Tracker[T]):
    __slots__ = ("_context", "_current", "_matrix", "_n", "_m")
    _context: Context
    _current: list[AffineForm]
    _matrix: type[T]
    _n: int | None
    _m: int

    def __init__(self, x0: T, n: int | None, m: int):
        ctx = getcontext()
        self._context = ctx.copy()
        self._matrix = type(x0)
        self._n = n
        self._m = m
        setcontext(self._context)

        try:
            self._current = [AffineForm(x) for x in x0]
        finally:
            setcontext(ctx)

    def hull(self) -> T:
        return self._matrix([x.range() for x in self._current])

    def sample(self) -> T:
        return self._matrix([x.mid() for x in self._current])

    def update(self, a: T, b: T) -> None:
        ctx = getcontext()
        setcontext(self._context)

        try:
            curr = self._current
            next: list[AffineForm] = []
            n = len(curr)

            for i in range(n):
                curr[i] -= curr[i].mid()

            for i in range(n):
                tmp = AffineForm(b[i])
                tmp += sum(a[i, j] * curr[j] for j in range(n))
                next.append(tmp)

            if self._n is not None:
                summarize(next, self._n, self._m)

            self._current = next
        finally:
            setcontext(ctx)


class direct[T: IntervalMatrix](TrackerFactory[T]):
    """Factory for creating the most obvious tracker."""

    __slots__ = ()

    def create(self, x0):
        return _DirectTracker(x0)


class _DirectTracker[T: IntervalMatrix](Tracker[T]):
    __slots__ = ("_current",)
    _current: T

    def __init__(self, x0: T):
        self._current = x0

    def hull(self) -> T:
        return self._current.copy()

    def sample(self) -> T:
        return self._current.__class__(self._current.mid())

    def update(self, a: T, b: T) -> None:
        curr = self._current
        self._current = a @ (curr - curr.mid()) + b


class qr[T: IntervalMatrix](TrackerFactory[T]):
    """Factory for creating :class:`Tracker` using a QR decomposition.

    This is an implementation of Evaluation 3 in [#Lo87]_, known as a Lohner's QR
    method.

    References
    ----------
    .. [#Lo87] R. J. Lohner, "Enclosing the Solutions of Ordinary Initial and Boundary
        Value Problem," in *Computerarithmetic*, E. Kaucher, U. Kulisch, and Ch.
        Ullrich, Eds. Stuttgart, Germany: B. G. Teubner, 1987, pp. 225--286.
    """

    def create(self, x0):
        return _QRTracker(x0)


class _QRTracker[T: IntervalMatrix](Tracker[T]):
    __slots__ = ("_b", "_m", "_r")
    _b: T
    _m: npt.NDArray
    _r: T

    def __init__(self, x0: T):
        self._b = x0.eye(len(x0))
        self._m = x0.mid()
        self._r = x0 - self._m

    def hull(self) -> T:
        return self._b @ self._r + self._m

    def sample(self) -> T:
        return self._b.__class__(self._m)

    def update(self, a: T, b: T) -> None:
        q0 = self._b
        tmp = a @ q0

        n = len(b)
        s = [(i, approx_norm(tmp[:, i], "two") * self._r[i].diam()) for i in range(n)]
        s.sort(key=lambda x: x[1], reverse=True)
        inf = a.inf[:, [x[0] for x in s]]
        sup = a.sup[:, [x[0] for x in s]]
        tmp = a.__class__(inf, sup) @ q0

        q1 = q0.__class__(approx_qr(tmp)[0])
        p1 = inv(q1, q1.mid().T)
        self._b = q1
        self._m = b.mid()
        self._r = (p1 @ a @ q0) @ self._r + p1 @ (b - self._m)


class doubleton[T: IntervalMatrix](TrackerFactory[T]):
    """Factory for creating :class:`Tracker` using doubleton method.

    This tracker is suitable for large initial intervals.

    This is an implementation of Evaluation 4 in [#Loh87]_, and the name "doubleton" is
    from [#MrZg00]_.

    Parameters
    ----------
    tracker : TrackerFactory | Callable[[], TrackerFactory], optional
        The default is :class:`qrtracker`.

    References
    ----------
    .. [#Loh87] R. J. Lohner, "Enclosing the Solutions of Ordinary Initial and Boundary
        Value Problem," in *Computerarithmetic*, E. Kaucher, U. Kulisch, and Ch.
        Ullrich, Eds. Stuttgart, Germany: B. G. Teubner, 1987, pp. 225--286.
    .. [#MrZg00] M. Mrozek and P. Zgliczy\u0144ski, "Set arithmetic and the enclosing
        problem in dynamics," *Ann. Pol. Math.*, vol. 74, pp. 237--259, 2000,
        :doi:`10.4064/ap-74-1-237-259`.
    """

    __slots__ = ("_factory",)
    _factory: TrackerFactory[T]

    def __init__(
        self, tracker: TrackerFactory[T] | Callable[[], TrackerFactory[T]] | None = None
    ):
        if tracker is None:
            self._factory = qr()
        elif isinstance(tracker, TrackerFactory):
            self._factory = tracker
        else:
            self._factory = tracker()

    def create(self, x0):
        return _DoubletonTracker(x0, self._factory)


class _DoubletonTracker[T: IntervalMatrix](Tracker[T]):
    __slots__ = ("_c", "_m", "_r0", "_tracker")
    _c: T
    _m: npt.NDArray
    _r0: T
    _tracker: Tracker[T]

    def __init__(self, x0: T, factory: TrackerFactory[T]):
        self._c = x0.eye(len(x0))
        self._m = x0.mid()
        self._r0 = x0 - self._m
        self._tracker = factory.create(x0.zeros_like())

    def hull(self) -> T:
        return self._m + self._tracker.hull() + self._c @ self._r0

    def sample(self):
        return self._c.__class__(self._m)

    def update(self, a: T, b: T) -> None:
        c0 = self._c
        c1 = a.mid() @ c0.inf
        self._c = c0.__class__(c1)
        self._m = b.mid()
        tmp = b - self._m + (a @ c0 - c1) @ self._r0 + a @ self._tracker.sample()
        self._tracker.update(a, tmp)
