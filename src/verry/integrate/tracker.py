from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy.typing as npt

from verry.affineform import AffineForm, Context, getcontext, setcontext, summarize
from verry.linalg.intervalmatrix import IntervalMatrix, approx_norm, approx_qr, inv


class Tracker[T: IntervalMatrix](ABC):
    """Abstract base class for trackers.

    Each instance of this class corresponds to a pair of a closed star set and its
    center.

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
        """Return an interval vector containing the center.

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
            The image of :meth:`hull` mapped by :math:`DF(x)`.
        b : IntervalMatrix, shape (n,)
            The value obtained by mapping :meth:`sample` with :math:`F(x)`.

        Notes
        -----
        This method updates a pair :math:`(S,c)\to(S',c')` with the following rule:

        .. math::

            S' \supseteq \{A(x-c)+b\mid\text{$A\in[A]$, $b\in[b]$, and $x\in S$}\}.

        Note that :math:`S'\supseteq F(S)` holds by the mean value theorem.
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
    """Factory for creating the most obvious :class:`Tracker`."""

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

    This is an implementation of Evaluation 3 in [#Loh87]_, known as a Lohner's QR
    algorithm.

    References
    ----------
    .. [#Loh87] R. J. Lohner, "Enclosing the Solutions of Ordinary Initial and Boundary
        Value Problem," in *Computerarithmetic*, E. Kaucher, U. Kulisch, and Ch.
        Ullrich, Eds. Stuttgart, Germany: B. G. Teubner, 1987, pp. 225--286.
    """

    def create(self, x0):
        return _QRTracker(x0)


class _QRTracker[T: IntervalMatrix](Tracker[T]):
    __slots__ = ("_c", "_q", "_r")
    _c: npt.NDArray
    _q: T
    _r: T

    def __init__(self, x0: T):
        self._c = x0.mid()
        self._q = x0.eye(len(x0))
        self._r = x0 - self._c

    def hull(self) -> T:
        return self._c + self._q @ self._r

    def sample(self) -> T:
        return self._q.__class__(self._c)

    def update(self, a: T, b: T) -> None:
        q0 = self._q
        tmp = a @ q0

        n = len(b)
        s = [(i, approx_norm(tmp[:, i], "two") * self._r[i].diam()) for i in range(n)]
        s.sort(key=lambda x: x[1], reverse=True)
        inf = a.inf[:, [x[0] for x in s]]
        sup = a.sup[:, [x[0] for x in s]]
        tmp = a.__class__(inf, sup) @ q0

        q1 = q0.__class__(approx_qr(tmp)[0])
        p1 = inv(q1, r=q1.mid().T)
        self._q = q1
        self._c = b.mid()
        self._r = (p1 @ a @ q0) @ self._r + p1 @ (b - self._c)


class doubleton[T: IntervalMatrix](TrackerFactory[T]):
    """Factory for creating :class:`Tracker` using doubleton method.

    This tracker is suitable for large initial intervals.

    This is an implementation of Evaluation 4 in [#Loh87]_, and the name "doubleton" is
    from [#MrZg00]_.

    Parameters
    ----------
    tracker : TrackerFactory | Callable[[], TrackerFactory], optional
        The default is :class:`qr`.

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
    __slots__ = ("_b", "_c", "_r", "_tracker")
    _b: npt.NDArray
    _c: npt.NDArray
    _r: T
    _tracker: Tracker[T]

    def __init__(self, x0: T, factory: TrackerFactory[T]):
        ZERO = x0.interval.operator.ZERO
        ONE = x0.interval.operator.ONE
        n = len(x0)
        eye = x0._emptyarray((n, n))

        for i in range(n):
            for j in range(n):
                eye[i, j] = ONE if i == j else ZERO

        self._b = eye
        self._c = x0.mid()
        self._r = x0 - self._c
        self._tracker = factory.create(x0.zeros_like())

    def hull(self) -> T:
        return self._c + self._tracker.hull() + self._b @ self._r

    def sample(self) -> T:
        return self._c + self._tracker.sample()

    def update(self, a: T, b: T) -> None:
        b0 = self._b
        b1 = a.mid() @ self._b
        c1 = b.mid()
        self._b = b1
        self._c = c1
        self._tracker.update(a, b - c1 + (a @ b0 - b1) @ self._r)
