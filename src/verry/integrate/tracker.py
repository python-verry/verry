import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, TypeIs

import numpy.typing as npt

from verry.affine import AffineForm, Context, getcontext, setcontext, summarize
from verry.linalg.intervalmatrix import IntervalMatrix, approx_norm, approx_qr, inv


class Tracker[T: IntervalMatrix](ABC):
    """Abstract base class for trackers.

    Each instance of :class:`Tracker` corresponds to a pair of a point and some closed
    convex set containing that point.

    Parameters
    ----------
    x0 : IntervalMatrix, shape (n,)
        Initial values.
    """

    __slots__ = ()

    @abstractmethod
    def __init__(self, x0: T):
        raise NotImplementedError

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


def affinetracker[T: IntervalMatrix = Any](
    n: int | None = None, m: int = 0
) -> type[Tracker[T]]:
    """Tracker using affine arithmetic, usually producing the most accurate result."""

    class Result(_affinetracker[T], n=n, m=m):
        __slots__ = ()

    return Result


class _affinetracker[T: IntervalMatrix](Tracker[T], ABC):
    __slots__ = ("_context", "_current", "_matrix")
    _cls_n: ClassVar[int | None]
    _cls_m: ClassVar[int]
    _context: Context
    _current: list[AffineForm]
    _matrix: type[T]

    def __init__(self, x0):
        ctx = getcontext()
        self._context = ctx.copy()
        self._matrix = type(x0)
        setcontext(self._context)

        try:
            self._current = [AffineForm(x) for x in x0]
        finally:
            setcontext(ctx)

    def hull(self) -> T:
        intvl = self._current[0].interval
        return self._matrix([x.range() for x in self._current], intvl=intvl)

    def sample(self) -> T:
        intvl = self._current[0].interval
        return self._matrix([x.mid() for x in self._current], intvl=intvl)

    def update(self, a: T, b: T) -> None:
        """Update the current pair.

        Parameters
        ----------
        a : IntervalMatrix, shape (n, n)
            The image of :meth:`hull` mapped by :math:`Df(x)`.
        b : IntervalMatrix, shape (n,)
            The value obtained by mapping :meth:`sample` with :math:`f(x)`.
        """
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

            if self._cls_n is not None:
                summarize(next, self._cls_n, self._cls_m)

            self._current = next
        finally:
            setcontext(ctx)

    def __init_subclass__(cls, /, n: int | None, m: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cls_n = n
        cls._cls_m = m


class directtracker[T: IntervalMatrix](Tracker[T]):
    """The most obvious tracker."""

    __slots__ = ("_current",)
    _current: T

    def __init__(self, x0: T):
        self._current = x0

    def hull(self) -> T:
        return self._current.copy()

    def sample(self) -> T:
        curr = self._current
        return curr.__class__(curr.mid(), intvl=curr.interval)

    def update(self, a: T, b: T) -> None:
        """Update the current pair.

        Parameters
        ----------
        a : IntervalMatrix, shape (n, n)
            The image of :meth:`hull` mapped by :math:`Df(x)`.
        b : IntervalMatrix, shape (n,)
            The value obtained by mapping :meth:`sample` with :math:`f(x)`.
        """
        curr = self._current
        self._current = a @ (curr - curr.mid()) + b


class qrtracker[T: IntervalMatrix](Tracker[T]):
    """Tracker using a QR decomposition.

    This is an implementation of Evaluation 3 in [#Lo87]_, known as a Lohner's QR
    method.

    References
    ----------
    .. [#Lo87] R. J. Lohner, "Enclosing the Solutions of Ordinary Initial and Boundary
        Value Problem," in *Computerarithmetic*, E. Kaucher, U. Kulisch, and Ch.
        Ullrich, Eds. Stuttgart, Germany: B. G. Teubner, 1987, pp. 225--286.
    """

    __slots__ = ("_b", "_m", "_r")
    _b: T
    _m: npt.NDArray
    _r: T

    def __init__(self, x0: T):
        self._b = x0.eye(len(x0), intvl=x0.interval)
        self._m = x0.mid()
        self._r = x0 - self._m

    def hull(self) -> T:
        return self._b @ self._r + self._m

    def sample(self) -> T:
        return self._b.__class__(self._m, intvl=self._b.interval)

    def update(self, a: T, b: T) -> None:
        """Update the current pair.

        Parameters
        ----------
        a : IntervalMatrix, shape (n, n)
            The image of :meth:`hull` mapped by :math:`Df(x)`.
        b : IntervalMatrix, shape (n,)
            The value obtained by mapping :meth:`sample` with :math:`f(x)`.
        """
        q0 = self._b
        tmp = a @ q0

        n = len(b)
        s = [(i, approx_norm(tmp[:, i], "two") * self._r[i].diam()) for i in range(n)]
        s.sort(key=lambda x: x[1], reverse=True)
        inf = a.inf[:, [x[0] for x in s]]
        sup = a.sup[:, [x[0] for x in s]]
        tmp = a.__class__(inf, sup, intvl=q0.interval) @ q0

        q1 = q0.__class__(approx_qr(tmp)[0], intvl=q0.interval)
        p1 = inv(q1, q1.mid().T)
        self._b = q1
        self._m = b.mid()
        self._r = (p1 @ a @ q0) @ self._r + p1 @ (b - self._m)


def doubletontracker[T: IntervalMatrix = Any](
    tracker: Callable[[], type[Tracker]] | type[Tracker] | None = None,
) -> type[Tracker[T]]:
    """Tracker suitable for large initial intervals.

    This is an implementation of Evaluation 4 in [#Lo87]_, and the name "doubleton" is
    from [#MrZg00]_.

    Parameters
    ----------
    tracker : type[Tracker], optional
        The default is :class:`qrtracker`.

    Returns
    -------
    type[Tracker]

    References
    ----------
    .. [#Lo87] R. J. Lohner, "Enclosing the Solutions of Ordinary Initial and Boundary
        Value Problem," in *Computerarithmetic*, E. Kaucher, U. Kulisch, and Ch.
        Ullrich, Eds. Stuttgart, Germany: B. G. Teubner, 1987, pp. 225--286.
    .. [#MrZg00] M. Mrozek and P. Zgliczy\u0144ski, "Set arithmetic and the enclosing
        problem in dynamics," *Ann. Pol. Math.*, vol. 74, pp. 237--259, 2000,
        :doi:`10.4064/ap-74-1-237-259`.
    """

    def is_tracker(x: object) -> TypeIs[type[Tracker]]:
        return inspect.isclass(x) and issubclass(x, Tracker)

    if tracker is None:
        tracker = qrtracker
    elif not is_tracker(tracker):
        tracker = tracker()

    if not issubclass(tracker, Tracker):
        raise TypeError

    class Result(_doubletontracker, tracker=tracker):
        __slots__ = ()

    return Result


class _doubletontracker[T: IntervalMatrix](Tracker[T], ABC):
    __slots__ = ("_c", "_m", "_r0", "_tracker")
    _cls_tracker: ClassVar[type[Tracker]]
    _tracker: Tracker[T]
    _c: T
    _m: npt.NDArray
    _r0: T

    def __init__(self, x0: T):
        self._c = x0.eye(len(x0), intvl=x0.interval)
        self._m = x0.mid()
        self._r0 = x0 - self._m
        self._tracker = self._cls_tracker(x0.zeros_like())

    def hull(self) -> T:
        return self._m + self._tracker.hull() + self._c @ self._r0

    def sample(self):
        return self._c.__class__(self._m, intvl=self._c.interval)

    def update(self, a: T, b: T) -> None:
        """Update the current pair.

        Parameters
        ----------
        a : IntervalMatrix, shape (n, n)
            The image of :meth:`hull` mapped by :math:`Df(x)`.
        b : IntervalMatrix, shape (n,)
            The value obtained by mapping :meth:`sample` with :math:`f(x)`.
        """
        c0 = self._c
        c1 = a.mid() @ c0.inf
        self._c = c0.__class__(c1, intvl=c0.interval)
        self._m = b.mid()
        tmp = b - self._m + (a @ c0 - c1) @ self._r0 + a @ self._tracker.sample()
        self._tracker.update(a, tmp)

    def __init_subclass__(cls, /, tracker: type[Tracker], **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cls_tracker = tracker
