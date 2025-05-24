import dataclasses
import itertools
from collections.abc import Callable
from typing import Literal

from verry.autodiff.autodiff import deriv, jacobian
from verry.interval.interval import Interval
from verry.linalg.intervalmatrix import IntervalMatrix, LinAlgError, approx_inv


@dataclasses.dataclass(frozen=True, slots=True)
class AllRootResult[T: IntervalMatrix]:
    """Output of :func:`allroot`.

    Attributes
    ----------
    exists : list[IntervalMatrix]
        Interval vectors in which at least one root exists.
    unique : list[IntervalMatrix]
        Interval vectors in which exactly one root exists.
    unknown : list[IntervalMatrix]
        Interval vectors in which the existence of roots could not be verified.
    """

    exists: list[T] = dataclasses.field(default_factory=list)
    unique: list[T] = dataclasses.field(default_factory=list)
    unknown: list[T] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True, slots=True)
class AllRootScalarResult[T: Interval]:
    """Output of :func:`allroot_scalar`.

    Attributes
    ----------
    exists : list[Interval]
        Intervals in which at least one root exists.
    unique : list[Interval]
        Intervals in which exactly one root exists.
    unknown : list[Interval]
        Intervals in which the existence of roots could not be verified.
    """

    exists: list[T] = dataclasses.field(default_factory=list)
    unique: list[T] = dataclasses.field(default_factory=list)
    unknown: list[T] = dataclasses.field(default_factory=list)


def allroot[T: IntervalMatrix](
    fun: Callable,
    domain: T,
    fprime: Callable | None = None,
    unique: bool = False,
    max_iter: int = 16,
) -> AllRootScalarResult[T]:
    """Find all roots of multivariate scalar-valued function.

    Parameters
    ----------
    fun : Callable
        Function to find a root of.
    domain : IntervalMatrix
        Interval vector for which roots are searched.
    fprime : Callable, optional
        Derivative of `fun` (the default is ``jacobian(fun)``).
    unique : bool, default=False
        If `unique` is ``True``, verification continues until the number of iterations
        reaches `max_iter` or the uniqueness of the root is verified, even if its
        existence has been established.
    max_iter : int, default=16
        Maximum number of iterations.

    Returns
    -------
    AllRootResult

    Warnings
    --------
    `fun` must be a :math:`C^1`-function on `domain`. Futhermore, `fun` must neither be
    a constant nor contain conditional branches (cf. :doc:`/userguide/pitfall`).

    See Also
    --------
    verry.autodiff.jacobian

    Examples
    --------
    >>> from verry.linalg import FloatIntervalMatrix as FIM
    >>> fun = lambda x, y: (x**2 - y - 3, -x + y**2 - 3)
    >>> r = allroot(fun, FIM(inf=[-3, -3], sup=[3, 3]), unique=True)
    >>> len(r.unique)
    4
    >>> root_min = min(r.unique, key=lambda x: x[0].inf)
    >>> -2 in root_min[0] and 1 in root_min[1]
    True
    """
    if not isinstance(domain, IntervalMatrix):
        raise TypeError

    if domain.ndim != 1:
        raise ValueError

    intvlmat: type[T] = type(domain)
    intvl = domain.interval

    if fprime is None:
        fprime = jacobian(fun)

    if max_iter <= 0:
        raise ValueError

    cands = [domain]
    result: AllRootResult[T] = AllRootResult()

    for i in range(max_iter):
        if not cands:
            break

        next_cands: list[T] = []

        for x in cands:
            match krawczyk(fun, x, fprime):
                case "EXISTS", root:
                    if i == max_iter - 1 or not unique:
                        result.exists.append(root)  # type: ignore
                        continue

                    tmp = ((intvl(y.inf, y.mid()), intvl(y.mid(), y.sup)) for y in x)
                    next_cands.extend(intvlmat(x) for x in itertools.product(*tmp))

                case "NOTEXISTS", _:
                    pass

                case "UNIQUE", root:
                    result.unique.append(root)  # type: ignore

                case "UNKNOWN", _:
                    if i == max_iter - 1:
                        result.unknown.append(x)
                        continue

                    tmp = ((intvl(y.inf, y.mid()), intvl(y.mid(), y.sup)) for y in x)
                    next_cands.extend(intvlmat(x) for x in itertools.product(*tmp))

        cands = next_cands

    return result


def allroot_scalar[T: Interval](
    fun: Callable,
    domain: T,
    fprime: Callable | None = None,
    unique: bool = False,
    max_iter: int = 16,
) -> AllRootScalarResult[T]:
    """Find all roots of univariate scalar-valued function.

    Parameters
    ----------
    fun : Callable
        Function to find a root of.
    domain : Interval
        Interval for which roots are searched.
    fprime : Callable, optional
        Derivative of `fun` (the default is ``deriv(fun)``).
    unique : bool, default=False
        If `unique` is ``True``, verification continues until the number of iterations
        reaches `max_iter` or the uniqueness of the root is verified, even if its
        existence has been established.
    max_iter : int, default=16
        Maximum number of iterations.

    Returns
    -------
    AllRootScalarResult

    Warnings
    --------
    `fun` must be a :math:`C^1`-function on `domain`. Futhermore, `fun` must neither be
    a constant nor contain conditional branches (cf. :doc:`/userguide/pitfall`).

    See Also
    --------
    verry.autodiff.deriv

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> from verry import function as vrf
    >>> r = allroot_scalar(lambda x: x**3 - 2 * x, FI(-2, 3), unique=True)
    >>> len(r.unique)
    3
    >>> root_max = max(r.unique, key=lambda x: x.sup)
    >>> root_max.issuperset(vrf.sqrt(FI(2)))
    True
    """
    if not isinstance(domain, Interval):
        raise TypeError

    intvl: type[T] = type(domain)

    if fprime is None:
        fprime = deriv(fun)

    if max_iter <= 0:
        raise ValueError

    cands = [domain]
    result = AllRootScalarResult()

    for i in range(max_iter):
        if not cands:
            break

        next_cands: list[T] = []

        for x in cands:
            match krawczyk_scalar(fun, x, fprime):
                case "EXISTS", root:
                    if i == max_iter - 1 or not unique:
                        result.exists.append(root)  # type: ignore
                        continue

                    next_cands.extend((intvl(x.inf, x.mid()), intvl(x.mid(), x.sup)))

                case "NOTEXISTS", _:
                    pass

                case "UNIQUE", root:
                    result.unique.append(root)  # type: ignore

                case "UNKNOWN", _:
                    if i == max_iter - 1:
                        result.unknown.append(x)
                        continue

                    next_cands.extend((intvl(x.inf, x.mid()), intvl(x.mid(), x.sup)))

        cands = next_cands

    return result


type _TestResult[T] = (
    tuple[Literal["EXISTS"], T]
    | tuple[Literal["NOTEXISTS"], None]
    | tuple[Literal["UNIQUE"], T]
    | tuple[Literal["UNKNOWN"], None]
)


def krawczyk[T: IntervalMatrix](
    fun: Callable, x: T, fprime: Callable | T | None = None
) -> _TestResult[T]:
    """Apply the Krawczyk test to the multivariate vector-valued function.

    Parameters
    ----------
    fun : Callable
        Function to find a root of.
    x : IntervalMatrix
        Candidate set for which a root is expected to exist.
    fprime : Callable | IntervalMatrix, optional
        Derivative of `fun`, or an image of `x` under the derivative (the default is
        ``jacobian(fun)``).

    Returns
    -------
    r0 : Literal["EXISTS", "NOTEXISTS", "UNIQUE", "UNKNOWN"]
    r1 : IntervalMatrix | None
        If `r0` is ``"EXISTS"`` or ``"UNIQUE"``, `r1` is the set containing the root;
        otherwise, `r1` is ``None``.

    Warnings
    --------
    `fun` must be a :math:`C^1`-function on `x`. Futhermore, `fun` must neither be a
    constant nor contain conditional branches (cf. :doc:`/userguide/pitfall`).

    See Also
    --------
    krawczyk_scalar, verry.autodiff.jacobian
    """
    if not isinstance(x, IntervalMatrix):
        raise TypeError

    if x.ndim != 1:
        raise ValueError

    intvlmat = type(x)
    intvl = x.interval

    match fprime:
        case Callable():  # type: ignore
            c = intvlmat(fprime(*x), intvl=intvl)

        case x.__class__():
            c = fprime

        case None:
            c = intvlmat(jacobian(fun)(*x), intvl=intvl)

        case _:
            raise TypeError

    x0 = intvlmat(x.mid(), intvl=intvl)

    if not x.interiorcontains(x0):
        return ("UNKNOWN", None)

    try:
        r = approx_inv(c)
    except LinAlgError:
        return ("UNKNOWN", None)

    tmp = intvlmat.eye(len(x), intvl=intvl) - r @ c
    x1 = x0 - r @ intvlmat(fun(*x0), intvl=intvl) + tmp @ (x - x0)

    if x.interiorcontains(x1):
        return ("UNIQUE", x1)

    if x.issuperset(x1):
        return ("EXISTS", x1)

    if x.isdisjoint(x1):
        return ("NOTEXISTS", None)

    return ("UNKNOWN", None)


def krawczyk_scalar[T: Interval](
    fun: Callable, x: T, fprime: Callable | T | None = None
) -> _TestResult[T]:
    """Apply the Krawczyk test to the univariate scalar-valued function.

    Parameters
    ----------
    fun : Callable
        Function to find a root of.
    x : Interval
        Candidate set for which a root is expected to exist.
    fprime : Callable | Interval, optional
        Derivative of `fun`, or an image of `x` under the derivative (the default is
        ``deriv(fun)``).

    Returns
    -------
    r0 : Literal["EXISTS", "NOTEXISTS", "UNIQUE", "UNKNOWN"]
    r1 : Interval | None
        If `r0` is ``"EXISTS"`` or ``"UNIQUE"``, `r1` is the set containing the root;
        otherwise, `r1` is ``None``.

    Warnings
    --------
    `fun` must be a :math:`C^1`-function on `x`. Futhermore, `fun` must neither be a
    constant nor contain conditional branches (cf. :doc:`/userguide/pitfall`).

    See Also
    --------
    krawczyk, verry.autodiff.deriv
    """
    if not isinstance(x, Interval):
        raise TypeError

    intvl = type(x)

    match fprime:
        case Callable():  # type: ignore
            c = fprime(x)

        case x.__class__():
            c = fprime

        case None:
            c = deriv(fun)(x)

        case _:
            raise TypeError

    if not isinstance(c, intvl):
        raise TypeError

    ONE = intvl.operator.ONE
    ZERO = intvl.operator.ZERO
    x0 = intvl(x.mid())

    if not x.interiorcontains(x0):
        return ("UNKNOWN", None)

    if (c0 := c.mid()) == ZERO:
        return ("UNKNOWN", None)

    r = ONE / c0
    x1 = x0 - r * fun(x0) + (1 - r * c) * (x - x0)

    if x.interiorcontains(x1):
        return ("UNIQUE", x1)

    if x.issuperset(x1):
        return ("EXISTS", x1)

    if x.isdisjoint(x1):
        return ("NOTEXISTS", None)

    return ("UNKNOWN", None)
