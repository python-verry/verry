from collections.abc import Callable, Sequence

from verry.autodiff.autodiff import jacobian
from verry.autodiff.dual import IntervalJet
from verry.interval.interval import Interval
from verry.linalg.intervalmatrix import IntervalMatrix
from verry.typing import ComparableScalar


def seriessol[T: ComparableScalar](
    fun: Callable, t0: T, y0: IntervalMatrix[T] | Sequence[Interval[T]], order: int
) -> tuple[IntervalJet[T], ...]:
    """Return the Taylor polynomial of the ODE solution.

    Parameters
    ----------
    fun : Callable
        Right-hand side of the system.
    t0 : Interval
        Initial time.
    y0 : IntervalMatrix | Sequence[Interval]
        Initial state.
    order : int
        Order of the Taylor polynomial.

    Returns
    -------
    tuple[IntervalJet, ...]

    Warnings
    --------
    `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> series = seriessolution(lambda t, y: (y,), FI(0), [FI(1)], 4)
    >>> print([format(x.mid(), ".3f") for x in series[0].coeffs])
    ['1.000', '1.000', '0.500', '0.167', '0.042']
    """
    intvl = type(t0)
    t = IntervalJet([t0])
    series = tuple(IntervalJet([x]) for x in y0)

    for k in range(1, order + 1):
        dydt = fun(t, *series)

        for i in range(len(series)):
            series[i].coeffs.append(dydt[i].coeffs[k - 1] / k)

        t.coeffs.append(intvl(1 if k == 1 else 0))

    return series


def variationaleq(fun: Callable, sol: Callable) -> Callable:
    r"""Return the right-hand side of variational equations.

    Parameters
    ----------
    fun : Callable
        Right-hand side of the system.
    sol : Callable
        Solution of ODEs.

    Returns
    -------
    Callable

    Warnings
    --------
    `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    Notes
    -----
    Variational equations of ODEs :math:`\mathrm{d}y/\mathrm{d}t=f(t,y)` are defined as

    .. math:: \frac{\mathrm{d}v}{\mathrm{d}t} = (v\cdot\nabla)f(t,y),

    where :math:`y=\phi(t)` is a solution of ODEs.
    """

    def result(t, *v):
        n = len(v)
        jac = jacobian(lambda *y: fun(t, *y))(*sol(t))
        return tuple(sum(jac[i][j] * v[j] for j in range(n)) for i in range(n))

    return result
