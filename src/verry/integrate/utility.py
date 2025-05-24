from collections.abc import Callable, Sequence

from verry.autodiff.autodiff import jacobian
from verry.interval.interval import Interval
from verry.intervalseries import IntervalSeries, localcontext
from verry.linalg.intervalmatrix import IntervalMatrix


def seriessolution[T: Interval](
    fun: Callable, t0: T, y0: IntervalMatrix[T] | Sequence[T], deg: int
) -> tuple[IntervalSeries[T], ...]:
    """Return the Taylor polynomial of the ODE solution.

    Parameters
    ----------
    fun : Callable
        Right-hand side of the system.
    t0 : Interval
        Initial time.
    y0 : IntervalMatrix | Sequence[Interval]
        Initial state.
    deg : int
        Degree of Taylor polynomial.

    Returns
    -------
    tuple[IntervalSeries, ...]

    Warnings
    --------
    `fun` must neither be a constant nor contain conditional branches
    (cf. :doc:`/userguide/pitfall`).

    Examples
    --------
    >>> from verry import FloatInterval as FI
    >>> series = seriessolution(lambda t, y: (y,), FI(0), [FI(1)], 4)
    >>> print([format(coeff.mid(), ".3f") for coeff in series[0].coeffs])
    ['1.000', '1.000', '0.500', '0.167', '0.042']
    """
    t = IntervalSeries([t0, 1], intvl=type(t0))
    y = (IntervalSeries([x], intvl=type(t0)) for x in y0)

    with localcontext(rounding="TYPE1", deg=0) as ctx:
        while ctx.deg < deg:
            y = (y0 + dy.integrate() for y0, dy in zip(y0, fun(t, *y)))
            ctx.deg += 1

    return tuple(y)


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
    Variational equations of ODEs :math:`\mathrm{d}x/\mathrm{d}t=f(t,x)` are defined as

    .. math:: \frac{\mathrm{d}v}{\mathrm{d}t} = (v\cdot\nabla)f(t,x),

    where :math:`x=\phi(t)` is a solution of ODEs.
    """

    def result(t, *v):
        n = len(v)
        jac = jacobian(lambda *y: fun(t, *y))(*sol(t))
        return tuple(sum(jac[i][j] * v[j] for j in range(n)) for i in range(n))

    return result
