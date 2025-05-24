from verry import function as vrf
from verry.integrate.integrator import eilo, kashi
from verry.integrate.solver import C0Solver, C1Solver
from verry.interval.floatinterval import FloatInterval as FI


def test_a2detest():
    solver = C0Solver(eilo)
    r = solver.solve(lambda _, x: (-(x**3) / 2,), FI(0), [FI(1)], FI(8))
    assert r.status == "SUCCESS"
    assert 1.0 / 3.0 in r.content.y[0]


def test_a4detest():
    a = FI("0.0125")
    t_bound = -4 * vrf.log(FI(9) / FI(19))
    solver = C1Solver(kashi)
    r = solver.solve(lambda _, x: (a * x * (20 - x),), FI(0), [FI(1)], t_bound)
    assert r.status == "SUCCESS"
    assert 2 in r.content.y[0]


def test_explosion():
    solver = C0Solver(eilo(min_step=1e-4))
    r = solver.solve(lambda _, x: (x**2,), FI(0), [FI(1)], FI(2))
    assert r.status == "FAILURE"
