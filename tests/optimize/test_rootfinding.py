from verry import FloatInterval as FI
from verry import function as vrf
from verry.linalg import FloatIntervalMatrix as FIM
from verry.optimize.rootfinding import allroot, allroot_scalar


def test_allroot():
    domain = FIM(inf=[-3, -3], sup=[3, 3])
    r = allroot(lambda x, y: (x**2 - y - 3, -x + y**2 - 3), domain, unique=True)
    assert len(r.unique) == 4
    root_min = min(r.unique, key=lambda x: x[0].inf)
    assert -2 in root_min[0] and 1 in root_min[1]


def test_allroot_scalar():
    domain = FI(-2, 3)
    r = allroot_scalar(lambda x: x**3 - 2 * x, domain, unique=True)
    assert len(r.unique) == 3
    root_max = max(r.unique, key=lambda x: x.sup)
    assert root_max.issuperset(vrf.sqrt(FI(2)))
