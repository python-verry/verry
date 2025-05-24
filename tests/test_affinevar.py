from verry.affine import AffineForm
from verry.interval import FloatInterval


def test():
    def rational(x):
        return x**2 / (x**3 + 2 * x + 1)

    var = 4 + 1e-4 * FloatInterval(-1, 1)
    assert rational(AffineForm(var)).range().issubset(rational(var))
