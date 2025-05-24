import pytest

from verry import function as vrf
from verry.autodiff import autodiff


def test_deriv():
    deriv1 = autodiff.deriv(lambda x: (x + vrf.sin(x**2)) / x)
    deriv2 = autodiff.deriv(deriv1)
    assert pytest.approx(deriv1(1.4), 1e-5) == -1.23095
    assert pytest.approx(deriv2(1.4), 1e-5) == -3.96476


def test_grad():
    grad = autodiff.grad(vrf.pow)
    assert pytest.approx(grad(4.5, -2.2), 1e-5) == (-0.0178707, 0.0549797)

    grad = autodiff.grad(lambda x, y: vrf.exp(y / x) + 2)
    assert pytest.approx(grad(1.2, 3.5), 1e-5) == (-44.9157, 15.3997)


def test_jacobian():
    jacobian = autodiff.jacobian(lambda x, y: (vrf.sin(x * y), x**2 - vrf.cos(y)))
    matrix = jacobian(2, 3)
    assert pytest.approx(matrix[0], 1e-5) == (2.88051, 1.92034)
    assert pytest.approx(matrix[1], 1e-5) == (4.00000, 0.14112)
