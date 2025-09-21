#####
Verry
#####

Verry is an open-source library for `verified computation <https://en.wikipedia.org/wiki/Validated_numerics>`_ written in Python 3.

********
Features
********

* Affine arithmetic
* Automatic differentiation
* Interval arithmetic
* Nonlinear equations solver
* ODE solver
* Quadrature

***************
Getting started
***************

Verry can be installed from PyPI: `pip install verry`.

Here is a simple example::

    >>> from verry import FloatInterval as FI
    >>> print(sum(FI("0.1") for _ in range(10)))
    [inf=0.999999, sup=1.00001]

*******
License
*******

Verry is distributed under the `BSD 3-Clause License <https://opensource.org/license/bsd-3-clause>`_.
