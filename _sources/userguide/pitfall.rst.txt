###############
Common pitfalls
###############

Initialization of intervals
---------------------------

The statement ``x = 0.1`` in Python means to assign to ``x`` a floating-point number
that approximates 1/10
(see `Python's tutorial <https://docs.python.org/3/tutorial/floatingpoint.html>`_).
This poses a problem when initializing an interval:

.. code-block:: python

    from verry import FloatInterval as FI
    x = FI(0.1)  # This is wrong.
    x = FI("0.1")  # The correct initialization is this.

It is recommended to specify endpoints by strings or integers whenever possible.

Limitations with constants and conditional branches
---------------------------------------------------

Currently, several features in Verry do not support a function that is a constant or
contains conditional branches. This limitation comes primarily from the automatic
differentiation (AD). Verified computation often requires the value of derivatives to
measure approximation errors, and derivatives are computed using AD. However, it is
difficult to make AD work with aforementioned functions.
