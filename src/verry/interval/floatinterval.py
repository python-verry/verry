import math

import gmpy2

from verry import function as vrf
from verry.interval import _floatoperator  # type: ignore
from verry.interval.interval import Converter, Interval, Operator, RoundingMode
from verry.misc.formatspec import FormatSpec


class FloatConverter(Converter[float]):
    __slots__ = ()

    def fromfloat(self, x, /):
        return x

    def fromstr(self, rnd, x, /) -> float:
        if rnd == RoundingMode.FAST:
            return float(x)

        ctx = gmpy2.ieee(64)

        if rnd == RoundingMode.FLOOR:
            ctx.round = gmpy2.RoundDown
        else:
            ctx.round = gmpy2.RoundUp

        return float(gmpy2.mpfr(x, context=ctx))

    def fromint(self, rnd, x, /) -> float:
        if abs(x) <= 0x1FFFFFFFFFFFFF:
            return float(x)

        return self.fromstr(rnd, str(x))

    def str(self, rnd, x, /):
        return self.format(rnd, x, FormatSpec())

    def format(self, rnd, x, spec, /):
        if rnd == RoundingMode.FAST:
            fmt = f".{spec.prec if spec.prec is not None else 6}"
            fmt += spec.type if spec.type is not None else "g"
            return format(x, fmt)

        fmt = f".{spec.prec if spec.prec is not None else 6}"
        fmt += "D" if rnd == RoundingMode.FLOOR else "U"
        fmt += spec.type if spec.type is not None else "g"
        return format(gmpy2.mpfr(x, context=gmpy2.ieee(64)), fmt)

    def repr(self, x, /):
        if not math.isfinite(x):
            return repr(x)

        return f"<{x.hex()}>"


class FloatOperator(Operator[float]):
    __slots__ = ()
    ZERO = 0.0
    ONE = 1.0
    INFINITY = math.inf

    def cadd(self, lhs, rhs):
        return _floatoperator.cadd(lhs, rhs)

    def cmul(self, lhs, rhs):
        return _floatoperator.cmul(lhs, rhs)

    def cdiv(self, lhs, rhs):
        return _floatoperator.cdiv(lhs, rhs)

    def csqr(self, value):
        return _floatoperator.csqr(value)

    def fsqr(self, value):
        return _floatoperator.fsqr(value)

    def mid(self, x, y):
        INFINITY = math.inf

        if x == -INFINITY:
            return 0.0 if y == INFINITY else x

        if y == INFINITY:
            return x

        if abs(x) >= 1 and abs(y) >= 1:
            return 0.5 * x + 0.5 * y

        return 0.5 * (x + y)


class FloatInterval(Interval[float]):
    """Double-precision inf-sup type interval.

    Parameters
    ----------
    inf : float | int | str | None, optional
        Infimum of the interval.
    sup : float | int | str | None, optional
        Supremum of the interval.

    Attributes
    ----------
    inf : float
        Infimum of the interval.
    sup : float
        Supremum of the interval.
    converter : Converter
    endtype : type[float]
    operator : Operator
    """

    __slots__ = ()
    converter = FloatConverter()
    operator = FloatOperator()
    endtype = float

    @classmethod
    def __exp_point(cls, x):
        SQRTE_INV_RD = float.fromhex("0x1.368b2fc6f9609p-1")
        SQRTE_RU = float.fromhex("0x1.a61298e1e069cp+0")
        h = cls(x) - round(x)
        r = vrf.e(h) ** round(x)
        a = cls(0.0)
        tmp = cls(1.0)

        for i in range(1, 16):
            a += tmp
            tmp *= h / i

        a += cls(SQRTE_INV_RD, SQRTE_RU) * tmp
        r *= a
        return r

    @classmethod
    def __log_point(cls, x):
        TWO_THIRDS_RU = float.fromhex("0x1.5555555555556p-1")
        ERROR = float.fromhex("0x1.973774dfc4858p+3")
        p = 0

        while x < TWO_THIRDS_RU:
            x *= 2.0
            p += 1

        while x > 2 * TWO_THIRDS_RU:
            x /= 2.0
            p -= 1

        u = cls(x)
        y = (u - 1.0) / (u + 1.0)
        r = -p * vrf.ln2(u) + 2.0 * y
        tmp = y

        for k in range(2, 14):
            tmp *= y**2
            r += 2.0 * tmp / (2.0 * k - 1.0)

        tmp *= y
        r += cls(-ERROR, ERROR) * tmp
        return r

    def _verry_overload_(self, fun, *args, **kwargs):
        if fun is vrf.e:
            E_INF = float.fromhex("0x1.5bf0a8b145769p+1")
            E_SUP = float.fromhex("0x1.5bf0a8b14576ap+1")
            return self.__class__(E_INF, E_SUP)

        if fun is vrf.exp:
            return self.__exp()

        if fun is vrf.ln2:
            LN2_INF = float.fromhex("0x1.62e42fefa39efp-1")
            LN2_SUP = float.fromhex("0x1.62e42fefa39f0p-1")
            return self.__class__(LN2_INF, LN2_SUP)

        if fun is vrf.log:
            return self.__log()

        if fun is vrf.pi:
            PI_INF = float.fromhex("0x1.921fb54442d18p+1")
            PI_SUP = float.fromhex("0x1.921fb54442d19p+1")
            return self.__class__(PI_INF, PI_SUP)

        if fun is vrf.pow:
            return vrf.exp(vrf.log(self.ensure(args[0])) * args[1])

        return super()._verry_overload_(fun, *args, **kwargs)

    def __exp(self):
        inf = self.__exp_point(self.inf).inf if math.isfinite(self.inf) else 0.0
        sup = self.__exp_point(self.sup).sup if math.isfinite(self.sup) else math.inf
        return self.__class__(inf, sup)

    def __log(self):
        if self.inf < 0.0:
            raise ValueError("math domain error")

        inf = self.__log_point(self.inf).inf if self.inf != 0.0 else -math.inf
        sup = self.__log_point(self.sup).sup if math.isfinite(self.sup) else math.inf
        return self.__class__(inf, sup)
