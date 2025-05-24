import decimal
import fractions
import math
import re
import sys
from typing import assert_never

from verry import function as vrf
from verry.interval import _floatoperator  # type: ignore
from verry.interval.interval import Converter, Interval, Operator, RoundingMode
from verry.misc.formatspec import FormatSpec


def _decimal_exponent(x: decimal.Decimal | fractions.Fraction) -> int:
    if x == 0:
        return 0

    result = 0

    while abs(x) < 1:
        x *= 10
        result -= 1

    while abs(x) >= 10:
        x /= 10
        result += 1

    return result


class FloatConverter(Converter[float]):
    __slots__ = ()

    def fromfloat(self, value, strict=True):
        return value

    def fromstr(self, value, rounding) -> float:
        if rounding == RoundingMode.ROUND_FAST:
            return float(value)

        if re.fullmatch("[-+]?inf(?:inity)?", value, re.I) is not None:
            return -math.inf if value[0] == "-" else math.inf

        try:
            frac = fractions.Fraction(value)
        except ValueError:
            raise ValueError(f"Could not convert string to float: '{value}'")

        if frac == 0:
            return 0.0

        if is_negative := frac < 0:
            frac *= -1

        if frac < sys.float_info.min:
            match rounding:
                case RoundingMode.ROUND_CEILING:
                    return 0.0 if is_negative else -sys.float_info.min

                case RoundingMode.ROUND_FLOOR:
                    return -sys.float_info.min if is_negative else 0.0

        if frac > sys.float_info.max:
            match rounding:
                case RoundingMode.ROUND_CEILING:
                    return -sys.float_info.max if is_negative else math.inf

                case RoundingMode.ROUND_FLOOR:
                    return -math.inf if is_negative else sys.float_info.max

        shift = 0

        while frac < 2**52:
            frac *= 2
            shift += 1

        while frac >= 2**53:
            frac /= 2
            shift -= 1

        div, mod = divmod(frac.numerator, frac.denominator)
        tmp = float(div)

        if is_negative:
            tmp *= -1.0

        if shift > 0:
            for _ in range(shift):
                tmp *= 0.5
        else:
            for _ in range(-shift):
                tmp *= 2.0

        if mod == 0:
            return tmp

        match rounding:
            case RoundingMode.ROUND_CEILING:
                return math.nextafter(tmp, float("inf"))

            case RoundingMode.ROUND_FLOOR:
                return math.nextafter(tmp, -float("inf"))

            case _ as unreachable:
                assert_never(unreachable)

    def fromint(self, value, rounding) -> float:
        if abs(value) <= 0x1FFFFFFFFFFFFF:
            return float(value)

        return self.fromstr(str(value), rounding)

    def tostr(self, value, mode):
        return self.format(value, FormatSpec(), mode)

    def format(self, value, spec, rounding):
        if value == 0.0 or not math.isfinite(value):
            return spec.format(value)

        context = decimal.Context(Emin=decimal.MIN_EMIN, Emax=decimal.MAX_EMAX)
        frac = fractions.Fraction(value)
        prec = spec.prec

        if prec is None:
            prec = 6

        match spec.type:
            case "e" | "E":
                prec_pow = 10**prec
                shift = 0

                while abs(frac) < prec_pow:
                    frac *= 10
                    shift += 1

                while abs(frac) >= 10 * prec_pow:
                    frac /= 10
                    shift -= 1

                with decimal.localcontext(context, prec=prec + 1):
                    match rounding:
                        case RoundingMode.ROUND_CEILING:
                            tmp = decimal.Decimal(math.ceil(frac))

                        case RoundingMode.ROUND_FLOOR:
                            tmp = decimal.Decimal(math.floor(frac))

                    if shift > 0:
                        for _ in range(shift):
                            tmp /= 10
                    else:
                        for _ in range(-shift):
                            tmp *= 10

                    return spec.format(tmp)

            case "f" | "F":
                exp = _decimal_exponent(frac)

                for _ in range(prec):
                    frac *= 10

                with decimal.localcontext(context, prec=max(2, prec + exp + 1)):
                    match rounding:
                        case RoundingMode.ROUND_CEILING:
                            tmp = decimal.Decimal(math.ceil(frac))

                        case RoundingMode.ROUND_FLOOR:
                            tmp = decimal.Decimal(math.floor(frac))

                    for _ in range(prec):
                        tmp /= 10

                    return spec.format(tmp)

            case "g" | "G" | None:
                exp = _decimal_exponent(round(frac, prec))

                if not -4 <= exp < prec:
                    spec = spec.replace(prec=prec - 1, type="e")
                    result = self.format(value, spec, rounding)
                    return result

                spec = spec.replace(prec=prec - 1 - exp, type="f")
                result = self.format(value, spec, rounding)

                while "." in result and result[-1] in {".", "0"}:
                    result = result[:-1]

                return result

    def repr(self, value):
        if not math.isfinite(value):
            return repr(value)

        return f"<{value.hex()}>"


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

    def mid(self) -> float:
        ZERO = self.operator.ZERO
        INFINITY = self.operator.INFINITY

        if self.inf == -INFINITY:
            return ZERO if self.sup == INFINITY else self.sup

        if self.sup == INFINITY:
            return self.inf

        if abs(self.inf) >= 1 and abs(self.sup) >= 1:
            return self.inf / 2 + self.sup / 2

        return (self.inf + self.sup) / 2

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
