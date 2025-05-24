from verry.interval.floatinterval import FloatConverter
from verry.interval.interval import RoundingMode
from verry.misc.formatspec import FormatSpec


def test_converter():
    ROUND_CEILING = RoundingMode.ROUND_CEILING
    ROUND_FLOOR = RoundingMode.ROUND_FLOOR
    converter = FloatConverter()

    assert converter.format(0.001, FormatSpec(".3f"), ROUND_CEILING) == "0.002"
    assert converter.format(0.001, FormatSpec(".3f"), ROUND_FLOOR) == "0.001"
    assert converter.format(-1.23, FormatSpec(".3g"), ROUND_CEILING) == "-1.22"
    assert converter.format(-1.23, FormatSpec(".3g"), ROUND_FLOOR) == "-1.23"
    assert converter.format(-12.1, FormatSpec(".3e"), ROUND_CEILING) == "-1.209e+1"
    assert converter.format(-12.1, FormatSpec(".3e"), ROUND_FLOOR) == "-1.210e+1"
