import re
from typing import Literal, Self, TypedDict, Unpack

_PATTERN = re.compile(
    r"(?:(?P<fill>[\s\S])?(?P<align>[<>=^]))?"
    r"(?P<sign>[-+ ])?"
    r"(?P<z>z)?"
    r"(?P<alt>#)?"
    r"(?P<zfill>0)?"
    r"(?P<width>\d+)?"
    r"(?P<grouping>[_,])?"
    r"(?:\.(?P<prec>\d+))?"
    r"(?P<type>[eEfFgGn%])?"
)


class _FormatSpecDict(TypedDict, total=False):
    fill: str
    align: Literal["<", ">", "=", "^"] | None
    sign: Literal["+", "-", "\u0020"]
    z: bool
    alt: bool
    zfill: bool
    width: int | None
    grouping: Literal["_", ","] | None
    prec: int | None
    type: Literal["e", "E", "f", "F", "g", "G", "n", "%"] | None


class FormatSpec:
    r"""Utility for managing format specifications.

    See `Python's documentation
    <https://docs.python.org/3/library/string.html#formatspec>`__ for the meaning of
    each parameter.

    Parameters
    ----------
    format_spec : str, default=""
    fill : str, default="\\u0020"
    align : Literal["<", ">", "=", "^"] | None, default=None
    sign : Literal["+", "-", "\\u0020"], default="-"
    z : bool, default=False
    alt : bool, default=False
    zfill : bool, default=False
    width : int | None, default=None
    grouping : Literal["_", ","] | None, default=None
    prec : int | None, default=None
    type : Literal["e", "E", "f", "F", "g", "G", "n", "%"] | None, default=None

    Attributes
    ----------
    fill : str
    align : Literal["<", ">", "=", "^"] | None
    sign : Literal["+", "-", "\\u0020"]
    z : bool
    alt : bool
    zfill : bool
    width : int | None
    grouping : Literal["_", ","] | None
    prec : int | None
    type : Literal["e", "E", "f", "F", "g", "G", "n", "%"] | None

    Examples
    --------
    >>> x = FormatSpec(".3f", prec=4)
    >>> x
    FormatSpec('.4f')
    >>> str(x)
    '.4f'
    >>> x.prec
    4
    >>> x.type
    'f'
    """

    __slots__ = (
        "fill",
        "align",
        "sign",
        "z",
        "alt",
        "zfill",
        "width",
        "grouping",
        "prec",
        "type",
    )

    fill: str
    align: Literal["<", ">", "=", "^"] | None
    sign: Literal["+", "-", "\u0020"]
    z: bool
    alt: bool
    zfill: bool
    width: int | None
    grouping: Literal["_", ","] | None
    prec: int | None
    type: Literal["e", "E", "f", "F", "g", "G", "n", "%"] | None

    def __init__(self, format_spec: str = "", **kwargs: Unpack[_FormatSpecDict]):
        self.fill = "\u0020"
        self.align = None
        self.sign = "-"
        self.z = False
        self.alt = False
        self.zfill = False
        self.width = None
        self.grouping = None
        self.prec = None
        self.type = None

        if format_spec:
            if (match := _PATTERN.fullmatch(format_spec)) is None:
                raise ValueError

            if (fill := match.group("fill")) is not None:
                self.fill = fill

            if (align := match.group("align")) is not None:
                self.align = align  # type: ignore

            if (sign := match.group("sign")) is not None:
                self.sign = sign  # type: ignore

            self.z = match.group("z") is not None
            self.alt = match.group("alt") is not None
            self.zfill = match.group("zfill") is not None

            if (width := match.group("width")) is not None:
                self.width = int(width)

            self.grouping = match.group("grouping")  # type: ignore

            if (prec := match.group("prec")) is not None:
                self.prec = int(prec)

            self.type = match.group("type")  # type: ignore

        for key, value in kwargs.items():
            setattr(self, key, value)

    def copy(self) -> Self:
        result = self.__class__()
        result.fill = self.fill
        result.align = self.align
        result.sign = self.sign
        result.z = self.z
        result.alt = self.alt
        result.zfill = self.zfill
        result.width = self.width
        result.grouping = self.grouping
        result.prec = self.prec
        result.type = self.type
        return result

    def format(self, value: object) -> str:
        """Shorthand for ``format(value, str(self))``."""
        return format(value, self.__str__())

    def replace(self, **changes: Unpack[_FormatSpecDict]) -> Self:
        """Create a new :class:`FormatSpec`, replacing fields with values from
        `changes`."""
        result = self.copy()

        for key, value in changes.items():
            setattr(result, key, value)

        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__str__()!r})"

    def __str__(self) -> str:
        result = ""

        if self.align is not None:
            if self.fill != "\u0020":
                result += self.fill

            result += self.align

        if self.sign != "-":
            result += self.sign

        if self.z:
            result += "z"

        if self.alt:
            result += "#"

        if self.zfill:
            result += "0"

        if self.width is not None:
            result += str(self.width)

        if self.grouping is not None:
            result += self.grouping

        if self.prec is not None:
            result += f".{self.prec}"

        if self.type is not None:
            result += self.type

        return result

    def __eq__(self, other: Self) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return (
            other.fill == self.fill
            and other.align == self.align
            and other.sign == self.sign
            and other.z == self.z
            and other.alt == self.alt
            and other.zfill == self.zfill
            and other.width == self.width
            and other.grouping == self.grouping
            and other.prec == self.prec
            and other.type == self.type
        )

    def __bool__(self) -> bool:
        return (
            self.fill != "\u0020"
            or self.align is not None
            or self.sign != "-"
            or self.z
            or self.alt
            or self.zfill
            or self.width is not None
            or self.grouping is not None
            or self.prec is not None
            or self.type is not None
        )

    def __copy__(self) -> Self:
        return self.copy()

    def __replace__(self, **changes: Unpack[_FormatSpecDict]) -> Self:
        return self.replace(**changes)
