from enum import IntEnum, StrEnum, auto
from dataclasses import dataclass




class TokenKind(IntEnum):
    IDENTIFIER = auto()      # Variable names, function names, etc.
    KEYWORD = auto()         # C keywords like int, return, void, etc.
    SYMBOL = auto()          # Symbols like {, }, (, ), ;, etc.
    OPERATOR = auto()        # Operators like +, -, *, /, etc.
    STRING_LITERAL = auto()  # String literals like "hello"
    NUMBER = auto()          # Numeric constants like 42, 3.14
    COMMENT = auto()         # Comments like // this is a comment
    WHITESPACE = auto()      # Spaces, tabs, newlines


class CKeywords(StrEnum):
    AUTO = "auto"
    BREAK = "break"
    CASE = "case"
    CHAR = "char"
    CONST = "const"
    CONTINUE = "continue"
    DEFAULT = "default"
    DO = "do"
    DOUBLE = "double"
    ELSE = "else"
    ENUM = "enum"
    EXTERN = "extern"
    FLOAT = "float"
    FOR = "for"
    GOTO = "goto"
    IF = "if"
    INLINE = "inline"
    INT = "int"
    LONG = "long"
    REGISTER = "register"
    RESTRICT = "restrict"
    RETURN = "return"
    SHORT = "short"
    SIGNED = "signed"
    SIZEOF = "sizeof"
    STATIC = "static"
    STRUCT = "struct"
    SWITCH = "switch"
    TYPEDEF = "typedef"
    UNION = "union"
    UNSIGNED = "unsigned"
    VOID = "void"
    VOLATILE = "volatile"
    WHILE = "while"
    _ALIGNAS = "_Alignas"
    _ALIGNOF = "_Alignof"
    _ATOMIC = "_Atomic"
    _BOOL = "_Bool"
    _COMPLEX = "_Complex"
    _GENERIC = "_Generic"
    _IMAGINARY = "_Imaginary"
    _NORETURN = "_Noreturn"
    _STATIC_ASSERT = "_Static_assert"
    _THREAD_LOCAL = "_Thread_local"


@dataclass
class Token:
    kind: TokenKind
    value: str
    start: int = -1
    end: int = -1