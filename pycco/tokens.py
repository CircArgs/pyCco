from enum import IntEnum, StrEnum, auto
from typing import Optional
from dataclasses import dataclass


class TokenKind(IntEnum):
    IDENTIFIER = auto()  # Variable names, function names, etc.
    KEYWORD = auto()  # C keywords like return, for, etc.
    TYPE = auto()  # C keywords like int, char, void, etc.
    SYMBOL = auto()  # Symbols like {, }, (, ), ;, etc.
    OPERATOR = auto()  # Operators like +, -, *, /, etc.
    STRING_LITERAL = auto()  # String literals like "hello"
    NUMBER = auto()  # Numeric constants like 42, 3.14
    COMMENT = auto()  # Comments like // this is a comment
    WHITESPACE = auto()  # Spaces, tabs, newlines
    EOF = auto()

    def __eq__(self, other):
        if isinstance(other, Token):
            return self == other.kind
        return super().__eq__(other)

    def __str__(self):
        return self.name


class CTypes(StrEnum):
    CHAR = "char"
    DOUBLE = "double"
    ENUM = "enum"
    FLOAT = "float"
    INT = "int"
    LONG = "long"
    SHORT = "short"
    SIGNED = "signed"
    STRUCT = "struct"
    UNION = "union"
    UNSIGNED = "unsigned"
    VOID = "void"
    _BOOL = "_Bool"
    _COMPLEX = "_Complex"
    _IMAGINARY = "_Imaginary"


class CKeywords(StrEnum):
    AUTO = "auto"
    BREAK = "break"
    CASE = "case"
    CONST = "const"
    CONTINUE = "continue"
    DEFAULT = "default"
    DO = "do"
    ELSE = "else"
    EXTERN = "extern"
    FOR = "for"
    GOTO = "goto"
    IF = "if"
    INLINE = "inline"
    REGISTER = "register"
    RESTRICT = "restrict"
    RETURN = "return"
    SIZEOF = "sizeof"
    STATIC = "static"
    SWITCH = "switch"
    TYPEDEF = "typedef"
    VOLATILE = "volatile"
    WHILE = "while"
    _ALIGNAS = "_Alignas"
    _ALIGNOF = "_Alignof"
    _ATOMIC = "_Atomic"
    _GENERIC = "_Generic"
    _NORETURN = "_Noreturn"
    _STATIC_ASSERT = "_Static_assert"
    _THREAD_LOCAL = "_Thread_local"


class MultiCharOperators(StrEnum):
    PLUS_ASSIGN = "+="
    MINUS_ASSIGN = "-="
    TIMES_ASSIGN = "*="
    DIV_ASSIGN = "/="
    MOD_ASSIGN = "%="
    AND_ASSIGN = "&="
    OR_ASSIGN = "|="
    XOR_ASSIGN = "^="
    SHIFT_LEFT_ASSIGN = "<<="
    SHIFT_RIGHT_ASSIGN = ">>="
    INC = "++"
    DEC = "--"
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"


class SingleCharOperators(StrEnum):
    PLUS = "+"
    MINUS = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    ASSIGN = "="
    BITWISE_AND = "&"
    BITWISE_OR = "|"
    BITWISE_XOR = "^"
    BITWISE_NOT = "~"
    LESS_THAN = "<"
    GREATER_THAN = ">"
    SHIFT_LEFT = "<<"
    SHIFT_RIGHT = ">>"


class Symbols(StrEnum):
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    LBRACE = "{"
    RBRACE = "}"
    SEMICOLON = ";"
    COMMA = ","
    DOT = "."
    COLON = ":"
    QUESTION = "?"
    BACKSLASH = "\\"


@dataclass
class Token:
    kind: TokenKind
    value: Optional[str] = None
    start: int = -1
    end: int = -1

    def __eq__(self, other):
        """
        Allows convenient comparisons like:
            token == TokenKind.IDENTIFIER
            token == "main"
            token1 == token2
        """
        if isinstance(other, TokenKind):
            return self.kind == other
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, Token) and self.kind == other.kind:
            return (self.value is None or other.value is None) or (
                self.value == other.value
            )
        return False

    def __str__(self):
        return f"{self.kind}({self.value!r})" if self.value else str(self.kind)
