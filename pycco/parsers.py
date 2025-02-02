from enum import StrEnum
from dataclasses import dataclass
from typing import List, Optional, Union
from pycco.parser import *
from pycco import ast


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
    INT = "int"
    LONG = "long"
    REGISTER = "register"
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


class BinaryOperators(StrEnum):
    MULTIPLICATION = "*"
    DIVISION = "/"
    MODULUS = "%"
    ADDITION = "+"
    SUBTRACTION = "-"
    LEFT_SHIFT = "<<"
    RIGHT_SHIFT = ">>"
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    EQUAL = "=="
    NOT_EQUAL = "!="
    BITWISE_AND = "&"
    BITWISE_XOR = "^"
    BITWISE_OR = "|"
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"


class UnaryOperators(StrEnum):
    INCREMENT = "++"
    DECREMENT = "--"
    ADDRESS_OF = "&"
    DEREFERENCE = "*"
    UNARY_PLUS = "+"
    UNARY_MINUS = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"


number_literal = (
    (regex(r"[0-9]+\.[0-9]+") | regex(r"[-+]?[0-9]+"))
    .map(lambda s: ast.Number(s))
    .desc("number")
)

string_literal = (
    regex(r"'[^']*'").map(lambda s: ast.StringLiteral(s[1:-1])).desc("string")
)

_ident = regex("[a-zA-Z][a-zA-Z0-9_]*").desc("identifier")
keyword = from_enum(CKeywords)


@Parser
def ident(stream, index):
    result = keyword(stream, index)
    if result.status:
        return Result.failure(index, "identifier")
    else:
        return _ident(stream, index)


ident = ident.map(ast.Ident)

open_paren = string("(")
close_paren = string(")")
open_brace = string("{")
close_brace = string("}")
semicolon = string(";")
comma = string(",")
equal = string("=")
space = regex(r"\s+").desc("whitespace")  # non-optional whitespace
padding = regex(r"\s*")  # optional whitespace

type_plain = from_enum(CTypes).map(ast.Type)
type_pointer = (type_plain << padding << string("*")).map(lambda t: t.set_pointer())
type_ = type_pointer | type_plain
type_ = type_.desc("type")

variable = seq(type=type_, _space=space, name=ident).combine_dict(ast.Variable)

void = string("void").result(ast.Type("void"))


@generate
def expression():
    ret = yield parens | op | ident | number_literal | string_literal | function_call
    return ret


assign = seq(
    var=variable,
    _lpad=padding,
    _eq=equal,
    _rpad=padding,
    value=expression,
    _semi=semicolon,
).combine_dict(ast.Assign)


@Parser
def match_paren(stream, index):
    parens = 1
    inner = []
    while True:
        if index >= len(stream):
            return Result.failure(index, ")" * parens)
        char = stream[index]
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        else:
            inner.append(char)
        index += 1
        if parens == 0:
            inner_result = expression("".join(inner), 0)
            if inner_result.status:
                return Result.success(index, ast.Parens(inner_result.value))
            return Result.failure(index + inner_result.index, inner_result.expected)


parens = open_paren >> match_paren


def op_parser(op: BinaryOperators):
    op_ = string(op)

    @generate
    def inner():
        stack = []
        left = yield any_char.until(op_).concat()
        yield op_
        right = yield expression
        if not left:
            return ast.UnaryOp(op, right)
        return ast.BinaryOp(expression.parse(left), op, right)

    return inner


binop = alt(*[op_parser(op) for op in BinaryOperators][::-1])
unop = alt(*[op_parser(op) for op in UnaryOperators])
op = binop | unop

return_ = seq(
    _ret=string("return"),
    _space=space,
    value=expression,
    _rpad=padding,
    _semi=semicolon,
).combine_dict(ast.Return)


@generate
def statement():
    ret = yield assign | function_def | function_call_statement
    return ret


@generate
def function_params():
    ret = []
    yield open_paren
    yield padding
    void_ = yield void.optional()
    if void_:
        ret = [void_]
    else:
        params = yield variable.sep_by(padding + comma + padding, min=1).optional()
        if params:
            ret = params
    yield padding
    yield close_paren
    return ret


@generate
def function_body():
    yield open_brace
    yield padding
    main = yield (return_ | code).sep_by(padding)
    yield padding
    ret = None
    for i, node in enumerate(main):
        if isinstance(node, ast.Return):
            ret = node
            main = main[:i]
            break
    yield close_brace
    return main, ret


@generate
def function_def():
    var = yield variable
    yield padding
    params = yield function_params
    yield padding
    main, ret = yield function_body

    return ast.Function(var, params, main, ret)


args = (
    open_paren
    >> padding
    >> expression.sep_by(padding + comma + padding)
    << padding
    << close_paren
)

function_call = seq(name=ident, args=args).combine_dict(ast.FunctionCall)

function_call_statement = (function_call << padding << semicolon).map(
    lambda f: f.set_statement()
)

code = statement | expression


@generate
def c_code():
    yield padding
    init = yield code
    yield padding
    code_ = yield code.sep_by(padding)
    yield padding

    code_ = [init, *code_]
    main = None

    for i, node in enumerate(code_):
        if isinstance(node, ast.Function) and str(node.var.name) == "main":
            main = code_[i]
            break
    code_ = code_[:i] + code_[i + 1 :]
    return ast.Code(code_, main)
