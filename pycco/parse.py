from enum import StrEnum
from dataclasses import dataclass
from typing import List, Optional, Union
from pycco.parser import *
from pycco import ast


class CTypes(StrEnum):
    CHAR = "char"
    DOUBLE = "double"
    FLOAT = "float"
    INT = "int"
    LONG = "long"
    SHORT = "short"
    SIGNED = "signed"
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


class PrefixUnaryOperators(StrEnum):
    INCREMENT = "++"
    DECREMENT = "--"
    ADDRESS_OF = "&"
    DEREFERENCE = "*"
    UNARY_PLUS = "+"
    UNARY_MINUS = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"


class PostfixUnaryOperators(StrEnum):
    INCREMENT = "++"
    DECREMENT = "--"


number_literal = (
    (regex(r"[0-9]+\.[0-9]+") | regex(r"[-+]?[0-9]+"))
    .map(lambda s: ast.Number(s))
    .desc("number")
)

char_literal = regex(r"'.'").map(lambda s: ast.Char(s[1:-1])).desc("char")

string_literal = (
    regex(r'".*?"').map(lambda s: ast.StringLiteral(s[1:-1])).desc("string")
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
struct_keyword = string("struct")

struct_type = struct_keyword >> padding >> ident

type_plain = (struct_type | from_enum(CTypes)).map(ast.Type)
type_pointer = (type_plain << padding << string("*")).map(lambda t: t.set_pointer())
type_ = type_pointer | type_plain
type_ = type_.desc("type")

variable_decl = seq(type=type_, _space=space, name=ident).combine_dict(ast.Variable)


@generate
def array_decl():
    t = yield type_
    yield space
    name = yield ident
    yield padding
    yield string("[")
    size = yield expression.optional()
    yield string("]")
    yield padding
    return ast.ArrayVariable(t, name, size)


variable = array_decl | variable_decl
variable_statement = variable << semicolon

void = string("void").result(ast.Type("void"))


@generate
def expression():
    ret = yield (
        function_call
        | parens
        | op
        | array
        | block
        | array_index
        | struct_access
        | char_literal
        | string_literal
        | ident
        | number_literal
    )
    return ret


padded_expression = padding >> expression << padding

assign = seq(
    var=(variable | ident),
    _lpad=padding,
    _eq=equal,
    _rpad=padding,
    value=expression,
    _semi=semicolon,
).combine_dict(ast.Assign)


@Parser
def match_paren(stream, index):
    indices = [index - 1]
    start_index = index
    inner = []
    while True:
        if index >= len(stream):
            raise PyCcoParserError(
                stream=stream, index=indices[-1], message="unmatched open paren"
            )
        char = stream[index]
        if char == "(":
            indices.append(index)
        elif char == ")":
            indices.pop()
        else:
            inner.append(char)
        index += 1
        if not indices:
            try:
                inner_result = expression.parse("".join(inner))
            except PyCcoParserError as e:
                e.stream = stream
                e.index += start_index
                raise e
            return Result.success(index, ast.Parens(inner_result))


parens = open_paren >> match_paren

array = seq(
    _lbrace=open_brace, values=padded_expression.sep_by(comma), _rbrace=close_brace
).combine_dict(ast.Array)


@generate
def block():
    yield open_brace
    yield padding
    main = yield statement.sep_by(padding)
    yield padding
    yield close_brace
    return ast.Block(main)


def binop_parser(op: BinaryOperators):
    @Parser
    def inner(stream, index):

        try:
            found = stream.index(op, index + 1)
        except ValueError:
            return Result.failure(index, "expression")

        left = stream[index:found]
        left_expr = padded_expression(left, 0)

        if not left_expr.status:
            return Result.failure(index, "expression")
        left_expr.index += len(op) + index

        return left_expr

    @generate
    def bop():
        left = yield inner
        right = yield padded_expression
        return ast.BinaryOp(left, op, right)

    return bop


binop = alt(*[binop_parser(op) for op in BinaryOperators][::-1])


def post_unop_parser(op: BinaryOperators):
    @Parser
    def inner(stream, index):

        try:
            found = stream.index(op, index + 1)
        except ValueError:
            return Result.failure(index, "expression")
        left = stream[index:found]
        left_expr = padded_expression(left, 0)
        if not left_expr.status:
            return Result.failure(index, "expression")
        left_expr.index += len(op) + index
        return left_expr

    @generate
    def uop():
        value = yield inner
        return ast.UnaryOp(op, value, True)

    return uop


post_unop = alt(*[post_unop_parser(op) for op in PostfixUnaryOperators])

pre_unop = seq(op=from_enum(PrefixUnaryOperators), value=expression).combine_dict(
    ast.UnaryOp
)

unop = post_unop | pre_unop

op = binop | unop


def post_op_parser(op: BinaryOperators):
    op_ = string(op)

    @generate
    def inner():
        stack = []
        left = yield any_char.until(op_).concat().desc("")
        yield op_
        right = yield expression
        if not left:
            return ast.UnaryOp(op, right)
        return ast.BinaryOp(expression.parse(left), op, right)

    return inner


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
    ret = yield (
        assign
        | variable_statement
        | return_
        | function_def
        | function_call_statement
        | if_statement
        | while_loop
        | struct_declaration  # Add struct declaration
        | typedef  # Add typedef
    )
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
def function_def():
    var = yield variable
    yield padding
    params = yield function_params
    yield padding
    main = yield block
    return ast.Function(var, main, params)


args = open_paren >> padded_expression.sep_by(comma) << close_paren

function_call = seq(name=ident, args=args).combine_dict(ast.FunctionCall)

function_call_statement = (function_call << padding << semicolon).map(
    lambda f: f.set_statement()
)


@generate
def if_statement():
    yield string("if")
    yield padding
    yield open_paren
    condition = yield padded_expression
    yield close_paren
    yield padding
    then_branch = yield block
    else_branch = None
    if (yield string("else").optional()):
        yield padding
        else_branch = yield block
    return ast.IfStatement(condition, then_branch, else_branch)


@generate
def while_loop():
    yield string("while")
    yield padding
    yield open_paren
    condition = yield padded_expression
    yield close_paren
    yield padding
    body = yield block
    return ast.WhileLoop(condition, body)


@generate
def array_index():
    array = yield ident
    yield string("[")
    index = yield padded_expression
    yield string("]")
    return ast.ArrayIndex(array, index)


@generate
def struct_access():
    obj = yield ident
    operator = yield string(".") | string("->")
    field = yield ident
    return ast.StructAccess(obj, operator, field)


@generate
def struct_definition():
    yield struct_keyword
    yield padding
    name = yield ident.optional()
    yield padding
    yield open_brace
    yield padding
    fields = yield variable_statement.sep_by(padding)
    yield padding
    yield close_brace
    return ast.Struct(name, fields)


struct_declaration = struct_definition << semicolon


@generate
def typedef():
    yield string("typedef")
    yield space
    original = yield (struct_definition | type_)
    yield space
    alias = yield ident
    yield semicolon
    return ast.Typedef(original, alias)


code = statement | expression


@generate
def c_code():
    yield padding
    code_ = yield code.sep_by(padding)
    yield padding
    return ast.Code(code_)


def parse(code: str) -> ast.Code:
    try:
        code_ast = c_code.parse(code)
        code_ast.validate_ast()
    except PyCcoParserError as e:
        raise PyCcoError(str(e))
    except ast.PyCcoASTError as e:
        index = e.node.parse_result.index
        message = str(e)
        raise PyCcoError(format_error(code, index, message))
    return code_ast
