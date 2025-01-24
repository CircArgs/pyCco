# pycco/parsers.py

from typing import List, Optional, Callable, Union
from pycco.parser import (
    Parser,
    match,
    match_fn,
    any_of,
    sequence,
    ParserResult,
    Position,
)
from pycco.tokens import Token, TokenKind
from pycco.ast import (
    Code,
    VariableDecl,
    Assign,
    Function,
    Arg,
    Return,
    Ident,
    Type,
    Number,
    StringLiteral,Expression
)
from pycco.tokenizers import tokenize
from itertools import chain


nest = lambda l: [l]
void_tok = Token(TokenKind.TYPE, 'void')
# ------------------------------------------------------------------------------
# Basic Token Matchers (Combinators)
# ------------------------------------------------------------------------------
type_ = match(TokenKind.TYPE)
star_op = match(Token(TokenKind.OPERATOR, "*"))
name = match(TokenKind.IDENTIFIER)
open_paren = match(Token(TokenKind.SYMBOL, "("))
close_paren = match(Token(TokenKind.SYMBOL, ")"))
open_brace = match(Token(TokenKind.SYMBOL, "{"))
close_brace = match(Token(TokenKind.SYMBOL, "}"))
semicolon = match(Token(TokenKind.SYMBOL, ";"))
comma = match(Token(TokenKind.SYMBOL, ","))
return_kw = match(Token(TokenKind.KEYWORD, "return"))
void = match(void_tok)


# ------------------------------------------------------------------------------
# Forward Declarations for Grammar
# ------------------------------------------------------------------------------
expression = Parser().set_description("expression")
statement = Parser().set_description("statement")


# ------------------------------------------------------------------------------
# Mapping Helpers
# ------------------------------------------------------------------------------
def map_ident(token: Token) -> Ident:
    """
    Create a new AST Ident from a Token.
    The new AST Ident requires Ident(name=...)
    """
    return Ident(name=token.value).set_tokens([token])


ident = name @ map_ident
ident.set_description("identifier")


def map_string(token: Token) -> StringLiteral:
    return StringLiteral(value=token.value).set_tokens([token])


string_lit = match(Token(TokenKind.STRING_LITERAL)) @ map_string


def map_number(token: Token) -> Number:
    """
    The new Number node is Number(value: str, type: Type).
    We'll do a quick guess of 'float' vs 'int' by '.' presence.
    """
    if "." in token.value:
        ty = Type(name="float")
    else:
        ty = Type(name="int")
    return Number(value=token.value, type=ty).set_tokens([token])


number_lit = match(Token(TokenKind.NUMBER)) @ map_number


# ------------------------------------------------------------------------------
# Variable Declaration Parsers
# ------------------------------------------------------------------------------

def map_var_no_semicolon(tokens: List[Token]) -> VariableDecl:
    """
    Map tokens to a VariableDecl without requiring a semicolon.
    """
    # tokens: [TYPE, (maybe STAR?), IDENT]
    ttype = tokens[0]
    pointer = False
    idx = 1
    if idx < len(tokens) and tokens[idx].kind == TokenKind.OPERATOR and tokens[idx].value == "*":
        pointer = True
        idx += 1
    if idx < len(tokens) and tokens[idx].kind == TokenKind.IDENTIFIER:
        name_tok = tokens[idx]
    else:
        raise ValueError("Expected identifier in variable declaration.")
    return VariableDecl(
        type=Type(name=ttype.value, pointer=pointer).set_tokens([ttype]),
        name=Ident(name=name_tok.value).set_tokens([name_tok]),
        semicolon=False
    ).set_tokens(tokens)


variable_no_semicolon = (type_ + star_op.optional() + name).map(map_var_no_semicolon)
variable_no_semicolon.set_description("type + optional '*' + identifier")


def set_semicolon_true(decl: VariableDecl) -> VariableDecl:
    decl.semicolon = True
    return decl


variable_decl = (variable_no_semicolon << semicolon).map(set_semicolon_true)
variable_decl.set_description("variable declaration with semicolon")


# ------------------------------------------------------------------------------
# Function Definition Parser
# ------------------------------------------------------------------------------

def convert_decl_to_arg(decl: VariableDecl) -> Arg:
    """
    Convert a single `VariableDecl(type=..., name=Ident(...))` into `Arg(type=..., name=...)`.
    """
    return Arg(name=decl.name, type=decl.type).set_tokens(decl.tokens)


def map_function(parts: List[object]) -> Function:
    """
    Map parsed parts to a Function AST node.
    parts = [variable_no_semicolon, list_of_args, list_of_body_nodes]
    """
    var_decl = parts[0]  # VariableDecl without semicolon
    raw_params = parts[1]  # List[VariableDecl]
    body_nodes = parts[2:]  # List[Statement | Expression]

    if raw_params==[void_tok]:
        raw_params = []
    # Convert VariableDecls to Args
    arg_list = [convert_decl_to_arg(d) for d in raw_params]

    # Extract Return statement if present
    ret_stmt = None
    body = []

    for node in body_nodes:
        if isinstance(node, Return):
            ret_stmt = node
        else:
            if ret_stmt:
                break
            body.append(node)

    return Function(
        var=var_decl,
        args=arg_list,
        body=body,
        ret=ret_stmt
    ).set_tokens(var_decl.tokens + list(chain.from_iterable(d.tokens for d in raw_params)))


# Parser for function parameters: (int a, float b)
params_list = (variable_no_semicolon.sep_by(comma)).set_description("function parameters")
function_args = ((open_paren >> params_list << close_paren) | (open_paren >> void @ nest << close_paren)).set_description("function arguments")

# Parser for function body: { ... }
function_body = (open_brace >> statement.many() << close_brace).set_description("function body")

function_def = (variable_no_semicolon + function_args @ nest + function_body).map(map_function)
function_def.set_description("function definition")


# ------------------------------------------------------------------------------
# Assignment Parser
# ------------------------------------------------------------------------------

def map_variable_assign(parts: List[Union[Ident, VariableDecl, Expression, Token]]) -> Assign:
    """
    Map parsed parts to an Assign AST node.
    parts = [Ident|VariableDecl, Expression]
    """
    left, right = parts
    return Assign(var=left, to=right).set_tokens(left.tokens + right.tokens)


assignment = (
    (ident | variable_no_semicolon)  # Left-hand side: Ident or VariableDecl
    << match(Token(TokenKind.OPERATOR, "="))  # Expect '=' operator
) + expression << semicolon  # Right-hand side: expression, followed by ';'

assignment = assignment.map(map_variable_assign)
assignment.set_description("assignment statement")


# ------------------------------------------------------------------------------
# Return Statement Parser
# ------------------------------------------------------------------------------

def map_return(expr: Expression) -> Return:
    return Return(value=expr).set_tokens([])  # Tokens are handled elsewhere


return_stmt = (return_kw >> expression << semicolon).map(map_return)
return_stmt.set_description("return statement")


# ------------------------------------------------------------------------------
# Statement Parser Definition
# ------------------------------------------------------------------------------

# Define the statement parser, ensuring function_def is tried first
statement.define(
    function_def
    | variable_decl
    | assignment
    | return_stmt
    # Add other statements like if, while, etc., as needed
)


# ------------------------------------------------------------------------------
# Expression Parser Definition
# ------------------------------------------------------------------------------

# For demonstration, we treat these as basic expressions
expression.define(
    number_lit
    | string_lit
    | ident
    # Placeholder for unary/binary operations, function calls, etc.
)
expression.set_description("expression")


# ------------------------------------------------------------------------------
# Top-level Parser: parse_tokens
# ------------------------------------------------------------------------------

parse_tokens = (statement | expression).many().set_description("many statements or expressions")


# ------------------------------------------------------------------------------
# Parse Error Context Helper
# ------------------------------------------------------------------------------

class ParseError(Exception):
    """
    Raised when the parser fails to match tokens properly.
    """

    def __init__(self, message: str):
        super().__init__(message)


def create_error_context(source: str, offset: int, context_lines: int = 2) -> str:
    """
    Given the entire source and an absolute `offset` into the string,
    create an error message with line/column context and a pointer.
    """
    lines = source.splitlines()
    # Safeguard: if offset goes beyond the source, clamp it
    offset = min(offset, len(source))

    # Find line number and column number
    line_number = source[:offset].count("\n")
    last_nl = source.rfind("\n", 0, offset)
    if last_nl == -1:
        column_number = offset
    else:
        column_number = offset - (last_nl + 1)

    # Show a few lines before and after for context
    start_line = max(0, line_number - context_lines)
    end_line = min(len(lines), line_number + context_lines + 1)
    snippet = lines[start_line:end_line]

    # Build the pointer (caret) line
    if line_number - start_line < len(snippet):
        error_line = snippet[line_number - start_line]
        pointer_line = " " * column_number + "^"
    else:
        error_line = ""
        pointer_line = ""

    # Combine everything:
    before_error = "\n".join(snippet[: line_number - start_line])
    after_error = "\n".join(snippet[line_number - start_line + 1 :])

    message = (
        f"Parse Error at line {line_number + 1}, column {column_number + 1}:\n\n"
        f"{before_error}\n{error_line}\n{pointer_line}\n{after_error}"
    )

    return message


# ------------------------------------------------------------------------------
# parse_c_code Function
# ------------------------------------------------------------------------------

def parse_c_code(source: str) -> Code:
    """
    Tokenize and parse the given C source into a Code AST node.
    Raises a ParseError with detailed line/column context on failure.
    Returns a top-level `Code` node containing the parsed statements.
    """
    # 1) Tokenize
    tokens = tokenize(source)
    if tokens and tokens[-1].kind == TokenKind.EOF:
        tokens.pop()  # remove EOF token to simplify the parse

    # 2) Attempt parse: parse_tokens is a Parser that returns ParserResult[List[Node]]
    parser_result = parse_tokens(tokens, Position(0))
    if not parser_result:  # i.e. parser_result.__bool__() is False => error
        offset = tokens[0].start if tokens else 0
        msg = create_error_context(source, offset)
        raise ParseError(msg)

    # Extract result
    node_list = parser_result.result  # The list of AST nodes
    consumed_index = parser_result.position.index

    # 3) If parser did not consume all tokens
    if consumed_index < len(tokens):
        leftover_token = tokens[consumed_index]
        offset = leftover_token.start
        msg = create_error_context(source, offset)
        raise ParseError(
            msg + f"\n\nUnexpected extra tokens starting at: {leftover_token}."
        )

    # 4) Ensure node_list is a list
    if not isinstance(node_list, list):
        node_list = [node_list]

    return Code(statements=node_list)
