"""Uses Parsers to transform a token stream into an AST"""
from typing import List, Optional
from pycco.parser import match, Parser, parser, S, T, U, ParserResult, any_of_enum, match_fn
from pycco.tokens import Token, TokenKind, BinaryOperator, UnaryOperator, OtherOperator
from pycco import ast
from pycco.tokenizers import tokenize

# utility to group sub-list items
group = lambda l: [l]

# Define token matchers
type = match(TokenKind.TYPE)
name = match(TokenKind.IDENTIFIER)
open_paren = match(Token(TokenKind.SYMBOL, "("))
close_paren = match(Token(TokenKind.SYMBOL, ")"))
open_bracket = match(Token(TokenKind.SYMBOL, "{"))
close_bracket = match(Token(TokenKind.SYMBOL, "}"))
comma = match(Token(TokenKind.SYMBOL, ","))
semicolon = match(Token(TokenKind.SYMBOL, ";"))
eq = match(Token(TokenKind.OPERATOR, "="))
star = match(Token(TokenKind.OPERATOR, "*"))

# forward declarations
expression = Parser().set_description('expression')
statement = Parser().set_description('expression')

# Grammar Components with Map Functions

def map_ident(token: Token)->ast.Ident:
    return ast.Ident(token.value).set_tokens([token])

ident = name @ map_ident
ident.set_description('ident')

# Variable Declaration
def map_variable_decl(tokens: List[Token]) -> ast.VariableDecl:
    """
    Map tokens to a VariableDecl.
    Assumes tokens are in the order: type [*] identifier ;
    """
    type = ast.Type(tokens[0].value).set_tokens([tokens[0]])
    pointer = semicolon = False
    if len(tokens) > 2:
        pointer = tokens[1].value == "*"
        semicolon = tokens[-1].value == ';'
    
    name_token = next(token for token in tokens[1:] if token.kind == TokenKind.IDENTIFIER)
    name = ast.Ident(name_token.value).set_tokens([name_token])
    type.pointer = pointer
    return ast.VariableDecl(type=type, name=name, semicolon=semicolon).set_tokens(tokens)


def map_number(token: Token)->ast.Number:
    type = ast.Type('int')
    if '.' in token.value:
        type = ast.Type('float')
    return ast.Number(token.value, type).set_tokens([token])
    
number = match(Token(TokenKind.NUMBER)) @ map_number

def map_string(token: Token)->ast.StringLiteral:
    return ast.StringLiteral(token.value).set_tokens([token])
    
string = match(Token(TokenKind.STRING_LITERAL)) @ map_string

variable_ = type + ~star + name
variable = variable_ @ map_variable_decl
# type with optional * with an identifier and optional ;
variable_decl = (variable_ + semicolon) @ map_variable_decl

args = open_paren >> variable.sep_by(Token(TokenKind.SYMBOL, ',')) << close_paren

def map_variable_assign(elements: List[Token|ast.Node])->ast.Assign:
    return ast.Assign(*elements)
    
variable_assign = (((name@map_ident | variable_@map_variable_decl) << Token(TokenKind.OPERATOR, '='))+expression<<semicolon)
variable_assign = variable_assign @ map_variable_assign

def map_return(expr: ast.Expression)->ast.Return:
    return ast.Return(expr)

return_ = match(Token(TokenKind.KEYWORD, 'return')) >> expression << semicolon
return_ = return_ @ map_return

def map_function(nodes: List[ast.Node])->ast.Function:
    var, args, *nodes = nodes
    ret = None
    if nodes and isinstance(nodes[-1], ast.Return):
        body, ret = nodes[:-1], nodes[-1]
    else:
        body = nodes
    return ast.Function(var, args, body, ret)

# need to declare in sta
function = variable + (args @ group)
function += open_bracket >> (statement.many() + ~return_) << close_bracket
function = function @ map_function

# Unary Operator Parser
def map_unary_op(elements: List[Token | ast.Expression]) -> ast.UnaryOp:
    op, operand = elements
    return ast.UnaryOp(operator=op.value, operand=operand)

unary_op = ((any_of_enum(UnaryOperator) | any_of_enum(BinaryOperator)) + expression) @ map_unary_op
unary_op.set_description('operator + expression')

# binary Operator Parser
def map_binop(elements: List[ast.Node]) -> ast.BinaryOp:
    left, unary = elements
    return ast.BinaryOp(left=left, operator=unary.operator, right=unary.operand)

binop = (match_fn(lambda tok: isinstance(tok, ast.Expression)) + match_fn(lambda node: isinstance(node, ast.UnaryOp))) @ map_binop

def binop_hook(exc: ast.PyCcoTypeError):
    if isinstance(exc.actual_value, list) and len(exc.actual_value)==2:
        if result:=binop(exc.actual_value):
            setattr(exc.node, exc.field_name, result.result)
            return
    raise exc

ast.Node.add_hook(binop_hook)

@parser
def parenthesized(stream, index):
    if stream[index]!='(':
        return ParserResult()
    count = 0
    def parenthesized_(stream, index):
        nonlocal count
        ret = []
        while 0<=index<len(stream):
            token = stream[index]
            index+=1
            if token!='(' and token!=')':
                if count==0:
                    return index-1, ret
                ret.append(token)
            elif token==')':
                count-=1
                if not ret:
                    return ParserResult()
                return index, ast.Parens(parse_tokens(ret).result)
            else:
                count+=1
                index, group = parenthesized_(stream, index)
                ret.append(group)
        if not ret:
            return ParserResult()
        return index, ast.Parens(parse_tokens(ret).result)
    index, ret = parenthesized_(stream, index)
    if count!=0:
        return ParserResult()
    return index, ret

statement.define(variable_decl | variable_assign | function)

expression.define( number | string | ident | unary_op | parenthesized)

parse_tokens = (expression | statement).many()

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

    line_number = source[:offset].count("\n")
    column_number = offset - (source.rfind("\n", 0, offset) + 1)

    # Show a few lines before and after for context
    start_line = max(0, line_number - context_lines)
    end_line = min(len(lines), line_number + context_lines + 1)
    snippet = lines[start_line:end_line]

    # Build the pointer (caret) line
    pointer_line = " " * column_number + "^"

    # Combine everything:
    before_error = "\n".join(snippet[: line_number - start_line])  # lines before
    error_line = snippet[line_number - start_line] if snippet else ""
    after_error = "\n".join(snippet[line_number - start_line + 1:])

    message = (
        f"Parse Error at line {line_number+1}, column {column_number+1}:\n\n"
        f"{before_error}\n"
        f"{error_line}\n"
        f"{pointer_line}\n"
        f"{after_error}"
    )

    return message


def parse_c_code(source: str) -> ast.Code:
    """
    Parse the given C source code into an AST using `parse_tokens`.
    Raises a ParseError with detailed line/column context on failure.
    Returns a top-level `ast.Code` node containing the parsed statements.
    """
    # 1) Tokenize
    tokens = tokenize(source)
    if tokens and tokens[-1].kind == TokenKind.EOF:
        tokens.pop()  # remove EOF token to simplify

    # 2) Attempt to parse from the beginning
    result = parse_tokens(tokens)
    if not result:
        # The parser returned an empty ParserResult -> no match at index 0
        # We'll point to the first token or the start of the file
        offset = 0
        if tokens:
            offset = tokens[0].start  # character offset in source
        error_msg = create_error_context(source, offset)
        raise ParseError(error_msg)

    (next_index, node_list) = result

    # 3) If the parser did not consume all tokens, treat that as an error
    if next_index < len(tokens):
        # next_index is where parsing "stopped"
        # We'll point to the leftover token
        leftover_token = tokens[next_index]
        offset = leftover_token.start
        error_msg = create_error_context(source, offset)
        raise ParseError(
            error_msg
            + "\n\n"
            + f"Unexpected extra tokens starting at: {leftover_token}."
        )

    # 4) Wrap the parser result in a top-level Code node
    # `parse_tokens` is defined as `.many()`, so `node_list` should be a list of statements/expressions
    if not isinstance(node_list, list):
        node_list = [node_list]  # In case it returned a single node

    # Construct a top-level Code node for convenience
    return ast.Code(statements=node_list)
