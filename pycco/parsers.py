from pycco.tokenizers import *
from pycco.ast import *
from pycco.parsers import *
from inspect import signature
from functools import reduce

"""Uses Parsers to transform a token stream into an AST"""
from typing import List, Optional
from pycco.parser import match, Parser, parser, S, T, U, ParserResult
from pycco.tokens import Token, TokenKind, BinaryOperator, UnaryOperator, OtherOperator
from pycco import ast

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
expression = Parser()
statement = Parser()

# Grammar Components with Map Functions

def map_ident(token: Token)->ast.Ident:
    return ast.Ident(token.value).set_tokens([token])

ident = name @ map_ident

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
variable_assign@=map_variable_assign

def map_return(expr: ast.Expression)->ast.Return:
    return ast.Return(expr)

return_ = match(Token(TokenKind.KEYWORD, 'return')) >> expression << semicolon
return_ @= map_return

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
function @= map_function

# Unary Operator Parser
def map_unary_op(elements: List[Token | ast.Expression]) -> ast.UnaryOp:
    op, operand = elements
    return ast.UnaryOp(operator=op.value, operand=operand)

unary_op = (any_of_enum(UnaryOperator) + expression) @ map_unary_op