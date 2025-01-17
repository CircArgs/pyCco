"""Uses Parsers to transform a token stream into an AST"""

from pycco.parser import any_of, anything, match, Parser
from pycco.tokens import Token, TokenKind, CKeywords

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


# Define grammar components
variable = type + star.optional() + name

_expression = Parser()  # Forward declaration for expressions


def expression():
    # Define the actual expression grammar here
    return variable | (
        open_paren >> _expression << close_paren
    )  # Example for simple expressions


_expression.parse_fn = (
    expression  # Set the parse function for the forward-declared parser
)

# Variable declaration: type [*] name;
variable_decl = variable + semicolon

# Statement placeholder
statement = variable_decl | _expression

# Function definition: return_type name(parameters) { body }
function = (
    type + name + (open_paren >> variable.until(comma.optional() + close_paren))
    << close_paren + open_bracket
    >> statement.until(close_bracket)
    << close_bracket
)

# # Parser starting point (placeholder)
# program = function.until(anything)
