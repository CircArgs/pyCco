"""Uses Parsers to tokenize C code"""
from typing import Callable, Iterable
from string import digits, ascii_letters
from pycco.parser import any_of, anything, match
from pycco.tokens import Token, TokenKind, CKeywords
from pycco.utils import flatten, NestedStr


def token_map(kind: TokenKind) -> Callable[[NestedStr], Token]:
    def create_token(result: NestedStr):
        return Token(kind, flatten(result))
    return create_token


# Basic parsers
whitespace = +any_of(' ', '\n', '\t', '\r') @ token_map(TokenKind.WHITESPACE)
letter = any_of(*ascii_letters)
digit = any_of(*digits)
underscore = '_'
double_quote = match('"')
single_quote = "'"

# Token parsers
identifier = (letter + (letter | digit | underscore).many()) @ token_map(TokenKind.IDENTIFIER)

string = (double_quote >> anything.many() << double_quote) @ token_map(TokenKind.STRING_LITERAL)

keywords = any_of(*[match(keyword) for keyword in CKeywords]) @ token_map(TokenKind.KEYWORD)

symbols = any_of("{", "}", "(", ")", ";", ",", "[", "]", ".", "->") @ token_map(TokenKind.SYMBOL)

operators = any_of(
    "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!",
    "&", "|", "^", "<<", ">>"
) @ token_map(TokenKind.OPERATOR)

number = (
    (digit.many(1) + (match(".") + digit.many(1)).optional())  # Match integers and floats
    @ token_map(TokenKind.NUMBER)
)

single_line_comment = (match("//") >> anything.until(match("\n"))) @ token_map(TokenKind.COMMENT)

multi_line_comment = (match("/*") >> anything.until(match("*/"))) @ token_map(TokenKind.COMMENT)

comments = single_line_comment | multi_line_comment

# Unified tokenizer parser
tokenizer = (
    whitespace |
    comments |
    keywords |
    symbols |
    operators |
    number |
    string |
    identifier
)


# Tokenizing function
def tokenize(source: str) -> Iterable[Token]:
    """
    Tokenizes the input C code into a sequence of tokens.

    Args:
        source (str): The input C code as a string.

    Returns:
        Iterable[Token]: A sequence of Token objects.
    """
    ret = []
    lines = source.splitlines()
    for line_number, line in enumerate(lines):
        index = 0
        while index < len(line):
            next_index, token = tokenizer(line, index)
            if next_index == -1:  # Parsing error
                before = "\n".join(lines[-5:line_number+1])
                pointer = " "*index+"^"
                after = "\n".join(lines[line_number+2:])
                raise ValueError(f"""Unexpected token at {line_number}:{index}:
{before}
{pointer}
{after}
""")
            if token.kind != TokenKind.WHITESPACE:
                ret.append(token)

            index = next_index
    return ret