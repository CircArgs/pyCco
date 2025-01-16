"""Uses Parsers to tokenize C code"""
from typing import Callable, Iterable, List
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

string = (double_quote >> anything.until(double_quote) << double_quote) @ token_map(TokenKind.STRING_LITERAL)

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


def iter_tokenize(source: str) -> Iterable[Token]:
    """
    Tokenizes the input C code into a sequence of tokens, with enhanced error reporting.

    Args:
        source (str): The input C code as a string.

    Returns:
        Iterable[Token]: A sequence of Token objects.

    Raises:
        ValueError: If an unexpected token is encountered, with detailed error context.
    """
    index = 0
    lines = source.splitlines()  # Split the source into lines for better error reporting

    while index < len(source):
        try:
            next_index, token = tokenizer(source, index)
        except Exception as e:
            # General error handling (e.g., tokenizer malfunction)
            raise ValueError(f"Parsing error at index {index}: {e}")

        if next_index == -1:  # Parsing error
            # Determine the line and column of the error
            line_number = source[:index].count("\n")
            column_number = index - (source.rfind("\n", 0, index) + 1)

            # Context for error display
            before_error = "\n".join(lines[max(0, line_number - 2):line_number])  # Lines before the error
            error_line = lines[line_number]
            after_error = "\n".join(lines[line_number + 1:min(len(lines), line_number + 3)])  # Lines after the error

            # Pointer to the exact column of the error
            pointer = " " * column_number + "^"

            # Raise detailed error
            raise ValueError(
                f"Unexpected token at line {line_number + 1}, column {column_number + 1}:\n\n"
                f"{before_error}\n"
                f"{error_line}\n"
                f"{pointer}\n"
                f"{after_error}\n"
            )

        if token.kind != TokenKind.WHITESPACE:  # Optionally, skip whitespace tokens
            yield token

        index = next_index

def tokenize(source: str)->List[Token]:
    return list(iter_tokenize(source))