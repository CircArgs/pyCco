"""Uses Parsers to tokenize C code"""

from typing import Callable, Iterable, List
from string import digits, ascii_letters
from pycco.parser import any_of, anything, match, any_of_enum
from pycco.tokens import (
    Token,
    TokenKind,
    CKeywords,
    CTypes,
    BinaryOperator,
    UnaryOperator,
    OtherOperator,
    Symbol,
)
from pycco.utils import flatten_str, NestedStr


class TokenizeError(ValueError): ...


def token_map(kind: TokenKind) -> Callable[[NestedStr], Token]:
    def create_token(result: NestedStr, index: int):
        flattened = flatten_str(result)
        return Token(kind, flattened, index, index + len(flattened) - 1)

    return create_token


# Basic parsers
whitespace = +any_of(" ", "\n", "\t", "\r") @ token_map(TokenKind.WHITESPACE)
whitespace.set_description("whitespace")

letter = any_of(*ascii_letters).set_description("letter")
digit = any_of(*digits).set_description("digit")
underscore = match("_")
double_quote = match('"')
single_quote = "'"

# Token parsers
identifier = (letter + (letter | digit | underscore).many()) @ token_map(
    TokenKind.IDENTIFIER
)
identifier.set_description("identifier")


string = (double_quote >> anything.until(double_quote) << double_quote) @ token_map(
    TokenKind.STRING_LITERAL
)
string.set_description("string literal")


keywords = any_of_enum(CKeywords) @ token_map(TokenKind.KEYWORD)
keywords.set_description("keywords")

types = any_of_enum(CTypes) @ token_map(TokenKind.TYPE)
types.set_description("keywords")

symbols = any_of_enum(Symbol).freeze_description() @ token_map(TokenKind.SYMBOL)


operators = any_of_enum(
    BinaryOperator, UnaryOperator, OtherOperator
).freeze_description() @ token_map(TokenKind.OPERATOR)

number = (
    (
        digit.many(1) + (match(".") + digit.many(1)).optional()
    )  # Match integers and floats
    @ token_map(TokenKind.NUMBER)
).set_description("number")

single_line_comment = (match("//") >> anything.until(match("\n"))) @ token_map(
    TokenKind.COMMENT
)

multi_line_comment = (
    match("/*") >> anything.until(match("*/")) << match("*/")
) @ token_map(TokenKind.COMMENT)

comments = single_line_comment | multi_line_comment
comments.set_description("comment")

tokenizer = any_of(
    whitespace,
    comments,
    keywords,
    types,
    symbols,
    operators,
    number,
    string,
    identifier,
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
    lines = (
        source.splitlines()
    )  # Split the source into lines for better error reporting

    while index < len(source):

        tokenized = tokenizer(source, index)
        next_index, token = tokenized
        description = tokenized.description

        if next_index == -1:  # Parsing error
            # Determine the line and column of the error
            line_number = source[:index].count("\n")
            column_number = index - (source.rfind("\n", 0, index) + 1)

            # Context for error display
            before_error = "\n".join(
                lines[max(0, line_number - 2) : line_number]
            )  # Lines before the error
            error_line = lines[line_number]
            after_error = "\n".join(
                lines[line_number + 1 : min(len(lines), line_number + 3)]
            )  # Lines after the error

            # Pointer to the exact column of the error
            pointer = " " * column_number + "^"

            # Raise detailed error
            raise TokenizeError(
                f"Tokenization Error:\n"
                f"Unexpected token at line {line_number + 1}, column {column_number + 1}:\n\n"
                f"{before_error}\n"
                f"{error_line}\n"
                f"{pointer}\n"
                f"{after_error}\n"
                f"Expected {description}.\n"
            )

        if token.kind not in (TokenKind.WHITESPACE, TokenKind.COMMENT):
            yield token

        index = next_index

    yield Token(TokenKind.EOF, start=index, end=index)


def tokenize(source: str) -> List[Token]:
    return list(iter_tokenize(source))
