from typing import Callable, Iterable, List
from string import ascii_letters, digits

from pycco.parser import (
    Parser,
    ParserResult,
    ParseError,
    Position,
    any_of,
    anything,
    match,
)
from pycco.tokens import (
    Token,
    TokenKind,
    CKeywords,
    CTypes,
    MultiCharOperators,
    SingleCharOperators,
    Symbols,
)
from pycco.utils import flatten_str, NestedStr


# -------------------------------------------------------------------------
# MATCHING MULTI-CHAR STRINGS
# -------------------------------------------------------------------------
def match_str(s: str) -> Parser[str, str]:
    """
    Match the exact string s at the current position.
    """

    def parse_fn(stream: List[str], position: Position) -> ParserResult[str]:
        start_idx = position.index
        end_idx = start_idx + len(s)
        if end_idx > len(stream):
            return ParserResult(
                position=position, error=ParseError(position, repr(s), None)
            )
        for i, ch in enumerate(s):
            if stream[start_idx + i] != ch:
                return ParserResult(
                    position=position,
                    error=ParseError(position, repr(s), stream[start_idx + i]),
                )
        return ParserResult(position=Position(end_idx), result=s)

    return Parser(parse_fn, description=repr(s))


# -------------------------------------------------------------------------
# END-OF-FILE PARSER
# -------------------------------------------------------------------------
@Parser
def eof_parser(stream: List[str], position: Position) -> ParserResult[str]:
    """
    A parser that succeeds only if we're at end of file (no more characters).
    """
    if position.index >= len(stream):
        return ParserResult(position=position, result="EOF")
    return ParserResult(
        position=position, error=ParseError(position, "EOF", stream[position.index])
    )


newline_or_eof = match_str("\n") | eof_parser.set_description("EOF")


# -------------------------------------------------------------------------
# ANY-OF-ENUM-STR
# -------------------------------------------------------------------------
def any_of_enum_str(*enum_classes) -> Parser[str, str]:
    """
    Similar to any_of_enum but uses match_str(...) for each .value.
    Allows multi-char tokens like '+=' or '/*' to match as single tokens.
    """
    choices = []
    for enum_cls in enum_classes:
        for e in enum_cls:
            choices.append(match_str(e.value))
    return any_of(*choices)


# -------------------------------------------------------------------------
# TOKEN MAP
# -------------------------------------------------------------------------
def token_map(kind: TokenKind) -> Callable[[str, Position], Token]:
    """
    Maps a parsed string + current position -> Token of a given kind.
    """

    def create_token(value: NestedStr, pos: Position) -> Token:
        end = pos.index - 1
        start = end - (len(value) - 1)
        flattened = flatten_str(value)
        return Token(kind, flattened, start, end)

    return create_token


# -------------------------------------------------------------------------
# BASIC PARSERS
# -------------------------------------------------------------------------
space_char = any_of(" ", "\t", "\n", "\r").set_description("space char")
whitespace = (+space_char) @ token_map(TokenKind.WHITESPACE)
whitespace.set_description("whitespace")

letter = any_of(*ascii_letters).set_description("letter")
digit = any_of(*digits).set_description("digit")
underscore = match("_").set_description("underscore")

identifier_body = letter + (letter | digit | underscore).many()
identifier = identifier_body.map(lambda lst, pos: flatten_str(lst)) @ token_map(
    TokenKind.IDENTIFIER
)
identifier.set_description("identifier")


# -------------------------------------------------------------------------
# KEYWORDS & TYPES
# -------------------------------------------------------------------------
keywords = any_of_enum_str(CKeywords) @ token_map(TokenKind.KEYWORD)
keywords.set_description("keyword")

types = any_of_enum_str(CTypes) @ token_map(TokenKind.TYPE)
types.set_description("type")


# -------------------------------------------------------------------------
# COMMENTS
# -------------------------------------------------------------------------
# 1) SINGLE-LINE COMMENT: "//" until newline OR EOF
single_line_comment = (match_str("//") >> anything.until(newline_or_eof)) @ token_map(
    TokenKind.COMMENT
)


# 2) MULTI-LINE COMMENT: "/*" ... until the last "*/"
def parse_nested_comment(stream: List[str], position: Position) -> ParserResult[str]:
    """
    Matches from '/*' up to the LAST '*/' in the remainder of the file.
    This ensures nested comments are recognized as one big comment,
    per the 'test_nested_comments' test.
    """
    start_idx = position.index
    # Confirm '/*' at current position
    if (
        start_idx + 2 > len(stream)
        or stream[start_idx] != "/"
        or stream[start_idx + 1] != "*"
    ):
        return ParserResult(position=position, error=ParseError(position, "'/*'", None))
    joined = "".join(stream)
    # Find the last occurrence of '*/'
    last_index = joined.rfind("*/", start_idx + 2)
    if last_index == -1:
        # No closing '*/' found
        return ParserResult(
            position=position, error=ParseError(position, "final '*/'", None)
        )
    # Extract the entire comment
    comment_text = joined[start_idx : last_index + 2]
    end_pos = last_index + 2
    return ParserResult(position=Position(end_pos), result=comment_text)


multi_line_comment = Parser(parse_nested_comment).map(lambda c, pos: c) @ token_map(
    TokenKind.COMMENT
)

comments = single_line_comment | multi_line_comment
comments.set_description("comment")


# -------------------------------------------------------------------------
# OPERATORS & SYMBOLS
# -------------------------------------------------------------------------
# Create parsers for multi-character operators, single-character operators, and symbols
multi_char_ops_parser = any_of_enum_str(MultiCharOperators)
single_char_ops_parser = any_of_enum_str(SingleCharOperators)
symbols_parser = any_of_enum_str(Symbols)

# Combine them, ensuring multi-character operators are matched first
op_syms_parser = any_of(
    multi_char_ops_parser,
    single_char_ops_parser,
    symbols_parser,
)


# Map to TokenKind.OPERATOR or TokenKind.SYMBOL accordingly
def classify_op_sym(value: str) -> TokenKind:
    if value in {e.value for e in MultiCharOperators} or value in {
        e.value for e in SingleCharOperators
    }:
        return TokenKind.OPERATOR
    elif value in {e.value for e in Symbols}:
        return TokenKind.SYMBOL
    return TokenKind.OPERATOR  # Fallback


def op_sym_token_mapper(data: tuple[str], pos: Position) -> Token:
    value = data[0]
    kind = classify_op_sym(value)
    end = pos.index - 1
    start = end - (len(value) - 1)
    return Token(kind, value, start, end)


op_syms_mapped = op_syms_parser.map(lambda val, pos: (val,)) @ op_sym_token_mapper
op_syms_mapped.set_description("operator or symbol")

operators_and_symbols = op_syms_mapped


# -------------------------------------------------------------------------
# NUMBER
# -------------------------------------------------------------------------
int_part = +digit
frac_part = match_str(".") + (+digit)
number_parser = int_part + frac_part.optional()
number_parser = number_parser.map(lambda parts, pos: flatten_str(parts))
number = number_parser @ token_map(TokenKind.NUMBER)
number.set_description("number")


# -------------------------------------------------------------------------
# STRING LITERAL
# -------------------------------------------------------------------------
string_body = anything.until(match_str('"'))
string_literal = (match_str('"') >> string_body << match_str('"')).map(
    lambda chars, pos: flatten_str(chars)
) @ token_map(TokenKind.STRING_LITERAL)
string_literal.set_description("string literal")


# -------------------------------------------------------------------------
# COMBINED TOKENIZER
# The order is crucial: whitespace, comments, keywords, types, operators/symbols, number, string_literal, identifier
# -------------------------------------------------------------------------
tokenizer = any_of(
    whitespace,
    comments,
    keywords,
    types,
    operators_and_symbols,
    number,
    string_literal,
    identifier,
)


class TokenizeError(ValueError):
    pass


def iter_tokenize(source: str) -> Iterable[Token]:
    """
    Tokenizes the input source, skipping whitespace/comments.
    Raises TokenizeError on parse failure.
    """
    stream = [*source, "\n"]
    index = 0
    lines = source.splitlines()

    while index < len(stream):
        result = tokenizer(stream, Position(index))
        if not result:
            # Parse error
            err = result.error
            line_number = source[:index].count("\n")
            last_nl = source.rfind("\n", 0, index)
            column_number = index - (last_nl + 1) if last_nl != -1 else index

            before_error = "\n".join(lines[max(0, line_number - 2) : line_number])
            error_line = lines[line_number] if line_number < len(lines) else ""
            after_error = "\n".join(lines[line_number + 1 : line_number + 3])
            pointer = " " * column_number + "^"

            raise TokenizeError(
                f"Tokenization Error:\n"
                f"Unexpected token at line {line_number + 1}, column {column_number + 1}:\n\n"
                f"{before_error}\n{error_line}\n{pointer}\n{after_error}\n"
                f"Expected {err.expected}, found {repr(err.found)}.\n"
            )

        index = result.position.index
        tok = result.result
        if tok.kind not in (TokenKind.WHITESPACE, TokenKind.COMMENT):
            yield tok

    # Emit EOF token
    yield Token(TokenKind.EOF, start=index, end=index)


def tokenize(source: str) -> List[Token]:
    return list(iter_tokenize(source))
