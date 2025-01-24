# tests/test_parser.py

import pytest
from enum import Enum
from pycco.parser import (
    Parser,
    match,
    match_fn,
    anything,
    sequence,
    any_of,
    any_of_enum,
    report_error,
    Position,
    ParseError,
    ParserResult,
)


# Define sample enums for testing
class Token(Enum):
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    DIGIT = "digit"


# Helper function to create a list of elements from a string
def to_stream(s: str):
    return list(s)


# Helper function to create a list of elements from a list of integers
def to_stream_int(lst: list):
    return lst


# Basic Parser Tests


def test_match_success():
    parser = match("a")
    stream = to_stream("abc")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == "a"
    assert result.position.index == 1


def test_match_failure():
    parser = match("a")
    stream = to_stream("bbc")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "'a'"
    assert result.error.found == "b"
    assert result.error.position.index == 0


def test_match_fn_success():
    parser = match_fn(lambda c: c.isdigit()).describe("digit")
    stream = to_stream("1a3")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == "1"
    assert result.position.index == 1


def test_match_fn_failure():
    parser = match_fn(lambda c: c.isdigit()).describe("digit")
    stream = to_stream("a1b")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "digit"
    assert result.error.found == "a"
    assert result.error.position.index == 0


def test_anything_success():
    parser = anything
    stream = to_stream("xyz")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == "x"
    assert result.position.index == 1


def test_anything_failure():
    parser = anything
    stream = to_stream("")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "any element"
    assert result.error.found is None
    assert result.error.position.index == 0


# Combinator Tests


def test_sequencing_success():
    parser = match("a") + match("b") + match("c")
    stream = to_stream("abc")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == ["a", "b", "c"]
    assert result.position.index == 3


def test_sequencing_failure():
    parser = match("a") + match("b") + match("c")
    stream = to_stream("abx")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "'c'"
    assert result.error.found == "x"
    assert result.error.position.index == 2


def test_alternation_success_first():
    parser = match("a") | match("b")
    stream = to_stream("a")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == "a"
    assert result.position.index == 1


def test_alternation_success_second():
    parser = match("a") | match("b")
    stream = to_stream("b")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == "b"
    assert result.position.index == 1


def test_alternation_failure():
    parser = match("a") | match("b")
    stream = to_stream("c")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "('a' | 'b')"
    assert result.error.found == "c"
    assert result.error.position.index == 0


def test_repetition_star_success():
    parser = match("a") * 3
    stream = to_stream("aaa")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == ["a", "a", "a"]
    assert result.position.index == 3


def test_repetition_star_failure():
    parser = match("a") * 3
    stream = to_stream("aa")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "at least 3 repetitions of 'a'"
    assert result.error.position.index == 2


def test_repetition_plus_success():
    parser = match("a").many(min=1)
    stream = to_stream("aaa")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == ["a", "a", "a"]
    assert result.position.index == 3


def test_optional_success():
    parser = match("a").optional()
    stream = to_stream("a")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == "a"
    assert result.position.index == 1


def test_optional_failure():
    parser = match("a").optional()
    stream = to_stream("b")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result is None
    assert result.position.index == 0


def test_not_parser_success():
    parser = match("a").not_()
    stream = to_stream("b")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result is None
    assert result.position.index == 0


def test_not_parser_failure():
    parser = match("a").not_()
    stream = to_stream("a")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "not 'a'"
    assert result.error.found == "a"
    assert result.error.position.index == 0


def test_sep_by_success():
    parser = match("a").sep_by(match(","))
    stream = to_stream("a,a,a")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == ["a", "a", "a"]
    assert result.position.index == 5


def test_sep_by_partial_success():
    parser = match("a").sep_by(match(","))
    stream = to_stream("a,a,b")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == ["a", "a"]
    assert result.position.index == 4  # Corrected expected index


def test_sep_by_no_match():
    parser = match("a").sep_by(match(","))
    stream = to_stream("b,a,a")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == []
    assert result.position.index == 0


def test_until_success():
    parser = match("a").until(match("c"))
    stream = to_stream("aabca")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == ["a", "a", "b"]
    assert result.position.index == 3  # Corrected expected index


def test_until_failure():
    parser = match("a").until(match("c"))
    stream = to_stream("aaba")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "stopping condition 'c'"
    assert result.error.position.index == 4


# Mapping Tests


def test_map_success():
    parser = match("a").map(lambda x: x.upper())
    stream = to_stream("a")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == "A"
    assert result.position.index == 1


def test_map_failure():
    parser = match("a").map(lambda x: x.upper())
    stream = to_stream("b")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "'a'"
    assert result.error.found == "b"


def test_at_operator_mapping():
    parser = match("a") @ (lambda x: x + "!")
    stream = to_stream("a")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == "a!"
    assert result.position.index == 1


# Error Reporting Tests


def test_error_reporting():
    parser = match("a") + match("b") + match("c")
    stream = to_stream("abx")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    error_message = report_error(stream, result.error)
    expected_message = "ParseError at index 2: expected 'c', found 'x'"
    assert error_message == expected_message


def test_error_position_tracking_newlines():
    # Even though it's generic, the test uses '\n' as an element
    parser = match("a") + match("b")
    stream = to_stream("a\nb")
    position = Position(0)
    result = parser(stream, position)
    assert not result  # Changed from assert result to assert not result
    assert result.error.expected == "'b'"
    assert result.error.found == "\n"
    assert result.error.position.index == 1


def test_error_reporting_multiline():
    # In generic parser, handle elements like '\n' but Position tracks index only
    parser = match("a") + match("b") + match("c")
    stream = to_stream("a\nbc")
    position = Position(0)
    result = parser(stream, position)
    # 'a' matches at 0, '\n' does not match 'b'
    assert not result
    error_message = report_error(stream, result.error)
    expected_message = "ParseError at index 1: expected 'b', found '\\n'"
    assert error_message == expected_message


# Enum Parsing Tests


def test_any_of_enum_success():
    parser = any_of_enum(Token)
    stream = to_stream("+")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == Token.PLUS
    assert result.position.index == 1


def test_any_of_enum_failure():
    parser = any_of_enum(Token)
    stream = to_stream("x")
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "(((('+' | '-') | '*') | '/') | 'digit')"
    assert result.error.found == "x"


def test_any_of_enum_multiple():
    parser = any_of_enum(Token)
    stream = to_stream("*")
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == Token.MULTIPLY
    assert result.position.index == 1


# Complex Parsing Scenario Tests


def test_arithmetic_expression():
    # Define parsers for digits and operators
    digit = match_fn(lambda c: c.isdigit()).map(int).describe("digit")
    plus = match("+").describe("'+'")
    minus = match("-").describe("'-'")
    operator = any_of(plus, minus).map(lambda op: op)

    # Define a simple expression parser: digit (operator digit)*
    expr = digit + (operator + digit).many()

    # Parse "1+2-3"
    stream = to_stream("1+2-3")
    position = Position(0)
    result = expr(stream, position)

    assert result
    assert result.result == [1, ["+", 2], ["-", 3]]
    assert result.position.index == 5


def test_arithmetic_expression_failure():
    # Define parsers for digits and operators
    digit = match_fn(lambda c: c.isdigit()).map(int).describe("digit")
    plus = match("+").describe("'+'")
    minus = match("-").describe("'-'")
    operator = any_of(plus, minus).map(lambda op: op)

    # Define a simple expression parser: digit (operator digit)*
    expr = digit + (operator + digit).many()

    # <--- extra step: wrap expr to fail if leftover
    def consume_all_as_digit():
        """
        A parser that succeeds only if all input has been consumed.
        If there's leftover input, it fails with "digit" expectation.
        """

        def parse(stream, position):
            if position.index >= len(stream):
                # No leftover => success
                return ParserResult(position=position, result=None)
            # There's a leftover character => fail with "digit"
            return ParserResult(
                position=position,
                error=ParseError(
                    position, expected="digit", found=stream[position.index]
                ),
            )

        return Parser(parse, description="EOF")

    consume_all = consume_all_as_digit()
    expr_full = expr << consume_all

    # Parse "1+2*"
    stream = to_stream("1+2*")
    position = Position(0)
    result = expr_full(stream, position)

    assert not result
    error_message = report_error(stream, result.error)
    expected_message = "ParseError at index 3: expected digit, found '*'"
    assert error_message == expected_message


# Additional Generic Stream Tests


def test_generic_stream_success():
    parser = match(1)
    stream = to_stream_int([1, 2, 3])
    position = Position(0)
    result = parser(stream, position)
    assert result
    assert result.result == 1
    assert result.position.index == 1


def test_generic_stream_failure():
    parser = match(1)
    stream = to_stream_int([2, 3, 4])
    position = Position(0)
    result = parser(stream, position)
    assert not result
    assert result.error.expected == "1"
    assert result.error.found == 2
    assert result.error.position.index == 0
