# tests/test_tokenizers.py

import pytest
from pycco.tokenizers import (
    tokenize,
    Token,
    TokenKind,
    TokenizeError,
)


def test_simple_declaration():
    """Test tokenizing a basic C declaration: int x = 42;"""
    src = "int x = 42;"
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "int", 0, 2),
        Token(TokenKind.IDENTIFIER, "x", 4, 4),
        Token(TokenKind.OPERATOR, "=", 6, 6),
        Token(TokenKind.NUMBER, "42", 8, 9),
        Token(TokenKind.SYMBOL, ";", 10, 10),
        Token(TokenKind.EOF, None, 11, 11),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_function_definition():
    """Test tokenizing a basic C function definition."""
    src = "int main() {\n  return 0;\n}"
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "int", 0, 2),
        Token(TokenKind.IDENTIFIER, "main", 4, 7),
        Token(TokenKind.SYMBOL, "(", 8, 8),
        Token(TokenKind.SYMBOL, ")", 9, 9),
        Token(TokenKind.SYMBOL, "{", 11, 11),
        Token(TokenKind.KEYWORD, "return", 14, 19),
        Token(TokenKind.NUMBER, "0", 21, 21),
        Token(TokenKind.SYMBOL, ";", 22, 22),
        Token(TokenKind.SYMBOL, "}", 24, 24),
        Token(TokenKind.EOF, None, 25, 25),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_comment_and_whitespace():
    """Ensure comments and whitespace are skipped in final output."""
    src = "  // some comment\nint /*multi\n line*/ x;"
    tokens = tokenize(src)
    # Expected tokens: TYPE('int'), IDENTIFIER('x'), SYMBOL(';'), EOF
    expected = [
        Token(TokenKind.TYPE, "int", 19, 21),
        Token(TokenKind.IDENTIFIER, "x", 35, 35),
        Token(TokenKind.SYMBOL, ";", 36, 36),
        Token(TokenKind.EOF, None, 37, 37),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_numeric_literals():
    """Test tokenizing numeric literals: integers and floats."""
    src = "int a = 10;\nfloat b = 3.14;"
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "int", 0, 2),
        Token(TokenKind.IDENTIFIER, "a", 4, 4),
        Token(TokenKind.OPERATOR, "=", 6, 6),
        Token(TokenKind.NUMBER, "10", 8, 9),
        Token(TokenKind.SYMBOL, ";", 10, 10),
        Token(TokenKind.TYPE, "float", 12, 16),
        Token(TokenKind.IDENTIFIER, "b", 18, 18),
        Token(TokenKind.OPERATOR, "=", 20, 20),
        Token(TokenKind.NUMBER, "3.14", 22, 25),
        Token(TokenKind.SYMBOL, ";", 26, 26),
        Token(TokenKind.EOF, None, 27, 27),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_array_declaration():
    """Test tokenizing an array declaration like int arr[10];"""
    src = "int arr[10];"
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "int", 0, 2),
        Token(TokenKind.IDENTIFIER, "arr", 4, 6),
        Token(TokenKind.SYMBOL, "[", 7, 7),
        Token(TokenKind.NUMBER, "10", 8, 9),
        Token(TokenKind.SYMBOL, "]", 10, 10),
        Token(TokenKind.SYMBOL, ";", 11, 11),
        Token(TokenKind.EOF, None, 12, 12),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_operators_and_symbols():
    """Test tokenizing various operators and symbols."""
    src = "a += b * (c - d) / e;"
    tokens = tokenize(src)
    # Expected tokens: IDENTIFIER('a'), OPERATOR('+='),
    # IDENTIFIER('b'), OPERATOR('*'),
    # SYMBOL('('), IDENTIFIER('c'), OPERATOR('-'), IDENTIFIER('d'), SYMBOL(')'),
    # OPERATOR('/'), IDENTIFIER('e'), SYMBOL(';'), EOF
    expected = [
        Token(TokenKind.IDENTIFIER, "a", 0, 0),
        Token(TokenKind.OPERATOR, "+=", 2, 3),
        Token(TokenKind.IDENTIFIER, "b", 5, 5),
        Token(TokenKind.OPERATOR, "*", 7, 7),
        Token(TokenKind.SYMBOL, "(", 9, 9),
        Token(TokenKind.IDENTIFIER, "c", 10, 10),
        Token(TokenKind.OPERATOR, "-", 12, 12),
        Token(TokenKind.IDENTIFIER, "d", 14, 14),
        Token(TokenKind.SYMBOL, ")", 15, 15),
        Token(TokenKind.OPERATOR, "/", 17, 17),
        Token(TokenKind.IDENTIFIER, "e", 19, 19),
        Token(TokenKind.SYMBOL, ";", 20, 20),
        Token(TokenKind.EOF, None, 21, 21),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_complex_expression():
    """Test a more complex C expression: int result = (a + b) * c - 42 / d;"""
    src = "int result = (a + b) * c - 42 / d;"
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "int", 0, 2),
        Token(TokenKind.IDENTIFIER, "result", 4, 9),
        Token(TokenKind.OPERATOR, "=", 11, 11),
        Token(TokenKind.SYMBOL, "(", 13, 13),
        Token(TokenKind.IDENTIFIER, "a", 14, 14),
        Token(TokenKind.OPERATOR, "+", 16, 16),
        Token(TokenKind.IDENTIFIER, "b", 18, 18),
        Token(TokenKind.SYMBOL, ")", 19, 19),
        Token(TokenKind.OPERATOR, "*", 21, 21),
        Token(TokenKind.IDENTIFIER, "c", 23, 23),
        Token(TokenKind.OPERATOR, "-", 25, 25),
        Token(TokenKind.NUMBER, "42", 27, 28),
        Token(TokenKind.OPERATOR, "/", 30, 30),
        Token(TokenKind.IDENTIFIER, "d", 32, 32),
        Token(TokenKind.SYMBOL, ";", 33, 33),
        Token(TokenKind.EOF, None, 34, 34),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_string_literal():
    """Test tokenizing a string literal with escaped characters."""
    src = 'char *msg = "Hello, world!";'
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "char", 0, 3),
        Token(TokenKind.OPERATOR, "*", 5, 5),
        Token(TokenKind.IDENTIFIER, "msg", 6, 8),
        Token(TokenKind.OPERATOR, "=", 10, 10),
        Token(TokenKind.STRING_LITERAL, "Hello, world!", 12, 26),
        Token(TokenKind.SYMBOL, ";", 27, 27),
        Token(TokenKind.EOF, None, 28, 28),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_comment_and_whitespace_again():
    """Another check for comments and whitespace."""
    src = "/* multiline comment */  int   x;"
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "int", 25, 27),
        Token(TokenKind.IDENTIFIER, "x", 31, 31),
        Token(TokenKind.SYMBOL, ";", 32, 32),
        Token(TokenKind.EOF, None, 33, 33),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_pointer_declaration():
    """Test a pointer declaration: char *str;"""
    src = "char *str;"
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "char", 0, 3),
        Token(TokenKind.OPERATOR, "*", 5, 5),
        Token(TokenKind.IDENTIFIER, "str", 6, 8),
        Token(TokenKind.SYMBOL, ";", 9, 9),
        Token(TokenKind.EOF, None, 10, 10),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_unrecognized_token_error():
    """
    If we have an unrecognized token, it should raise TokenizeError with a descriptive message.
    """
    src = "int $$ invalid;"
    with pytest.raises(TokenizeError) as excinfo:
        _ = tokenize(src)
    assert "Unexpected token" in str(excinfo.value)
    assert "$" in str(excinfo.value)


def test_multiple_tokens():
    """Test a sequence with multiple types: float pi = 3.14; // Initialize pi => 6 tokens total, ignoring comment."""
    src = "float pi = 3.14; // Initialize pi"
    tokens = tokenize(src)
    # Expected tokens: TYPE('float'), IDENTIFIER('pi'), OPERATOR('='),
    # NUMBER('3.14'), SYMBOL(';'), EOF
    expected = [
        Token(TokenKind.TYPE, "float", 0, 4),
        Token(TokenKind.IDENTIFIER, "pi", 6, 7),
        Token(TokenKind.OPERATOR, "=", 9, 9),
        Token(TokenKind.NUMBER, "3.14", 11, 14),
        Token(TokenKind.SYMBOL, ";", 15, 15),
        Token(TokenKind.EOF, None, 17, 17),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_nested_comments():
    """
    '/* outer /* inner */ outer end */ int x;'
    Expect:
    1) COMMENT('/* outer /* inner */ outer end */')
    2) TYPE('int')
    3) IDENTIFIER('x')
    4) SYMBOL(';')
    5) EOF
    """
    src = "/* outer /* inner */ outer end */ int x;"
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.TYPE, "int", 32, 34),
        Token(TokenKind.IDENTIFIER, "x", 36, 36),
        Token(TokenKind.SYMBOL, ";", 37, 37),
        Token(TokenKind.EOF, None, 38, 38),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    for i, exp_tok in enumerate(expected):
        assert (
            tokens[i] == exp_tok
        ), f"Token {i} mismatch: expected {exp_tok}, got {tokens[i]}"


def test_only_whitespace():
    """
    Test tokenizing a string with only whitespace.
    We expect a single EOF token.
    """
    src = "   \n\t  "
    tokens = tokenize(src)
    expected = [
        Token(TokenKind.EOF, None, 6, 6),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    assert tokens[0] == expected[0], f"Expected EOF token, got {tokens[0]}"


def test_empty_input():
    """Test tokenizing an empty string."""
    src = ""
    tokens = tokenize(src)
    # Just EOF
    expected = [
        Token(TokenKind.EOF, None, 0, 0),
    ]
    assert len(tokens) == len(
        expected
    ), f"Expected {len(expected)} tokens, got {len(tokens)}"
    assert tokens[0] == expected[0], f"Expected EOF token, got {tokens[0]}"
