# tests/test_parsers.py

import pytest

from pycco.parsers import parse_c_code, ParseError
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
    StringLiteral,
)


def test_empty_source():
    """
    Parsing an empty string should result in an empty Code node
    with no statements.
    """
    source = ""
    ast = parse_c_code(source)
    assert isinstance(ast, Code)
    assert len(ast.statements) == 0


def test_simple_var_decl():
    """
    Test parsing of a single variable declaration: `int x;`
    """
    source = "int x;"
    ast = parse_c_code(source)
    assert isinstance(ast, Code)
    assert len(ast.statements) == 1
    decl = ast.statements[0]
    assert isinstance(decl, VariableDecl)
    assert decl.type.name == "int"
    assert decl.type.pointer is False
    assert decl.name.name == "x"
    assert decl.semicolon is True


def test_pointer_var_decl():
    """
    Test parsing a pointer declaration: `float* ptr;`
    """
    source = "float* ptr;"
    ast = parse_c_code(source)
    assert isinstance(ast, Code)
    assert len(ast.statements) == 1
    decl = ast.statements[0]
    assert isinstance(decl, VariableDecl)
    assert decl.type.name == "float"
    assert decl.type.pointer is True
    assert decl.name.name == "ptr"
    assert decl.semicolon is True


def test_assignment():
    """
    Test parsing a variable assignment: `x = 42;`
    """
    source = "int x; x = 42;"
    ast = parse_c_code(source)
    assert isinstance(ast, Code)
    assert len(ast.statements) == 2

    decl = ast.statements[0]
    assign = ast.statements[1]

    assert isinstance(decl, VariableDecl)
    assert isinstance(assign, Assign)
    assert isinstance(assign.var, Ident)
    assert assign.var.name == "x"
    assert isinstance(assign.to, Number)
    assert assign.to.value == "42"
    assert assign.to.type.name == "int"


def test_function_no_args():
    """
    Test parsing a simple function with no parameters:
        int main() { return 0; }
    """
    source = "int main() { return 0; }"
    ast = parse_c_code(source)

    assert isinstance(ast, Code)
    assert len(ast.statements) == 1
    func = ast.statements[0]
    assert isinstance(func, Function)

    # Check function's "var" (return type + function name)
    assert isinstance(func.var, VariableDecl)
    assert func.var.type.name == "int"
    assert isinstance(func.var.name, Ident)
    assert func.var.name.name == "main"

    # Check args
    assert len(func.args) == 0

    # Check body
    assert len(func.body) == 0
    assert isinstance(func.ret, Return)
    assert isinstance(func.ret.value, Number)
    assert func.ret.value.value == "0"


def test_function_with_args():
    """
    Test parsing a function with parameters:
        int sum(int a, int b) {
            return a;
        }
    (simplified for demonstration)
    """
    source = "int sum(int a, int b) { return a; }"
    ast = parse_c_code(source)
    assert isinstance(ast, Code)
    assert len(ast.statements) == 1

    func = ast.statements[0]
    assert isinstance(func, Function)
    # function var: "int sum"
    assert isinstance(func.var, VariableDecl)
    assert func.var.type.name == "int"
    assert func.var.type.pointer is False
    assert isinstance(func.var.name, Ident)
    assert func.var.name.name == "sum"

    # arguments
    assert len(func.args) == 2
    arg1, arg2 = func.args
    assert isinstance(arg1, Arg)
    assert arg1.type.name == "int"
    assert arg1.type.pointer is False
    assert isinstance(arg1.name, Ident)
    assert arg1.name.name == "a"

    assert isinstance(arg2, Arg)
    assert arg2.type.name == "int"
    assert arg2.type.pointer is False
    assert isinstance(arg2.name, Ident)
    assert arg2.name.name == "b"

    # Check body
    assert len(func.body) == 0
    assert isinstance(func.ret, Return)
    assert isinstance(func.ret.value, Ident)
    assert func.ret.value.name == "a"


def test_string_literal():
    """
    Test parsing a string literal expression: `char* s; s = "Hello";`
    """
    source = 'char* s; s = "Hello";'
    ast = parse_c_code(source)
    assert isinstance(ast, Code)
    assert len(ast.statements) == 2

    var_decl = ast.statements[0]
    assign = ast.statements[1]

    assert isinstance(var_decl, VariableDecl)
    assert var_decl.type.name == "char"
    assert var_decl.type.pointer is True
    assert var_decl.name.name == "s"

    assert isinstance(assign, Assign)
    assert isinstance(assign.var, Ident)
    assert assign.var.name == "s"
    assert isinstance(assign.to, StringLiteral)
    assert assign.to.value == "Hello"


def test_parse_error():
    """
    Test that an invalid snippet raises a ParseError.
    Example: missing semicolon or unmatched brace
    """
    source = "int x"
    with pytest.raises(ParseError):
        _ = parse_c_code(source)
