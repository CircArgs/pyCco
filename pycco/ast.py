from dataclasses import field, dataclass, fields
import typing
from typing import (
    List,
    Optional,
    TypeVar,
    Iterator,
    Any,
    Callable,
    Type as TypeType,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from itertools import zip_longest, chain
from copy import deepcopy
from pycco.utils import flatten
from abc import ABC, abstractmethod
from pycco.parser import Result, PyCcoError, PyCcoParseError
from textwrap import indent

if TYPE_CHECKING:
    from pycco.visitor import Visitor, CompilationContext


def get_compiler_backend(backend: str) -> Tuple["CompilationContext", "Visitor"]:
    if backend.strip() in ("arm_v8", "armv8", "arm8"):
        from pycco.backends.armv8 import compiler

        return compiler()
    raise ValueError(f"Unknown backend `{backend}`.")


PRIMITIVES = {int, float, str, bool, type(None)}


# typevar used for node methods that return self
# so the typesystem can correlate the self type with the return type
TNode = TypeVar("TNode", bound="Node")  # pylint: disable=C0103


class PyCcoASTError(PyCcoError): ...


class PyCcoNodeValidateError(PyCcoASTError):
    """
    Raised by a node during `validate`
    """

    def __init__(self, message, node):
        self.message = message
        self.node = node

    def __str__(self):
        return self.message


class PyCcoNodeTypeError(PyCcoASTError):
    """
    Raised when a field in a PyCco AST node does not match its expected type.
    """

    def __init__(
        self, node: "Node", field_name: str, expected_type: type, actual_value: Any
    ):
        self.node = node
        self.node_type = node.__class__
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.actual_type = type(actual_value).__name__

        super().__init__(
            self._generate_message(),
        )

    def _generate_message(self) -> str:
        """
        Generate a detailed error message.
        """
        value = (
            self.actual_value
            if not isinstance(self.actual_value, list)
            else "[" + ", ".join(map(str, self.actual_value)) + "]"
        )
        return (
            f"Type mismatch for '{self.field_name}' of {self.node_type.__name__}:\n"
            f"  Expected type: {self._prettify_type(self.expected_type)}\n"
            f"  Got value: {value} (type: {self._prettify_type(self.actual_type)})"
        )

    @staticmethod
    def _prettify_type(type_) -> str:
        """
        Clean up type names for better readability.
        Handles Union, ForwardRef, and other complex types.
        """
        from typing import Union, get_origin, get_args, ForwardRef

        # Handle simple string types (for ForwardRefs or direct strings)
        if isinstance(type_, str):
            return type_

        # Handle Union types
        origin = get_origin(type_)
        if origin is Union:
            args = get_args(type_)
            prettified_args = ", ".join(
                PyCcoNodeTypeError._prettify_type(arg) for arg in args
            )
            return f"one of {prettified_args}"

        # Handle ForwardRefs (e.g., "MyNode")
        if isinstance(type_, ForwardRef):
            return type_.__forward_arg__

        # Handle generic collections like List, Dict
        if origin is not None:
            args = get_args(type_)
            prettified_args = ", ".join(
                PyCcoNodeTypeError._prettify_type(arg) for arg in args
            )
            return f"{origin.__name__}[{prettified_args}]"

        # Handle built-in and standard types (int, str, etc.)
        if hasattr(type_, "__name__"):
            return type_.__name__

        # Fallback to string representation for unknown types
        return str(type_)


class Node(ABC):
    """Base class for all PyCco AST nodes.

    PyCco nodes are python dataclasses with the following patterns:
        - Attributes are either
            - PRIMITIVES (int, float, str, bool, None)
            - iterable from (list, tuple, set)
            - Enum
            - descendant of `Node`
        - Attributes starting with '_' are "obfuscated" and are not included in `children`


    Tokens
        Node attributes should be listed in the order of their token sublists
        otherwise if not possible to guarantee, tokens MUST be set on the node
    """

    parent: Optional["Node"] = None
    parent_key: Optional[str] = None

    _is_compiled: bool = False
    _hooks: List[Callable[["Node", PyCcoNodeTypeError], None]] = []

    def __post_init__(self):
        self.add_self_as_parent()
        # try:

        #     # self.validate_field_types()
        # except PyCcoNodeTypeError as e:
        #     self._run_hooks(e)

    @property
    def parse_result(self) -> "Result":
        return self._result

    def set_parse_result(self, res: "Result"):
        """
        Set the result from parsing on the node
        """
        self._result = res

    @classmethod
    def add_hook(cls, hook: Callable[["Node", Exception], None]) -> None:
        """
        Add a class-level hook to be executed if an exception occurs in __post_init__.

        Args:
            hook (Callable[["Node", Exception], None]): A callback function that
                                                       takes the node instance and an exception.
        """
        if not callable(hook):
            raise ValueError("Hook must be a callable that accepts (self, exception).")
        cls._hooks.append(hook)

    @classmethod
    def _run_hooks(cls, exception: Exception) -> None:
        """
        Run all registered hooks, passing the current instance and the exception.

        Args:
            exception (Exception): The exception raised during __post_init__.
        """
        for hook in cls._hooks:
            hook(exception)

    def validate_field_types(self):
        """
        Validates that the fields of the node match their expected types.
        Raises:
              PyCcoNodeTypeError: If any field does not match its expected type.
        """
        for field in self.__dataclass_fields__.values():
            field_name = field.name
            expected_type = field.type
            actual_value = getattr(self, field_name)

            # Skip obfuscated fields starting with '_'
            if field_name.startswith("_"):
                continue

            # Check type compatibility
            if not self.is_type_compatible(actual_value, expected_type):
                raise PyCcoNodeTypeError(
                    node=self,
                    field_name=field_name,
                    expected_type=expected_type,
                    actual_value=actual_value,
                )

    def validate_ast(self):
        """
        Children can validate themselves within the ast
        """
        self.validate()
        for child in self.children:
            child.validate_ast()

    def validate(self):
        """
        Validate this node
        """

    @staticmethod
    def is_type_compatible(value, expected_type):
        """
        Checks if a value matches the expected type.
        """
        # Handle ForwardRef by comparing type names
        if isinstance(expected_type, typing.ForwardRef):
            return type(value).__name__ == expected_type.__forward_arg__

        # Handle Union types (Optional is a Union with NoneType)
        if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
            return any(
                Node.is_type_compatible(value, t) for t in expected_type.__args__
            )

        # Handle lists, tuples, sets, etc.
        if hasattr(expected_type, "__origin__"):
            if isinstance(value, expected_type.__origin__):
                # Check if inner elements match the expected type
                if expected_type.__args__:
                    return all(
                        Node.is_type_compatible(v, expected_type.__args__[0])
                        for v in value
                    )
                return True

        # Handle Node descendants and primitives
        if isinstance(value, expected_type):
            return True

        # Handle NoneType (None)
        if value is None and expected_type is type(None):
            return True

        return False

    @property
    def depth(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.depth + 1

    def clear_parent(self: TNode) -> TNode:
        """
        Remove parent from the node
        """
        self.parent = None
        return self

    def set_parent(self: TNode, parent: "Node", parent_key: str) -> TNode:
        """
        Add parent to the node
        """
        self.parent = parent
        self.parent_key = parent_key
        return self

    def add_self_as_parent(self: TNode) -> TNode:
        """
        Adds self as a parent to all children
        """
        for name, child in self.fields(
            flat=True,
            nodes_only=True,
            obfuscated=False,
            nones=False,
            named=True,
        ):
            child.set_parent(self, name)
        return self

    def __setattr__(self, key: str, value: Any):
        """
        Facilitates setting children using `.` syntax ensuring parent is attributed
        """
        if key == "parent":
            object.__setattr__(self, key, value)
            return

        object.__setattr__(self, key, value)
        for child in flatten(value):
            if isinstance(child, Node) and not key.startswith("_"):
                child.set_parent(self, key)

    def swap(self: TNode, other: "Node") -> TNode:
        """
        Swap the Node for another
        """
        if not (self.parent and self.parent_key):
            return self
        parent_attr = getattr(self.parent, self.parent_key)
        if parent_attr is self:
            setattr(self.parent, self.parent_key, other)
            return self.clear_parent()

        new = []
        for iterable_type in (list, tuple, set):
            if isinstance(parent_attr, iterable_type):
                for element in parent_attr:
                    if self is element:
                        new.append(other)
                    else:
                        new.append(element)
                new = iterable_type(new)
                break

        setattr(self.parent, self.parent_key, new)
        return self.clear_parent()

    def copy(self: TNode) -> TNode:
        """
        Create a deep copy of the `self`
        """
        return deepcopy(self)

    def get_nearest_parent_of_type(
        self: "Node",
        node_type: TypeType[TNode],
    ) -> Optional[TNode]:
        """
        Traverse up the tree until you find a node of `node_type` or hit the root
        """
        if isinstance(self.parent, node_type):
            return self.parent
        if self.parent is None:
            return None
        return self.parent.get_nearest_parent_of_type(node_type)

    def get_furthest_parent(
        self: "Node",
    ) -> Optional[TNode]:
        """
        Traverse up the tree until you find a node of `node_type` or hit the root
        """
        if self.parent is None:
            return None
        curr_parent = self.parent
        while True:
            if curr_parent.parent is None:
                return curr_parent
            curr_parent = curr_parent.parent

    def flatten(self) -> Iterator["Node"]:
        """
        Flatten the sub-ast of the node as an iterator
        """
        return self.filter(lambda _: True)

    # pylint: disable=R0913
    def fields(
        self,
        flat: bool = True,
        nodes_only: bool = True,
        obfuscated: bool = False,
        nones: bool = False,
        named: bool = False,
    ) -> Iterator:
        """
        Returns an iterator over fields of a node with particular filters

        Args:
            flat: return a flattened iterator (if children are iterable)
            nodes_only: do not yield children that are not Nodes (trumped by `obfuscated`)
            obfuscated: yield fields that have leading underscores
                (typically accessed via a property)
            nones: yield values that are None
                (optional fields without a value); trumped by `nodes_only`
            named: yield pairs `(field name: str, field value)`
        Returns:
            Iterator: returns all children of a node given filters
                and optional flattening (by default Iterator[Node])
        """

        def make_child_generator():
            """
            Makes a generator enclosing self to return
            not obfuscated fields (fields without starting `_`)
            """
            for self_field in fields(self):
                if (
                    not self_field.name.startswith("_") if not obfuscated else True
                ) and (self_field.name in self.__dict__):
                    value = self.__dict__[self_field.name]
                    values = [value]
                    if flat:
                        values = flatten(value)
                    for value in values:
                        if named:
                            yield (self_field.name, value)
                        else:
                            yield value

        # `iter`s used to satisfy mypy (`child_generator` type changes between generator, filter)
        child_generator = iter(make_child_generator())

        if nodes_only:
            child_generator = iter(
                filter(
                    lambda child: (
                        isinstance(child, Node)
                        if not named
                        else isinstance(child[1], Node)
                    ),
                    child_generator,
                ),
            )

        if not nones:
            child_generator = iter(
                filter(
                    lambda child: (
                        (child is not None) if not named else (child[1] is not None)
                    ),
                    child_generator,
                ),
            )  # pylint: disable=C0301

        return child_generator

    @property
    def children(self) -> Iterator["Node"]:
        """
        Returns an iterator of all nodes that are one
        step from the current node down including through iterables
        """
        return self.fields(
            flat=True,
            nodes_only=True,
            obfuscated=False,
            nones=False,
            named=False,
        )

    def replace(  # pylint: disable=invalid-name
        self,
        from_: "Node",
        to: "Node",
        compare: Optional[Callable[[Any, Any], bool]] = None,
        times: int = -1,
        copy: bool = True,
    ):
        """
        Replace a node `from_` with a node `to` in the subtree
        """
        replacements = 0
        compare_ = (lambda a, b: a is b) if compare is None else compare
        for node in self.flatten():
            if compare_(node, from_):
                other = to.copy() if copy else to
                node.swap(other)
                replacements += 1
            if replacements == times:
                return

    def filter(self, func: Callable[["Node"], bool]) -> Iterator["Node"]:
        """
        Find all nodes that `func` returns `True` for
        """
        if func(self):
            yield self

        for node in chain(*[child.filter(func) for child in self.children]):
            yield node

    def contains(self, other: "Node") -> bool:
        """
        Checks if the subtree of `self` contains the node
        """
        return any(self.filter(lambda node: node is other))

    def is_ancestor_of(self, other: Optional["Node"]) -> bool:
        """
        Checks if `self` is an ancestor of the node
        """
        return bool(other) and other.contains(self)

    def find_all(self, node_type: TypeType[TNode]) -> Iterator[TNode]:
        """
        Find all nodes of a particular type in the node's sub-ast
        """
        return self.filter(lambda n: isinstance(n, node_type))  # type: ignore

    def apply(self, func: Callable[["Node"], None]):
        """
        Traverse ast and apply func to each Node
        """
        func(self)
        for child in self.children:
            child.apply(func)

    def compare(
        self,
        other: "Node",
    ) -> bool:
        """
        Compare two ASTs for deep equality
        """
        if type(self) != type(other):  # pylint: disable=unidiomatic-typecheck
            return False
        if id(self) == id(other):
            return True
        return hash(self) == hash(other)

    def diff(self, other: "Node") -> List[Tuple["Node", "Node"]]:
        """
        Compare two ASTs for differences and return the pairs of differences
        """

        def _diff(self, other: "Node"):
            if self != other:
                diffs.append((self, other))
            else:
                for child, other_child in zip_longest(self.children, other.children):
                    _diff(child, other_child)

        diffs: List[Tuple["Node", "Node"]] = []
        _diff(self, other)
        return diffs

    def similarity_score(self, other: "Node") -> float:
        """
        Determine how similar two nodes are with a float score
        """
        self_nodes = list(self.flatten())
        other_nodes = list(other.flatten())
        intersection = [
            self_node for self_node in self_nodes if self_node in other_nodes
        ]
        union = (
            [self_node for self_node in self_nodes if self_node not in intersection]
            + [
                other_node
                for other_node in other_nodes
                if other_node not in intersection
            ]
            + intersection
        )
        return len(intersection) / len(union)

    def __eq__(self, other) -> bool:
        """
        Compares two nodes for "top level" equality.

        Checks for type equality and primitive field types for full equality.
        Compares all others for type equality only. No recursing.
        Note: Does not check (sub)AST. See `Node.compare` for comparing (sub)ASTs.
        """
        return type(self) == type(other) and all(  # pylint: disable=C0123
            (
                s == o
                if type(s) in PRIMITIVES  # pylint: disable=C0123
                else type(s) == type(o)
            )  # pylint: disable=C0123
            for s, o in zip(
                (self.fields(False, False, False, True)),
                (other.fields(False, False, False, True)),
            )
        )

    def __hash__(self) -> int:
        """
        Hash a node
        """
        return hash(
            tuple(
                chain(
                    (type(self),),
                    self.fields(
                        flat=True,
                        nodes_only=False,
                        obfuscated=False,
                        nones=True,
                        named=False,
                    ),
                ),
            ),
        )

    @abstractmethod
    def __str__(self) -> str:
        """
        Get the string of a node
        """
        raise NotImplementedError()

    def compile(self, backend: str):
        ctx, visitor = get_compiler_backend(backend)
        return visitor(ctx, self)


@dataclass
class Statement(Node):
    """
    Base class for all AST nodes that represent statements (i.e., no return value).
    These typically appear inside a function body, blocks, or control-flow structures.
    """

    pass


@dataclass
class Expression(Node):
    """
    Base class for all AST nodes that produce a value when evaluated.
    Subclasses must implement a `type` property to indicate the Expression's type.
    """

    def parenthesize(self) -> bool:
        """
        Indicates whether this expression needs parentheses when converted to string.
        Subclasses or the parent node can override how parentheses are used.
        """
        return True


# ------------------------------------------------------------------------------
# BLOCK & TOP-LEVEL CODE
# ------------------------------------------------------------------------------


@dataclass
class Block(Expression):
    """
    Represents a block of statements (e.g. a function body or an 'if' branch).
    Inherits from Expression for convenience in some compilers, but it's used
    like a statement block in many C-like languages.
    """

    statements: List[Statement]

    @property
    def type(self) -> "Type":
        """
        A Block isn't evaluated for a value in most C-like languages.
        Returning 'void' type for convenience.
        """
        return Type(name="void")

    def validate(self):
        """
        Ensures all items in the statements list are indeed Statement nodes.
        """
        from pycco.utils import flatten  # or your flatten utility

        for stmt in flatten(self.statements):
            if not isinstance(stmt, Statement):
                raise PyCcoNodeValidateError("Block can only contain statements.", self)

    def __str__(self):
        from textwrap import indent

        statements_str = indent("\n".join(map(str, self.statements)), " " * 4)
        return f"{{\n{statements_str}\n}}"


@dataclass
class Code(Statement):
    """
    Represents the top-level container of an entire program (AST).
    Typically holds multiple statements (global declarations, functions, etc.),
    plus a reference to the 'main' function if present.
    """

    statements: List[Statement]
    main: Optional["Function"] = None

    def __post_init__(self):
        # Ensure parent references are updated (if using a Node-like base).
        # Then locate main if it exists.
        self.main = next(
            (
                stmt
                for stmt in self.statements
                if isinstance(stmt, Function) and stmt.var.name.name == "main"
            ),
            None,
        )

    def validate(self):
        if not self.main:
            raise PyCcoNodeValidateError("No main function found in the code.", self)

    def __str__(self):
        return "\n".join(map(str, self.statements))


# ------------------------------------------------------------------------------
# FUNCTION & RETURN
# ------------------------------------------------------------------------------


@dataclass
class Function(Statement):
    """
    A function definition with a Variable node for its 'signature', a Block as its body,
    zero or more parameters (list of Variable), and optionally a Return node if it exists
    in the body.
    """

    var: "Variable"
    body: Block
    params: List["Variable"] = field(default_factory=list)

    def validate(self):
        """
        Ensures that:
        - If the function's type is not 'void', it must have a Return node.
        - If there's a Return, it should match the function type (enforced by Return's validation).
        """
        # If it's not void, we at least require some Return somewhere in the body.
        if self.var.type.name != "void":
            # This is a simplistic check; the actual requirement might be more complex
            # (e.g., all control paths must return).
            # But we provide a simple presence check for demonstration.
            has_return = any(isinstance(stmt, Return) for stmt in self.body.statements)
            if not has_return:
                raise PyCcoNodeValidateError(
                    f"Non-void function '{self.var.name.name}' must have a return statement.",
                    self,
                )

    def __str__(self):
        params = "void" if not self.params else ", ".join(map(str, self.params))
        return f"""
{self.var}({params})
{self.body}
"""


@dataclass
class Return(Statement):
    """
    A return statement returning an Expression. Must appear inside a Function body.
    """

    value: Expression

    def validate(self):
        """
        Ensures that a Return statement is inside a Function and that the Expression type
        matches the function's declared return type (unless the function is void).
        """
        func = self.get_nearest_parent_of_type(Function)
        if not func:
            raise PyCcoNodeValidateError(
                "Return statement outside of a function.", self
            )
        # If the function is non-void, the expression type must match (simplified check).
        if func.var.type.name != "void":
            # Simple check if names match.
            if self.value.type.name != func.var.type.name:
                raise PyCcoNodeValidateError(
                    f"Expected return type '{func.var.type.name}', got '{self.value.type.name}'.",
                    self,
                )

    def __str__(self):
        return f"return {self.value};"


# ------------------------------------------------------------------------------
# PRIMITIVES (Ident, Type, Number, Char, StringLiteral)
# ------------------------------------------------------------------------------


@dataclass
class Ident(Expression):
    """
    An identifier (e.g., a variable name).
    """

    name: str

    @property
    def type(self) -> "Type":
        """
        Naive approach: an Ident alone doesn't strictly define type without context.
        We'll treat the identifier as if its name indicates a type.
        (In a real compiler, you'd do symbol-table lookups.)
        """
        return Type(self.name)

    def __str__(self):
        return self.name


@dataclass
class Type(Statement):
    """
    A 'Type' node can represent 'int', 'char', 'myStruct', etc. plus whether it's a pointer.
    """

    name: str
    pointer: bool = False

    @property
    def type(self) -> "Type":
        # A Type node is a 'Type' itself, so we return self.
        return self

    def set_pointer(self) -> "Type":
        self.pointer = True
        return self

    def __str__(self):
        pointer = "" if not self.pointer else "*"
        return f"{self.name}{pointer}"


@dataclass
class Number(Expression):
    """
    Represents numeric constants (ints in this simplified AST).
    """

    value: str

    @property
    def type(self) -> "Type":
        """
        Attempts to infer the type of the number based on:
        - The variable it is assigned to (if applicable).
        - The type of surrounding expressions (e.g., BinaryOp).
        - Defaults to 'int' if no context is available.
        """
        parent = self.get_nearest_parent_of_type((Assign, BinaryOp))
        if isinstance(parent, Assign):
            return parent.var.type
        elif isinstance(parent, BinaryOp):
            return parent.left.type if parent.left is not self else parent.right.type
        return Type("float") if "." in self.value else Type("int")  # Default assumption

    def __str__(self):
        return self.value


@dataclass
class Char(Expression):
    """
    Represents a character constant (e.g. 'a', 'b').
    """

    value: str

    @property
    def type(self) -> "Type":
        return Type("char")

    def __str__(self):
        return f"'{self.value}'"


@dataclass
class StringLiteral(Expression):
    """
    Represents a string constant (e.g. "hello").
    """

    value: str

    @property
    def type(self) -> "Type":
        return Type("char", pointer=True)

    def __str__(self):
        return f'"{self.value}"'


# ------------------------------------------------------------------------------
# OPERATORS
# ------------------------------------------------------------------------------


@dataclass
class BinaryOp(Expression):
    """
    Represents a binary operation, e.g. left + right, left - right, etc.
    """

    left: Expression
    op: str
    right: Expression

    @property
    def type(self) -> "Type":
        """
        Naively returns the left operand type.
        Realistically, you'd also check operator rules & type compatibility.
        """
        return self.left.type

    def __str__(self):
        ret = f"{self.left} {self.op} {self.right}"
        if self.parenthesize():
            return f"({ret})"
        return ret


@dataclass
class UnaryOp(Expression):
    """
    Represents a unary operation, e.g. -value, !value, value++ (postfix=true).
    """

    op: str
    value: Expression
    postfix: bool = False

    @property
    def type(self) -> "Type":
        return self.value.type

    def __str__(self):
        if not self.postfix:
            ret = f"{self.operator}{self.operand}"
        else:
            ret = f"{self.operand}{self.operator}"
        if self.parenthesize:
            return f"({ret})"
        return ret


@dataclass
class Parens(Expression):
    """
    Explicit parentheses around an expression.
    """

    inner: Expression

    @property
    def type(self) -> "Type":
        """
        The parenthesized expression is the same type as the inner expression.
        """
        return self.inner.type

    def __str__(self):
        ret = str(self.inner)
        if self.parenthesize:
            return f"({ret})"
        return ret


# ------------------------------------------------------------------------------
# VARIABLES & ASSIGNMENT
# ------------------------------------------------------------------------------


@dataclass
class Variable(Statement):
    """
    Represents a variable declaration: type + name.
    """

    type: Type
    name: Ident

    def __str__(self):
        semicolon = ";" if not isinstance(self.parent, Statement) else ""
        return f"{self.type} {self.name}{semicolon}"


@dataclass
class ArrayVariable(Variable):
    """
    Represents a variable that is an array: type + name + size (optional).
    """

    size: Optional[Expression] = None

    @property
    def semicolon(self) -> bool:
        return True  # for printing, if needed

    def __str__(self):
        size_str = f"[{self.size}]" if self.size is not None else "[]"
        semicolon = ";" if self.semicolon else ""
        return f"{self.type} {self.name}{size_str}{semicolon}"


@dataclass
class Assign(Statement):
    """
    Represents an assignment: var = value;
    var can be an Ident or a Variable node, value is an Expression.
    """

    var: Union[Ident, Variable]
    value: Expression

    def validate(self):
        """
        Ensures that var and value have matching types (basic name-based check).
        """
        # If var is a Variable node, use var.type. If it's an Ident, use Ident's type.
        var_type = self.var.type if isinstance(self.var, Variable) else self.var.type
        if (
            var_type.name != self.value.type.name
            or var_type.pointer != self.value.type.pointer
        ):
            raise PyCcoNodeValidateError(
                f"Assignment type mismatch: var '{var_type}' vs value '{self.value.type}'",
                self,
            )

    def __str__(self):
        return f"{self.var} = {self.value};"


# ------------------------------------------------------------------------------
# CALLS & ARRAYS
# ------------------------------------------------------------------------------


@dataclass
class FunctionCall(Expression):
    """
    Represents a function call: name(args...).
    """

    name: Ident
    args: List[Expression]

    @property
    def type(self) -> "Type":
        """
        Naively tries to find the nearest parent Function and use its return type.
        Otherwise fallback to 'void'.
        """
        func = (
            self.get_nearest_parent_of_type(Function)
            if hasattr(self, "get_nearest_parent_of_type")
            else None
        )
        return func.var.type if func else Type("void")

    def set_statement(self, value: bool = True):
        self._statement = value
        return self

    @property
    def statement(self):
        return self._statement if hasattr(self, "_statement") else False

    def __str__(self):
        args_str = ", ".join(map(str, self.args))
        semicolon = ";" if self.statement else ""
        return f"{self.name}({args_str}){semicolon}"


@dataclass
class Array(Expression):
    """
    Represents an array literal: {val1, val2, ...}.
    """

    values: List[Expression]

    @property
    def type(self) -> "Type":
        """
        Determines the array type based on:
        - The first element’s type if values exist.
        - The assignment target’s type if in an Assign node.
        - Defaults to 'int[]' if no other context is available.
        """
        if self.values:
            return Type(self.values[0].type.name, pointer=True)
        parent = self.get_nearest_parent_of_type(Assign)
        if isinstance(parent, Assign):
            return parent.var.type
        return Type("int", pointer=True)  # Default assumption

    def validate(self):
        """
        Ensures all array elements have the same type (basic name-based check).
        """
        if not self.values:
            return
        first_type = self.values[0].type
        for val in self.values[1:]:
            if (
                val.type.name != first_type.name
                or val.type.pointer != first_type.pointer
            ):
                raise PyCcoNodeValidateError(
                    "All array elements must have the same type.", self
                )

    def __str__(self):
        return f"{self.array}[{self.index}]"


@dataclass
class ArrayIndex(Expression):
    """
    Represents indexing into an array: array[index].
    """

    array: Expression
    index: Expression

    @property
    def type(self) -> "Type":
        """
        If array is a pointer or an array, the result is the 'base' type.
        e.g. int[] -> int, or int* -> int.
        """
        arr_type = self.array.type
        # If it's something like int*, the base is int.
        # For a real compiler, we might store array dimension info.
        if arr_type.pointer:
            return Type(name=arr_type.name)  # same name, but pointer=False
        # fallback
        return Type(name="unknown")

    def validate(self):
        """
        Ensures that 'array' is something indexable and 'index' is an integer.
        """
        if not self.index.type.name == "int":
            raise PyCcoNodeValidateError("Array index must be of type int.", self)
        arr_type = self.array.type
        if not arr_type.pointer:
            raise PyCcoNodeValidateError(
                f"Cannot index non-pointer/array type '{arr_type}'.", self
            )

    def __str__(self):
        return f"{self.array}[{self.index}]"


# ------------------------------------------------------------------------------
# STRUCTS & TYPEDEF
# ------------------------------------------------------------------------------


@dataclass
class Struct(Expression):
    """
    Represents a C-style struct definition or usage (with optional name).
    Declarations inside are typically field variables.
    """

    name: Optional[Ident]
    decls: List[Variable]

    @property
    def type(self) -> "Type":
        """
        If named, we treat it like 'struct MyStruct'; else just 'struct'.
        """
        return Type(self.name.name if self.name else "struct")

    def validate(self):
        """
        Ensure each declaration inside the struct is a valid Variable node.
        """
        for decl in self.decls:
            if not isinstance(decl, Variable):
                raise PyCcoNodeValidateError(
                    "Struct fields must be Variable nodes.", self
                )

    def __str__(self):
        from textwrap import indent

        name_str = f" {self.name}" if self.name else ""
        fields_str = indent("\n".join(str(field) for field in self.decls), " " * 4)
        return f"struct{name_str} {{\n{fields_str}\n}};"


@dataclass
class StructAccess(Expression):
    """
    Represents accessing a field within a struct or a pointer to a struct,
    using either '.' or '->'.
    """

    obj: Expression  # the struct or pointer
    operator: str  # '.' or '->'
    field: Ident

    @property
    def type(self) -> "Type":
        """
        Naive approach:
        In a real compiler, you'd look up the struct type (obj.type.name) in a symbol table,
        find the field's declared type. Here, we'll just return 'int' or 'unknown'.
        """
        # Simplify to 'int' for demonstration, or 'unknown' if uncertain.
        return Type("int")

    def validate(self):
        """
        Basic check for '.' vs '->'. If the struct is a pointer, we expect '->'.
        If the struct is not a pointer, we expect '.'.
        This is still a simplification for demonstration.
        """
        obj_t = self.obj.type
        is_pointer = obj_t.pointer
        if is_pointer and self.operator == ".":
            raise PyCcoNodeValidateError("Use '->' on pointer to struct.", self)
        if not is_pointer and self.operator == "->":
            raise PyCcoNodeValidateError("Use '.' on struct (non-pointer).", self)

    def __str__(self):
        return f"{self.obj}{self.operator}{self.field}"


@dataclass
class Typedef(Expression):
    """
    Represents a typedef: typedef original alias.
    original can be a Type or a Struct, alias is an Ident.
    """

    original: Union[Type, Struct]
    alias: Ident

    @property
    def type(self) -> "Type":
        # The result of a typedef is effectively the alias as a new type.
        return Type(self.alias.name)

    def __str__(self):
        return f"typedef {self.original} {self.alias};"


# ------------------------------------------------------------------------------
# IF & WHILE STATEMENTS
# ------------------------------------------------------------------------------


@dataclass
class IfStatement(Statement):
    """
    Represents an 'if' statement, optional 'else' branch.
    """

    condition: Expression
    then_branch: Block
    else_branch: Optional[Block] = None

    def validate(self):
        """
        Basic check that condition is an int (following C-like tradition).
        """
        if self.condition.type.name != "int":
            raise PyCcoNodeValidateError(
                "If condition must be an integer expression.", self
            )

    def __str__(self):
        else_part = f"else {self.else_branch}" if self.else_branch else ""
        return f"if ({self.condition}) {self.then_branch} {else_part}"


@dataclass
class WhileLoop(Statement):
    """
    Represents a 'while' loop with a condition and a body (Block).
    """

    condition: Expression
    body: Block

    def validate(self):
        """
        Basic check that condition is an int (following C-like tradition).
        """
        if self.condition.type.name != "int":
            raise PyCcoNodeValidateError(
                "While condition must be an integer expression.", self
            )

    def __str__(self):
        return f"while ({self.condition}) {self.body}"
