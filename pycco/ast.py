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
    get_type_hints,
    TYPE_CHECKING,
)
from itertools import zip_longest, chain
from pycco.tokens import Token
from copy import deepcopy
from pycco.utils import flatten
from abc import ABC, abstractmethod

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


from typing import Any, Union, get_origin, get_args, ForwardRef


class PyCcoTypeError(TypeError):
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

        super().__init__(self._generate_message())

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
                PyCcoTypeError._prettify_type(arg) for arg in args
            )
            return f"one of {prettified_args}"

        # Handle ForwardRefs (e.g., "MyNode")
        if isinstance(type_, ForwardRef):
            return type_.__forward_arg__

        # Handle generic collections like List, Dict
        if origin is not None:
            args = get_args(type_)
            prettified_args = ", ".join(
                PyCcoTypeError._prettify_type(arg) for arg in args
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
    _tokens: List[Token] = []

    _is_compiled: bool = False
    _hooks: List[Callable[["Node", PyCcoTypeError], None]] = []

    def __post_init__(self):
        try:
            self.add_self_as_parent()
            self.validate_field_types()
        except PyCcoTypeError as e:
            self._run_hooks(e)

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
            PyCcoTypeError: If any field does not match its expected type.
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
                raise PyCcoTypeError(
                    node=self,
                    field_name=field_name,
                    expected_type=expected_type,
                    actual_value=actual_value,
                )

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
    def tokens(self) -> List[Token]:
        if self._tokens:
            return self._tokens
        ret = []
        for child in self.children:
            ret += child.tokens
        return ret

    def set_tokens(self, tokens: List[Token]) -> TNode:
        self._tokens = tokens
        return self

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


class Expression(Node):
    def type(self) -> "Type":
        raise NotImplementedError()

    def parenthesize(self) -> str:
        if isinstance(self.parent, Parens):
            return False
        return True


class Statement(Node): ...


@dataclass
class Code(Node):
    statements: List[Node]
    main: Optional["Function"] = None

    def __str__(self):
        return "\n".join(map(str, self.children))


@dataclass
class Ident(Expression):
    name: str

    def __str__(self):
        return self.name


@dataclass
class Type(Statement):
    name: str
    pointer: bool = False

    def set_pointer(self):
        self.pointer = True
        return self

    def __str__(self):
        pointer = "" if not self.pointer else "*"
        return f"{self.name}{pointer}"


@dataclass
class Variable(Statement):
    type: Type
    name: Ident

    def __str__(self):
        return f"{self.type} {self.name}"


@dataclass
class Assign(Statement):
    var: Variable
    value: Expression

    def __str__(self):
        return f"{self.var} = {self.value};"


@dataclass
class Return(Statement):
    value: Expression

    def __str__(self):
        return f"return {self.value};"


@dataclass
class Number(Expression):
    value: str

    def __str__(self):
        return self.value


@dataclass
class Function(Statement):
    var: Variable
    params: List[Variable] = field(default_factory=list)
    body: List[Expression | Statement] = field(default_factory=list)
    ret: Optional[Return] = None

    def __str__(self):
        params = "void" if not self.params else ", ".join(map(str, self.params))
        body = "" if not self.body else "\n".join(map(lambda s: f"  {s}", self.body))
        ret = "" if self.ret is None else f"  {self.ret}"
        return f"""
{self.var}({params})
{{
{body}
{ret}
}}

"""


@dataclass
class FunctionCall(Expression):
    name: Ident  # The function being called
    args: List[Expression]  # The arguments passed to the function
    statement: bool = False

    def set_statement(self):
        self.statement = True
        return self

    def __str__(self):
        args_str = ", ".join(map(str, self.args))
        semicolon = ";" if self.statement else ""
        return f"{self.name}({args_str}){semicolon}"


@dataclass
class IfStatement(Statement):
    condition: Expression
    then_branch: List[Statement]
    else_branch: Optional[List[Statement]] = None

    def __str__(self):
        else_part = "" if not self.else_branch else f"else {{\n{self.else_branch}\n}}"
        return f"if ({self.condition}) {{\n{self.then_branch}\n}} {else_part}"


@dataclass
class WhileLoop(Statement):
    condition: Expression
    body: List[Statement]

    def __str__(self):
        return f"while ({self.condition}) {{\n{self.body}\n}}"


@dataclass
class BinaryOp(Expression):
    left: Expression
    op: str
    right: Expression

    def __str__(self):
        ret = f"{self.left} {self.op} {self.right}"
        if self.parenthesize():
            return f"({ret})"
        return ret


@dataclass
class UnaryOp(Expression):
    op: str
    value: Expression
    postfix: bool = False

    def __str__(self):
        if not self.postfix:
            ret = f"{self.operator}{self.operand}"
        else:
            ret = f"{self.operand}{self.operator}"
        if self.parenthesize:
            return f"({ret})"
        return ret


@dataclass
class Char(Expression):
    value: str

    def __str__(self):
        return f"'{self.value}'"


@dataclass
class StringLiteral(Expression):
    value: str

    def __str__(self):
        return f'"{self.value}"'


@dataclass
class ArrayIndex(Expression):
    array: Expression  # The array being indexed
    index: Expression  # The index used to access the array

    def __str__(self):
        return f"{self.array}[{self.index}]"


@dataclass
class StructAccess(Expression):
    obj: Expression  # The struct or pointer to a struct
    operator: str  # Either '.' or '->'
    field: Ident  # The field being accessed

    def __str__(self):
        return f"{self.obj}{self.operator}{self.field}"


@dataclass
class Parens(Expression):
    inner: Expression

    def __str__(self):
        ret = str(self.inner)
        if self.parenthesize:
            return f"({ret})"
        return ret
