from dataclasses import field, dataclass, fields
from typing import (
    List,
    Optional,
    TypeVar,
    Iterator,
    Any,
    Callable,
    Type as TypeType,
    Tuple,
)
from itertools import zip_longest, chain
from pycco.tokens import Token
from copy import deepcopy
from pycco.utils import flatten
from abc import ABC, abstractmethod

PRIMITIVES = {int, float, str, bool, type(None)}


# typevar used for node methods that return self
# so the typesystem can correlate the self type with the return type
TNode = TypeVar("TNode", bound="Node")  # pylint: disable=C0103


class Node(ABC):
    """Base class for all PyCco AST nodes.

    PyCco nodes are python dataclasses with the following patterns:
        - Attributes are either
            - PRIMITIVES (int, float, str, bool, None)
            - iterable from (list, tuple, set)
            - Enum
            - descendant of `Node`
        - Attributes starting with '_' are "obfuscated" and are not included in `children`

    """

    parent: Optional["Node"] = None
    parent_key: Optional[str] = None

    _is_compiled: bool = False

    def __post_init__(self):
        self.add_self_as_parent()

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


class Expression(Node): ...


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

    def __str__(self):
        return f"*{self.name}"


@dataclass
class Arg(Statement):
    name: Ident
    type: Type

    def __str__(self):
        return f"{self.type} {self.name}"


@dataclass
class Return(Statement):
    value: Expression

    def __str__(self):
        return f"return {self.value}"


@dataclass
class Function(Statement):
    name: Ident
    type: Type
    args: List[Arg] = field(default_factory=list)
    body: List[Expression] = field(default_factory=list)
    ret: Optional[Return] = None

    def __str__(self):
        args = ", ".join(map(str, self.args))
        body = "\n".join(map(lambda s: f"\t{s}", self.body))
        ret = "" if self.ret is None else self.ret
        return f"""
{self.type} {self.name}({args}){{
{body}
{ret}
}}

"""
