from dataclasses import field, dataclass
from typing import List, Optional
from pycco.tokens import Token

@dataclass
class Node:
    token: Token

class Expression(Node):
    ...

class Statement(Node):
    ...

@dataclass
class Code(Node):
    children: List[Node]

@dataclass
class Ident(Expression):
    name: str

@dataclass
class Type(Statement):
    name: Ident
    pointer: bool = False

@dataclass
class Arg(Statement):
    name: Ident
    type: Type

@dataclass
class Return(Statement):
    value: Expression

@dataclass
class Function(Statement):
    name: Ident
    args: List[Arg] = field(default_factory=list)
    body: List[Expression] = field(default_factory=list)
    ret: Optional[Return] = None