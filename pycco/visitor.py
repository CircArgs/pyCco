"""
Visitor and Context for compile backends
"""

import inspect
from typing import Dict, Callable, get_type_hints, Type, Tuple
from pycco.ast import Node


class CompilationContext:
    """
    Holds data needed during compilation or code generation.
    For example:
      - backend = "arm" or "js" or ...
      - symbol_table = {}
      - current_function = None
      - ...
    """

    def __init__(self, backend: str = "none"):
        self.backend = backend
        # Add more fields as needed


Compiler = Callable[[], Tuple[CompilationContext, "Visitor"]]


class Visitor:
    """
    A single visitor that dispatches on the *second* argument's type,
    which must be a subclass of Node. The *first* argument must be
    a CompilationContext.
    """

    def __init__(self):
        # { node_class: function(context, node), ... }
        self.registry: Dict[Type[Node], Callable[..., any]] = {}

    def register(self, func: Callable[..., any]):
        """
        Decorator that inspects the function signature to ensure:
          - first param is CompilationContext
          - second param is a Node subclass
        Then it stores func in self.registry under that Node type.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # We expect exactly 2 params: (ctx: CompilationContext, node: NodeSubclass)
        if len(params) < 2:
            raise ValueError(
                f"Function {func.__name__} must have at least two parameters: "
                "CompilationContext and a Node subclass."
            )

        first_param, second_param = params[0], params[1]

        # Check the first param is typed to CompilationContext
        first_anno = first_param.annotation
        if (
            first_anno is inspect.Parameter.empty
            or first_anno is not CompilationContext
        ):
            raise TypeError(
                f"Function {func.__name__} first parameter must be annotated with CompilationContext."
            )

        # Check the second param is typed to a Node subclass
        second_anno = second_param.annotation
        if second_anno is inspect.Parameter.empty:
            raise TypeError(
                f"Function {func.__name__} second parameter must have a type annotation (Node subclass)."
            )
        if not issubclass(second_anno, Node):
            raise TypeError(
                f"Function {func.__name__} second parameter must be a subclass of Node, not {second_anno}."
            )

        # Store in the registry
        if second_anno in self.registry:
            raise ValueError(
                f"A visitor is already registered for node type {second_anno.__name__}."
            )

        self.registry[second_anno] = func
        return func

    def __call__(self, context: CompilationContext, node: Node):
        """
        Invoke the visitor with (context, node).
        Looks up the function in the registry by node's type.
        Raises an error if no matching function is found.
        """
        node_cls = type(node)
        visit_func = self.registry.get(node_cls)
        if not visit_func:
            raise TypeError(
                f"No visitor function registered for node type {node_cls.__name__}."
            )
        return visit_func(context, node)
