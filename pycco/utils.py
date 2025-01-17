from typing import Iterable, Iterator, Any
from itertools import chain

NestedStr = str | Iterable["NestedStr"]


def flatten(maybe_iterables: Any) -> Iterator:
    """
    Flattens `maybe_iterables` by descending into items that are Iterable
    """

    if not isinstance(maybe_iterables, (list, tuple, set, Iterator)):
        return iter([maybe_iterables])
    return chain.from_iterable(
        (flatten(maybe_iterable) for maybe_iterable in maybe_iterables)
    )


def flatten_str(nested_str: NestedStr) -> str:
    if isinstance(nested_str, str):
        return nested_str
    return "".join(flatten_str(c) for c in nested_str)
