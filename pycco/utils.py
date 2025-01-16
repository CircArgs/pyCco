from typing import Iterable

NestedStr = str | Iterable["NestedStr"] 


def flatten(nested_str: NestedStr)->str:
    if isinstance(nested_str, str):
        return nested_str
    return "".join(flatten(c) for c in nested_str)