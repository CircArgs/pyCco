from functools import reduce
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    Iterator,
    Type,
)
from pycco.utils import _len
from inspect import signature
from enum import Enum

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")

Stream = Sequence[S]


class ParserResult(Generic[T]):
    def __init__(
        self,
        index: int = -1,
        result: Optional[T] = None,
        description: Optional[str] = None,
    ):
        self.index = index
        self.result = result
        self.description = description

    def __getitem__(self, item: int) -> Union[int, Optional[T], Optional[str]]:
        if item == 0:
            return self.index
        elif item == 1:
            return self.result
        elif item == 2:
            return self.description
        else:
            raise IndexError("ParserResult index out of range")

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> Iterator[Union[int, Optional[T]]]:
        yield self.index
        yield self.result

    def __repr__(self) -> str:
        return f"ParserResult(index={self.index}, result={self.result}, description={self.description})"

    def __eq__(self, other):
        if not isinstance(other, ParserResult):
            return False
        return (
            self.index == other.index
            and self.result == other.result
            and self.description == other.description
        )


ParserFn = Callable[[Stream[S], int], ParserResult[T]]


class Parser(Generic[S, T]):
    """
    A generic parser combinator class that encapsulates parsing logic and provides combinator
    operations for building complex parsers. Each parser maintains a `description` attribute
    for a human-readable DSL representation.
    """

    def __init__(
        self,
        parse_fn: Optional[ParserFn[S, T]] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize the parser with a parsing function and an optional description.

        Args:
            parse_fn (ParserFn[S, T]): A function that takes a stream and index, and returns a parsing result. OR another parser (for forward declarations)
            description (Optional[str]): A human-readable description of the parser.
        """
        self.parse_fn = parse_fn
        self.description = description
        self.frozen_description = False

    def define(self: "Parser[S, T]", other: "Parser[S, U]") -> "Parser[S, U]":
        self.parse_fn = other.parse_fn
        self.description = other.description
        return self

    def freeze_description(self: "Parser[S, T]") -> "Parser[S, T]":
        self.frozen_description = True
        return self

    def describe(self: "Parser[S, T]", description: str) -> "Parser[S, T]":
        if not self.frozen_description:
            self.description = description
        return self

    def __call__(self, stream: Stream[S], index: int = 0) -> ParserResult[T]:
        """
        Invoke the parser on the given stream starting at the given index.

        Args:
            stream (Stream[S]): The input stream to parse.
            index (int): The starting index in the stream.

        Returns:
            ParserResult[T]: A tuple of the next index and the parsed result (or None on failure).
        """
        description = (
            self.description
            if not isinstance(self.parse_fn, Parser)
            else self.parse_fn.description
        )
        index, result = (
            self.parse_fn(stream, index)
            if not isinstance(self.parse_fn, Parser)
            else self.parse_fn.parse_fn
        )
        return ParserResult(index, result, description)

    def __str__(self) -> str:
        return self.description

    def __repr__(self) -> str:
        return f"Parser({self.description})"

    def __or__(self, other: Union["Parser[S, T]", S]) -> "Parser[S, T]":
        """
        Combine this parser with another parser as an alternative.
        """
        if not isinstance(other, Parser):
            other = self._auto_convert(other)

        description = f"({self.description} | {other.description})"

        def parse(stream: Stream[S], index: int) -> ParserResult[T]:
            result = self(stream, index)
            if result[0] != -1:
                return result
            return other(stream, index)

        return Parser(parse).describe(description)

    def map(
        self, mapper: Union[Callable[[T], U], Callable[[T, int], U]]
    ) -> "Parser[S, U]":
        """
        Transform the result of this parser using a mapping function.

        Args:
            mapper (Callable[[T], U]): A function to transform the parsed result.

        Returns:
            Parser[S, U]: A parser with the transformed result.
        """
        name = "unknown" if not hasattr(mapper, "__name__") else mapper.__name__

        description = f"({self.description} @ {name})"

        def parse(stream: Stream[S], index: int) -> ParserResult[U]:
            i, res = self(stream, index)
            if i == -1 or res is None:
                return ParserResult()

            if len(signature(mapper).parameters) == 1:
                mapped = mapper(res)
            else:
                mapped = mapper(res, i)

            return ParserResult(i, mapped)

        return Parser(parse).describe(description)

    def __matmul__(
        self, mapper: Union[Callable[[T], U], Callable[[T], U]]
    ) -> "Parser[S, U]":
        return self.map(mapper)

    def __add__(
        self, other: Union["Parser[S, U]", S]
    ) -> "Parser[S, List[Union[T, U]]]":
        """
        Sequence this parser with another parser, combining their results into a single list.

        Args:
            other (Union[Parser[S, U], S]): Another parser or value to sequence with this parser.

        Returns:
            Parser[S, List[Union[T, U]]]: A parser that combines the results of both parsers into a list.
        """
        if not isinstance(other, Parser):
            other = self._auto_convert(other)

        description = f"({self.description} + {other.description})"

        def parse(stream: Stream[S], index: int) -> ParserResult[List[Union[T, U]]]:
            i, res1 = self(stream, index)
            if i == -1:
                return ParserResult()  # First parser failed
            j, res2 = other(stream, i)
            if j == -1:
                return ParserResult()  # Second parser failed
            # Combine results into a flat list
            combined_result = []
            if isinstance(res1, list):
                combined_result.extend(res1)
            elif res1 is not None:
                combined_result.append(res1)
            if isinstance(res2, list):
                combined_result.extend(res2)
            elif res2 is not None:
                combined_result.append(res2)
            return ParserResult(j, combined_result)

        return Parser(parse).describe(description)

    def __mul__(self, times: int) -> "Parser[S, List[T]]":
        """
        Repeat this parser a fixed number of times.
        """
        description = f"{self.description}*{times}"

        def parse(stream: Stream[S], index: int) -> ParserResult[List[T]]:
            results = []
            current_index = index
            for _ in range(times):
                next_index, result = self(stream, current_index)
                if next_index == -1:
                    return ParserResult()
                results.append(result)
                current_index = next_index
            return ParserResult(current_index, results)

        return Parser(parse).describe(description)

    def optional(self) -> "Parser[S, Optional[T]]":
        """
        Make this parser optional.
        """
        description = f"{self.description}?"

        def parse(stream: Stream[S], index: int) -> ParserResult[Optional[T]]:
            result = self(stream, index)
            if result[0] == -1:
                return ParserResult(index)
            return result

        return Parser(parse).describe(description)

    def __invert__(self) -> "Parser[S, Optional[T]]":
        return self.optional()

    def __neg__(self) -> "Parser[S, Optional[T]]":
        description = f"not {self.description}"

        def parse(stream: Stream[S], index: int) -> ParserResult[S]:
            result_index, _ = self(stream, index)
            if result_index != -1:
                # The parser succeeded, so `~` negates to failure
                return ParserResult()
            if index < _len(stream):
                # Consume one element since the parser failed
                return ParserResult(index + 1, stream[index])
            return ParserResult()

        return Parser(parse).describe(description)

    def __pos__(self) -> "Parser[S, Optional[T]]":
        return self.many(1)

    def many(self, min: int = 0, max: Optional[int] = None) -> "Parser[S, List[T]]":
        """
        Match this parser repeatedly.
        """
        if min == 0 and max is None:
            description = f"{self.description}*"
        elif min == 1 and max is None:
            description = f"{self.description}+"
        elif max is None:
            description = self.description + "{" + f"{min}" + "}"
        else:
            description = self.description + "{" + f"{min}, {max}" + "}"

        def parse(stream: Stream[S], index: int) -> ParserResult[List[T]]:
            current_index = index
            results = []
            count = 0

            while current_index < _len(stream) and (max is None or count < max):
                next_index, result = self(stream, current_index)
                if next_index == -1:
                    break
                results.append(result)
                current_index = next_index
                count += 1

            if count < min:
                return ParserResult()

            return ParserResult(current_index, results)

        return Parser(parse).describe(description)

    def until(self, stopping_parser: Union["Parser[S, T]", S]) -> "Parser[S, List[T]]":
        """
        Create a parser that consumes input until the specified parser matches.
        """
        if not isinstance(stopping_parser, Parser):
            stopping_parser = self._auto_convert(stopping_parser)

        description = f"({self.description} until {stopping_parser.description})"

        def parse(stream: Stream[S], index: int) -> ParserResult[List[T]]:
            results = []
            current_index = index

            while current_index < _len(stream):
                stop_index, _ = stopping_parser(stream, current_index)
                if stop_index != -1:
                    return ParserResult(current_index, results)

                results.append(stream[current_index])
                current_index += 1

            return ParserResult()

        return Parser(parse).describe(description)

    def sep_by(self, separator: Union["Parser[S, U]", S]) -> "Parser[S, List[T]]":
        """
        Create a parser that matches this parser separated by a specified separator.
        """
        if not isinstance(separator, Parser):
            separator = self._auto_convert(separator)

        description = f"({self.description} sep_by {separator.description})"

        def parse(stream: Stream[S], index: int) -> ParserResult[List[T]]:
            results = []
            current_index = index

            first_index, first_result = self(stream, current_index)
            if first_index == -1:
                return ParserResult(index, [])  # No match at all

            results.append(first_result)
            current_index = first_index

            while current_index < _len(stream):
                sep_index, _ = separator(stream, current_index)
                if sep_index == -1:
                    break

                next_index, next_result = self(stream, sep_index)
                if next_index == -1:
                    break

                results.append(next_result)
                current_index = next_index

            return ParserResult(current_index, results)

        return Parser(parse).describe(description)

    def __rshift__(self, other: Union["Parser[S, U]", S]) -> "Parser[S, U]":
        """
        Right-associative sequencing: Combine with another parser but keep only the result of the second parser.
        """
        if not isinstance(other, Parser):
            other = self._auto_convert(other)

        description = f"({self.description} >> {other.description})"

        def parse(stream: Stream[S], index: int) -> ParserResult[U]:
            i, _ = self(stream, index)
            if i == -1:
                return ParserResult()
            j, res2 = other(stream, i)
            if j == -1:
                return ParserResult()
            return ParserResult(j, res2)

        return Parser(parse).describe(description)

    def __lshift__(self, other: Union["Parser[S, U]", S]) -> "Parser[S, T]":
        """
        Left-associative sequencing: Combine with another parser but keep only the result of the first parser.
        """
        if not isinstance(other, Parser):
            other = self._auto_convert(other)

        description = f"({self.description} << {other.description})"

        def parse(stream: Stream[S], index: int) -> ParserResult[T]:
            i, res1 = self(stream, index)
            if i == -1:
                return ParserResult()
            j, _ = other(stream, i)
            if j == -1:
                return ParserResult()
            return ParserResult(j, res1)

        return Parser(parse).describe(description)

    @staticmethod
    def _auto_convert(item: S) -> "Parser[S, S]":
        """
        Automatically convert an item into a parser if it is not already a parser.
        """
        if isinstance(item, Parser):
            return item
        return match(item)


# Utility functions to create basic parsers
def match(element: Stream[S]) -> Parser[S, Stream[S]]:
    """
    Create a parser that matches a specific sequence of elements.

    Args:
        element (Stream[S]): The sequence of elements to match.

    Returns:
        Parser[S, Stream[S]]: A parser that matches the specified sequence.
    """

    def parse(s: Stream[S], index: int) -> ParserResult[Stream[S]]:
        element_len = _len(element)
        end_index = index + element_len
        if element_len == 1:
            if s[index] == element:
                return ParserResult(end_index, s[index])
            return ParserResult()

        # Check if the slice matches the element
        s_slice = s[index:end_index]
        if s_slice == element:
            return ParserResult(end_index, s_slice)
        return ParserResult()  # Indicate failure

    return Parser(parse, f'"{str(element)}"')


def match_fn(
    predicate: Callable[[S], bool]
) -> Callable[[Stream[S], int], ParserResult[S]]:
    """
    Create a parser that matches elements in the stream based on a predicate function.

    Args:
        predicate (Callable[[S], bool]): A function that takes an element of the stream and returns True if it matches.

    Returns:
        Callable[[Stream[S], int], ParserResult[S]]: A parser function that matches based on the predicate.
    """

    def parse(s: Stream[S], index: int) -> ParserResult[S]:
        if index < _len(s) and predicate(s[index]):
            return index + 1, s[index]
        return -1, None

    return parse


def _anything() -> Parser[S, S]:
    """
    Create a parser that matches any single element.

    Returns:
        Parser[S, S]: A parser that matches any element.
    """

    def parse(s: Stream[S], index: int) -> ParserResult[S]:
        if index < _len(s):
            return ParserResult(index + 1, s[index])
        return ParserResult()

    return Parser(parse, ".")


anything = _anything()


def sequence(*elements: Union[Parser[S, U], S]) -> Parser[S, List[T]]:
    """
    Create a parser that matches a sequence of parsers.

    Args:
        parsers (Parser[S, T]): The parsers to sequence.

    Returns:
        Parser[S, List[T]]: A parser that parses the input sequentially with the given parsers.
    """
    return reduce(lambda a, b: a + b, map(Parser._auto_convert, elements)).describe(
        f'sequence of {" ".join(map(str, elements))}'
    )


def any_of(*elements: Union[Parser[S, U], S]) -> Parser[S, S]:
    """
    Create a parser that matches any one of the specified elements.

    Args:
        elements (S): The elements to match.

    Returns:
        Parser[S, S]: A parser that matches one of the elements.
    """
    return reduce(lambda a, b: a | b, map(Parser._auto_convert, elements)).describe(
        f'one of {", ".join(map(str, elements))}'
    )


TEnum = TypeVar("TEnum", bound=Enum)


def any_of_enum(*enum_classes: Type[TEnum]) -> Parser[Type[TEnum], TEnum]:
    """
    Create a parser that matches any member of the specified Enums.

    Args:
        *enum_classes (Type[TEnum]): The Enum classes to transform into a parser.

    Returns:
        Parser[Token, TEnum]: A parser that matches one of the members from the Enums.
    """
    return any_of(
        *[match(enum_member) for enum_cls in enum_classes for enum_member in enum_cls]
    ).describe(
        f"one of {', '.join(enum_cls.__name__ for enum_cls in enum_classes)} members"
    )


def parser(fn: ParserFn) -> Parser:
    """
    Decorator to convert a parsing function into a Parser object.

    Args:
        fn (ParserFn): The parsing function to wrap.

    Returns:
        Parser: The wrapped parser object.
    """
    return Parser(fn)
