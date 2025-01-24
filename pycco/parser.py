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
from inspect import signature
from enum import Enum
from dataclasses import dataclass

# Type variables
S = TypeVar("S")  # Element type in the stream
T = TypeVar("T")  # Result type of the parser
U = TypeVar("U")  # Generic type variable
V = TypeVar("V")  # Generic type variable
TEnum = TypeVar("TEnum", bound=Enum)  # Enum type variable

# Stream type
Stream = Sequence[S]


@dataclass
class Position:
    index: int

    def advance(self, char: Optional[S] = None) -> "Position":
        return Position(self.index + 1)


class ParseError(Exception):
    def __init__(self, position: Position, expected: str, found: Optional[S]):
        self.position = position
        self.expected = expected
        self.found = found
        super().__init__(self.__str__())

    def __str__(self):
        return f"ParseError at index {self.position.index}: expected {self.expected}, found {repr(self.found)}"


class ParserResult(Generic[T]):
    def __init__(
        self,
        position: Position,
        result: Optional[T] = None,
        description: Optional[str] = None,
        error: Optional[ParseError] = None,
    ):
        self.position = position
        self.result = result
        self.description = description
        self.error = error

    def __getitem__(
        self, item: int
    ) -> Union[Position, Optional[T], Optional[str], Optional[ParseError]]:
        if item == 0:
            return self.position
        elif item == 1:
            return self.result
        elif item == 2:
            return self.description
        elif item == 3:
            return self.error
        else:
            raise IndexError("ParserResult index out of range")

    def __len__(self) -> int:
        return 4

    def __iter__(
        self,
    ) -> Iterator[Union[Position, Optional[T], Optional[str], Optional[ParseError]]]:
        yield self.position
        yield self.result
        yield self.description
        yield self.error

    def __repr__(self) -> str:
        return f"ParserResult(position={self.position}, result={self.result}, description={self.description}, error={self.error})"

    def __eq__(self, other):
        if not isinstance(other, ParserResult):
            return False
        return (
            self.position == other.position
            and self.result == other.result
            and self.description == other.description
            and self.error == other.error
        )

    def __bool__(self) -> bool:
        if self.error is not None:
            return False
        return True

    def update_on_failure(self, new_error: ParseError):
        if self.error is None or new_error.position.index > self.error.position.index:
            self.error = new_error


ParserFn = Callable[[Stream[S], Position], ParserResult[T]]


class Parser(Generic[S, T]):
    """
    A generic parser combinator class that encapsulates parsing logic and provides combinator
    operations for building complex parsers. Each parser maintains a `description` attribute
    for a human-readable DSL representation.
    """

    def __init__(
        self,
        parse_fn: Optional[ParserFn[S, T]] = None,
        description: str = "",
        frozen_description: bool = False,
    ):
        """
        Initialize the parser with a parsing function and an optional description.

        Args:
            parse_fn (ParserFn[S, T]): A function that takes a stream and position, and returns a parsing result.
            description (Optional[str]): A human-readable description of the parser.
        """
        self.parse_fn = parse_fn
        self.description = description
        self.frozen_description = frozen_description

    def define(self: "Parser[S, T]", other: "Parser[S, U]") -> "Parser[S, U]":
        self.parse_fn = other.parse_fn
        self.description = other.description
        return self

    def freeze_description(self: "Parser[S, T]", value: bool = True) -> "Parser[S, T]":
        self.frozen_description = value
        return self

    def set_description(self: "Parser[S, T]", description: str) -> "Parser[S, T]":
        if not self.frozen_description:
            self.description = description
        return self

    def describe(self: "Parser[S, T]", description: str) -> "Parser[S, T]":
        # Wrap the existing parse_fn to update error.expected and description dynamically
        original_parse_fn = self.parse_fn

        def new_parse_fn(stream: Stream[S], position: Position) -> ParserResult[T]:
            result = original_parse_fn(stream, position)
            if result.error:
                result.error.expected = description
            if result.description:
                result.description = description
            return result

        self.parse_fn = new_parse_fn
        self.description = description
        return self

    def __call__(
        self, stream: Stream[S], position: Position = Position(0)
    ) -> ParserResult[T]:
        """
        Invoke the parser on the given stream starting at the given position.

        Args:
            stream (Stream[S]): The input stream to parse.
            position (Position): The starting position in the stream.

        Returns:
            ParserResult[T]: The parsing result with updated position, result, and error information.
        """

        try:
            result = self.parse_fn(stream, position)
            return result
        except ParseError as e:
            return ParserResult(position=e.position, error=e)

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

        def parse(stream: Stream[S], position: Position) -> ParserResult[T]:
            result1 = self.parse_fn(stream, position)
            if result1:
                return result1

            result2 = other.parse_fn(stream, position)
            if result2:
                return result2

            # Both failed
            if result1.error and result2.error:
                pos1 = result1.error.position.index
                pos2 = result2.error.position.index
                if pos1 > pos2:
                    return ParserResult(
                        position=result1.error.position, error=result1.error
                    )
                elif pos2 > pos1:
                    return ParserResult(
                        position=result2.error.position, error=result2.error
                    )
                else:
                    # Same failure position: combine them using 'description'
                    found_char = None
                    if position.index < len(stream):
                        found_char = stream[position.index]
                    return ParserResult(
                        position=position,
                        error=ParseError(position, description, found_char),
                    )
            # Fallback
            return ParserResult(
                position=position, error=ParseError(position, description, None)
            )

        return Parser(parse, description=description).set_description(description)

    def map(
        self, mapper: Union[Callable[[T], U], Callable[[T, Position], U]]
    ) -> "Parser[S, U]":
        """
        Transform the result of this parser using a mapping function.

        Args:
            mapper (Callable[[T], U] or Callable[[T, Position], U]): A function to transform the parsed result.

        Returns:
            Parser[S, U]: A parser with the transformed result.
        """
        name = "unknown" if not hasattr(mapper, "__name__") else mapper.__name__

        description = f"({self.description} @ {name})"

        def parse(stream: Stream[S], position: Position) -> ParserResult[U]:
            result = self.parse_fn(stream, position)
            if not result:
                return result  # Propagate failure

            try:
                params = len(signature(mapper).parameters)
            except ValueError:
                # Assume single argument mapper for built-in functions like int
                params = 1

            if params == 1:
                mapped = mapper(result.result)
            else:
                mapped = mapper(result.result, result.position)

            return ParserResult(
                position=result.position, result=mapped, description=description
            )

        return Parser(parse, description=description).set_description(description)

    def __matmul__(
        self, mapper: Union[Callable[[T], U], Callable[[T], U]]
    ) -> "Parser[S, U]":
        return self.map(mapper).set_description(self.description)

    def __imatmul__(
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

        def parse(
            stream: Stream[S], position: Position
        ) -> ParserResult[List[Union[T, U]]]:
            result1 = self.parse_fn(stream, position)
            if not result1:
                return result1  # Propagate failure
            result2 = other.parse_fn(stream, result1.position)
            if not result2:
                return result2  # Propagate failure
            # Combine results into a flat list
            combined_result = []
            if isinstance(result1.result, list):
                combined_result.extend(result1.result)
            elif result1.result is not None:
                combined_result.append(result1.result)
            if isinstance(result2.result, list):
                combined_result.extend(result2.result)
            elif result2.result is not None:
                combined_result.append(result2.result)
            return ParserResult(
                position=result2.position,
                result=combined_result,
                description=description,
            )

        return Parser(parse, description=description).set_description(description)

    def __mul__(self, times: int) -> "Parser[S, List[T]]":
        """
        Repeat this parser a fixed number of times.

        Args:
            times (int): Number of repetitions.

        Returns:
            Parser[S, List[T]]: A parser that matches the pattern the specified number of times.
        """
        return self.many(min=times, max=times)

    def optional(self) -> "Parser[S, Optional[T]]":
        """
        Make this parser optional.

        Returns:
            Parser[S, Optional[T]]: A parser that returns the result or None without failing.
        """
        description = f"{self.description}?"

        def parse(stream: Stream[S], position: Position) -> ParserResult[Optional[T]]:
            result = self.parse_fn(stream, position)
            if result:
                return result
            return ParserResult(position=position, result=None, description=description)

        return Parser(parse, description=description).set_description(description)

    def __invert__(self) -> "Parser[S, Optional[T]]":
        return self.optional()

    def not_(self) -> "Parser[S, Optional[S]]":
        description = f"not {self.description}"

        def parse(stream: Stream[S], position: Position) -> ParserResult[Optional[S]]:
            result = self.parse_fn(stream, position)
            if result:
                # The parser succeeded, so `not_` negates to failure
                error = ParseError(position, description, result.result)
                return ParserResult(position=position, error=error)
            # The parser failed, so `not_` succeeds by consuming no input
            return ParserResult(position=position, result=None, description=description)

        return Parser(parse, description=description).set_description(description)

    def __pos__(self) -> "Parser[S, List[T]]":
        return self.many(1)

    def many(self, min: int = 0, max: Optional[int] = None) -> "Parser[S, List[T]]":
        """
        Match this parser repeatedly.

        Args:
            min (int): Minimum number of repetitions.
            max (Optional[int]): Maximum number of repetitions.

        Returns:
            Parser[S, List[T]]: A parser that matches the pattern repeatedly within the specified bounds.
        """
        if min == 0 and max is None:
            description = f"{self.description}*"
        elif min == 1 and max is None:
            description = f"{self.description}+"
        elif max is None:
            description = f"{self.description}{{{min}}}"
        else:
            description = f"{self.description}{{{min}, {max}}}"

        def parse(stream: Stream[S], position: Position) -> ParserResult[List[T]]:
            current_position = position
            results = []
            count = 0

            while (max is None or count < max) and current_position.index < len(stream):
                result = self.parse_fn(stream, current_position)
                if not result:
                    break
                results.append(result.result)
                current_position = result.position
                count += 1

            if count < min:
                error = ParseError(
                    position=current_position,
                    expected=f"at least {min} repetitions of {self.description}",
                    found=None,
                )
                return ParserResult(position=current_position, error=error)

            return ParserResult(
                position=current_position, result=results, description=description
            )

        return Parser(parse, description=description).set_description(description)

    def until(self, stopping_parser: Union["Parser[S, V]", S]) -> "Parser[S, List[T]]":
        """
        Create a parser that consumes input until the specified parser matches.

        Args:
            stopping_parser (Union[Parser[S, V], S]): The parser that indicates when to stop.

        Returns:
            Parser[S, List[T]]: A parser that collects results until the stopping parser matches.
        """
        if not isinstance(stopping_parser, Parser):
            stopping_parser = self._auto_convert(stopping_parser)

        description = f"({self.description} until {stopping_parser.description})"

        def parse(stream: Stream[S], position: Position) -> ParserResult[List[T]]:
            results = []
            current_position = position

            while current_position.index < len(stream):
                stop_result = stopping_parser.parse_fn(stream, current_position)
                if stop_result and stop_result.result is not None:
                    return ParserResult(
                        position=current_position,
                        result=results,
                        description=description,
                    )
                # Consume one element
                any_result = anything.parse_fn(stream, current_position)
                if not any_result:
                    break  # Cannot consume further
                results.append(any_result.result)
                current_position = any_result.position

            error = ParseError(
                position=current_position,
                expected=f"stopping condition {stopping_parser.description}",
                found=None,
            )
            return ParserResult(position=current_position, error=error)

        return Parser(parse, description=description).set_description(description)

    def sep_by(self, separator: Union["Parser[S, U]", S]) -> "Parser[S, List[T]]":
        """
        Create a parser that matches this parser separated by a specified separator.

        Args:
            separator (Union[Parser[S, U], S]): The separator parser or element.

        Returns:
            Parser[S, List[T]]: A parser that matches a list of elements separated by the separator.
        """
        if not isinstance(separator, Parser):
            separator = self._auto_convert(separator)

        description = f"({self.description} sep_by {separator.description})"

        def parse(stream: Stream[S], position: Position) -> ParserResult[List[T]]:
            results = []
            current_position = position

            first_result = self.parse_fn(stream, current_position)
            if not first_result:
                # No matches, return empty list
                return ParserResult(
                    position=position, result=[], description=description
                )

            results.append(first_result.result)
            current_position = first_result.position

            while current_position.index < len(stream):
                sep_result = separator.parse_fn(stream, current_position)
                if not sep_result:
                    break  # No more separators

                current_position = sep_result.position

                element_result = self.parse_fn(stream, current_position)
                if not element_result:
                    break  # Separator found but no element after it

                results.append(element_result.result)
                current_position = element_result.position

            return ParserResult(
                position=current_position, result=results, description=description
            )

        return Parser(parse, description=description).set_description(description)

    def __rshift__(self, other: Union["Parser[S, U]", S]) -> "Parser[S, U]":
        """
        Right-associative sequencing: Combine with another parser but keep only the result of the second parser.
        """
        if not isinstance(other, Parser):
            other = self._auto_convert(other)

        description = f"({self.description} >> {other.description})"

        def parse(stream: Stream[S], position: Position) -> ParserResult[U]:
            result1 = self.parse_fn(stream, position)
            if not result1:
                return result1  # Propagate failure
            result2 = other.parse_fn(stream, result1.position)
            if not result2:
                return result2  # Propagate failure
            return ParserResult(
                position=result2.position,
                result=result2.result,
                description=description,
            )

        return Parser(parse, description=description).set_description(description)

    def __lshift__(self, other: Union["Parser[S, U]", S]) -> "Parser[S, T]":
        """
        Left-associative sequencing: Combine with another parser but keep only the result of the first parser.
        """
        if not isinstance(other, Parser):
            other = self._auto_convert(other)

        description = f"({self.description} << {other.description})"

        def parse(stream: Stream[S], position: Position) -> ParserResult[T]:
            result1 = self.parse_fn(stream, position)
            if not result1:
                return result1  # Propagate failure
            result2 = other.parse_fn(stream, result1.position)
            if not result2:
                return result2  # Propagate failure
            return ParserResult(
                position=result2.position,
                result=result1.result,
                description=description,
            )

        return Parser(parse, description=description).set_description(description)

    def expect(self, expected: str) -> "Parser[S, T]":
        """
        Attach an explicit expectation to this parser for better error messages.

        Args:
            expected (str): Description of what is expected.

        Returns:
            Parser[S, T]: The parser with the attached expectation.
        """

        def new_parse_fn(stream: Stream[S], position: Position) -> ParserResult[T]:
            result = self.parse_fn(stream, position)
            if result:
                return result
            # If failed, attach the expectation
            error = ParseError(position, expected, None)
            return ParserResult(position=position, error=error)

        self.parse_fn = new_parse_fn
        self.description = expected
        return self

    @staticmethod
    def _auto_convert(item: S) -> "Parser[S, S]":
        """
        Automatically convert an item into a parser if it is not already a parser.

        Args:
            item (S): The item to convert.

        Returns:
            Parser[S, S]: The converted parser.
        """
        if isinstance(item, Parser):
            return item
        return match(item)


# Utility functions to create basic parsers


def match(element: S) -> Parser[S, S]:
    """
    Create a parser that matches a specific single element.

    Args:
        element (S): The element to match.

    Returns:
        Parser[S, S]: A parser that matches the specified element.
    """

    def parse(s: Stream[S], position: Position) -> ParserResult[S]:
        if position.index >= len(s):
            error = ParseError(position, repr(element), None)
            return ParserResult(position=position, error=error)
        current_element = s[position.index]
        if current_element == element:
            new_position = position.advance(current_element)
            return ParserResult(
                position=new_position, result=current_element, description=repr(element)
            )
        error = ParseError(position, repr(element), current_element)
        return ParserResult(position=position, error=error)

    return Parser(parse, repr(element))


def match_fn(predicate: Callable[[S], bool]) -> Parser[S, S]:
    """
    Create a parser that matches elements in the stream based on a predicate function.

    Args:
        predicate (Callable[[S], bool]): A function that takes an element of the stream and returns True if it matches.

    Returns:
        Parser[S, S]: A parser that matches based on the predicate.
    """

    # Handle lambda functions which might not have a __name__
    if hasattr(predicate, "__name__"):
        predicate_name = predicate.__name__
    else:
        predicate_name = "<lambda>"

    description = f"fn`{predicate_name}`"

    def parse(s: Stream[S], position: Position) -> ParserResult[S]:
        if position.index >= len(s):
            error = ParseError(position, description, None)
            return ParserResult(position=position, error=error)
        current_element = s[position.index]
        if predicate(current_element):
            new_position = position.advance(current_element)
            return ParserResult(
                position=new_position, result=current_element, description=description
            )
        error = ParseError(position, description, current_element)
        return ParserResult(position=position, error=error)

    return Parser(parse, description)


def _anything() -> Parser[S, S]:
    """
    Create a parser that matches any single element.

    Returns:
        Parser[S, S]: A parser that matches any element.
    """

    def parse(s: Stream[S], position: Position) -> ParserResult[S]:
        if position.index >= len(s):
            error = ParseError(position, "any element", None)
            return ParserResult(position=position, error=error)
        current_element = s[position.index]
        new_position = position.advance(current_element)
        return ParserResult(
            position=new_position, result=current_element, description="."
        )

    return Parser(parse, ".")


anything = _anything()


def sequence(*elements: Union[Parser[S, U], S]) -> Parser[S, List[U]]:
    """
    Create a parser that matches a sequence of parsers.

    Args:
        elements (Union[Parser[S, U], S]): The parsers or elements to sequence.

    Returns:
        Parser[S, List[U]]: A parser that parses the input sequentially with the given parsers.
    """
    return reduce(
        lambda a, b: a + b, map(Parser._auto_convert, elements)
    ).set_description(f'sequence of {" ".join(map(str, elements))}')


def any_of(*elements: Union[Parser[S, U], S]) -> Parser[S, S]:
    """
    Create a parser that matches any one of the specified elements.

    Args:
        elements (Union[Parser[S, U], S]): The parsers or elements to match.

    Returns:
        Parser[S, S]: A parser that matches one of the elements.
    """
    return reduce(
        lambda a, b: a | b, map(Parser._auto_convert, elements)
    ).set_description(f"({' | '.join(map(str, elements))})")


def any_of_enum(*enum_classes: Type[TEnum]) -> Parser[S, TEnum]:
    """
    Create a parser that matches any member of the specified Enums.

    Args:
        *enum_classes (Type[TEnum]): The Enum classes to transform into a parser.

    Returns:
        Parser[S, TEnum]: A parser that matches one of the members from the Enums.
    """
    return (
        any_of(
            *[
                match(enum_member.value)
                for enum_cls in enum_classes
                for enum_member in enum_cls
            ]
        )
        .map(
            lambda x: next(
                enum_member
                for enum_cls in enum_classes
                for enum_member in enum_cls
                if enum_member.value == x
            )
        )
        .set_description(
            f"one of {', '.join(enum_cls.__name__ for enum_cls in enum_classes)} members"
        )
    )


def expect(parser: Parser[S, T], expected: str) -> Parser[S, T]:
    """
    Decorator to attach an expectation to a parser.

    Args:
        parser (Parser[S, T]): The parser to decorate.
        expected (str): The expected description.

    Returns:
        Parser[S, T]: The decorated parser.
    """
    return parser.expect(expected)


def parser(fn: ParserFn[S, T]) -> Parser[S, T]:
    """
    Decorator to convert a parsing function into a Parser object.

    Args:
        fn (ParserFn[S, T]): The parsing function to wrap.

    Returns:
        Parser[S, T]: The wrapped parser object.
    """
    return Parser(fn)


def report_error(stream: Stream[S], error: ParseError) -> str:
    """
    Format and present the parsing error in a user-friendly manner.

    Args:
        stream (Stream[S]): The input stream being parsed.
        error (ParseError): The parsing error to report.

    Returns:
        str: A formatted error message.
    """
    return str(error)
