from typing import Callable, Any, TypeVar, Tuple, Optional, List, Sequence

S = TypeVar('S')
T = TypeVar('T')
Stream = Sequence[S]
ParserResult = Tuple[int, Optional[T]]
Parser = Callable[[Stream[S], int], ParserResult[T]]

def word_parser(word: str) -> Parser[str, str]:
    def parse(s: str, i: int) -> ParserResult[str]:
        if s[i : i + len(word)] == word:
            return i + len(word), word
        return -1, None

    return parse

def sequence(*parsers: Parser[S, T]) -> Parser[S, List[T]]:
    def parse(s: Stream[S], i: int) -> ParserResult[List[T]]:
        current_index = i
        results: List[T] = []
        for parser in parsers:
            next_index, result = parser(s, current_index)
            if next_index == -1:
                return -1, None
            results.append(result)  # type: ignore
            current_index = next_index
        return current_index, results

    return parse

def optional(parser: Parser[S, T]) -> Parser[S, Optional[T]]:
    def parse(s: Stream[S], i: int) -> ParserResult[Optional[T]]:
        index, result = parser(s, i)
        if index == -1:
            return i, None
        return index, result

    return parse

def many(parser: Parser[S, T], times: Optional[int] = None) -> Parser[S, List[T]]:
    def parse(s: Stream[S], i: int) -> ParserResult[List[T]]:
        nonlocal times
        current_index = i
        results: List[T] = []
        count = 0
        while times is None or count < times:
            next_index, result = parser(s, current_index)
            if next_index == -1:
                break
            results.append(result)  # type: ignore
            current_index = next_index
            count += 1
        return current_index, results

    return parse

def map(parser: Parser[S, T], to: Callable[[T], Any]) -> Parser[S, Any]:
    def parse(s: Stream[S], i: int) -> ParserResult[Any]:
        index, result = parser(s, i)
        if index == -1 or result is None:
            return -1, None
        return index, to(result)

    return parse
