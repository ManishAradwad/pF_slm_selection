"""Source-preserving structural text utilities."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Pattern

from .types import Clause, EvidenceSpan


_CLAUSE_BREAK = re.compile(r"(?:[\r\n]+|(?<=[.!?;])\s+|\s+(?:but|however|while|whereas)\s+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class StructuralView:
    normalized: str
    source_index_by_normalized_char: tuple[int, ...]

    def finditer(self, pattern: Pattern[str]) -> tuple[StructuralMatch, ...]:
        return tuple(StructuralMatch(self, match) for match in pattern.finditer(self.normalized))

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end <= start or end > len(self.source_index_by_normalized_char):
            raise ValueError("normalized match is outside the structural view")
        return (
            self.source_index_by_normalized_char[start],
            self.source_index_by_normalized_char[end - 1] + 1,
        )


@dataclass(frozen=True, slots=True)
class StructuralMatch:
    view: StructuralView
    match: re.Match[str]

    def group(self, name: str) -> str:
        return self.match.group(name)

    def start(self, name: str | None = None) -> int:
        return self.span(name)[0]

    def end(self, name: str | None = None) -> int:
        return self.span(name)[1]

    def span(self, name: str | None = None) -> tuple[int, int]:
        normalized_span = self.match.span(name) if name is not None else self.match.span()
        return self.view.source_span(*normalized_span)

    @property
    def normalized_start(self) -> int:
        return self.match.start()

    @property
    def normalized_end(self) -> int:
        return self.match.end()


def build_structural_view(source: str) -> StructuralView:
    """Return a searchable view with a mapping back to unchanged source characters."""

    chars: list[str] = []
    indexes: list[int] = []
    previous_space = False
    for source_index, source_char in enumerate(source):
        normalized = unicodedata.normalize("NFKC", source_char).casefold()
        for char in normalized:
            if char.isspace():
                if previous_space:
                    continue
                char = " "
                previous_space = True
            else:
                previous_space = False
            chars.append(char)
            indexes.append(source_index)
    return StructuralView("".join(chars), tuple(indexes))


def split_clauses(source: str) -> tuple[Clause, ...]:
    if not source:
        return ()
    clauses: list[Clause] = []
    cursor = 0
    for match in _CLAUSE_BREAK.finditer(source):
        start, end = _trim_bounds(source, cursor, match.start())
        if end > start:
            clauses.append(_clause(source, len(clauses), start, end))
        cursor = match.end()
    start, end = _trim_bounds(source, cursor, len(source))
    if end > start:
        clauses.append(_clause(source, len(clauses), start, end))
    if not clauses and source:
        clauses.append(_clause(source, 0, 0, len(source)))
    return tuple(clauses)


def clause_for_span(clauses: tuple[Clause, ...], start: int, end: int) -> str | None:
    for clause in clauses:
        evidence = clause.evidence
        if start >= evidence.start_char and end <= evidence.end_char:
            return clause.clause_id
    return None


def _trim_bounds(source: str, start: int, end: int) -> tuple[int, int]:
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


def _clause(source: str, index: int, start: int, end: int) -> Clause:
    return Clause(clause_id=f"cl{index}", evidence=EvidenceSpan.from_source(source, start, end))
