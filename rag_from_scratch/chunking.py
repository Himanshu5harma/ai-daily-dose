"""Chunkers. YOU implement these (checkpoint 2).

Both chunkers return character spans ``(start, end)`` (half-open) into
``Document.text`` instead of strings. The plumbing turns spans into chunk text and
maps them back to gold anchors with ``corpus.anchors_in_span``.
"""

from .corpus import Document

Span = tuple[int, int]


def fixed_size_spans(text: str, size: int, overlap: int = 0) -> list[Span]:
    """Split ``text`` into windows of at most ``size`` characters, ignoring structure.

    Contract:
    - Returns half-open ``(start, end)`` offsets into ``text``, sorted by ``start``.
    - Every window satisfies ``0 < end - start <= size``.
    - The first window starts at 0. Each next window starts ``size - overlap``
      characters after the previous one (the stride).
    - Stop as soon as a window reaches ``len(text)``: every character is covered,
      and no window is emitted that lies entirely inside the previous one.
      Example: ``len(text) == 10, size=4, overlap=2`` ->
      ``[(0, 4), (2, 6), (4, 8), (6, 10)]`` (no trailing ``(8, 10)``).
    - Empty ``text`` returns ``[]``.

    Raises:
        ValueError: if ``size <= 0``, ``overlap < 0`` or ``overlap >= size``.
    """
    if size <= 0:
        raise ValueError("Size must be greater than 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between 0 and size")

    if not text:
        return []

    length = len(text)
    windows = []
    start = 0
    stride = size - overlap

    while start < length:
        end = min(start + size, length)
        windows.append((start, end))
        if end == length:
            break
        start += stride

    return windows
      
    
    


def section_spans(document: Document, max_chars: int) -> list[Span]:
    """Chunk ``document`` along its sections, never crossing a section boundary.

    A section's *units* are its blocks in order; ``section.blocks[0]`` is the
    heading. Contract:
    - A section with no body (only its heading block) produces no span.
    - A section whose span ``section.end - section.start <= max_chars`` produces
      exactly one span, ``(section.start, section.end)``.
    - A longer section is packed greedily: start a span at the first unit, keep
      adding the next unit while ``unit.end - span_start <= max_chars``, otherwise
      close the span at the previous unit's end and start a new one at this unit.
      Spans therefore start at a unit's ``start`` and end at a unit's ``end``.
    - A single unit longer than ``max_chars`` (e.g. a huge code block) is the only
      thing ever cut mid-way: split it with ``fixed_size_spans(..., max_chars, 0)``
      over that unit's range (shifted to document offsets), and start the next
      span after it.
    - Output: sorted, non-overlapping spans, each ``0 < end - start <= max_chars``.

    Raises:
        ValueError: if ``max_chars <= 0``.
    """
    raise NotImplementedError
