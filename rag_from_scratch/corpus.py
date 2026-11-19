"""Load the parsed docs into flat text with character offsets for sections and blocks.

Every document becomes one string. Sections and blocks remember where they sit in
that string (half-open ``[start, end)`` offsets), so any chunker that returns
character spans can be mapped back to the gold anchors used by the questions.

Layout of ``Document.text``::

    <section heading>\\n\\n<block>\\n\\n<block> ... \\n\\n<next section heading> ...

The heading is stored as the first block of its section (``kind == "heading"``),
so a section whose only block is its heading has no body content.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SEPARATOR = "\n\n"
# A chunk "contains" a block when it overlaps at least this many of the block's
# characters (or the whole block, if shorter). This stops a chunk that merely
# clips the last few characters of a block from counting as evidence.
MIN_EVIDENCE_OVERLAP = 50


@dataclass(frozen=True)
class Block:
    kind: str  # "heading", "paragraph", "signature", "code", ...
    anchor: str
    start: int
    end: int


@dataclass(frozen=True)
class Section:
    heading: str
    heading_path: tuple[str, ...]
    anchor: str
    start: int
    end: int
    blocks: tuple[Block, ...]  # blocks[0] is always the heading

    @property
    def has_body(self) -> bool:
        return len(self.blocks) > 1


@dataclass(frozen=True)
class Document:
    doc_id: str  # e.g. "python-3.13-tempfile"; matches question source_id
    version: str
    title: str
    text: str
    sections: tuple[Section, ...]


def build_document(
    doc_id: str,
    sections: Iterable[Mapping],
    *,
    version: str = "",
    title: str = "",
) -> Document:
    """Build a Document from DocLens-style section dicts.

    Each section needs ``heading`` and ``anchor`` and may have ``heading_path``
    and ``blocks`` (dicts with ``kind``, ``text`` and optional ``anchor``, which
    defaults to the section anchor). Tests use this to make tiny documents.
    """
    parts: list[str] = []
    cursor = 0
    built: list[Section] = []

    def append(text: str) -> tuple[int, int]:
        nonlocal cursor
        if parts:
            parts.append(SEPARATOR)
            cursor += len(SEPARATOR)
        start = cursor
        parts.append(text)
        cursor += len(text)
        return start, cursor

    for raw in sections:
        anchor = raw["anchor"]
        heading = raw["heading"]
        h_start, h_end = append(heading)
        blocks = [Block("heading", anchor, h_start, h_end)]
        for raw_block in raw.get("blocks", ()):
            text = raw_block["text"]
            if not text:
                continue
            b_start, b_end = append(text)
            blocks.append(
                Block(raw_block["kind"], raw_block.get("anchor") or anchor, b_start, b_end)
            )
        built.append(
            Section(
                heading=heading,
                heading_path=tuple(raw.get("heading_path") or (heading,)),
                anchor=anchor,
                start=h_start,
                end=blocks[-1].end,
                blocks=tuple(blocks),
            )
        )
    return Document(doc_id, version, title, "".join(parts), tuple(built))


def load_document(path: Path) -> Document:
    raw = json.loads(path.read_text(encoding="utf-8"))
    source = raw["source"]
    return build_document(
        source["source_id"], raw["sections"], version=source["version"], title=raw["title"]
    )


def load_corpus(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load every parsed page under ``data_dir/corpus``, sorted by doc_id."""
    paths = sorted((data_dir / "corpus").rglob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no corpus under {data_dir / 'corpus'}; run "
            "`python -m rag_from_scratch.copy_corpus --doclens ../doclens`"
        )
    return sorted((load_document(p) for p in paths), key=lambda d: d.doc_id)


def anchors_in_span(document: Document, start: int, end: int) -> set[str]:
    """Return the anchors whose content a ``[start, end)`` chunk meaningfully covers.

    A block counts when the chunk overlaps at least ``MIN_EVIDENCE_OVERLAP`` of its
    characters (or all of it, if shorter). Each counted block contributes its own
    anchor and its section's anchor, so gold labels can name either an API entry
    (``shutil.rmtree``) or a whole section (``exit-on-error``).
    """
    found: set[str] = set()
    for section in document.sections:
        if section.end <= start or section.start >= end:
            continue
        for block in section.blocks:
            overlap = min(end, block.end) - max(start, block.start)
            if overlap >= min(MIN_EVIDENCE_OVERLAP, block.end - block.start) and overlap > 0:
                found.add(block.anchor)
                found.add(section.anchor)
    return found


def section_of(document: Document, offset: int) -> Section | None:
    """Return the section containing ``offset`` (separators belong to no section)."""
    for section in document.sections:
        if section.start <= offset < section.end:
            return section
    return None
