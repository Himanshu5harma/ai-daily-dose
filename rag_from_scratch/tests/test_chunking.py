import pytest

from rag_from_scratch.chunking import fixed_size_spans, section_spans
from rag_from_scratch.corpus import build_document

# ---------------------------------------------------------------- fixed size


def test_fixed_empty_text():
    assert fixed_size_spans("", 10) == []


def test_fixed_text_shorter_than_size():
    assert fixed_size_spans("abc", 10) == [(0, 3)]


def test_fixed_exact_multiple_no_overlap():
    assert fixed_size_spans("x" * 9, 3) == [(0, 3), (3, 6), (6, 9)]


def test_fixed_short_tail_no_overlap():
    assert fixed_size_spans("x" * 10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_fixed_overlap_does_not_emit_contained_tail():
    assert fixed_size_spans("x" * 10, 4, overlap=2) == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_fixed_overlap_with_partial_tail():
    assert fixed_size_spans("x" * 11, 4, overlap=1) == [(0, 4), (3, 7), (6, 10), (9, 11)]


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)])
def test_fixed_rejects_bad_arguments(size, overlap):
    with pytest.raises(ValueError):
        fixed_size_spans("hello world", size, overlap)


@pytest.mark.parametrize("length,size,overlap", [(1, 1, 0), (97, 10, 3), (500, 64, 63), (31, 7, 6)])
def test_fixed_invariants(length, size, overlap):
    spans = fixed_size_spans("y" * length, size, overlap)
    assert spans[0][0] == 0 and spans[-1][1] == length
    for (s1, e1), (s2, e2) in zip(spans, spans[1:], strict=False):
        assert s2 - s1 == size - overlap
        assert e2 > e1  # every window adds new text
    assert all(0 < e - s <= size for s, e in spans)


# ---------------------------------------------------------------- section based


def _doc():
    return build_document(
        "doc",
        [
            {
                "heading": "Intro",
                "anchor": "intro",
                "blocks": [{"kind": "paragraph", "text": "i" * 30}],
            },
            {"heading": "Empty", "anchor": "empty", "blocks": []},
            {
                "heading": "Big",
                "anchor": "big",
                "blocks": [
                    {"kind": "paragraph", "text": "p" * 40, "anchor": "b1"},
                    {"kind": "paragraph", "text": "q" * 40, "anchor": "b2"},
                    {"kind": "code", "text": "c" * 130, "anchor": "b3"},
                    {"kind": "paragraph", "text": "r" * 20, "anchor": "b4"},
                ],
            },
        ],
    )


def test_section_rejects_bad_max():
    with pytest.raises(ValueError):
        section_spans(_doc(), 0)


def test_section_small_section_is_one_span():
    doc = _doc()
    intro = doc.sections[0]
    assert (intro.start, intro.end) in section_spans(doc, 100)


def test_section_heading_only_section_is_skipped():
    doc = _doc()
    empty = doc.sections[1]
    spans = section_spans(doc, 100)
    assert not any(s <= empty.start < e for s, e in spans)


def test_section_large_section_packs_whole_blocks_and_splits_only_oversized_unit():
    doc = _doc()
    heading, b1, b2, code, b4 = doc.sections[2].blocks
    assert section_spans(doc, 100) == [
        (doc.sections[0].start, doc.sections[0].end),
        (heading.start, b2.end),
        (code.start, code.start + 100),
        (code.start + 100, code.end),
        (b4.start, b4.end),
    ]


def test_section_whole_doc_fits_still_one_span_per_section():
    doc = _doc()
    spans = section_spans(doc, 10_000)
    assert spans == [(s.start, s.end) for s in doc.sections if s.has_body]


def _check_section_invariants(doc, max_chars):
    spans = section_spans(doc, max_chars)
    assert spans == sorted(spans)
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 <= s2
    for start, end in spans:
        assert 0 < end - start <= max_chars
        owners = [s for s in doc.sections if s.start <= start and end <= s.end]
        assert len(owners) == 1, "a span must stay inside one section"


@pytest.mark.parametrize("max_chars", [1, 50, 100, 500])
def test_section_invariants_on_fixture(max_chars):
    _check_section_invariants(_doc(), max_chars)


@pytest.mark.parametrize("max_chars", [500, 2000])
def test_section_invariants_on_real_corpus(real_corpus, max_chars):
    for doc in real_corpus:
        _check_section_invariants(doc, max_chars)


@pytest.mark.parametrize("size,overlap", [(500, 0), (500, 100), (2000, 200)])
def test_fixed_covers_real_documents(real_corpus, size, overlap):
    for doc in real_corpus:
        spans = fixed_size_spans(doc.text, size, overlap)
        assert spans[0][0] == 0 and spans[-1][1] == len(doc.text)
