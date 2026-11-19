"""Tests for the plumbing (corpus/question loading). These should pass already."""

from rag_from_scratch.corpus import anchors_in_span, build_document, section_of
from rag_from_scratch.questions import load_questions


def _doc():
    return build_document(
        "d",
        [
            {
                "heading": "Top",
                "anchor": "top",
                "blocks": [{"kind": "paragraph", "text": "intro text"}],
            },
            {
                "heading": "API",
                "anchor": "api",
                "blocks": [
                    {"kind": "signature", "text": "f(x)", "anchor": "mod.f"},
                    {"kind": "paragraph", "text": "F" * 120, "anchor": "mod.f"},
                    {"kind": "paragraph", "text": "G" * 120, "anchor": "mod.g"},
                ],
            },
        ],
    )


def test_offsets_point_at_the_right_text():
    doc = _doc()
    top, api = doc.sections
    assert doc.text[top.start : top.end] == "Top\n\nintro text"
    sig = api.blocks[1]
    assert doc.text[sig.start : sig.end] == "f(x)"
    assert api.blocks[0].kind == "heading"
    assert section_of(doc, sig.start) is api
    assert section_of(doc, top.end) is None  # separator


def test_blocks_inherit_section_anchor():
    assert _doc().sections[0].blocks[1].anchor == "top"


def test_anchors_in_span_requires_meaningful_overlap():
    doc = _doc()
    _, sig, f_body, g_body = doc.sections[1].blocks
    assert anchors_in_span(doc, sig.start, f_body.end) == {"api", "mod.f"}
    # clipping 10 chars of g's paragraph is not enough to count as evidence
    assert "mod.g" not in anchors_in_span(doc, f_body.start, g_body.start + 10)
    assert "mod.g" in anchors_in_span(doc, g_body.start, g_body.start + 60)
    # short blocks count when fully covered
    assert "mod.f" in anchors_in_span(doc, sig.start, sig.end)


def test_real_questions_and_corpus_line_up(real_corpus):
    questions = load_questions()
    assert len(questions) == 21  # 24 dev questions minus 3 missing-evidence
    assert {q.query_type for q in questions} == {"exact", "conceptual", "multi"}
    docs = {d.doc_id: d for d in real_corpus}
    for q in questions:
        for group in q.evidence_groups:
            for ref in group.references:
                doc = docs[ref.source_id]
                found = anchors_in_span(doc, 0, len(doc.text))
                assert ref.anchor in found, (q.qid, ref)
