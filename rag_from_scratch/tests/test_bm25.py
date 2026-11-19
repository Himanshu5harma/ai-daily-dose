import pytest

from rag_from_scratch.bm25 import BM25, tokenize

# Tiny corpus you can check by hand: N = 4, average length = 6 / 4 = 1.5.
DOCS = [["a", "b"], ["a", "c", "c"], ["d"], []]


# ---------------------------------------------------------------- tokenize


@pytest.mark.parametrize(
    "text,expected",
    [
        ("shutil.copytree(dirs_exist_ok=True)", ["shutil", "copytree", "dirs_exist_ok", "true"]),
        ("Path.from_uri() -- NEW in 3.13!", ["path", "from_uri", "new", "in", "3", "13"]),
        ("", []),
        ("   \n\t ...", []),
        ("Ünïcode café", ["ünïcode", "café"]),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


# ---------------------------------------------------------------- idf


def test_idf_values():
    bm = BM25(DOCS)
    assert bm.idf("a") == pytest.approx(0.693147, abs=1e-6)  # df = 2
    assert bm.idf("c") == pytest.approx(1.203973, abs=1e-6)  # df = 1
    assert bm.idf("zzz") == pytest.approx(2.302585, abs=1e-6)  # unseen, df = 0


def test_idf_never_negative_for_very_common_terms():
    bm = BM25([["the"]] * 10)
    assert bm.idf("the") > 0


# ---------------------------------------------------------------- scores


@pytest.mark.parametrize(
    "query,expected",
    [
        (["a"], [0.602737, 0.478033, 0.0, 0.0]),
        (["c"], [0.0, 1.301592, 0.0, 0.0]),
        (["a", "c"], [0.602737, 1.779625, 0.0, 0.0]),
        (["a", "a"], [1.205473, 0.956065, 0.0, 0.0]),  # repeated query token counts twice
        (["zzz"], [0.0, 0.0, 0.0, 0.0]),
        ([], [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_scores_default_parameters(query, expected):
    assert BM25(DOCS).scores(query) == pytest.approx(expected, abs=1e-6)


def test_b_zero_ignores_document_length():
    assert BM25(DOCS, b=0.0).scores(["a"]) == pytest.approx([0.693147, 0.693147, 0, 0], abs=1e-6)


def test_k1_zero_ignores_term_frequency():
    assert BM25(DOCS, k1=0.0).scores(["c"]) == pytest.approx([0, 1.203973, 0, 0], abs=1e-6)


def test_term_frequency_saturates():
    bm = BM25([["x"] * n + ["pad"] * (20 - n) for n in (1, 2, 4, 8, 16)])
    s = bm.scores(["x"])
    gains = [b - a for a, b in zip(s, s[1:], strict=False)]
    assert all(g > 0 for g in gains)
    assert all(later < earlier for earlier, later in zip(gains, gains[1:], strict=False))


def test_all_empty_documents_score_zero():
    assert BM25([[], []]).scores(["a"]) == [0.0, 0.0]


@pytest.mark.parametrize("k1,b", [(-0.1, 0.75), (1.2, -0.1), (1.2, 1.1)])
def test_rejects_bad_parameters(k1, b):
    with pytest.raises(ValueError):
        BM25(DOCS, k1=k1, b=b)


# ---------------------------------------------------------------- top_k


def test_top_k_orders_and_drops_zero_scores():
    bm = BM25(DOCS)
    result = bm.top_k(["a"], 10)
    assert [i for i, _ in result] == [0, 1]
    assert result[0][1] == pytest.approx(0.602737, abs=1e-6)


def test_top_k_truncates_and_breaks_ties_by_index():
    bm = BM25([["x"], ["y"], ["x"], ["x"]])
    assert [i for i, _ in bm.top_k(["x"], 2)] == [0, 2]


def test_top_k_rejects_bad_k():
    with pytest.raises(ValueError):
        BM25(DOCS).top_k(["a"], 0)
