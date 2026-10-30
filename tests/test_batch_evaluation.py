"""Behavioral tests for batch evaluation over many queries."""

import random
import sys
from math import log2

import pytest

from ai_engineering.evaluations import evaluate, read_qrels, read_run, write_run

QRELS = {
    "q1": {"a": 2, "b": 0, "c": 1},
    "q2": {"x": 1},
    "q3": {"z": 0},
    "q5": {"m": 1},
}
RUN = {
    "q1": ["a", "d", "c"],
    "q2": {"y": 0.9, "x": 0.5},
    "q3": ["z"],
    "q4": ["a"],
}
METRICS = ["recall@2", "precision@2", "mrr@3", "ndcg@3", "judged@3"]


def test_per_query_scores_and_macro_averages():
    report = evaluate(QRELS, RUN, METRICS)

    assert report.per_query["recall@2"] == {"q1": 0.5, "q2": 1.0, "q5": 0.0}
    assert report.per_query["precision@2"] == {"q1": 0.5, "q2": 0.5, "q5": 0.0}
    assert report.per_query["mrr@3"] == {"q1": 1.0, "q2": 0.5, "q5": 0.0}
    assert report.per_query["ndcg@3"]["q1"] == pytest.approx(2.5 / (2 + 1 / log2(3)))
    assert report.per_query["ndcg@3"]["q2"] == pytest.approx(1 / log2(3))
    assert report.per_query["judged@3"] == pytest.approx({"q1": 2 / 3, "q2": 0.5, "q5": 0.0})
    assert report.means["recall@2"] == pytest.approx(1.5 / 3)
    assert report.means["mrr@3"] == pytest.approx(1.5 / 3)


def test_reports_skipped_missing_and_unjudged_queries():
    report = evaluate(QRELS, RUN, ["recall@2"])

    assert report.evaluated_queries == ("q1", "q2", "q5")
    assert report.skipped_queries == ("q3",)
    assert report.unjudged_queries == ("q4",)


def test_zero_policy_scores_queries_without_relevant_documents():
    report = evaluate(QRELS, RUN, ["recall@2", "judged@3"], empty_queries="zero")

    assert report.per_query["recall@2"]["q3"] == 0.0
    assert report.per_query["judged@3"]["q3"] == 1.0
    assert report.means["recall@2"] == pytest.approx(1.5 / 4)
    assert report.skipped_queries == ()


def test_raise_policy_rejects_queries_without_relevant_documents():
    with pytest.raises(ValueError, match="q3"):
        evaluate(QRELS, RUN, ["recall@2"], empty_queries="raise")


def test_relevance_threshold_changes_binary_relevance_only():
    report = evaluate(QRELS, RUN, ["recall@3", "ndcg@3"], relevance_threshold=2)

    assert report.evaluated_queries == ("q1",)
    assert report.skipped_queries == ("q2", "q3", "q5")
    assert report.per_query["recall@3"]["q1"] == 1.0
    assert report.per_query["ndcg@3"]["q1"] == pytest.approx(2.5 / (2 + 1 / log2(3)))


def test_scored_runs_are_sorted_by_score_with_deterministic_ties():
    run = {"q": {"b": 1.0, "a": 1.0, "c": 0.5}}

    assert evaluate({"q": {"a": 1}}, run, ["mrr@3"]).means["mrr@3"] == 1.0
    assert evaluate({"q": {"b": 1}}, run, ["mrr@3"]).means["mrr@3"] == 0.5
    assert evaluate({"q": {"c": 1}}, run, ["mrr@3"]).means["mrr@3"] == pytest.approx(1 / 3)


def test_metric_names_are_case_insensitive_and_deduplicated():
    report = evaluate(QRELS, RUN, ["MRR@3", " mrr@3 "])
    assert list(report.means) == ["mrr@3"]


def test_exponential_gain_is_passed_to_ndcg():
    report = evaluate(QRELS, RUN, ["ndcg@3"], gain="exponential")
    assert report.per_query["ndcg@3"]["q1"] == pytest.approx(3.5 / (3 + 1 / log2(3)))


@pytest.mark.parametrize("metric", ["recall", "recall@0", "recall@-1", "map@10", "ndcg@3.5"])
def test_unknown_metrics_are_rejected(metric):
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate(QRELS, RUN, [metric])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metrics": []}, "at least one metric"),
        ({"empty_queries": "ignore"}, "empty_queries"),
        ({"relevance_threshold": 0}, "relevance_threshold"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    arguments = {"metrics": ["recall@2"], **kwargs}
    with pytest.raises(ValueError, match=message):
        evaluate(QRELS, RUN, **arguments)


def test_fails_when_no_query_is_left_to_evaluate():
    with pytest.raises(ValueError, match="no queries"):
        evaluate({"q": {"a": 0}}, {"q": ["a"]}, ["recall@1"])


def test_string_ranking_is_rejected():
    with pytest.raises(TypeError, match="not a string"):
        evaluate({"q": {"a": 1}}, {"q": "abc"}, ["recall@1"])


@pytest.mark.parametrize(("grade", "error"), [(-1, ValueError), (float("nan"), ValueError)])
def test_invalid_grades_are_rejected(grade, error):
    with pytest.raises(error, match="grade"):
        evaluate({"q": {"a": grade}}, {"q": ["a"]}, ["recall@1"])


def test_non_finite_retrieval_scores_are_rejected():
    with pytest.raises(ValueError, match="retrieval score"):
        evaluate({"q": {"a": 1}}, {"q": {"a": float("nan")}}, ["recall@1"])


def test_many_queries_match_single_query_results():
    qrels = {q: {f"d{q}": 1} for q in range(2_000)}
    run = {q: [f"x{q}", f"d{q}"] if q % 2 else [f"d{q}"] for q in range(2_000)}

    report = evaluate(qrels, run, ["mrr@2"])

    assert len(report.evaluated_queries) == 2_000
    assert report.means["mrr@2"] == pytest.approx(0.75)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        evaluate(QRELS, RUN, ["recall@2"], backend="pytrec")


def test_ranx_backend_reports_a_clear_error_when_ranx_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "ranx", None)
    with pytest.raises(ImportError, match=r"ai-daily-dose\[ranx\]"):
        evaluate(QRELS, RUN, ["recall@2"], backend="ranx")


def _random_judged_run(seed=0, queries=300):
    rng = random.Random(seed)
    docs = [f"d{i}" for i in range(60)]
    qrels, run = {}, {}
    for number in range(queries):
        query_id = f"q{number}"  # q10 sorts before q2, so ordering bugs would show
        judged = rng.sample(docs, rng.randint(0, 12))
        qrels[query_id] = {doc: rng.choice([0, 0, 1, 2, 3]) for doc in judged}
        if number % 25 != 0:  # every 25th query is missing from the run
            retrieved = rng.sample(docs, rng.randint(0, 30))
            # scores rounded to one decimal, so ties are common
            run[query_id] = {doc: round(rng.random(), 1) for doc in retrieved}
    run["unjudged-query"] = {"d1": 1.0}
    return qrels, run


@pytest.fixture
def ranx_available():
    try:
        import ranx  # noqa: F401
    except ImportError:
        pytest.skip("ranx is not installed or cannot be imported")


@pytest.mark.usefixtures("ranx_available")
@pytest.mark.parametrize("gain", ["linear", "exponential"])
@pytest.mark.parametrize("threshold", [1, 2, 3.0])
@pytest.mark.parametrize("empty_queries", ["skip", "zero"])
def test_ranx_backend_matches_reference_backend(gain, threshold, empty_queries):
    qrels, run = _random_judged_run()
    metrics = ["recall@5", "recall@20", "precision@10", "mrr@10", "ndcg@10", "ndcg@3"]
    options = {"relevance_threshold": threshold, "empty_queries": empty_queries, "gain": gain}

    reference = evaluate(qrels, run, metrics, **options)
    fast = evaluate(qrels, run, metrics, backend="ranx", **options)

    assert fast.evaluated_queries == reference.evaluated_queries
    assert fast.skipped_queries == reference.skipped_queries
    assert fast.unjudged_queries == reference.unjudged_queries
    for metric in metrics:
        assert fast.per_query[metric] == pytest.approx(reference.per_query[metric], abs=1e-6)
        assert fast.means[metric] == pytest.approx(reference.means[metric], abs=1e-6)


@pytest.mark.usefixtures("ranx_available")
def test_ranx_backend_supports_a_single_metric_and_list_rankings():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": ["x", "a"], "q2": ["b", "b", "y"]}  # a repeated ID keeps its first position

    report = evaluate(qrels, run, ["mrr@3"], backend="ranx")

    assert report.per_query["mrr@3"] == pytest.approx({"q1": 0.5, "q2": 1.0})


@pytest.mark.usefixtures("ranx_available")
def test_ranx_backend_rejects_unsupported_options():
    with pytest.raises(ValueError, match="judged@k"):
        evaluate(QRELS, RUN, ["judged@3"], backend="ranx")
    with pytest.raises(ValueError, match="integer"):
        evaluate(QRELS, RUN, ["recall@2"], backend="ranx", relevance_threshold=1.5)


@pytest.mark.usefixtures("ranx_available")
def test_ranx_backend_rejects_ids_that_collide_as_strings():
    with pytest.raises(ValueError, match="collide"):
        evaluate({1: {"a": 1}, "1": {"a": 1}}, {1: ["a"], "1": ["a"]}, ["recall@1"], backend="ranx")


def test_trec_files_round_trip_into_evaluation(tmp_path):
    (tmp_path / "qrels.txt").write_text("# judgments\nq1 0 a 2\nq1 0 b 0\n\nq2 0 x 1\n")
    write_run(tmp_path / "run.txt", {"q1": {"a": 0.9, "b": 0.5}, "q2": {"y": 0.9, "x": 0.5}}, "sys")

    qrels = read_qrels(tmp_path / "qrels.txt")
    run = read_run(tmp_path / "run.txt")

    assert qrels == {"q1": {"a": 2.0, "b": 0.0}, "q2": {"x": 1.0}}
    assert (tmp_path / "run.txt").read_text().splitlines()[0] == "q1 Q0 a 1 0.9 sys"
    assert evaluate(qrels, run, ["mrr@2"]).per_query["mrr@2"] == {"q1": 1.0, "q2": 0.5}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("q1 0 a\n", "expected 4 fields"),
        ("q1 0 a high\n", "not a number"),
        ("q1 0 a -1\n", "negative"),
        ("q1 0 a nan\n", "finite"),
        ("q1 0 a 1\nq1 0 a 2\n", "conflicting"),
    ],
)
def test_malformed_qrels_files_are_rejected(tmp_path, content, message):
    (tmp_path / "qrels.txt").write_text(content)
    with pytest.raises(ValueError, match=message):
        read_qrels(tmp_path / "qrels.txt")


def test_run_files_keep_the_highest_score_for_repeated_documents(tmp_path):
    (tmp_path / "run.txt").write_text("q1 Q0 a 1 0.2 s\nq1 Q0 a 2 0.8 s\nq1 Q0 b 3 0.5 s\n")
    assert read_run(tmp_path / "run.txt") == {"q1": {"a": 0.8, "b": 0.5}}
    (tmp_path / "bad.txt").write_text("q1 Q0 a 1 inf s\n")
    with pytest.raises(ValueError, match="finite"):
        read_run(tmp_path / "bad.txt")
