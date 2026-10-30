"""Metrics for evaluating retrieval quality."""

from .batch import EvaluationReport, evaluate
from .ndcg_at_k import ndcg_at_k
from .precision_at_k import precision_at_k
from .recall_at_k import recall_at_k
from .reciprocal_rank_at_k import reciprocal_rank_at_k
from .statistics import (
    ConfidenceInterval,
    SignificanceResult,
    bootstrap_ci,
    paired_randomization_test,
)
from .trec import read_qrels, read_run, write_run

__all__ = [
    "ConfidenceInterval",
    "EvaluationReport",
    "SignificanceResult",
    "bootstrap_ci",
    "evaluate",
    "ndcg_at_k",
    "paired_randomization_test",
    "precision_at_k",
    "recall_at_k",
    "read_qrels",
    "read_run",
    "reciprocal_rank_at_k",
    "write_run",
]
