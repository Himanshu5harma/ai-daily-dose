"""Load the DocLens dev questions and their gold evidence anchors.

Gold labels are ``(source_id, anchor)`` pairs, not chunk IDs, so the same labels
work for every chunking strategy. A question has one or more *evidence groups*;
a group is satisfied by any one of its alternative references.

Query types used in the experiment tables:

- ``exact``: names an API or option directly (DocLens category ``exact``).
- ``conceptual``: describes the need in other words (DocLens ``paraphrase``).
- ``multi``: needs several pieces of evidence (``multi_source`` and ``comparison``).

``missing_evidence`` questions have no gold evidence and are skipped by default.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .corpus import DATA_DIR

QUERY_TYPES = {
    "exact": "exact",
    "paraphrase": "conceptual",
    "multi_source": "multi",
    "comparison": "multi",
}


@dataclass(frozen=True)
class Reference:
    source_id: str
    anchor: str


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    references: tuple[Reference, ...]  # any one of these satisfies the group


@dataclass(frozen=True)
class Question:
    qid: str
    query: str
    category: str
    query_type: str
    versions: tuple[str, ...]
    evidence_groups: tuple[EvidenceGroup, ...]


def load_questions(
    path: Path = DATA_DIR / "dev.json", answerable_only: bool = True
) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for q in raw["questions"]:
        if answerable_only and not q["answerable"]:
            continue
        groups = tuple(
            EvidenceGroup(
                f"{q['id']}:{g['id']}",
                tuple(Reference(r["source_id"], r["anchor"]) for r in g["references"]),
            )
            for g in q["evidence_groups"]
        )
        questions.append(
            Question(
                qid=q["id"],
                query=q["query"],
                category=q["category"],
                query_type=QUERY_TYPES.get(q["category"], "none"),
                versions=tuple(q.get("versions", ())),
                evidence_groups=groups,
            )
        )
    return questions
