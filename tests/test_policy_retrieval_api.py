import pytest
from fastapi.testclient import TestClient

from ai_engineering.retrieval.api import app

client = TestClient(app)

USER = {
    "role": "support_engineer",
    "access_level": "internal",
    "region": "eu",
    "allowed_product_versions": ["v2", "v3"],
}


def doc(doc_id, score, access_level="public", region="eu", product_version="v3"):
    return {
        "id": doc_id,
        "content": f"content of {doc_id}",
        "score": score,
        "metadata": {
            "access_level": access_level,
            "region": region,
            "product_version": product_version,
        },
    }


def post(candidates, user=None, query="how do I reset my password?"):
    return client.post(
        "/retrieve",
        json={"query": query, "user_context": user or USER, "candidate_documents": candidates},
    )


def result_ids(response):
    return [item["id"] for item in response.json()["results"]]


@pytest.mark.parametrize(
    ("access_level", "allowed"),
    [("public", True), ("internal", True), ("confidential", False), ("restricted", False)],
)
def test_access_level_must_not_exceed_user_clearance(access_level, allowed):
    response = post([doc("d1", 0.9, access_level=access_level)])

    assert response.status_code == 200
    assert result_ids(response) == (["d1"] if allowed else [])


def test_unauthorized_content_is_not_leaked_in_response():
    secret = doc("secret-doc", 0.99, access_level="restricted")
    secret["content"] = "TOP-SECRET roadmap"

    response = post([secret, doc("ok", 0.1)])

    assert result_ids(response) == ["ok"]
    assert "secret-doc" not in response.text
    assert "TOP-SECRET" not in response.text


@pytest.mark.parametrize(
    ("region", "allowed"),
    [("eu", True), ("EU", True), ("global", True), ("us", False), ("apac", False)],
)
def test_region_must_match_user_or_be_global(region, allowed):
    response = post([doc("d1", 0.9, region=region)])

    assert result_ids(response) == (["d1"] if allowed else [])


@pytest.mark.parametrize(
    ("product_version", "allowed"),
    [("v2", True), ("v3", True), ("v1", False), ("v4", False), ("V3", False)],
)
def test_product_version_must_be_in_allowed_list(product_version, allowed):
    response = post([doc("d1", 0.9, product_version=product_version)])

    assert result_ids(response) == (["d1"] if allowed else [])


def test_empty_allowed_versions_admits_nothing():
    user = {**USER, "allowed_product_versions": []}

    response = post([doc("d1", 0.9)], user=user)

    assert response.status_code == 200
    assert result_ids(response) == []


def test_document_must_pass_every_policy():
    candidates = [
        doc("all-pass", 0.1),
        doc("bad-access", 0.9, access_level="confidential"),
        doc("bad-region", 0.9, region="us"),
        doc("bad-version", 0.9, product_version="v1"),
    ]

    assert result_ids(post(candidates)) == ["all-pass"]


def test_filtering_happens_before_top_k_selection():
    # The three highest scores are ineligible; a post-filter would return nothing.
    candidates = [
        doc("blocked-1", 0.99, access_level="restricted"),
        doc("blocked-2", 0.98, region="us"),
        doc("blocked-3", 0.97, product_version="v1"),
        doc("ok-1", 0.50),
        doc("ok-2", 0.40),
        doc("ok-3", 0.30),
        doc("ok-4", 0.20),
    ]

    response = post(candidates)

    assert result_ids(response) == ["ok-1", "ok-2", "ok-3"]


def test_returns_top_three_ranked_by_score_with_contract():
    candidates = [doc("a", 0.2), doc("b", 0.9), doc("c", 0.5), doc("d", 0.7), doc("e", 0.1)]

    body = post(candidates, query="q").json()

    assert body["query"] == "q"
    assert body["top_k"] == 3
    assert body["count"] == 3
    assert [(r["id"], r["rank"], r["score"]) for r in body["results"]] == [
        ("b", 1, 0.9),
        ("d", 2, 0.7),
        ("c", 3, 0.5),
    ]
    assert body["results"][0]["metadata"] == candidates[1]["metadata"]


def test_score_ties_are_broken_by_id():
    candidates = [doc("z", 0.5), doc("a", 0.5), doc("m", 0.5), doc("b", 0.5)]

    assert result_ids(post(candidates)) == ["a", "b", "m"]


def test_fewer_than_three_eligible_returns_all_eligible():
    response = post([doc("a", 0.2), doc("b", 0.9, region="us"), doc("c", 0.5)])

    assert result_ids(response) == ["c", "a"]
    assert response.json()["count"] == 2


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        [doc("x", 0.9, access_level="restricted"), doc("y", 0.8, region="us")],
    ],
    ids=["no-candidates", "none-pass-policy"],
)
def test_empty_result_is_success_with_empty_list(candidates):
    response = post(candidates)

    assert response.status_code == 200
    assert response.json() == {
        "query": "how do I reset my password?",
        "top_k": 3,
        "count": 0,
        "results": [],
    }


def _without_metadata_field(field):
    candidate = doc("d1", 0.9)
    del candidate["metadata"][field]
    return {"candidate_documents": [candidate]}


@pytest.mark.parametrize(
    "override",
    [
        {"query": ""},
        {"user_context": {**USER, "access_level": "admin"}},
        {"user_context": {k: v for k, v in USER.items() if k != "region"}},
        {"user_context": {**USER, "acess_level": "restricted"}},
        _without_metadata_field("access_level"),
        _without_metadata_field("region"),
        _without_metadata_field("product_version"),
        {"candidate_documents": [{**doc("d1", 0.9), "score": "high"}]},
        {"candidate_documents": [doc("d1", 0.9)] * 1001},
    ],
    ids=[
        "empty-query",
        "unknown-access-level",
        "missing-user-region",
        "misspelled-user-field",
        "missing-doc-access-level",
        "missing-doc-region",
        "missing-doc-version",
        "non-numeric-score",
        "too-many-candidates",
    ],
)
def test_invalid_payload_is_rejected_with_422(override):
    payload = {"query": "q", "user_context": USER, "candidate_documents": [], **override}

    response = client.post("/retrieve", json=payload)

    assert response.status_code == 422


def test_non_finite_score_is_rejected_with_422():
    body = (
        '{"query": "q", "user_context": {"role": "r", "access_level": "public", "region": "eu",'
        ' "allowed_product_versions": ["v3"]}, "candidate_documents": [{"id": "d1",'
        ' "content": "c", "score": NaN, "metadata": {"access_level": "public", "region": "eu",'
        ' "product_version": "v3"}}]}'
    )

    response = client.post("/retrieve", content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 422


def test_validation_error_does_not_echo_rejected_input():
    response = post([doc("d1", 0.9, access_level="top-secret")])

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == [
        "body",
        "candidate_documents",
        0,
        "metadata",
        "access_level",
    ]
    assert "top-secret" not in response.text


def test_wrong_method_returns_405():
    assert client.get("/retrieve").status_code == 405
