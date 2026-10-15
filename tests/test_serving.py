"""Offline/online serving parity and request-contract tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recommender.evaluation.metrics import rerank_pairs
from recommender.models import LambdaMARTReranker
from recommender.models.bm25 import BM25Index, job_document
from recommender.serving import build_topn_payloads, parse_request


def test_published_topn_matches_offline_two_stage_ranking(store, split, catalogue):
    X, y, group = store.feature_matrix(split.train)
    reranker = LambdaMARTReranker(n_estimators=20).fit(X, y, group)

    query_index = int(split.train["query_index"].iloc[0])
    payload = build_topn_payloads(
        store, reranker, catalogue, split, k=5, query_indices=[query_index]
    )[0]
    published = [int(row["job_id"]) for row in payload["recommendations"]]

    # Recompute the same ranking independently and assert byte-for-byte parity.
    frame = catalogue.frame
    pool = split.candidate_pool(split.train)
    bm25 = BM25Index().fit(
        pool,
        [
            job_document(
                title=frame.at[int(idx), "title"],
                canonical_skills=frame.at[int(idx), "canonical_skills"],
                seniority=frame.at[int(idx), "seniority"],
                location=frame.at[int(idx), "location"],
            )
            for idx in pool
        ],
    )
    profile = frame.at[query_index, "canonical_skills"]
    timestamp = int(frame.at[query_index, "posted_at"])
    retrieved = bm25.top_k(" ".join(profile), 5)
    pairs = rerank_pairs(store, query_index, profile, timestamp, retrieved)
    scores = reranker.predict(store.pair_frame(pairs))
    order = np.argsort(-scores, kind="stable")
    offline = [int(catalogue.job_ids[int(idx)]) for idx in retrieved[order]]

    assert published == offline
    json.dumps(payload)


def test_request_parser_validates_inputs():
    assert parse_request({"query_job_id": "12", "k": "5"}) == ("12", 5)


def test_request_parser_rejects_missing_query_job_id():
    with pytest.raises(ValueError, match="query_job_id"):
        parse_request({})


def test_request_parser_rejects_out_of_range_k():
    with pytest.raises(ValueError, match="between 1 and"):
        parse_request({"query_job_id": "1", "k": "1000"})


def test_build_topn_payloads_rejects_non_positive_k(store, split, catalogue):
    reranker = LambdaMARTReranker(n_estimators=5)
    X, y, group = store.feature_matrix(split.train)
    reranker.fit(X, y, group)
    with pytest.raises(ValueError, match="k must be positive"):
        build_topn_payloads(store, reranker, catalogue, split, k=0)
