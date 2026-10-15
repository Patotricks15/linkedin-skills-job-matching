"""Build the top-N payloads published to DynamoDB.

This is the offline half of the serving path: it runs the *exact* two-stage
pipeline (BM25 retrieval, then LambdaMART re-ranking) that
``recommender.evaluation.metrics.evaluate_two_stage`` scores, which is what
lets ``tests/test_serving.py`` assert the published ranking is byte-for-byte
what the offline evaluation saw -- the classic training/serving skew bug.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from recommender.data.loader import JobPostingData
from recommender.data.splitting import Split
from recommender.evaluation.metrics import rerank_pairs
from recommender.features.pipeline import FeatureStore
from recommender.models.bm25 import BM25Index, job_document
from recommender.models.lambdamart import LambdaMARTReranker


def build_topn_payloads(
    store: FeatureStore,
    reranker: LambdaMARTReranker,
    data: JobPostingData,
    split: Split,
    k: int = 20,
    query_indices: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Materialize per-profile top-``k`` job lists in DynamoDB item shape.

    Args:
        store: A fitted :class:`FeatureStore`.
        reranker: A fitted :class:`LambdaMARTReranker`.
        data: The job catalogue.
        split: The split whose training pool defines the retrieval universe.
        k: List length to store.
        query_indices: Job indices to publish a profile for; defaults to
            every job that appeared as a query in training.

    Returns:
        One dict per profile, ready for ``table.put_item``.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    frame = data.frame
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

    targets = (
        sorted(split.train["query_index"].unique()) if query_indices is None else query_indices
    )

    payloads = []
    for query_index in targets:
        profile_skills = frame.at[int(query_index), "canonical_skills"]
        timestamp = int(frame.at[int(query_index), "posted_at"])

        retrieved = bm25.top_k(" ".join(profile_skills), k)
        candidate_pairs = rerank_pairs(
            store, int(query_index), profile_skills, timestamp, retrieved
        )
        scores = reranker.predict(store.pair_frame(candidate_pairs))
        order = np.argsort(-scores, kind="stable")
        ranked = retrieved[order]

        payloads.append(
            {
                "query_job_id": str(int(data.job_ids[int(query_index)])),
                "model": "bm25-lambdamart",
                "recommendations": [
                    {"rank": rank, "job_id": str(int(data.job_ids[int(job_index)]))}
                    for rank, job_index in enumerate(ranked, start=1)
                ],
            }
        )
    return payloads
