"""Two-stage evaluation protocol and the pure ranking metrics it is built from.

Stage 1 (retrieval) and stage 2 (re-ranking) are measured separately and
deliberately:

* **Recall@K** (stage 1) measures only whether lexical retrieval kept the
  true profile-job pair in play at all, independent of how it is ordered.
* **NDCG@K / MAP@K** (stage 2) measure ranking quality *within* whatever
  stage 1 already retrieved, using the proxy relevance label.

Evaluating both over the same candidate pool, with the same K, is what makes
"stage 1 recall" and "stage 2 ranking quality" a genuine decomposition of
one end-to-end number rather than two unrelated statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.loader import JobPostingData
from recommender.features.pipeline import FeatureStore
from recommender.models.baselines import LexicalOnlyBaseline, PopularityBaseline
from recommender.models.lambdamart import LambdaMARTReranker


def recall_at_k(relevant_rank: int | None, k: int) -> float:
    """Whether a single relevant item survived into the top-``k``.

    Args:
        relevant_rank: 0-based position of the relevant item in the *full*
            ranked list, or ``None`` if it was not retrieved at all.
        k: Cut-off.

    Returns:
        ``1.0`` if retrieved within the top-``k``, else ``0.0``.
    """
    return 1.0 if relevant_rank is not None and relevant_rank < k else 0.0


def ndcg_at_k(relevance: Sequence[int], k: int) -> float:
    """Normalized discounted cumulative gain of a ranked relevance sequence.

    Args:
        relevance: Binary (or graded) relevance, in ranked order.
        k: Cut-off.

    Returns:
        NDCG@K in ``[0, 1]``; ``0.0`` if no relevant item exists at all.
    """
    truncated = list(relevance)[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(truncated))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision_at_k(
    relevance: Sequence[int], k: int, total_relevant: int | None = None
) -> float:
    """Average precision at K for a single query.

    Args:
        relevance: Binary relevance, in ranked order.
        k: Cut-off.
        total_relevant: Total number of relevant items for this query,
            which may exceed what appears within the top-``k``. Defaults to
            the count observed within ``relevance`` itself (correct for this
            project's proxy supervision, where exactly one item is ever
            relevant).

    Returns:
        Average precision in ``[0, 1]``.
    """
    truncated = list(relevance)[:k]
    denominator = total_relevant if total_relevant is not None else sum(truncated)
    if denominator <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, rel in enumerate(truncated, start=1):
        if rel:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / denominator


@dataclass(frozen=True)
class EvaluationReport:
    model: str
    k: int
    recall: float
    ndcg: float
    map: float
    n_queries: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "k": self.k,
            "recall": round(self.recall, 4),
            "ndcg": round(self.ndcg, 4),
            "map": round(self.map, 4),
            "n_queries": self.n_queries,
        }


def rerank_pairs(
    store: FeatureStore,
    query_index: int,
    profile_skills: list[str],
    timestamp: int,
    candidates: np.ndarray,
) -> pd.DataFrame:
    """Build pair rows for re-ranking a retrieved candidate set.

    Re-uses the fitted :class:`FeatureStore`'s own ``pair_frame``, which is
    what makes stage-2 scoring at evaluation time identical in mechanism to
    stage-2 scoring at training time.
    """
    n = len(candidates)
    return pd.DataFrame(
        {
            "query_index": np.full(n, query_index, dtype=np.int64),
            "candidate_index": candidates,
            "profile_skills": [profile_skills] * n,
            "timestamp": np.full(n, timestamp, dtype=np.int64),
        }
    )


def evaluate_two_stage(
    store: FeatureStore,
    reranker: LambdaMARTReranker,
    data: JobPostingData,
    pairs_partition: pd.DataFrame,
    k: int = 10,
) -> dict[str, EvaluationReport]:
    """Evaluate the popularity, lexical-only and two-stage models on one partition.

    For every query group in ``pairs_partition``, stage 1 retrieves from the
    *partition's own* candidate pool (the realistic retrieval universe for
    that evaluation slice), and stage 2 re-ranks exactly that retrieved set.

    Args:
        store: A :class:`FeatureStore` fitted on the training partition.
        reranker: A fitted :class:`LambdaMARTReranker`.
        data: The job catalogue.
        pairs_partition: Pair rows for one evaluation partition (validation
            or test), one contiguous group per query.
        k: Cut-off shared by every metric and every model.

    Returns:
        A dict keyed by model name: ``"Popularity"``, ``"LexicalOnly"``,
        ``"TwoStage"``.

    Raises:
        ValueError: If the partition contains no groups.
    """
    pool = pairs_partition["candidate_index"].unique()
    pool.sort()
    if pool.size == 0:
        raise ValueError("Cannot evaluate an empty partition.")

    lexical = LexicalOnlyBaseline.fit(data, pool)
    popularity = PopularityBaseline(data=data)

    recalls: dict[str, list[float]] = {"Popularity": [], "LexicalOnly": [], "TwoStage": []}
    ndcgs: dict[str, list[float]] = {"Popularity": [], "LexicalOnly": [], "TwoStage": []}
    maps: dict[str, list[float]] = {"Popularity": [], "LexicalOnly": [], "TwoStage": []}

    for query_index, group in pairs_partition.groupby("query_index", sort=False):
        profile_skills = group["profile_skills"].iloc[0]
        timestamp = int(group["timestamp"].iloc[0])
        positives = group.loc[group["label"] == 1, "candidate_index"]
        if positives.empty:
            continue
        true_candidate = int(positives.iloc[0])

        # Popularity: no retrieval, scores the whole partition pool directly.
        pop_ranked = popularity.rank(profile_skills, pool, k)
        _accumulate(recalls, ndcgs, maps, "Popularity", pop_ranked, true_candidate, k)

        # Stage 1: lexical retrieval over the partition pool.
        stage1_ranked = lexical.rank(profile_skills, k)
        _accumulate(recalls, ndcgs, maps, "LexicalOnly", stage1_ranked, true_candidate, k)

        # Stage 2: re-rank exactly what stage 1 retrieved.
        if stage1_ranked.size:
            candidate_pairs = rerank_pairs(
                store, int(query_index), profile_skills, timestamp, stage1_ranked
            )
            scores = reranker.predict(store.pair_frame(candidate_pairs))
            order = np.argsort(-scores, kind="stable")
            stage2_ranked = stage1_ranked[order]
        else:
            stage2_ranked = stage1_ranked
        _accumulate(recalls, ndcgs, maps, "TwoStage", stage2_ranked, true_candidate, k)

    return {
        name: EvaluationReport(
            model=name,
            k=k,
            recall=float(np.mean(recalls[name])) if recalls[name] else 0.0,
            ndcg=float(np.mean(ndcgs[name])) if ndcgs[name] else 0.0,
            map=float(np.mean(maps[name])) if maps[name] else 0.0,
            n_queries=len(recalls[name]),
        )
        for name in ("Popularity", "LexicalOnly", "TwoStage")
    }


def _accumulate(
    recalls: dict[str, list[float]],
    ndcgs: dict[str, list[float]],
    maps: dict[str, list[float]],
    name: str,
    ranked: np.ndarray,
    true_candidate: int,
    k: int,
) -> None:
    matches = np.flatnonzero(ranked == true_candidate)
    rank = int(matches[0]) if matches.size else None
    relevance = [1 if int(job) == true_candidate else 0 for job in ranked[:k]]

    recalls[name].append(recall_at_k(rank, k))
    ndcgs[name].append(ndcg_at_k(relevance, k))
    maps[name].append(average_precision_at_k(relevance, k, total_relevant=1))
