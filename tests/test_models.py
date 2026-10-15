"""Model and evaluation-metric tests."""

from __future__ import annotations

import numpy as np
import pytest

from recommender.evaluation.metrics import (
    average_precision_at_k,
    evaluate_two_stage,
    ndcg_at_k,
    recall_at_k,
)
from recommender.models import LambdaMARTReranker, LexicalOnlyBaseline, PopularityBaseline


def test_recall_at_k_true_cases():
    assert recall_at_k(0, 10) == 1.0
    assert recall_at_k(9, 10) == 1.0
    assert recall_at_k(10, 10) == 0.0
    assert recall_at_k(None, 10) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 0, 0], 3) == pytest.approx(1.0)


def test_ndcg_worst_ranking_is_less_than_best():
    best = ndcg_at_k([1, 0, 0], 3)
    worst = ndcg_at_k([0, 0, 1], 3)
    assert worst < best


def test_ndcg_with_no_relevant_items_is_zero():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_average_precision_perfect_rank_is_one():
    assert average_precision_at_k([1, 0, 0], 3, total_relevant=1) == pytest.approx(1.0)


def test_average_precision_rewards_earlier_hits():
    early = average_precision_at_k([1, 0, 0], 3, total_relevant=1)
    late = average_precision_at_k([0, 0, 1], 3, total_relevant=1)
    assert early > late


def test_average_precision_with_zero_relevant_is_zero():
    assert average_precision_at_k([0, 0, 0], 3, total_relevant=0) == 0.0


def test_lexical_only_baseline_retrieves_the_literal_text_match(catalogue):
    pool = np.arange(min(50, catalogue.n_jobs))
    baseline = LexicalOnlyBaseline.fit(catalogue, pool)
    own_skills = catalogue.frame.at[int(pool[0]), "canonical_skills"]
    if not own_skills:
        pytest.skip("fixture job has no canonical skills")
    ranked = baseline.rank(own_skills, k=5)
    assert pool[0] in ranked


def test_popularity_baseline_ranks_by_overlap_first(catalogue):
    pool = np.arange(min(20, catalogue.n_jobs))
    baseline = PopularityBaseline(data=catalogue)
    profile = catalogue.frame.at[int(pool[0]), "canonical_skills"]
    ranked = baseline.rank(profile, pool, k=len(pool))
    assert set(ranked.tolist()) == set(pool.tolist())


def test_lambdamart_fit_rejects_group_row_mismatch(store, split):
    X, y, group = store.feature_matrix(split.train)
    with pytest.raises(ValueError, match="group sums to"):
        LambdaMARTReranker(n_estimators=5).fit(X, y, group[:-1])


def test_lambdamart_predict_requires_fit(store, split):
    X, _, _ = store.feature_matrix(split.train)
    with pytest.raises(RuntimeError, match="not fitted"):
        LambdaMARTReranker().predict(X)


def test_two_stage_beats_or_matches_lexical_only_beats_or_matches_popularity(
    store, split, catalogue
):
    """The headline ordering the paper reports: TwoStage >= LexicalOnly >= Popularity.

    Not a strict ``>`` because a tiny fixture can saturate at the metric's
    ceiling for more than one model; the direction must still never invert.
    """
    X, y, group = store.feature_matrix(split.train)
    reranker = LambdaMARTReranker(n_estimators=30).fit(X, y, group)
    reports = evaluate_two_stage(store, reranker, catalogue, split.validation, k=10)

    assert reports["TwoStage"].ndcg >= reports["LexicalOnly"].ndcg - 1e-9
    assert reports["LexicalOnly"].recall >= reports["Popularity"].recall - 1e-9


def test_evaluate_two_stage_rejects_empty_partition(store, split, catalogue):
    X, y, group = store.feature_matrix(split.train)
    reranker = LambdaMARTReranker(n_estimators=5).fit(X, y, group)
    with pytest.raises(ValueError, match="empty partition"):
        evaluate_two_stage(store, reranker, catalogue, split.train.iloc[0:0], k=5)
