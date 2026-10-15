"""Offline/online feature parity tests.

The pipeline is used two ways: batched over a whole partition (training,
evaluation) and one query-group at a time (serving). If the per-group
computation depended on anything outside that group -- a shared running
statistic, accidental global state -- the two paths would disagree. These
tests pin down that they cannot.
"""

from __future__ import annotations

import pandas as pd

from recommender.data.loader import build_proxy_pairs


def test_single_group_matches_batched_computation(store, split):
    """Scoring one query's rows alone must equal scoring it inside the batch."""
    batched = store.pair_frame(split.train)

    one_query = split.train["query_index"].iloc[0]
    subset = split.train[split.train["query_index"] == one_query]
    isolated = store.pair_frame(subset)

    pd.testing.assert_frame_equal(
        batched.loc[subset.index].reset_index(drop=True),
        isolated.reset_index(drop=True),
    )


def test_feature_matrix_group_array_matches_lightgbm_contract(store, split):
    """The ``group`` array must sum to the row count and respect row order."""
    X, y, group = store.feature_matrix(split.train)
    assert group.sum() == len(X)

    # Row order must stay grouped by query_index, contiguous, which is what
    # LightGBM's `group` parameter assumes.
    query_order = split.train["query_index"].to_numpy()
    boundaries = group.cumsum()[:-1]
    # Every boundary must fall exactly where query_index changes (or the
    # groups are not actually contiguous in row order).
    changes = (query_order[1:] != query_order[:-1]).nonzero()[0] + 1
    assert set(boundaries.tolist()) <= set(changes.tolist()) | {len(query_order)}


def test_rebuilding_pairs_with_same_seed_is_deterministic(catalogue):
    """Determinism is required for the training/serving split to be reproducible."""
    first = build_proxy_pairs(catalogue, n_negatives=5, seed=3)
    second = build_proxy_pairs(catalogue, n_negatives=5, seed=3)
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_job_vector_matches_catalogue_wide_frame(store):
    """``job_vector`` for a known job must agree with the exported frame."""
    frames = store.to_frames()
    jobs = frames["jobs"].set_index("job_index")
    sample_job = int(jobs.index[0])
    vector = store.job_vector(sample_job)
    assert vector["candidate_job_popularity_rank"] == jobs.loc[
        sample_job, "candidate_job_popularity_rank"
    ]
    assert vector["candidate_job_trailing_postings"] == jobs.loc[
        sample_job, "candidate_job_trailing_postings"
    ]
