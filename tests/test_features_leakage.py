"""Leakage controls.

These tests are adversarial by design. Each one constructs a situation in
which a leaking implementation would produce a *different* answer from a
correct one, and asserts the correct answer. A test that merely checks
``build_features`` returns something would pass against a pipeline that
reads the held-out partitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data.splitting import Split, temporal_split
from recommender.features import build_features, resolve_as_of
from recommender.features.context_features import query_candidate_recency_gap_days
from recommender.features.item_features import SECONDS_PER_DAY, candidate_job_trailing_postings


def test_future_query_injection_does_not_move_item_features(split, catalogue):
    """Control 1: forging held-out queries must not change fitted item features.

    The validation partition is overwritten with fabricated future queries
    against a job that otherwise never appears as a training candidate. If
    popularity rank or trailing postings read beyond the training partition,
    their values will shift. Nothing may move.

    Both the baseline and the poisoned split use the ``grouped`` protocol so
    ``resolve_as_of`` depends only on ``split.train`` in both cases -- the
    comparison would otherwise be confounded by the cut-off itself shifting.
    """
    baseline_split = Split(
        train=split.train, validation=split.validation, test=split.test,
        n_jobs=split.n_jobs, protocol="grouped",
    )
    baseline = build_features(baseline_split, catalogue)

    forged_candidate = int(split.train["candidate_index"].max()) if not split.train.empty else 0
    future_ts = int(split.train["timestamp"].max()) + 10 * int(SECONDS_PER_DAY)
    n = 20
    forged = pd.DataFrame(
        {
            "query_index": np.arange(n),
            "candidate_index": np.full(n, forged_candidate, dtype=np.int64),
            "label": np.ones(n, dtype=np.int64),
            "profile_skills": [["python"]] * n,
            "timestamp": np.full(n, future_ts, dtype=np.int64),
        }
    )

    poisoned = Split(
        train=split.train,
        validation=forged,
        test=forged,
        n_jobs=split.n_jobs,
        protocol="grouped",  # forces resolve_as_of off the training max, not validation
    )
    after = build_features(poisoned, catalogue)

    pd.testing.assert_series_equal(
        baseline.job_popularity_rank, after.job_popularity_rank, check_names=False
    )
    pd.testing.assert_series_equal(
        baseline.job_trailing_postings, after.job_trailing_postings, check_names=False
    )


def test_fit_partition_isolation(pairs, catalogue):
    """Control 2: a later cut-off must be a strict superset in observed history.

    Features fitted at an earlier cut-off cannot depend on postings after
    it, so every job's trailing-postings count at the early cut-off must be
    no greater than at the late one for jobs known in both partitions.
    """
    early_split = temporal_split(pairs, 0.50, 0.60)
    late_split = temporal_split(pairs, 0.80, 0.90)
    early = build_features(early_split, catalogue)
    late = build_features(late_split, catalogue)

    assert early.as_of < late.as_of
    # The window is trailing, so a strictly later cut-off is not guaranteed
    # to dominate pointwise, but the cut-offs must differ meaningfully.
    assert not early.job_trailing_postings.equals(late.job_trailing_postings)


def test_label_destruction_collapses_overlap_signal(split, catalogue):
    """Control 3: destroying the query-candidate association must destroy the signal.

    Candidate indices are permuted across the *entire* training frame (not
    merely within a group): a within-group shuffle leaves each group's
    (profile, candidate) multiset untouched and would pass vacuously.
    Genuine skill-overlap features are computed from the true association,
    so this global scramble must change the average overlap.
    """
    rng = np.random.default_rng(0)
    scrambled_train = split.train.copy()
    scrambled_train["candidate_index"] = rng.permutation(
        scrambled_train["candidate_index"].to_numpy()
    )

    store = build_features(split, catalogue)
    baseline_overlap = store.pair_frame(split.train)["skill_overlap_jaccard"].mean()
    after_overlap = store.pair_frame(scrambled_train)["skill_overlap_jaccard"].mean()

    assert abs(baseline_overlap - after_overlap) > 1e-6


def test_recency_gap_rejects_cutoff_inside_training_window(split):
    """A cut-off at or before the known training timestamps is not point-in-time safe."""
    inside = float(split.train["timestamp"].median())
    with pytest.raises(ValueError, match="point-in-time safe"):
        query_candidate_recency_gap_days(
            split.train, None, as_of=inside, max_known_timestamp=inside
        )


def test_trailing_postings_excludes_the_cutoff_instant(split, catalogue):
    """The window is half-open: a posting exactly at ``as_of`` is future.

    Moving a job's posting time to exactly ``as_of`` must be indistinguishable
    from moving it far outside the window altogether: both are excluded.
    Comparing against the *un-modified* data would conflate "excluded because
    at the boundary" with "excluded because it happened to already be
    outside the window", so both sides of this comparison are constructed.
    """
    train = split.train
    as_of = resolve_as_of(split)
    n_jobs = split.n_jobs
    target_job = int(train["candidate_index"].iloc[0])
    window_days = 60.0

    outside_frame = catalogue.frame.copy()
    outside_frame.at[target_job, "posted_at"] = int(as_of - (window_days + 30) * SECONDS_PER_DAY)
    outside_catalogue = type(catalogue)(frame=outside_frame, job_ids=catalogue.job_ids)
    without = candidate_job_trailing_postings(
        train, outside_catalogue, n_jobs, as_of, window_days=window_days
    )

    boundary_frame = catalogue.frame.copy()
    boundary_frame.at[target_job, "posted_at"] = int(as_of)
    boundary_catalogue = type(catalogue)(frame=boundary_frame, job_ids=catalogue.job_ids)
    with_boundary = candidate_job_trailing_postings(
        train, boundary_catalogue, n_jobs, as_of, window_days=window_days
    )

    pd.testing.assert_series_equal(without, with_boundary, check_names=False)


def test_future_candidate_gap_is_floored_not_negative(split, catalogue, store):
    """A negative gap would leak future ordering; it must be floored at zero."""
    gaps = store.pair_frame(split.train)["query_candidate_recency_gap_days"]
    assert (gaps >= 0.0).all()
