"""Item-family transforms: catalogue-wide aggregates about candidate jobs.

Both transforms materialize one value per ``job_index`` across the *whole*
catalogue, fitted only from jobs that appear as a candidate somewhere in the
training partition -- an unseen job at serving time gets the declared fill
rather than a crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data.loader import JobPostingData
from recommender.features.taxonomy import dominant_category

SECONDS_PER_DAY = 86400.0


def candidate_job_popularity_rank(
    train_pairs: pd.DataFrame,
    data: JobPostingData,
    n_jobs: int,
    normalize: bool = True,
) -> pd.Series:
    """Normalized dense rank of each job's taxonomy-category popularity.

    Popularity is measured by how often a job's dominant category appears
    among the *training* candidates, not by the job's own individual
    frequency (each job appears at most once as a candidate's own positive).

    Args:
        train_pairs: The training partition's pair rows.
        data: The job catalogue.
        n_jobs: Size of the catalogue (defines the output index range).
        normalize: If true, scale ranks to ``[0, 1]`` with 0 the most popular.

    Returns:
        A series indexed ``0..n_jobs-1``.
    """
    categories = data.frame["canonical_skills"].apply(dominant_category)

    candidate_categories = categories.iloc[train_pairs["candidate_index"].to_numpy()]
    category_counts = candidate_categories.value_counts()

    job_scores = categories.map(category_counts).fillna(0.0).to_numpy(dtype=np.float64)
    if not normalize:
        return pd.Series(job_scores, index=np.arange(n_jobs))

    # Higher raw count -> more popular -> rank 0. Ties share the same rank.
    series_scores = pd.Series(job_scores)
    tie_adjusted = series_scores.rank(method="average", ascending=False) - 1.0
    normalized = tie_adjusted.to_numpy() / max(n_jobs - 1, 1)
    return pd.Series(normalized, index=np.arange(n_jobs))


def candidate_job_trailing_postings(
    train_pairs: pd.DataFrame,
    data: JobPostingData,
    n_jobs: int,
    as_of: float,
    window_days: float = 60.0,
    log_transform: bool = True,
) -> pd.Series:
    """Count of same-category postings in a trailing window before ``as_of``.

    Only jobs that appear as a candidate in the training partition
    contribute to the count: a job the training process never observed
    cannot inform a "how common is this category right now" estimate.

    Args:
        train_pairs: The training partition's pair rows.
        data: The job catalogue.
        n_jobs: Size of the catalogue (defines the output index range).
        as_of: Point-in-time cut-off. Postings at or after this instant are
            strictly excluded -- never ``<=``.
        window_days: Trailing window length, in days.
        log_transform: If true, apply ``log1p`` to compress the long tail.

    Returns:
        A series indexed ``0..n_jobs-1``.
    """
    categories = data.frame["canonical_skills"].apply(dominant_category).to_numpy()
    posted_at = data.frame["posted_at"].to_numpy(dtype=np.float64)

    known_jobs = np.zeros(n_jobs, dtype=bool)
    known_jobs[train_pairs["candidate_index"].unique()] = True

    window_start = as_of - window_days * SECONDS_PER_DAY
    # Strict `<` against as_of: an event exactly at the cut-off is future,
    # never part of the trailing history.
    eligible = known_jobs & (posted_at < as_of) & (posted_at >= window_start)

    counts = np.zeros(n_jobs, dtype=np.float64)
    eligible_categories = categories[eligible]
    category_counts = pd.Series(eligible_categories).value_counts()

    for job_index in range(n_jobs):
        category = categories[job_index]
        if category is not None:
            counts[job_index] = float(category_counts.get(category, 0))

    if log_transform:
        counts = np.log1p(counts)
    return pd.Series(counts, index=np.arange(n_jobs))
