"""Context-family transforms: constraints and timing, independent of skill content."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data.loader import JobPostingData

SECONDS_PER_DAY = 86400.0


def location_match(pairs: pd.DataFrame, data: JobPostingData) -> pd.Series:
    """Binary indicator: same location, or either side is Remote."""
    location = data.frame["location"].to_numpy()
    query_location = location[pairs["query_index"].to_numpy()]
    candidate_location = location[pairs["candidate_index"].to_numpy()]
    match = (
        (query_location == candidate_location)
        | (query_location == "Remote")
        | (candidate_location == "Remote")
    )
    return pd.Series(match.astype(np.float64), index=pairs.index)


def query_candidate_recency_gap_days(
    pairs: pd.DataFrame,
    data: JobPostingData,
    as_of: float,
    max_known_timestamp: float | None = None,
) -> pd.Series:
    """Days between the cut-off and the candidate's posting date.

    Args:
        pairs: Pair rows to score.
        data: The job catalogue.
        as_of: Point-in-time cut-off the feature store was fitted with.
        max_known_timestamp: The latest query timestamp the store was
            fitted on (``split.train["timestamp"].max()``). Defaults to
            this call's own ``pairs["timestamp"].max()`` when not given,
            which is only safe when ``pairs`` *is* the training partition.

    Returns:
        A series aligned to ``pairs.index``. Candidates that happen to post
        after the cut-off (an artifact of sampling negatives from the whole
        catalogue) report a floor of ``0.0`` rather than a negative gap that
        would let the model infer future ordering.

    Raises:
        ValueError: If ``as_of`` is not strictly after the known training
            timestamps, which would mean the cut-off itself is not
            point-in-time safe.
    """
    if max_known_timestamp is not None:
        bound = float(max_known_timestamp)
    else:
        bound = float(pairs["timestamp"].max())
    if as_of <= bound:
        raise ValueError(
            "as_of must be strictly after every training query timestamp to "
            f"remain point-in-time safe; got as_of={as_of} <= {bound}."
        )

    candidate_posted_at = data.frame["posted_at"].to_numpy(dtype=np.float64)[
        pairs["candidate_index"].to_numpy()
    ]
    gap_days = np.maximum(as_of - candidate_posted_at, 0.0) / SECONDS_PER_DAY
    return pd.Series(gap_days, index=pairs.index)
