"""Dataset partitioning for proxy-labeled (profile, job) pairs.

Two protocols are implemented deliberately:

``temporal_split``
    A global wall-clock cut-off on each query's own posting time. This is
    the protocol reported as primary, because it is the only one under
    which point-in-time job-popularity features cannot see a query that
    postdates another query's held-out group.

``grouped_query_split``
    A random assignment of whole query groups to partitions, analogous to
    the leave-one-out protocol in the sibling repositories: it has no
    temporal meaning but guarantees no query is split across folds, so it
    is a sanity check that the headline numbers are not an artifact of the
    temporal boundary.

Both protocols keep an entire query's group (its positive plus its sampled
negatives) inside a single partition: splitting a LightGBM ``lambdarank``
group across folds would silently corrupt the ranking supervision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.loader import PairData


@dataclass(frozen=True)
class Split:
    """Train/validation/test partitions over contiguous job indices."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_jobs: int
    protocol: str = "temporal"

    def __post_init__(self) -> None:
        if self.train.empty:
            raise ValueError("The training partition is empty.")

    def group_sizes(self, frame: pd.DataFrame) -> np.ndarray:
        """LightGBM ``group`` array: pair counts per query, in row order."""
        return (
            frame.groupby("query_index", sort=False)["candidate_index"]
            .size()
            .to_numpy(dtype=np.int64)
        )

    def candidate_pool(self, frame: pd.DataFrame) -> np.ndarray:
        """Distinct candidate job indices present in a partition."""
        return np.sort(frame["candidate_index"].unique())


def _assert_groups_disjoint(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    train_q = set(train["query_index"].unique())
    val_q = set(validation["query_index"].unique())
    test_q = set(test["query_index"].unique())
    overlap = (train_q & val_q) | (train_q & test_q) | (val_q & test_q)
    if overlap:
        raise AssertionError(f"Query groups leaked across partitions: {sorted(overlap)[:5]}...")


def temporal_split(
    pairs: PairData,
    train_quantile: float = 0.80,
    validation_quantile: float = 0.90,
) -> Split:
    """Split whole query groups on the query's own posting time.

    Args:
        pairs: Proxy-labeled pairs, one ``timestamp`` per group.
        train_quantile: Quantile of the group-timestamp distribution ending training.
        validation_quantile: Quantile ending validation.

    Returns:
        A :class:`Split` with ``protocol="temporal"``.

    Raises:
        ValueError: If the quantiles are not ordered within ``(0, 1)``.
    """
    if not 0 < train_quantile < validation_quantile < 1:
        raise ValueError(
            "Expected 0 < train_quantile < validation_quantile < 1, got "
            f"{train_quantile} and {validation_quantile}."
        )

    frame = pairs.frame
    group_ts = frame.groupby("query_index")["timestamp"].first()
    train_cutoff = group_ts.quantile(train_quantile)
    val_cutoff = group_ts.quantile(validation_quantile)

    # Strict `<` on the training side: a query at exactly the cut-off belongs
    # to the future partition, never to the history used to fit features.
    train_queries = group_ts[group_ts < train_cutoff].index
    val_queries = group_ts[(group_ts >= train_cutoff) & (group_ts < val_cutoff)].index
    test_queries = group_ts[group_ts >= val_cutoff].index

    train = frame[frame["query_index"].isin(train_queries)].reset_index(drop=True)
    validation = frame[frame["query_index"].isin(val_queries)].reset_index(drop=True)
    test = frame[frame["query_index"].isin(test_queries)].reset_index(drop=True)

    _assert_groups_disjoint(train, validation, test)
    return Split(train=train, validation=validation, test=test, n_jobs=pairs.n_jobs, protocol="temporal")


def grouped_query_split(
    pairs: PairData,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    seed: int = 20260115,
) -> Split:
    """Randomly assign whole query groups to partitions, ignoring time.

    Args:
        pairs: Proxy-labeled pairs.
        train_fraction: Fraction of query groups assigned to training.
        validation_fraction: Fraction assigned to validation (the remainder
            goes to test).
        seed: Controls the random assignment.

    Returns:
        A :class:`Split` with ``protocol="grouped"``.

    Raises:
        ValueError: If the fractions are not a valid partition of ``(0, 1)``.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must be in (0, 1).")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1.")

    frame = pairs.frame
    queries = frame["query_index"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(queries)

    n = len(shuffled)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * validation_fraction))

    train_queries = set(shuffled[:n_train])
    val_queries = set(shuffled[n_train : n_train + n_val])
    test_queries = set(shuffled[n_train + n_val :])

    train = frame[frame["query_index"].isin(train_queries)].reset_index(drop=True)
    validation = frame[frame["query_index"].isin(val_queries)].reset_index(drop=True)
    test = frame[frame["query_index"].isin(test_queries)].reset_index(drop=True)

    _assert_groups_disjoint(train, validation, test)
    return Split(train=train, validation=validation, test=test, n_jobs=pairs.n_jobs, protocol="grouped")
