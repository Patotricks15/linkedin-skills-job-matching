"""Feature pipeline: materializes every declared feature for a set of pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data.loader import JobPostingData
from recommender.data.splitting import Split
from recommender.features import (
    context_features,
    interaction_features,
    item_features,
    user_features,
)
from recommender.features.registry import validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs
from recommender.models.bm25 import BM25Index, job_document


def resolve_as_of(split: Split) -> float:
    """Derive the point-in-time cut-off implied by a partition.

    For the temporal protocol the cut-off is the first held-out query's
    timestamp: every training group precedes it by construction. For the
    grouped protocol there is no global boundary, so the cut-off is one
    second past the last training timestamp -- the earliest instant at
    which all of training is strictly past.

    Args:
        split: The partition features will be fitted on.

    Returns:
        A Unix timestamp strictly greater than every training timestamp.
    """
    last_train = float(split.train["timestamp"].max())
    if split.protocol == "temporal" and not split.validation.empty:
        return float(split.validation["timestamp"].min())
    return last_train + 1.0


def _fit_bm25_index(split: Split, data: JobPostingData) -> BM25Index:
    """Fit a BM25 index over exactly the candidate documents training saw."""
    job_indices = split.candidate_pool(split.train)
    frame = data.frame
    documents = [
        job_document(
            title=frame.at[int(idx), "title"],
            canonical_skills=frame.at[int(idx), "canonical_skills"],
            seniority=frame.at[int(idx), "seniority"],
            location=frame.at[int(idx), "location"],
        )
        for idx in job_indices
    ]
    return BM25Index().fit(job_indices, documents)


@dataclass
class FeatureStore:
    """Materialized features, fitted on the training partition only."""

    job_popularity_rank: pd.Series
    job_trailing_postings: pd.Series
    bm25_index: BM25Index
    data: JobPostingData
    n_jobs: int
    as_of: float
    max_known_timestamp: float
    specs: list[FeatureSpec] = field(default_factory=list)

    def job_vector(self, job_index: int) -> dict[str, float]:
        """Item-family feature values for one job, with declared fills applied."""
        popularity = float(self.job_popularity_rank.get(job_index, 1.0))
        trailing = float(self.job_trailing_postings.get(job_index, 0.0))
        return {
            "candidate_job_popularity_rank": popularity,
            "candidate_job_trailing_postings": trailing,
        }

    def pair_frame(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Compute every declared feature for a set of (profile, job) pairs.

        Args:
            pairs: Rows with ``query_index``, ``candidate_index``,
                ``profile_skills``, ``timestamp`` (see
                :func:`recommender.data.loader.build_proxy_pairs`).

        Returns:
            A frame with one column per declared feature name, aligned to
            ``pairs.index``, in spec declaration order.
        """
        columns: dict[str, pd.Series] = {
            "profile_skill_count": user_features.profile_skill_count(pairs),
            "profile_seniority_level": user_features.profile_seniority_level(pairs, self.data),
            "profile_taxonomy_coverage": user_features.profile_taxonomy_coverage(pairs),
            "candidate_job_popularity_rank": self.job_popularity_rank.reindex(
                pairs["candidate_index"]
            ).fillna(1.0).to_numpy(),
            "candidate_job_trailing_postings": self.job_trailing_postings.reindex(
                pairs["candidate_index"]
            ).fillna(0.0).to_numpy(),
            "bm25_score": interaction_features.bm25_score(pairs, self.bm25_index),
            "skill_overlap_raw": interaction_features.skill_overlap_raw(pairs, self.data),
            "skill_overlap_jaccard": interaction_features.skill_overlap_jaccard(pairs, self.data),
            "seniority_match": interaction_features.seniority_match(pairs, self.data),
            "location_match": context_features.location_match(pairs, self.data),
            "query_candidate_recency_gap_days": context_features.query_candidate_recency_gap_days(
                pairs, self.data, self.as_of, max_known_timestamp=self.max_known_timestamp
            ),
        }
        ordered = [spec.name for spec in self.specs]
        out = pd.DataFrame({name: np.asarray(columns[name]) for name in ordered}, index=pairs.index)
        return out

    def feature_matrix(self, pairs: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """``(X, y, group)`` ready for a LightGBM ``lambdarank`` model.

        Args:
            pairs: Rows from one partition, assumed grouped contiguously by
                ``query_index`` (true of every :class:`Split` partition).

        Returns:
            Feature matrix, integer labels, and LightGBM group-size array.
        """
        X = self.pair_frame(pairs)
        y = pairs["label"].to_numpy(dtype=np.int64)
        group = (
            pairs.groupby("query_index", sort=False)["candidate_index"]
            .size()
            .to_numpy(dtype=np.int64)
        )
        return X, y, group

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Export the catalogue-wide, job-indexed features as a tidy frame."""
        index = range(self.n_jobs)
        jobs = pd.DataFrame(
            {
                "job_index": np.arange(self.n_jobs),
                "candidate_job_popularity_rank": self.job_popularity_rank.reindex(index)
                .fillna(1.0)
                .to_numpy(),
                "candidate_job_trailing_postings": self.job_trailing_postings.reindex(index)
                .fillna(0.0)
                .to_numpy(),
            }
        )
        return {"jobs": jobs}


def build_features(
    split: Split,
    data: JobPostingData,
    spec_path: str | Path | None = None,
) -> FeatureStore:
    """Materialize every declared feature from the training partition.

    Only ``split.train`` reaches the train-only transforms (popularity rank,
    trailing postings, the BM25 index), which is what enforces the
    ``fit_on: train_only`` contract declared in ``conf/features.yaml``.

    Args:
        split: Leakage-safe partition.
        data: The job catalogue.
        spec_path: Optional override of ``conf/features.yaml``.

    Returns:
        A populated :class:`FeatureStore`.
    """
    specs = load_feature_specs(spec_path)
    validate_registry(specs)

    as_of = resolve_as_of(split)
    params = {spec.transform: spec.params for spec in specs}

    job_popularity_rank = item_features.candidate_job_popularity_rank(
        split.train,
        data,
        n_jobs=split.n_jobs,
        normalize=params["candidate_job_popularity_rank"].get("normalize", True),
    )
    job_trailing_postings = item_features.candidate_job_trailing_postings(
        split.train,
        data,
        n_jobs=split.n_jobs,
        as_of=as_of,
        window_days=params["candidate_job_trailing_postings"].get("window_days", 60.0),
        log_transform=params["candidate_job_trailing_postings"].get("log_transform", True),
    )
    bm25_index = _fit_bm25_index(split, data)

    return FeatureStore(
        job_popularity_rank=job_popularity_rank,
        job_trailing_postings=job_trailing_postings,
        bm25_index=bm25_index,
        data=data,
        n_jobs=split.n_jobs,
        as_of=as_of,
        max_known_timestamp=float(split.train["timestamp"].max()),
        specs=specs,
    )
