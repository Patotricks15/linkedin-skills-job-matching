"""Interaction-family transforms: pairwise profile-candidate comparisons.

These are the features that make the two-stage design legible: ``bm25_score``
carries the stage-1 retriever's own belief into stage 2, and the two overlap
features are a deliberate minimal pair that demonstrates why taxonomy
normalization matters (see ``conf/features.yaml`` for the rationale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data.loader import JobPostingData
from recommender.features.taxonomy import SENIORITY_LEVELS, seniority_rank
from recommender.models.bm25 import BM25Index


def bm25_score(pairs: pd.DataFrame, bm25_index: BM25Index) -> pd.Series:
    """Stage-1 BM25 score of the profile query against the candidate document."""
    scores = [
        bm25_index.score_for(" ".join(profile), int(candidate_index))
        for profile, candidate_index in zip(
            pairs["profile_skills"], pairs["candidate_index"], strict=True
        )
    ]
    return pd.Series(scores, index=pairs.index, dtype=np.float64)


def skill_overlap_raw(pairs: pd.DataFrame, data: JobPostingData) -> pd.Series:
    """Count of profile skills matching a candidate's raw skill mentions verbatim.

    Deliberately naive: a profile skill "javascript" does not match a
    candidate's raw mention "js" unless the posting happened to spell it
    identically, which is exactly the gap ``skill_overlap_jaccard`` closes.
    """
    raw_skills = data.frame["raw_skills"].to_numpy()
    overlaps = []
    for profile, candidate_index in zip(
        pairs["profile_skills"], pairs["candidate_index"], strict=True
    ):
        candidate_raw = {token.strip().lower() for token in raw_skills[candidate_index]}
        overlaps.append(float(sum(1 for skill in profile if skill in candidate_raw)))
    return pd.Series(overlaps, index=pairs.index)


def skill_overlap_jaccard(pairs: pd.DataFrame, data: JobPostingData) -> pd.Series:
    """Jaccard overlap between taxonomy-normalized profile and candidate skills."""
    canonical_skills = data.frame["canonical_skills"].to_numpy()
    scores = []
    for profile, candidate_index in zip(
        pairs["profile_skills"], pairs["candidate_index"], strict=True
    ):
        profile_set = set(profile)
        candidate_set = set(canonical_skills[candidate_index])
        union = profile_set | candidate_set
        scores.append(len(profile_set & candidate_set) / len(union) if union else 0.0)
    return pd.Series(scores, index=pairs.index)


def seniority_match(pairs: pd.DataFrame, data: JobPostingData) -> pd.Series:
    """One minus the normalized ordinal seniority distance of a pair."""
    seniority = data.frame["seniority"].to_numpy()
    query_rank = np.array([seniority_rank(seniority[i]) for i in pairs["query_index"]])
    candidate_rank = np.array([seniority_rank(seniority[i]) for i in pairs["candidate_index"]])
    max_distance = len(SENIORITY_LEVELS) - 1
    scores = 1.0 - np.abs(query_rank - candidate_rank) / max_distance
    return pd.Series(scores, index=pairs.index)
