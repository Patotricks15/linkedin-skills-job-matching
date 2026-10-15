"""User-family transforms: features describing the candidate profile side.

A "profile" here is a resampled subset of a job's own canonical skills
(see ``recommender.data.loader.build_proxy_pairs``), so every one of these
is a pure function of the pair's own ``profile_skills`` and the originating
job's metadata -- no label signal, no interaction history, no leakage risk.
"""

from __future__ import annotations

import pandas as pd

from recommender.data.loader import JobPostingData
from recommender.features.taxonomy import seniority_rank, skill_category


def profile_skill_count(pairs: pd.DataFrame) -> pd.Series:
    """Number of canonical skills in the candidate profile query."""
    return pairs["profile_skills"].apply(len).astype(float)


def profile_seniority_level(pairs: pd.DataFrame, data: JobPostingData) -> pd.Series:
    """Ordinal seniority of the profile's originating posting."""
    seniority = data.frame["seniority"].to_numpy()
    query_index = pairs["query_index"].to_numpy()
    ranks = [float(seniority_rank(seniority[i])) for i in query_index]
    return pd.Series(ranks, index=pairs.index)


def profile_taxonomy_coverage(pairs: pd.DataFrame) -> pd.Series:
    """Number of distinct taxonomy categories represented in the profile."""
    def _coverage(skills: list[str]) -> float:
        categories = {skill_category(skill) for skill in skills}
        categories.discard(None)
        return float(len(categories))

    return pairs["profile_skills"].apply(_coverage)
