"""Registry binding declared feature names to their implementations."""

from __future__ import annotations

from collections.abc import Callable

from recommender.features import (
    context_features,
    interaction_features,
    item_features,
    user_features,
)
from recommender.features.spec import FeatureSpec

TRANSFORM_REGISTRY: dict[str, Callable[..., object]] = {
    "profile_skill_count": user_features.profile_skill_count,
    "profile_seniority_level": user_features.profile_seniority_level,
    "profile_taxonomy_coverage": user_features.profile_taxonomy_coverage,
    "candidate_job_popularity_rank": item_features.candidate_job_popularity_rank,
    "candidate_job_trailing_postings": item_features.candidate_job_trailing_postings,
    "bm25_score": interaction_features.bm25_score,
    "skill_overlap_raw": interaction_features.skill_overlap_raw,
    "skill_overlap_jaccard": interaction_features.skill_overlap_jaccard,
    "seniority_match": interaction_features.seniority_match,
    "location_match": context_features.location_match,
    "query_candidate_recency_gap_days": context_features.query_candidate_recency_gap_days,
}

#: Transforms that must receive an explicit ``as_of`` cut-off.
POINT_IN_TIME_TRANSFORMS = frozenset(
    {"candidate_job_trailing_postings", "query_candidate_recency_gap_days"}
)


def validate_registry(specs: list[FeatureSpec]) -> None:
    """Assert the declaration and the implementation agree in both directions.

    A one-directional check would let dead transforms accumulate, or let a
    feature be computed without ever being declared and therefore without
    ever being covered by the leakage tests.

    Args:
        specs: Parsed feature specifications.

    Raises:
        ValueError: If a declared feature has no implementation, an
            implemented transform is never declared, or a transform that
            reads event times is not declared ``point_in_time``.
    """
    declared = {spec.transform for spec in specs}
    implemented = set(TRANSFORM_REGISTRY)

    problems: list[str] = []
    if unimplemented := declared - implemented:
        problems.append(f"declared but not implemented: {sorted(unimplemented)}")
    if undeclared := implemented - declared:
        problems.append(f"implemented but not declared: {sorted(undeclared)}")

    mislabelled = sorted(
        spec.name
        for spec in specs
        if spec.transform in POINT_IN_TIME_TRANSFORMS and not spec.point_in_time
    )
    if mislabelled:
        problems.append(f"time-dependent but not declared point_in_time: {mislabelled}")

    if problems:
        raise ValueError("Feature registry mismatch -- " + "; ".join(problems))
