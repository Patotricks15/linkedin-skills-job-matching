"""Classical, retrieval-free and retrieval-only baselines.

Both exist to make the two-stage system's contribution measurable rather
than assumed: :class:`PopularityBaseline` has no retrieval step at all, and
:class:`LexicalOnlyBaseline` has retrieval but no re-ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recommender.data.loader import JobPostingData
from recommender.models.bm25 import BM25Index, job_document


@dataclass
class PopularityBaseline:
    """Rank a candidate pool by raw skill-overlap count, tie-broken by popularity.

    No retrieval stage: every candidate in the pool is scored directly. This
    is the floor a real two-stage system must clear, representing what a
    system with no lexical index and no learned ranker could do with the
    cheapest possible signal.
    """

    data: JobPostingData

    def rank(self, profile_skills: list[str], pool: np.ndarray, k: int) -> np.ndarray:
        """Return up to ``k`` job indices from ``pool``, most promising first.

        Args:
            profile_skills: Canonical skills of the candidate profile query.
            pool: Candidate job indices to rank (no retrieval narrows this).
            k: Number of results to return.

        Returns:
            Up to ``k`` job indices, ordered best first.
        """
        raw_skills = self.data.frame["raw_skills"].to_numpy()
        profile_lower = {skill.lower() for skill in profile_skills}

        overlap = np.array(
            [
                sum(1 for token in raw_skills[idx] if token.strip().lower() in profile_lower)
                for idx in pool
            ],
            dtype=np.float64,
        )
        popularity_count = np.array(
            [len(self.data.frame.at[int(idx), "canonical_skills"]) for idx in pool],
            dtype=np.float64,
        )
        # Primary key: raw overlap. Secondary key: a cheap popularity proxy
        # (richer postings skew more "generic"/popular in this fixture).
        score = overlap * 1_000.0 + popularity_count
        order = np.argsort(-score, kind="stable")[: min(k, len(pool))]
        return pool[order]


@dataclass
class LexicalOnlyBaseline:
    """BM25 retrieval truncated at K, with no re-ranking stage at all."""

    index: BM25Index = field(default_factory=BM25Index)

    @classmethod
    def fit(cls, data: JobPostingData, pool: np.ndarray) -> LexicalOnlyBaseline:
        """Fit a BM25 index over exactly ``pool``.

        Args:
            data: The job catalogue.
            pool: Candidate job indices the index should cover.

        Returns:
            A fitted :class:`LexicalOnlyBaseline`.
        """
        frame = data.frame
        documents = [
            job_document(
                title=frame.at[int(idx), "title"],
                canonical_skills=frame.at[int(idx), "canonical_skills"],
                seniority=frame.at[int(idx), "seniority"],
                location=frame.at[int(idx), "location"],
            )
            for idx in pool
        ]
        return cls(index=BM25Index().fit(pool, documents))

    def rank(self, profile_skills: list[str], k: int) -> np.ndarray:
        """Top-``k`` job indices for a profile query, best first."""
        return self.index.top_k(" ".join(profile_skills), k)
