"""BM25 lexical retrieval: the stage-1 model in the two-stage pipeline.

Wraps ``rank_bm25.BM25Okapi`` with a fixed, deterministic tokenizer and a
document-index mapping so it can be reused both as the retrieval model
itself (``top_k``) and as the source of the ``bm25_score`` re-ranking
feature consumed by stage 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


def tokenize(text: str) -> list[str]:
    """Lowercase, punctuation-stripped tokenization shared by index and query."""
    return _TOKEN_RE.findall(text.lower())


def job_document(title: str, canonical_skills: list[str], seniority: str, location: str) -> str:
    """Build the searchable document text for one job posting.

    Skills are repeated once extra: BM25's term-frequency component should
    weigh an explicitly required skill more than an incidental word in the
    free-text title, and skills are the signal this project cares about.
    """
    skills_text = " ".join(canonical_skills)
    return f"{title} {skills_text} {skills_text} {seniority} {location}"


@dataclass
class BM25Index:
    """A BM25 index fit over a fixed pool of candidate job documents.

    Attributes:
        job_indices: Catalogue ``job_index`` values aligned to document
            position in the underlying ``rank_bm25`` corpus.
    """

    job_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    _bm25: BM25Okapi | None = field(default=None, repr=False)
    _position: dict[int, int] = field(default_factory=dict, repr=False)

    def fit(self, job_indices: np.ndarray, documents: list[str]) -> BM25Index:
        """Fit the index over exactly the given documents, in the given order.

        Args:
            job_indices: Catalogue job indices, one per document.
            documents: Raw document text, positionally aligned to ``job_indices``.

        Returns:
            ``self``.

        Raises:
            ValueError: If the inputs are empty or misaligned.
        """
        if len(job_indices) != len(documents):
            raise ValueError(
                f"job_indices has {len(job_indices)} entries but documents has {len(documents)}."
            )
        if len(documents) == 0:
            raise ValueError("Cannot fit a BM25 index over zero documents.")

        self.job_indices = np.asarray(job_indices, dtype=np.int64)
        tokenized = [tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._position = {int(job_index): pos for pos, job_index in enumerate(self.job_indices)}
        return self

    def scores(self, query_text: str) -> np.ndarray:
        """BM25 scores for every indexed document, aligned to ``job_indices``."""
        if self._bm25 is None:
            raise RuntimeError("Index is not fitted; call fit() first.")
        return np.asarray(self._bm25.get_scores(tokenize(query_text)), dtype=np.float64)

    def score_for(self, query_text: str, job_index: int) -> float:
        """BM25 score of one specific indexed document; ``0.0`` if not indexed."""
        position = self._position.get(int(job_index))
        if position is None:
            return 0.0
        return float(self.scores(query_text)[position])

    def top_k(self, query_text: str, k: int) -> np.ndarray:
        """Top-``k`` indexed ``job_index`` values for a query, best first.

        Args:
            query_text: Free-text query (typically a candidate profile).
            k: Number of results to return.

        Returns:
            Up to ``k`` job indices.

        Raises:
            ValueError: If ``k`` is not positive.
        """
        if k < 1:
            raise ValueError(f"k must be positive, got {k}.")
        scores = self.scores(query_text)
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        ordered = top[np.argsort(-scores[top], kind="stable")]
        return self.job_indices[ordered]
