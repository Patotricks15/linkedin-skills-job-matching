"""Load job postings and construct proxy supervision pairs.

The Kaggle LinkedIn dataset has job postings with required skills but **no
real candidate profiles and no real candidate-job interaction log**. To get a
learning-to-rank signal at all, this project builds a documented proxy: each
job posting's own (partial, resampled) skill set stands in for "a candidate
profile that matches this job", paired against other postings' profiles as
negatives, grouped by the originating job for LightGBM's ``lambdarank``
group structure. See ``paper/sections/02-dataset.tex`` for the full
discussion of what this construction can and cannot demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Deferred to function scope (see build_job_catalogue): recommender.features
# imports recommender.data.loader for FeatureStore construction, so a
# top-level import here would be circular.

REQUIRED_COLUMNS = {
    "job_id", "title", "description", "skills", "seniority", "location", "posted_at",
}


@dataclass(frozen=True)
class JobPostingData:
    """Job postings with contiguous internal indices, sorted by posting time.

    Attributes:
        frame: One row per posting: ``job_index``, ``title``, ``description``,
            ``raw_skills`` (list[str]), ``canonical_skills`` (list[str]),
            ``seniority``, ``location``, ``posted_at``.
        job_ids: Original ``job_id`` values, positionally aligned to ``job_index``.
    """

    frame: pd.DataFrame
    job_ids: np.ndarray

    @property
    def n_jobs(self) -> int:
        return int(self.job_ids.shape[0])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def load_postings(path: str | Path) -> pd.DataFrame:
    """Read ``job_postings.csv`` and validate its contract.

    Args:
        path: Path to a job-postings CSV file.

    Returns:
        The raw frame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py --synthetic"
        )
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    return frame


def build_job_catalogue(postings: pd.DataFrame) -> JobPostingData:
    """Reindex postings to contiguous indices, sorted by posting time.

    Args:
        postings: Raw frame validated by :func:`load_postings`.

    Returns:
        A :class:`JobPostingData` with canonical skills precomputed.
    """
    frame = postings.drop_duplicates(subset=["job_id"]).copy()
    frame = frame.sort_values(["posted_at", "job_id"]).reset_index(drop=True)

    from recommender.features.taxonomy import normalize_skills

    job_ids = frame["job_id"].to_numpy()
    raw_skills = frame["skills"].fillna("").apply(
        lambda cell: [token for token in cell.split("|") if token]
    )
    canonical_skills = raw_skills.apply(normalize_skills)

    out = pd.DataFrame(
        {
            "job_index": np.arange(len(frame), dtype=np.int64),
            "title": frame["title"].astype(str),
            "description": frame["description"].astype(str),
            "raw_skills": raw_skills,
            "canonical_skills": canonical_skills,
            "seniority": frame["seniority"].astype(str),
            "location": frame["location"].astype(str),
            "posted_at": frame["posted_at"].astype(np.int64),
        }
    )
    return JobPostingData(frame=out, job_ids=job_ids)


@dataclass(frozen=True)
class PairData:
    """Proxy-labeled (query profile, candidate job) pairs for LTR.

    Attributes:
        frame: Columns ``query_index`` (the originating job, i.e. "profile"),
            ``candidate_index`` (the job being scored), ``label`` (1 for the
            query's own posting, 0 for sampled negatives), ``profile_skills``
            (list[str], the resampled subset of the query's own canonical
            skills used as the candidate-profile query), ``timestamp`` (the
            *query's own* ``posted_at`` -- i.e. "when this ranking request
            happens" -- shared by every row of a group, which is what keeps
            a group from being split across train/validation/test).
        n_jobs: Size of the job catalogue the indices are drawn from.
    """

    frame: pd.DataFrame
    n_jobs: int

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def build_proxy_pairs(
    data: JobPostingData,
    n_negatives: int = 9,
    profile_fraction_range: tuple[float, float] = (0.5, 0.9),
    min_profile_skills: int = 2,
    seed: int = 20260115,
) -> PairData:
    """Construct self-supervised (profile, job) ranking pairs.

    Each job contributes exactly one group: a positive pair against itself
    (profile = a resampled subset of its own canonical skills, which keeps
    the query from being a trivial verbatim duplicate of the candidate
    document) and ``n_negatives`` negative pairs against other, randomly
    sampled jobs.

    Args:
        data: The job catalogue.
        n_negatives: Negatives sampled per positive.
        profile_fraction_range: Uniform range for the fraction of a job's own
            canonical skills retained in its profile query.
        min_profile_skills: Jobs with fewer canonical skills than this are
            skipped: there is nothing to subsample.
        seed: Controls profile subsampling and negative sampling.

    Returns:
        A :class:`PairData` with one contiguous group per query, positive row
        first within each group.

    Raises:
        ValueError: If ``n_negatives`` is not positive or every job is
            skipped for having too few canonical skills.
    """
    if n_negatives < 1:
        raise ValueError(f"n_negatives must be positive, got {n_negatives}.")

    rng = np.random.default_rng(seed)
    frame = data.frame
    n_jobs = len(frame)
    canonical = frame["canonical_skills"].to_numpy()
    posted_at = frame["posted_at"].to_numpy()

    rows = []
    for query_index in range(n_jobs):
        skills = canonical[query_index]
        if len(skills) < min_profile_skills:
            continue

        fraction = rng.uniform(*profile_fraction_range)
        keep = max(min_profile_skills, int(round(len(skills) * fraction)))
        keep = min(keep, len(skills))
        profile = sorted(rng.choice(skills, size=keep, replace=False).tolist())

        query_timestamp = int(posted_at[query_index])
        rows.append(
            {
                "query_index": query_index,
                "candidate_index": query_index,
                "label": 1,
                "profile_skills": profile,
                "timestamp": query_timestamp,
            }
        )

        candidates = rng.choice(
            [i for i in range(n_jobs) if i != query_index],
            size=min(n_negatives, n_jobs - 1),
            replace=False,
        )
        for candidate_index in candidates:
            rows.append(
                {
                    "query_index": query_index,
                    "candidate_index": int(candidate_index),
                    "label": 0,
                    "profile_skills": profile,
                    "timestamp": query_timestamp,
                }
            )

    if not rows:
        raise ValueError(
            "No job produced a valid profile; every posting has fewer than "
            f"{min_profile_skills} canonical skills."
        )

    out = pd.DataFrame(rows)
    return PairData(frame=out, n_jobs=n_jobs)
