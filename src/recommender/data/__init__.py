"""Data loading, proxy-pair construction, and leakage-safe splitting."""

from recommender.data.loader import (
    JobPostingData,
    PairData,
    build_job_catalogue,
    build_proxy_pairs,
    load_postings,
)
from recommender.data.splitting import Split, grouped_query_split, temporal_split

__all__ = [
    "JobPostingData",
    "PairData",
    "Split",
    "build_job_catalogue",
    "build_proxy_pairs",
    "grouped_query_split",
    "load_postings",
    "temporal_split",
]
