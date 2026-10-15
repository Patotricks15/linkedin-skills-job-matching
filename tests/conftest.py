"""Shared fixtures.

The fixture is generated with real posting timestamps and real category
drift, so the point-in-time machinery is exercised rather than merely
instantiated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from download_data import generate_synthetic  # noqa: E402

from recommender.data import (  # noqa: E402
    build_job_catalogue,
    build_proxy_pairs,
    grouped_query_split,
    temporal_split,
)
from recommender.features import build_features  # noqa: E402


@pytest.fixture(scope="session")
def raw_postings():
    """A small synthetic job-postings fixture."""
    return generate_synthetic(n_postings=360, seed=7)


@pytest.fixture(scope="session")
def catalogue(raw_postings):
    return build_job_catalogue(raw_postings)


@pytest.fixture(scope="session")
def pairs(catalogue):
    return build_proxy_pairs(catalogue, n_negatives=9, seed=7)


@pytest.fixture(scope="session")
def split(pairs):
    return temporal_split(pairs)


@pytest.fixture(scope="session")
def grouped_split(pairs):
    return grouped_query_split(pairs, seed=7)


@pytest.fixture(scope="session")
def store(split, catalogue):
    return build_features(split, catalogue)
