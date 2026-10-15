"""Fetch the Kaggle LinkedIn jobs/skills dataset, or generate a synthetic fixture.

The real dataset (``asaniczka/1-3m-linkedin-jobs-and-skills-2024``) requires a
Kaggle account and is not redistributed with this repository. The synthetic
generator exists so the pipeline, its tests, and CI run with no network
access and no credentials -- and so the taxonomy-normalization story is
testable against data where the "ground truth" canonical skill set is known
by construction.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.features.taxonomy import (  # noqa: E402
    CANONICAL_SKILLS,
    LOCATIONS,
    SENIORITY_LEVELS,
    SKILL_CATEGORIES,
    SYNONYMS,
)

DATA_DIR = ROOT / "data"
REQUIRED_COLUMNS = [
    "job_id", "title", "description", "skills", "seniority", "location", "posted_at",
]

TITLE_TEMPLATES = {
    "languages": "Software Engineer",
    "data": "Data Engineer",
    "data-science": "Data Scientist",
    "cloud": "Platform Engineer",
    "web": "Frontend Engineer",
    "soft-skills": "Product Manager",
}

# Surface forms grouped by canonical skill, used to pick a *spelling* for each
# posting rather than always writing the canonical token verbatim -- this is
# what makes taxonomy normalization a real test of the pipeline rather than a
# no-op over data that already uses canonical names.
SURFACE_FORMS: dict[str, list[str]] = {}
for surface, canonical in SYNONYMS.items():
    SURFACE_FORMS.setdefault(canonical, []).append(surface)


def _dominant_category(rng: np.random.Generator) -> str:
    return str(rng.choice(sorted(set(SKILL_CATEGORIES.values()))))


def _sample_skills(rng: np.random.Generator, category: str, n_skills: int) -> list[str]:
    """Sample raw (non-canonical) skill mentions, mostly from one category."""
    in_category = [s for s, c in SKILL_CATEGORIES.items() if c == category]
    other = [s for s in CANONICAL_SKILLS if s not in in_category]
    n_in = min(len(in_category), max(1, int(round(n_skills * 0.7))))
    n_out = max(0, n_skills - n_in)
    chosen = list(rng.choice(in_category, size=n_in, replace=False))
    if n_out and other:
        chosen += list(rng.choice(other, size=min(n_out, len(other)), replace=False))
    return [str(rng.choice(SURFACE_FORMS[skill])) for skill in chosen]


def generate_synthetic(
    n_postings: int = 1500,
    seed: int = 20260115,
) -> pd.DataFrame:
    """Build a job-postings fixture with embedded skill synonyms and real time drift.

    Three properties matter for this project specifically:

    * **Synonym variance**: the same canonical skill is spelled multiple
      ways across postings, so taxonomy normalization has real work to do.
    * **Category clustering**: skills cluster by job family (a "Data
      Scientist" posting draws mostly from ``data-science``), which gives
      BM25 retrieval and the overlap features recoverable structure instead
      of pure noise.
    * **Posting-time drift**: ``posted_at`` spans roughly two years with
      category popularity drifting over that span, so a temporal split is
      genuinely harder than a random one and trailing-window features have
      something real to track.

    Args:
        n_postings: Number of synthetic job postings.
        seed: Seed controlling the whole generation.

    Returns:
        A frame matching ``REQUIRED_COLUMNS``.
    """
    rng = np.random.default_rng(seed)
    py_random = random.Random(seed)

    categories = sorted(set(SKILL_CATEGORIES.values()))
    # Each category has a popularity "era" on the time axis, mirroring the
    # trailing-popularity story used in the sibling repositories.
    category_era = {cat: rng.uniform(0.1, 0.9) for cat in categories}

    start_ts = 1_700_000_000  # 2023-11-14, arbitrary anchor
    span_seconds = 2 * 365 * 24 * 3600

    rows = []
    for i in range(n_postings):
        category = str(rng.choice(categories))
        era = category_era[category]
        window = rng.uniform(0.08, 0.3)
        offset = float(np.clip(rng.normal(era, window), 0.0, 1.0))
        posted_at = int(start_ts + offset * span_seconds)

        seniority = str(rng.choice(SENIORITY_LEVELS, p=[0.08, 0.22, 0.28, 0.26, 0.11, 0.05]))
        location = str(rng.choice(LOCATIONS))
        n_skills = int(np.clip(rng.poisson(6), 3, 14))
        raw_skills = _sample_skills(rng, category, n_skills)

        job_id = i + 1
        title = f"{seniority} {TITLE_TEMPLATES[category]}"
        description = (
            f"We are hiring a {seniority.lower()} professional for our "
            f"{category.replace('-', ' ')} team in {location}. "
            f"Required skills: {', '.join(raw_skills)}. "
            f"The role is based in {location} and reports into the "
            f"{category.replace('-', ' ')} organization."
        )
        rows.append(
            {
                "job_id": job_id,
                "title": title,
                "description": description,
                "skills": "|".join(raw_skills),
                "seniority": seniority,
                "location": location,
                "posted_at": posted_at,
            }
        )

    py_random.shuffle(rows)
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    return frame.sort_values("posted_at").reset_index(drop=True)


def download_kaggle(out_dir: Path) -> pd.DataFrame:
    """Download the real Kaggle LinkedIn jobs/skills dataset.

    Requires ``KAGGLE_USERNAME``/``KAGGLE_KEY`` environment variables and the
    ``kaggle`` package. The raw release has no reliable posting-time column,
    so a ``posted_at`` is synthesized here (documented as a limitation in
    ``paper/sections/02-dataset.tex``) by assigning a uniform-random date over
    a plausible scraping window.

    Args:
        out_dir: Directory the Kaggle API should download the raw CSV into.

    Returns:
        A frame matching ``REQUIRED_COLUMNS``.

    Raises:
        RuntimeError: If credentials are missing or the download fails.
    """
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise RuntimeError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set to download the real dataset. "
            "Use --synthetic for a credential-free fixture."
        )
    try:
        import kaggle  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pip install kaggle to use --kaggle.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(
        "asaniczka/1-3m-linkedin-jobs-and-skills-2024", path=str(out_dir), unzip=True
    )
    raw_files = list(out_dir.glob("*.csv"))
    if not raw_files:
        raise RuntimeError(f"Kaggle download produced no CSV files in {out_dir}.")

    raw = pd.read_csv(raw_files[0])
    rng = np.random.default_rng(20260115)
    n = len(raw)
    span_seconds = 2 * 365 * 24 * 3600
    synthesized_posted_at = (1_700_000_000 + rng.uniform(0, span_seconds, size=n)).astype(int)

    frame = pd.DataFrame(
        {
            "job_id": np.arange(1, n + 1),
            "title": raw.get("job_title", raw.iloc[:, 0]).astype(str),
            "description": raw.get("job_summary", raw.get("description", "")).astype(str),
            "skills": raw.get("job_skills", raw.get("skills", "")).astype(str),
            "seniority": raw.get("job_level", "Mid").astype(str),
            "location": raw.get("job_location", "Remote").astype(str),
            "posted_at": synthesized_posted_at,
        }
    )
    return frame.sort_values("posted_at").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", action="store_true", default=True, help="Generate a local fixture (default)."
    )
    source.add_argument("--kaggle", action="store_true", help="Download the real Kaggle dataset.")
    parser.add_argument("--postings", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260115)
    args = parser.parse_args()

    if args.kaggle:
        postings = download_kaggle(DATA_DIR / "kaggle_raw")
    else:
        postings = generate_synthetic(args.postings, args.seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "job_postings.csv"
    postings.to_csv(out_path, index=False)

    span = postings["posted_at"].max() - postings["posted_at"].min()
    print(f"postings: {len(postings):,} rows -> {out_path}")
    print(f"span:     {span / 86400:.0f} days")
    print(f"locations:{postings['location'].nunique():,}")
    print(f"seniority:{sorted(postings['seniority'].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
