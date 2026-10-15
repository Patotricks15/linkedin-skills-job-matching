"""Materialize the declared features for the training partition and write them to Parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import (  # noqa: E402
    build_job_catalogue,
    build_proxy_pairs,
    grouped_query_split,
    load_postings,
    temporal_split,
)
from recommender.features import build_features  # noqa: E402

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postings", type=Path, default=DATA_DIR / "job_postings.csv")
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--n-negatives", type=int, default=9)
    parser.add_argument(
        "--protocol", choices=["temporal", "grouped"], default="temporal",
        help="Evaluation protocol the features are fitted for.",
    )
    args = parser.parse_args()

    postings = load_postings(args.postings)
    data = build_job_catalogue(postings)
    pairs = build_proxy_pairs(data, n_negatives=args.n_negatives)

    split = (
        temporal_split(pairs) if args.protocol == "temporal" else grouped_query_split(pairs)
    )
    store = build_features(split, data)

    args.out.mkdir(parents=True, exist_ok=True)
    frames = store.to_frames()
    for name, frame in frames.items():
        target = args.out / f"features_{name}_{args.protocol}.parquet"
        frame.to_parquet(target, index=False)
        print(f"{name:6s} {frame.shape[0]:>6,} rows x {frame.shape[1]} cols -> {target}")

    X, y, group = store.feature_matrix(split.train)
    X.assign(label=y).to_parquet(args.out / f"train_pairs_{args.protocol}.parquet", index=False)

    print()
    print(f"protocol        {split.protocol}")
    print(f"as_of           {store.as_of:.0f}")
    print(
        f"train/val/test  {len(split.train):,} / {len(split.validation):,} / "
        f"{len(split.test):,} pair rows"
    )
    print(f"groups          {len(group):,}")
    print(f"features        {len(store.specs)} declared, all materialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
