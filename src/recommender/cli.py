"""Command-line entry points for training, evaluation, and ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender.data import build_job_catalogue, build_proxy_pairs, load_postings, temporal_split
from recommender.evaluation import evaluate_two_stage
from recommender.evaluation.metrics import rerank_pairs
from recommender.features import build_features
from recommender.models import LambdaMARTReranker

ROOT = Path(__file__).resolve().parents[2]


def _load():
    postings = load_postings(ROOT / "data" / "job_postings.csv")
    data = build_job_catalogue(postings)
    pairs = build_proxy_pairs(data, n_negatives=9)
    split = temporal_split(pairs)
    return data, split


def _fit_reranker(store, split) -> LambdaMARTReranker:
    X, y, group = store.feature_matrix(split.train)
    return LambdaMARTReranker(n_estimators=100).fit(X, y, group)


def train_command(_: argparse.Namespace) -> None:
    data, split = _load()
    store = build_features(split, data)
    reranker = _fit_reranker(store, split)
    importance = reranker.feature_importance()
    print(
        json.dumps(
            {
                "n_train_pairs": len(split.train),
                "n_jobs": split.n_jobs,
                "as_of": store.as_of,
                "top_feature": importance.index[0],
            }
        )
    )


def evaluate_command(args: argparse.Namespace) -> None:
    data, split = _load()
    store = build_features(split, data)
    reranker = _fit_reranker(store, split)
    partition = getattr(split, args.partition)
    reports = evaluate_two_stage(store, reranker, data, partition, k=args.k)
    print(json.dumps([report.as_row() for report in reports.values()], indent=2))


def rank_command(args: argparse.Namespace) -> None:
    data, split = _load()
    store = build_features(split, data)
    reranker = _fit_reranker(split=split, store=store)

    frame = data.frame
    matches = (data.job_ids == args.query_job_id).nonzero()[0]
    if matches.size == 0:
        raise SystemExit(f"Unknown job id {args.query_job_id}.")
    query_index = int(matches[0])
    profile_skills = frame.at[query_index, "canonical_skills"]
    timestamp = int(frame.at[query_index, "posted_at"])

    pool = split.candidate_pool(split.train)
    candidate_pairs = rerank_pairs(store, query_index, profile_skills, timestamp, pool)
    scores = reranker.predict(store.pair_frame(candidate_pairs))
    order = scores.argsort()[::-1][: args.k]
    ranked = pool[order]

    payload = [
        {
            "rank": rank,
            "job_id": int(data.job_ids[job_index]),
            "title": str(frame.at[int(job_index), "title"]),
            "location": str(frame.at[int(job_index), "location"]),
            "seniority": str(frame.at[int(job_index), "seniority"]),
        }
        for rank, job_index in enumerate(ranked, start=1)
    ]
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.set_defaults(func=train_command)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--partition", choices=["validation", "test"], default="test")
    evaluate_parser.add_argument("--k", type=int, default=10)
    evaluate_parser.set_defaults(func=evaluate_command)

    rank = commands.add_parser("rank")
    rank.add_argument("--query-job-id", type=int, required=True)
    rank.add_argument("--k", type=int, default=10)
    rank.set_defaults(func=rank_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
