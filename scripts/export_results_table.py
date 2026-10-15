"""Run the classical two-stage benchmark and export Markdown and LaTeX tables."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import build_job_catalogue, build_proxy_pairs, load_postings, temporal_split  # noqa: E402
from recommender.evaluation import evaluate_two_stage  # noqa: E402
from recommender.features import build_features  # noqa: E402
from recommender.models import LambdaMARTReranker  # noqa: E402


def main() -> None:
    postings = load_postings(ROOT / "data" / "job_postings.csv")
    data = build_job_catalogue(postings)
    pairs = build_proxy_pairs(data, n_negatives=9)
    split = temporal_split(pairs)
    store = build_features(split, data)

    X, y, group = store.feature_matrix(split.train)
    reranker = LambdaMARTReranker(n_estimators=100).fit(X, y, group)

    reports = evaluate_two_stage(store, reranker, data, split.test, k=10)
    rows = [reports[name].as_row() for name in ("Popularity", "LexicalOnly", "TwoStage")]

    markdown = [
        "| Model | Recall@10 | NDCG@10 | MAP@10 |",
        "|---|---:|---:|---:|",
    ]
    latex = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Recall@10 & NDCG@10 & MAP@10 \\",
        r"\midrule",
    ]
    for row in rows:
        markdown.append(
            f"| {row['model']} | {row['recall']:.4f} | {row['ndcg']:.4f} | {row['map']:.4f} |"
        )
        latex.append(
            f"{row['model']} & {row['recall']:.4f} & {row['ndcg']:.4f} & {row['map']:.4f} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])

    table_dir = ROOT / "paper" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "results.tex").write_text("\n".join(latex) + "\n")

    readme = ROOT / "README.md"
    content = readme.read_text()
    replacement = "<!-- RESULTS:START -->\n" + "\n".join(markdown) + "\n<!-- RESULTS:END -->"
    content = re.sub(
        r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )
    readme.write_text(content)
    print(f"wrote {table_dir / 'results.tex'}")


if __name__ == "__main__":
    main()
