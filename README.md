# linkedin-skills-job-matching

Classical, fully non-neural two-stage retrieve-then-rank recommender for
matching candidate skill profiles to job postings: **BM25 lexical retrieval**
(stage 1) followed by **LambdaMART (LightGBM `lambdarank`) re-ranking** on
handcrafted, point-in-time-safe features (stage 2).

> Status: scaffolding in progress. See `paper/main.pdf` for the full write-up
> once built, and the sections below as they are completed.

## Why classical, two-stage retrieval?

Before dense neural retrieval, production search and recommendation systems
were built on exactly this pattern: a cheap lexical/statistical retriever
(BM25) narrows a huge catalogue down to a shortlist, and a boosted-tree
ranker — trained on explicit, auditable features — reorders that shortlist
for quality. This project reproduces that pipeline end to end for the
LinkedIn skills/job-matching domain, with no neural networks or learned
embeddings anywhere in the stack.

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/download_data.py --synthetic
pytest -q
python -m recommender.cli train
python -m recommender.cli evaluate --partition validation
python -m recommender.cli rank --query-job-id 1 --k 5
```

## Repository layout

```
conf/features.yaml          declarative feature specification
data/                        local datasets (gitignored, synthetic by default)
docs/                        architecture diagrams (draw.io + rasterized SVG/PNG)
paper/                       LaTeX technical report
scripts/                     data generation, feature export, diagram export
src/recommender/             library code (data, features, models, evaluation, serving)
terraform/                   Floci-backed (or real AWS) serving infrastructure
tests/                       pytest suite, including dedicated leakage controls
```

<!-- RESULTS:START -->
<!-- RESULTS:END -->

## License

MIT, see `LICENSE`.
