"""Generate the two architecture diagrams as editable draw.io files.

Run:

    python scripts/generate_diagrams.py

Then rasterize with the draw.io exporter:

    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f svg -o . --output-mode relative --remove-page-suffix .
    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f png -o . --output-mode relative --remove-page-suffix .

    Do not pass -t/--transparent to the PNG export: it produces a
    transparent background instead of the white one the README expects.
    After rasterizing, run scripts/patch_svg_background.py (invoked
    automatically by `make diagrams` / the CI check) to insert a literal
    white <rect> GitHub's sanitizer will not strip.
"""

from __future__ import annotations

from pathlib import Path

import drawpyo
from drawpyo.diagram import Edge

DOCS = Path(__file__).resolve().parents[1] / "docs"

AWS_SHAPE = (
    "sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#{fill};"
    "strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;"
    "align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;"
    "shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{icon};"
)
BOX = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#{fill};strokeColor=#{stroke};"
    "fontSize=11;fontColor=#232F3E;align=center;verticalAlign=middle;"
)

ORANGE, BLUE, GREEN, PURPLE = "D45B07", "4D72F3", "277116", "8C4FFF"


def aws_node(page, title, icon, fill, x, y):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(AWS_SHAPE.format(fill=fill, icon=icon))
    node.position = (x, y)
    node.width, node.height = 78, 78
    return node


def box(page, title, x, y, width=190, height=58, fill="FFFFFF", stroke="9AA5B1"):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(BOX.format(fill=fill, stroke=stroke))
    node.position = (x, y)
    node.width, node.height = width, height
    return node


def link(page, source, target, label=""):
    """Connect two nodes.

    drawpyo's Edge has no apply_style_string, so styling is passed through the
    constructor instead.
    """
    return Edge(
        page=page,
        source=source,
        target=target,
        label=label,
        fontSize=10,
        fontColor="#4A5568",
        strokeColor="#4A5568",
        endArrow="blockThin",
        rounded=1,
    )


def build_aws_architecture() -> None:
    """Offline training on the left, online serving on the right."""
    file = drawpyo.File(file_name="aws-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="AWS architecture")

    box(page, "Offline — nightly retraining", 30, 20, 480, 36, fill="FFF4E6", stroke="D45B07")
    box(
        page, "Online — request path (p99 < 50 ms)",
        590, 20, 530, 36, fill="EEF2FF", stroke="4D72F3",
    )

    source = box(page, "Job postings\nKaggle or synthetic fixture", 30, 100, 220, 58)
    raw = aws_node(page, "S3\nraw", "bucket", GREEN, 320, 90)
    build = aws_node(page, "ECS Fargate\ntraining job", "machine_learning", ORANGE, 320, 260)
    schedule = aws_node(page, "EventBridge\nnightly 03:00", "event_time_based", PURPLE, 90, 260)
    artifacts = aws_node(page, "S3\nartifacts", "bucket", GREEN, 320, 430)
    topn = aws_node(page, "DynamoDB\ntop-N job matches", "dynamodb", BLUE, 630, 430)

    client = aws_node(page, "Client", "client", BLUE, 630, 90)
    api = aws_node(page, "API Gateway", "api_gateway", PURPLE, 850, 90)
    lam = aws_node(page, "Lambda\ninference", "lambda_function", ORANGE, 850, 260)
    logs = aws_node(page, "CloudWatch\nlogs & metrics", "cloudwatch_2", PURPLE, 1040, 430)

    link(page, source, raw, "ingest")
    link(page, schedule, build, "cron")
    link(page, raw, build, "read")
    link(page, build, artifacts, "BM25 index + model")
    link(page, artifacts, topn, "publish top-N")
    link(page, client, api, "GET /rank?query_job_id=")
    link(page, api, lam, "invoke")
    link(page, lam, topn, "GetItem")
    link(page, lam, logs, "emit")

    file.write()
    print(f"wrote {DOCS / 'aws-architecture.drawio'}")


def build_recommendation_architecture() -> None:
    """The modelling pipeline, from raw postings to a re-ranked list.

    Laid out as a single vertical spine with side branches so that no edge
    has to cross a node box.
    """
    file = drawpyo.File(file_name="recommendation-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="Recommendation architecture")

    spine_x = 420
    w, gap = 260, 118

    raw = box(
        page, "Raw job postings\ntitle · skills · seniority · location · time",
        spine_x, 40, w, 58, fill="F7FAFC",
    )
    taxonomy = box(
        page,
        "Skill extraction + taxonomy normalization\nsynonyms → canonical skill",
        spine_x, 40 + gap, w, 72, fill="FFF4E6", stroke="D45B07",
    )
    proxy = box(
        page, "Proxy supervision\nself-positive + sampled negatives, grouped",
        spine_x, 40 + 2 * gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    split = box(
        page, "Temporal + grouped splits\ntrain < τ₁ · val < τ₂ · test",
        spine_x, 40 + 3 * gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    features = box(
        page, "Point-in-time feature build\nfit on training only",
        spine_x, 40 + 4 * gap, w, 58, fill="E6FFFA", stroke="277116",
    )
    stage1 = box(
        page, "Stage 1 — BM25 retrieval\nrank_bm25, lexical top-K",
        spine_x, 40 + 5 * gap, w, 58, fill="EEF2FF", stroke="4D72F3",
    )
    stage2 = box(
        page,
        "Stage 2 — LambdaMART re-rank\nLightGBM lambdarank over handcrafted features",
        spine_x, 55 + 6 * gap, w, 72, fill="EEF2FF", stroke="4D72F3",
    )
    evaluation = box(
        page,
        "Evaluation\nRecall@K (stage 1) · NDCG@K / MAP@K (stage 2)",
        spine_x, 70 + 7 * gap, w, 72, fill="FFF5F5", stroke="C53030",
    )

    spec = box(
        page,
        "Feature registry\nconf/features.yaml\nstrict cut-off (<)",
        spine_x - 330, 40 + 4 * gap - 6, 250, 72, fill="E6FFFA", stroke="277116",
    )
    baselines = box(
        page, "Baselines\nPopularity (no retrieval) · lexical-only (no re-rank)",
        spine_x - 330, 70 + 7 * gap + 14, 250, 58, fill="F7FAFC",
    )
    serving = box(
        page, "Top-N publication\nDynamoDB · parity-tested",
        spine_x + 330, 40 + 5 * gap + 14, 250, 58, fill="EEF2FF", stroke="4D72F3",
    )

    link(page, raw, taxonomy)
    link(page, taxonomy, proxy)
    link(page, proxy, split)
    link(page, split, features)
    link(page, features, stage1, "feature matrix")
    link(page, stage1, stage2, "top-K candidates")
    link(page, stage2, evaluation)
    link(page, split, spec, "train only")
    link(page, spec, features)
    link(page, baselines, evaluation)
    link(page, stage1, serving, "top-K")

    file.write()
    print(f"wrote {DOCS / 'recommendation-architecture.drawio'}")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    build_aws_architecture()
    build_recommendation_architecture()
