"""Schema, registry and split-integrity tests."""

from __future__ import annotations

import pytest
import yaml

from recommender.data.splitting import temporal_split
from recommender.features import load_feature_specs, validate_registry
from recommender.features.registry import TRANSFORM_REGISTRY
from recommender.features.spec import FeatureSpec, default_spec_path


def test_spec_file_parses_and_is_complete():
    specs = load_feature_specs()
    assert len(specs) == 11
    validate_registry(specs)
    assert {s.transform for s in specs} == set(TRANSFORM_REGISTRY)


def test_every_feature_documents_a_rationale():
    for spec in load_feature_specs():
        assert len(spec.rationale.strip()) > 40, spec.name


def test_declared_dtypes_match_materialized_series(store):
    attr_by_name = {
        "candidate_job_popularity_rank": "job_popularity_rank",
        "candidate_job_trailing_postings": "job_trailing_postings",
    }
    for name, attr in attr_by_name.items():
        spec = next(s for s in load_feature_specs() if s.name == name)
        series = getattr(store, attr)
        assert str(series.dtype) == spec.dtype, name


def test_point_in_time_features_are_declared_consistently():
    pit = {s.name for s in load_feature_specs() if s.point_in_time}
    assert pit == {"candidate_job_trailing_postings", "query_candidate_recency_gap_days"}


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown family"):
        FeatureSpec("x", "nonsense", "float64", "t", {}, "none", False, {}, "why")


def test_point_in_time_feature_must_fit_on_train_only():
    with pytest.raises(ValueError, match="point-in-time but fits on"):
        FeatureSpec("x", "user", "float64", "t", {}, "catalogue", True, {}, "a reason")


def test_missing_rationale_is_rejected():
    with pytest.raises(ValueError, match="must document a rationale"):
        FeatureSpec("x", "user", "float64", "t", {}, "none", False, {}, "   ")


def test_duplicate_declaration_is_rejected(tmp_path):
    entry = {
        "name": "dup", "family": "user", "dtype": "float64",
        "transform": "profile_skill_count", "rationale": "duplicated on purpose for this test",
    }
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump({"features": [entry, entry]}))
    with pytest.raises(ValueError, match="Duplicate feature declaration"):
        load_feature_specs(path)


def test_registry_rejects_undeclared_transform():
    specs = [s for s in load_feature_specs() if s.transform != "location_match"]
    with pytest.raises(ValueError, match="implemented but not declared"):
        validate_registry(specs)


def test_spec_path_resolves_from_package():
    assert default_spec_path().exists()


def test_feature_matrix_has_no_nulls(store, split):
    X, y, group = store.feature_matrix(split.train)
    assert not X.isna().any().any()
    assert group.sum() == len(X) == len(y)


def test_feature_matrix_columns_match_declared_specs(store, split):
    X, _, _ = store.feature_matrix(split.train)
    assert list(X.columns) == [s.name for s in store.specs]


def test_unknown_job_gets_declared_fills(store):
    vector = store.job_vector(10**9)
    assert vector["candidate_job_popularity_rank"] == 1.0
    assert vector["candidate_job_trailing_postings"] == 0.0


def test_partitions_are_disjoint_and_ordered(pairs):
    split = temporal_split(pairs)
    train_ts = split.train.groupby("query_index")["timestamp"].first()
    val_ts = split.validation.groupby("query_index")["timestamp"].first()
    test_ts = split.test.groupby("query_index")["timestamp"].first()
    assert train_ts.max() < val_ts.min()
    assert val_ts.max() < test_ts.min()
    assert len(split.validation) > 0 and len(split.test) > 0


def test_grouped_split_keeps_each_query_in_exactly_one_partition(grouped_split):
    train_q = set(grouped_split.train["query_index"])
    val_q = set(grouped_split.validation["query_index"])
    test_q = set(grouped_split.test["query_index"])
    assert not (train_q & val_q)
    assert not (train_q & test_q)
    assert not (val_q & test_q)


@pytest.mark.parametrize(
    "train_q,val_q", [(0.9, 0.8), (0.0, 0.5), (0.5, 1.0), (0.5, 0.5)]
)
def test_invalid_quantiles_are_rejected(pairs, train_q, val_q):
    with pytest.raises(ValueError, match="train_quantile"):
        temporal_split(pairs, train_q, val_q)


def test_group_sizes_match_pair_counts(split):
    sizes = split.group_sizes(split.train)
    assert sizes.sum() == len(split.train)
    assert (sizes == 10).all()  # 1 positive + 9 negatives, by fixture construction
