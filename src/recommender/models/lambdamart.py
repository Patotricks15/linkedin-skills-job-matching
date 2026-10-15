"""Stage-2 re-ranking: LambdaMART (LightGBM ``lambdarank``) over handcrafted features.

LambdaMART optimizes a pairwise surrogate for a listwise ranking metric: for
a pair ``i`` (relevant) and ``j`` (irrelevant) in the same group, the
gradient with respect to the model's score is

    lambda_ij = sigma(s_j - s_i) * |delta NDCG_ij|

which pushes the relevant document's score up proportionally to how much
swapping the pair would actually change NDCG, not merely by whether the
pair is currently mis-ordered. See ``paper/sections/04-methodology.tex`` for
the full derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


@dataclass
class LambdaMARTReranker:
    """Thin wrapper around ``lightgbm.LGBMRanker`` with an explicit feature contract."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    num_leaves: int = 15
    random_state: int = 20260115
    feature_names_: list[str] | None = field(default=None, init=False, repr=False)
    model_: LGBMRanker | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        group: np.ndarray,
        eval_set: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
    ) -> LambdaMARTReranker:
        """Fit the ranker.

        Args:
            X: Feature matrix, one row per (profile, job) pair.
            y: Integer relevance labels (``1`` positive, ``0`` negative).
            group: LightGBM group-size array; must sum to ``len(X)``.
            eval_set: Optional ``(X, y, group)`` for early-stopping validation.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``group`` does not sum to the row count.
        """
        if group.sum() != len(X):
            raise ValueError(
                f"group sums to {group.sum()} but X has {len(X)} rows; they must match."
            )

        self.feature_names_ = list(X.columns)
        model = LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            verbosity=-1,
        )
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            eval_X, eval_y, eval_group = eval_set
            fit_kwargs["eval_set"] = [(eval_X[self.feature_names_], eval_y)]
            fit_kwargs["eval_group"] = [eval_group]
        model.fit(X[self.feature_names_], y, group=group, **fit_kwargs)
        self.model_ = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Relevance scores for a feature matrix, higher is more relevant.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model_ is None or self.feature_names_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        return np.asarray(self.model_.predict(X[self.feature_names_]), dtype=np.float64)

    def feature_importance(self) -> pd.Series:
        """Gain-based feature importance, indexed by feature name.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model_ is None or self.feature_names_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        importances = self.model_.booster_.feature_importance(importance_type="gain")
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)
