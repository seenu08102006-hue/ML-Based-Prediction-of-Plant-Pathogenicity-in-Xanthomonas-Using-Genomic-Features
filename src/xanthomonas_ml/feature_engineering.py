"""Leakage-safe feature filters and descriptive feature statistics.

The two filters mirror the unsupervised domain filtering of te Molder et al.
(2021) but are implemented as scikit-learn transformers so that they are
*fitted on the training fold only* inside every cross-validation split:

* :class:`PrevalenceFilter` removes near-invariant domains (present in more
  than ``max_prevalence`` or fewer than ``min_prevalence`` of training strains).
  Such domains are almost uninformative and unstable in small folds.
* :class:`CorrelationFilter` removes one domain of every pair whose absolute
  Pearson correlation exceeds ``threshold``. Domains encoded by the same operon
  or gene cluster often co-occur perfectly; keeping one representative avoids
  splitting importance among redundant copies. The dropped->kept mapping is
  stored so that interpretation can report the whole correlated group.

The functions at the bottom compute *descriptive* class-wise prevalences for
EDA and interpretation only; they are never used to select model features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _as_frame(X: Any, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    return pd.DataFrame(arr, columns=columns if columns is not None else [f"x{i}" for i in range(arr.shape[1])])


class PrevalenceFilter(TransformerMixin, BaseEstimator):
    """Keep features whose training prevalence lies within [min_prevalence, max_prevalence].

    Prevalence is the fraction of training samples in which a binary feature is 1.
    Features with prevalence 0 or 1 in the training fold are always removed.

    Args:
        min_prevalence: Lower bound (inclusive) on prevalence.
        max_prevalence: Upper bound (inclusive) on prevalence.
    """

    def __init__(self, min_prevalence: float = 0.025, max_prevalence: float = 0.975):
        self.min_prevalence = min_prevalence
        self.max_prevalence = max_prevalence

    def fit(self, X: Any, y: Any = None) -> "PrevalenceFilter":
        frame = _as_frame(X)
        if not 0 <= self.min_prevalence <= self.max_prevalence <= 1:
            raise ValueError("require 0 <= min_prevalence <= max_prevalence <= 1")
        prevalence = frame.to_numpy(dtype=float).mean(axis=0)
        # Constant features (prevalence 0 or 1) are always dropped, whatever the bounds are.
        keep = (
            (prevalence >= self.min_prevalence)
            & (prevalence <= self.max_prevalence)
            & (prevalence > 0)
            & (prevalence < 1)
        )
        if not keep.any():
            raise ValueError("PrevalenceFilter removed every feature; relax the prevalence bounds")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.support_ = keep
        self.prevalence_ = prevalence
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "support_")
        return _select(X, self.feature_names_in_, self.support_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.feature_names_in_[self.support_]


def _select(X: Any, names: np.ndarray, support: np.ndarray) -> pd.DataFrame:
    """Return the supported columns, checking that the input matches the fitted schema."""
    frame = _as_frame(X, list(names))
    if frame.shape[1] != len(names) or not np.array_equal(np.asarray(frame.columns, dtype=object), names):
        raise ValueError("input columns differ from the columns seen during fit")
    # Positional selection is much faster than label-based .loc on wide frames.
    return frame.iloc[:, np.flatnonzero(support)]


class CorrelationFilter(TransformerMixin, BaseEstimator):
    """Greedily drop features highly correlated with an already-kept feature.

    Features are visited in input order (Pfam accession order); a feature is
    kept if its |Pearson r| with every previously kept feature is <= threshold.
    The rule is deterministic and label-free.

    Args:
        threshold: Absolute correlation above which a feature is dropped.
            ``None`` disables the filter (all features are kept).
    """

    def __init__(self, threshold: float | None = 0.8):
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> "CorrelationFilter":
        frame = _as_frame(X)
        names = np.asarray(frame.columns, dtype=object)
        n = frame.shape[1]
        self.feature_names_in_ = names
        self.n_features_in_ = n
        self.dropped_to_kept_: dict[str, str] = {}
        if self.threshold is None or n <= 1:
            self.support_ = np.ones(n, dtype=bool)
            return self
        values = frame.to_numpy(dtype=float)
        # Pearson correlation via standardised columns; constant columns have
        # zero standard deviation and are treated as uncorrelated with everything.
        centred = values - values.mean(axis=0)
        std = np.sqrt((centred**2).sum(axis=0))
        std[std == 0] = np.inf
        z = centred / std
        corr = np.abs(z.T @ z)
        # A feature is dropped if it exceeds the threshold with an earlier kept
        # feature. Pairs among earlier columns are resolved first, so a single
        # forward pass with a running "kept" mask reproduces the greedy rule.
        keep = np.zeros(n, dtype=bool)
        above = corr > self.threshold
        for j in range(n):
            partners = np.flatnonzero(above[j, :j] & keep[:j])
            if partners.size:
                best = partners[np.argmax(corr[j, partners])]
                self.dropped_to_kept_[str(names[j])] = str(names[best])
            else:
                keep[j] = True
        self.support_ = keep
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "support_")
        return _select(X, self.feature_names_in_, self.support_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.feature_names_in_[self.support_]

    def correlated_groups(self) -> dict[str, list[str]]:
        """Map each kept feature to the features it represents (dropped as correlated)."""
        check_is_fitted(self, "support_")
        groups: dict[str, list[str]] = {}
        for dropped, kept in self.dropped_to_kept_.items():
            groups.setdefault(kept, []).append(dropped)
        return groups


def feature_prevalence_by_class(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Descriptive prevalence of each feature in each class, with Fisher's exact test.

    Args:
        X: Binary feature matrix.
        y: Encoded target (1 = pathogenic, 0 = non-pathogenic), aligned to X.

    Returns:
        DataFrame indexed by feature with prevalence per class, the difference
        (pathogenic - non-pathogenic), the odds-ratio direction, the raw
        two-sided Fisher p-value and the Benjamini-Hochberg adjusted q-value.

    Note:
        This is a univariate *association* summary for EDA/interpretation. It is
        computed on the whole dataset and therefore must never be used to select
        features for model training (that would leak validation labels).
    """
    y = y.loc[X.index]
    pos, neg = X[y == 1], X[y == 0]
    a = pos.sum().to_numpy()
    c = neg.sum().to_numpy()
    b, d = len(pos) - a, len(neg) - c
    # One 2x2 table per feature: rows = class (pathogenic, non-pathogenic), columns = feature present/absent.
    pvals = np.array([fisher_exact([[a[i], b[i]], [c[i], d[i]]])[1] for i in range(X.shape[1])])
    out = pd.DataFrame(
        {
            "prevalence_pathogenic": a / max(len(pos), 1),
            "prevalence_non_pathogenic": c / max(len(neg), 1),
            "n_pathogenic_with_feature": a,
            "n_non_pathogenic_with_feature": c,
            "fisher_p": pvals,
        },
        index=X.columns,
    )
    out["prevalence_difference"] = out["prevalence_pathogenic"] - out["prevalence_non_pathogenic"]
    out["more_common_in"] = np.where(
        out["prevalence_difference"] > 0, "pathogenic", np.where(out["prevalence_difference"] < 0, "non-pathogenic", "equal")
    )
    out["fisher_q_bh"] = benjamini_hochberg(out["fisher_p"].to_numpy())
    out.index.name = "feature"
    return out.sort_values("fisher_p")


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted q-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    # Scale each sorted p-value by n / rank, then make the q-values non-decreasing
    # by walking from the largest p-value downwards.
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out
