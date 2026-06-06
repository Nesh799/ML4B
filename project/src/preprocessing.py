from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def detect_numeric_sensor_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """
    Detect numeric sensor feature columns in a dataframe.

    - Uses pandas dtype to select numeric columns
    - Excludes any explicitly provided column names (case-sensitive exact match)
    """
    exclude_set = set(exclude)
    numeric_cols = [
        c for c in df.columns if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols


@dataclass(frozen=True)
class PreprocessingArtifact:
    """Reusable preprocessing artifact for training + prediction."""

    pipeline: Pipeline
    feature_columns: Sequence[str]


def build_preprocessor(feature_columns: Sequence[str]) -> PreprocessingArtifact:
    """
    Build a preprocessing pipeline that:
    - imputes missing values (median)
    - scales features (StandardScaler)
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return PreprocessingArtifact(pipeline=pipeline, feature_columns=list(feature_columns))


def fit_transform(df: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[PreprocessingArtifact, np.ndarray]:
    artifact = build_preprocessor(feature_columns=feature_columns)
    X = artifact.pipeline.fit_transform(df[list(feature_columns)])
    return artifact, X


def transform(df: pd.DataFrame, artifact: PreprocessingArtifact) -> np.ndarray:
    """
    Transform new data using a fitted preprocessing artifact.
    """
    missing = [c for c in artifact.feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in prediction data: {missing}")
    X = artifact.pipeline.transform(df[list(artifact.feature_columns)])
    return X

