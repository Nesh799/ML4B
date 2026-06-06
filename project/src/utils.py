from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


LABEL_CANDIDATES: tuple[str, ...] = ("label", "activity", "class", "target")


class DataFormatError(ValueError):
    """Raised when the input CSV does not match expected format."""


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def detect_label_column(columns: Iterable[str]) -> str:
    cols = list(columns)
    lower_to_original = {c.lower(): c for c in cols}
    for cand in LABEL_CANDIDATES:
        if cand in lower_to_original:
            return lower_to_original[cand]
    raise DataFormatError(
        "Could not find label column. Expected one of: "
        + ", ".join([repr(c) for c in LABEL_CANDIDATES])
        + f". Found columns: {cols}"
    )


def safe_stem(path: Path) -> str:
    return path.stem if path.stem else "output"


@dataclass(frozen=True)
class ModelBundle:
    """Serializable model bundle stored via joblib."""

    preprocessor: object
    model: object
    label_classes: Sequence[str]
    feature_columns: Sequence[str]

