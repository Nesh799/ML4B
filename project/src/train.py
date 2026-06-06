from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import detect_numeric_sensor_columns, fit_transform
from src.utils import ModelBundle, detect_label_column, ensure_parent_dir, read_csv


def train_model(df: pd.DataFrame) -> ModelBundle:
    label_col = detect_label_column(df.columns)

    y_raw = df[label_col].astype(str)
    feature_cols = detect_numeric_sensor_columns(df, exclude=[label_col])
    if not feature_cols:
        raise ValueError(
            "No numeric sensor columns found. Ensure your sensor features are numeric columns."
        )

    pre_artifact, X = fit_transform(df=df, feature_columns=feature_cols)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Training accuracy: {acc:.4f}")

    return ModelBundle(
        preprocessor=pre_artifact,
        model=model,
        label_classes=list(le.classes_),
        feature_columns=list(feature_cols),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train activity classifier.")
    p.add_argument("--data", type=str, required=True, help="Path to training CSV.")
    p.add_argument(
        "--out",
        type=str,
        default=str(Path("models") / "model.pkl"),
        help="Output path for model bundle (joblib).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_path = Path(args.out)

    df = read_csv(data_path)
    bundle = train_model(df)

    ensure_parent_dir(out_path)
    joblib.dump(bundle, out_path)
    print(f"Saved model bundle to: {out_path}")


if __name__ == "__main__":
    main()

