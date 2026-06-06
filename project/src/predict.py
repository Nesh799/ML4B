from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import PreprocessingArtifact, transform
from src.utils import ModelBundle, ensure_parent_dir, read_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict activities from a CSV using a trained model.")
    p.add_argument("--model", type=str, required=True, help="Path to model.pkl (joblib bundle).")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV for prediction.")
    p.add_argument("--out", type=str, default=None, help="Optional output CSV path.")
    return p.parse_args()


def load_bundle(path: Path) -> ModelBundle:
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        # joblib can load across code reloads; allow duck-typing but validate fields
        required = ("preprocessor", "model", "label_classes", "feature_columns")
        for r in required:
            if not hasattr(bundle, r):
                raise TypeError(f"Loaded object is not a valid ModelBundle (missing {r}).")
    return bundle


def predict_df(df: pd.DataFrame, bundle: ModelBundle) -> pd.Series:
    pre: PreprocessingArtifact = bundle.preprocessor  # type: ignore[assignment]
    X = transform(df=df, artifact=pre)
    y_idx = bundle.model.predict(X)
    labels = pd.Series([bundle.label_classes[int(i)] for i in y_idx], name="prediction")
    return labels


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else None

    bundle = load_bundle(model_path)
    df = read_csv(csv_path)

    preds = predict_df(df, bundle)
    print(preds.to_string(index=False))

    if out_path is not None:
        ensure_parent_dir(out_path)
        out_df = df.copy()
        out_df["prediction"] = preds.values
        out_df.to_csv(out_path, index=False)
        print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()

