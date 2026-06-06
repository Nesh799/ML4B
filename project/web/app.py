from __future__ import annotations

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# optional debug print
print("Using ROOT_DIR:", ROOT_DIR)

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import load_bundle, predict_df
from src.utils import ModelBundle

def try_parse_datetime_index(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return a datetime series to use as timeline x-axis.
    Prefers a 'timestamp' column; otherwise uses first parseable datetime-like column.
    """
    preferred = [c for c in df.columns if c.lower() in ("timestamp", "time", "datetime", "date")]
    candidates = preferred + [c for c in df.columns if c not in preferred]

    for c in candidates:
        if df[c].dtype == object or "datetime" in str(df[c].dtype).lower():
            parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
            if parsed.notna().sum() >= max(3, int(0.5 * len(df))):
                return parsed
    return None


def load_model_from_default_path() -> ModelBundle:
    model_path = Path("models") / "model.pkl"
    return load_bundle(model_path)


def main() -> None:
    st.set_page_config(page_title="Activity Classifier", layout="wide")
    st.title("Activity Classifier – Sitzen / Stehen / Gehen / Chillen")

    with st.sidebar:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    col_left, col_right = st.columns([1.2, 1.0], gap="large")

    if uploaded is None:
        st.info("Upload a CSV in the sidebar to get predictions.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    with col_left:
        st.subheader("Data preview")
        st.dataframe(df.head(50), use_container_width=True)

    if not predict_clicked:
        return

    try:
        bundle = load_model_from_default_path()
    except Exception as e:
        st.error(
            "Could not load model from `models/model.pkl`. Train a model first:\n\n"
            "`python -m src.train --data data/raw_data.csv --out models/model.pkl`\n\n"
            f"Error: {e}"
        )
        return

    try:
        preds = predict_df(df, bundle)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    result = df.copy()
    result["prediction"] = preds.values

    with col_left:
        st.subheader("Predictions")
        st.dataframe(result[["prediction"]].join(df).head(200), use_container_width=True)

    with col_right:
        st.subheader("Class distribution")
        dist = result["prediction"].value_counts().reset_index()
        dist.columns = ["class", "count"]
        fig_bar = px.bar(dist, x="class", y="count", text="count")
        fig_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Timeline")
        t = try_parse_datetime_index(df)
        if t is None:
            t = pd.Series(range(len(result)), name="index")
            x_label = "index"
        else:
            x_label = "time"

        timeline = pd.DataFrame({x_label: t, "prediction": result["prediction"].astype(str)})
        fig_tl = px.scatter(
            timeline,
            x=x_label,
            y="prediction",
            category_orders={"prediction": list(bundle.label_classes)},
        )
        fig_tl.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tl, use_container_width=True)


if __name__ == "__main__":
    main()

