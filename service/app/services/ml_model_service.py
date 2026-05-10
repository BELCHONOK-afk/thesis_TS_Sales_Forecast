import re
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.services.s3_service import get_s3_client, S3_BUCKET_NAME


ML_MODEL_NAMES = {"LightGBM", "XGBoost"}


def is_ml_model(model_name: str) -> bool:
    return model_name in ML_MODEL_NAMES


def download_model_from_s3(
    model_name: str,
    horizon: int,
) -> Path:
    s3_key = f"models/{model_name}_horizon_{horizon}.pkl"

    tmp_dir = Path(tempfile.gettempdir()) / "mars_models"
    tmp_dir.mkdir(exist_ok=True)

    local_path = tmp_dir / f"{model_name}_horizon_{horizon}.pkl"

    if local_path.exists():
        return local_path

    client = get_s3_client()

    client.download_file(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
        Filename=str(local_path),
    )

    return local_path


def load_ml_artifact(
    model_name: str,
    horizon: int,
):
    model_path = download_model_from_s3(
        model_name=model_name,
        horizon=horizon,
    )

    artifact = joblib.load(model_path)

    return artifact


def get_feature_default_values(
    features: list[str],
    latest_row: pd.Series | None = None,
) -> dict:
    defaults = {}

    for feature in features:
        if latest_row is not None and feature in latest_row.index:
            value = latest_row[feature]

            if pd.notna(value):
                defaults[feature] = value
                continue

        defaults[feature] = 0

    return defaults


def build_future_features_recursive(
    history_df: pd.DataFrame,
    features: list[str],
    horizon: int,
) -> pd.DataFrame:

    df = history_df.copy()
    df = df.sort_values("date")

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    sales_values = df["sales"].dropna().tolist()
    last_date = df["date"].max()

    future_rows = []

    defaults = get_feature_default_values(features)

    for step in range(1, horizon + 1):
        future_date = last_date + pd.DateOffset(months=step)

        row = defaults.copy()

        for feature in features:

            # calendar features
            if feature == "year":
                row[feature] = future_date.year

            elif feature == "month":
                row[feature] = future_date.month

            elif feature == "quarter":
                row[feature] = future_date.quarter

            elif feature == "year_month":
                row[feature] = int(f"{future_date.year}{future_date.month:02d}")

            # lag features
            elif "lag" in feature.lower():
                numbers = re.findall(r"\d+", feature)

                if numbers:
                    lag = int(numbers[-1])

                    if len(sales_values) >= lag:
                        row[feature] = sales_values[-lag]
                    else:
                        row[feature] = sales_values[-1] if sales_values else 0

            # rolling mean
            elif "rolling" in feature.lower() and "mean" in feature.lower():
                numbers = re.findall(r"\d+", feature)

                if numbers:
                    window = int(numbers[-1])
                    values = sales_values[-window:]

                    row[feature] = float(np.mean(values)) if values else 0

            # rolling std
            elif "rolling" in feature.lower() and "std" in feature.lower():
                numbers = re.findall(r"\d+", feature)

                if numbers:
                    window = int(numbers[-1])
                    values = sales_values[-window:]

                    row[feature] = float(np.std(values)) if values else 0

        future_rows.append(row)

        # temporary placeholder; будет заменено после predict
        sales_values.append(np.nan)

    return pd.DataFrame(future_rows)[features]


def make_ml_forecast(
    history_df: pd.DataFrame,
    model_name: str,
    horizon: int,
):
    artifact = load_ml_artifact(
        model_name=model_name,
        horizon=horizon,
    )

    model = artifact["model"]
    features = artifact["features"]

    future_X = build_future_features_recursive(
        history_df=history_df,
        features=features,
        horizon=horizon,
    )

    prediction = model.predict(future_X)
    prediction = np.maximum(prediction, 0)

    return prediction.tolist()