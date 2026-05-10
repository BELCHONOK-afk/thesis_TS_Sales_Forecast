import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.s3_service import read_csv_from_s3, object_exists


BASE_DIR = Path(__file__).resolve().parents[2]

USE_S3 = os.getenv("USE_S3", "false").lower() == "true"

BEST_MODELS_PATH = BASE_DIR / "data" / "processed" / "best_by_ts.csv"
SALES_PATH = BASE_DIR / "data" / "processed" / "data_4_forecast.csv"
BUSINESS_METRICS_PATH = BASE_DIR / "data" / "processed" / "business_data.csv"

BEST_MODELS_S3_KEY = "raw/best_models.csv"
SALES_S3_KEY = "raw/sales_history.csv"
BUSINESS_METRICS_S3_KEY = "raw/business_metrics.csv"


def read_csv_with_fallback(
    local_path: Path,
    s3_key: str,
) -> pd.DataFrame:
    if USE_S3:
        try:
            if object_exists(s3_key):
                return read_csv_from_s3(s3_key)
        except Exception as e:
            print(f"S3 read failed for {s3_key}. Fallback to local. Error: {e}")

    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    return pd.read_csv(local_path)


def load_best_models() -> pd.DataFrame:
    df = read_csv_with_fallback(
        local_path=BEST_MODELS_PATH,
        s3_key=BEST_MODELS_S3_KEY,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    required_columns = {
        "market",
        "product",
        "model",
        "model_params",
        "horizon",
        "smape",
        "mape",
        "rmse",
        "mae",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in best_models: {missing}")

    return df


def load_sales_history() -> pd.DataFrame:
    df = read_csv_with_fallback(
        local_path=SALES_PATH,
        s3_key=SALES_S3_KEY,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    required_columns = {
        "date",
        "market",
        "product",
        "sales",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in sales_history: {missing}")

    df["date"] = pd.to_datetime(df["date"])

    return df


def load_business_metrics() -> pd.DataFrame:
    df = read_csv_with_fallback(
        local_path=BUSINESS_METRICS_PATH,
        s3_key=BUSINESS_METRICS_S3_KEY,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    required_columns = {
        "date",
        "market",
        "product",
        "revenue",
        "volume",
        "avg_price",
        "penetration",
        "frequency",
        "spend_per_trip",
        "volume_per_trip",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in business_metrics: {missing}")

    df["date"] = pd.to_datetime(df["date"])

    return df