import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from app.services.s3_service import upload_file_to_s3

BASE_DIR = Path(__file__).resolve().parents[2]

FORECASTS_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "forecasts.parquet"
)


def save_forecast(
    aggregation_level,
    market,
    product,
    horizon,
    model_name,
    model_params,
    forecast_dates,
    forecast_values,
):
    forecast_run_id = str(uuid.uuid4())

    created_at = datetime.utcnow()

    df = pd.DataFrame({
        "forecast_run_id": forecast_run_id,
        "created_at": created_at,
        "aggregation_level": aggregation_level,
        "market": market,
        "product": product,
        "horizon": horizon,
        "model_name": model_name,
        "model_params": str(model_params),
        "forecast_date": forecast_dates,
        "forecast_value": forecast_values,
    })

    if FORECASTS_PATH.exists():

        old_df = pd.read_parquet(FORECASTS_PATH)

        df = pd.concat(
            [old_df, df],
            ignore_index=True,
        )

    df.to_parquet(
        FORECASTS_PATH,
        index=False,
    )
    upload_file_to_s3(
        local_path=FORECASTS_PATH,
        s3_key="forecasts/forecasts.parquet",
    )

    return forecast_run_id


def load_forecasts():

    if not FORECASTS_PATH.exists():
        return pd.DataFrame()

    return pd.read_parquet(FORECASTS_PATH)