from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd
from pydantic import BaseModel
from app.services.forecasting.forecast_service import make_forecast
from app.services.model_selector import (
    get_available_series,
    get_best_model,
)
from app.services.data_loader import load_sales_history, load_best_models
from app.services.aggregation_service import aggregate_sales_history
from app.services.storage_service import save_forecast, load_forecasts
from app.services.data_loader import load_business_metrics
from app.services.aggregation_service import aggregate_business_metrics
from app.services.s3_service import upload_file_to_s3, ensure_bucket_exists
from pathlib import Path
from app.services.s3_service import get_s3_client, S3_BUCKET_NAME
from app.services.ml_model_service import is_ml_model, make_ml_forecast

router = APIRouter()

class ForecastRequest(BaseModel):
    aggregation_level: str = "market_product"
    market: str | None = None
    product: str | None = None
    horizon: int

def get_metrics_for_run(
    market: str | None,
    product: str | None,
    horizon: int,
):
    if market is None or product is None:
        return None

    try:
        df = load_best_models()

        filtered = df[
            (df["market"] == market)
            & (df["product"] == product)
            & (df["horizon"] == horizon)
        ]

        if filtered.empty:
            return None

        row = filtered.sort_values("smape").iloc[0]

        return {
            "mape": row.get("mape"),
            "smape": row.get("smape"),
            "rmse": row.get("rmse"),
            "mae": row.get("mae"),
        }

    except Exception:
        return None

@router.get("/")
def root():
    return {
        "service": "Mars Forecasting API",
        "status": "running",
        "version": "0.1.0",
    }

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/series")
def series():
    return get_available_series()


@router.get("/best-model")
def best_model(
    market: str = Query(...),
    product: str = Query(...),
    horizon: int | None = Query(None),
):
    result = get_best_model(
        market=market,
        product=product,
        horizon=horizon,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Best model not found for selected series",
        )

    return result


@router.get("/history")
def history(
    aggregation_level: str = Query("market_product"),
    market: str | None = Query(None),
    product: str | None = Query(None),
):
    df = load_sales_history()

    try:
        filtered = aggregate_sales_history(
            df=df,
            aggregation_level=aggregation_level,
            market=market,
            product=product,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    if filtered.empty:
        raise HTTPException(
            status_code=404,
            detail="Sales history not found for selected filters",
        )

    filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")
    filtered = filtered.replace([np.inf, -np.inf], np.nan)
    filtered = filtered.replace({np.nan: None})

    return filtered.to_dict(orient="records")

@router.post("/forecast")
def forecast(request: ForecastRequest):

    sales_df = load_sales_history()

    try:
        filtered = aggregate_sales_history(
            df=sales_df,
            aggregation_level=request.aggregation_level,
            market=request.market,
            product=request.product,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    if filtered.empty:
        raise HTTPException(
            status_code=404,
            detail="Time series not found",
        )

    if request.aggregation_level == "market_product":
        best_model = get_best_model(
            market=request.market,
            product=request.product,
            horizon=request.horizon,
        )
        SUPPORTED_MODELS = [
            'Holt - Winters',
            'ARIMA',
            'SARIMA',
            'XGBoost',
            'LightGBM'
        ]
        if best_model['model'] not in SUPPORTED_MODELS:
            print(f"Unsupported model {best_model}, fallback to Holt-Winters")
            best_model = {"model":"Holt-Winters",
                          "model_params": '{"trend": "add", "seasonal": "add", "seasonal_periods": 12}',
                          }
    else:
        best_model = {
            "model": "Holt-Winters",
            "model_params": '{"trend": "add", "seasonal": "add", "seasonal_periods": 12}',
        }

    ts = filtered["sales"]

    try:
        if is_ml_model(best_model["model"]):
            forecast_values = make_ml_forecast(
                history_df=filtered,
                model_name=best_model["model"],
                horizon=request.horizon,
            )
        else:
            forecast_values = make_forecast(
                ts=filtered["sales"],
                model_name=best_model["model"],
                model_params=best_model["model_params"],
                horizon=request.horizon,
            )

        last_date = filtered["date"].max()

        forecast_dates = pd.date_range(
            start=last_date,
            periods=request.horizon + 1,
            freq="MS",
        )[1:]

        forecast_run_id = save_forecast(
            aggregation_level=request.aggregation_level,
            market=request.market,
            product=request.product,
            horizon=request.horizon,
            model_name=best_model["model"],
            model_params=best_model["model_params"],
            forecast_dates=forecast_dates,
            forecast_values=forecast_values,
        )

        metrics_for_run = get_metrics_for_run(
            market=request.market,
            product=request.product,
            horizon=request.horizon,
        )
        # try:
        #     log_forecast_run(
        #     forecast_run_id=forecast_run_id,
        #     aggregation_level=request.aggregation_level,
        #     market=request.market,
        #     product=request.product,
        #     horizon=request.horizon,
        #     model_name=best_model["model"],
        #     model_params=best_model["model_params"],
        #     forecast_dates=forecast_dates,
        #     forecast_values=forecast_values,
        #     metrics=metrics_for_run,
        # )
        # except Exception as e:
        #     print(f"MLflow logging failed: {e}")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")
    filtered = filtered.replace([np.inf, -np.inf], np.nan)
    filtered = filtered.replace({np.nan: None})

    return {
        "forecast_run_id": forecast_run_id,
        "aggregation_level": request.aggregation_level,
        "market": request.market,
        "product": request.product,
        "model_name": best_model["model"],
        "model_params": best_model["model_params"],
        "forecast": forecast_values,
        "history": filtered.to_dict(orient="records"),
    }

@router.get("/metrics")
def metrics(
    market: str | None = Query(None),
    product: str | None = Query(None),
    horizon: int | None = Query(None),
):
    df = load_best_models()

    if market is not None:
        df = df[df["market"] == market]

    if product is not None:
        df = df[df["product"] == product]

    if horizon is not None:
        df = df[df["horizon"] == horizon]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="Metrics not found",
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace({np.nan: None})

    return df.to_dict(orient="records")

@router.get("/forecast-history")
def forecast_history():

    df = load_forecasts()

    if df.empty:
        return []

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace({np.nan: None})

    return df.to_dict(orient="records")

@router.get("/business-metrics")
def business_metrics(
    aggregation_level: str = Query("market_product"),
    market: str | None = Query(None),
    product: str | None = Query(None),
):
    df = load_business_metrics()

    try:
        result = aggregate_business_metrics(
            df=df,
            aggregation_level=aggregation_level,
            market=market,
            product=product,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Business metrics not found",
        )

    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.replace({np.nan: None})

    return result.to_dict(orient="records")

@router.post("/s3/upload-raw-data")
def upload_raw_data_to_s3():
    ensure_bucket_exists()

    base_dir = Path(__file__).resolve().parents[2]

    files = {
        "raw/sales_history.csv": base_dir / "data" / "processed" / "data_4_forecast.csv",
        "raw/best_models.csv": base_dir / "data" / "processed" / "best_by_ts.csv",
        "raw/business_metrics.csv": base_dir / "data" / "processed" / "business_data.csv",
    }

    uploaded = []

    for s3_key, local_path in files.items():
        if not local_path.exists():
            uploaded.append({
                "s3_key": s3_key,
                "status": "skipped",
                "reason": f"Local file not found: {local_path}",
            })
            continue

        result = upload_file_to_s3(
            local_path=local_path,
            s3_key=s3_key,
        )

        uploaded.append({
            "s3_key": s3_key,
            "status": "uploaded",
            "bucket": result["bucket"],
        })

    return {
        "bucket": "mars-forecasting",
        "uploaded": uploaded,
    }

@router.get("/s3/list-raw-data")
def list_raw_data_from_s3():
    ensure_bucket_exists()

    client = get_s3_client()

    response = client.list_objects_v2(
        Bucket=S3_BUCKET_NAME,
        Prefix="raw/",
    )

    objects = response.get("Contents", [])

    return [
        {
            "key": obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"].isoformat(),
        }
        for obj in objects
    ]