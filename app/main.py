from fastapi import FastAPI
import pandas as pd

from app.schemas import UploadRequest, ForecastRequest
from app.storage import save_data, get_data, clear_data
from app.services.forecast import run_forecast

app = FastAPI(title="Forecast API (Holt-Winters)")

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/upload")
def upload_data(request: UploadRequest):

    df = pd.DataFrame([x.dict() for x in request.data])
    df["date"] = pd.to_datetime(df["date"])

    save_data(df)

    return {
        "status": "ok",
        "rows_added": len(df)
    }


@app.get("/data")
def show_data():
    df = get_data()
    return df.head(50).to_dict(orient="records")



@app.delete("/data")
def delete_data():
    rows_removed = clear_data()

    return {
        "status": "deleted",
        "rows_removed": rows_removed
    }


@app.post("/forecast")
def forecast(request: ForecastRequest):

    df = get_data()

    if df.empty:
        return {"error": "No data uploaded"}

    result_df = run_forecast(
        df,
        request.horizons,
        request.group_by,
        request.trend,
        request.seasonal,
        request.seasonal_periods
    )

    return result_df.to_dict(orient="records")