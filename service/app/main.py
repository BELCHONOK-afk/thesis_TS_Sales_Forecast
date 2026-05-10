from fastapi import FastAPI
from app.api.routes import router


app = FastAPI(
    title="Mars Sales Forecasting API",
    description="Service for selecting best forecasting models and serving time series forecasts.",
    version="0.1.0",
)

app.include_router(router)