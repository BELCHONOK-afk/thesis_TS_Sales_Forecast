from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date


class DataPoint(BaseModel):
    date: date
    value: float
    product: Optional[str] = None
    market: Optional[str] = None


class UploadRequest(BaseModel):
    data: List[DataPoint]


class ForecastRequest(BaseModel):
    horizons: List[int] = Field(..., example=[3, 6, 12])

    group_by: Literal[
        "total",
        "product",
        "market",
        "market_product"
    ]

    seasonal_periods: int = 12
    trend: Optional[Literal["add", "mul"]] = "add"
    seasonal: Optional[Literal["add", "mul"]] = "add"


class ForecastResult(BaseModel):
    product: Optional[str] = None
    market: Optional[str] = None
    horizon: int
    model: str
    MAE: float
    RMSE: float
    MAPE: float
    SMAPE: float