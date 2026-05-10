import json
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


def make_forecast(
    ts: pd.Series,
    model_name: str,
    model_params: str,
    horizon: int,
):
    """
    Train model and make forecast.
    """

    params = json.loads(model_params)

    # =========================
    # Holt-Winters
    # =========================
    if model_name == "Holt-Winters":

        model = ExponentialSmoothing(
            ts,
            trend=params.get("trend"),
            seasonal=params.get("seasonal"),
            seasonal_periods=params.get("seasonal_periods"),
        )

        fitted = model.fit()

        forecast = fitted.forecast(horizon)

        return forecast.tolist()

    # =========================
    # ARIMA
    # =========================
    elif model_name == "ARIMA":

        model = ARIMA(
            ts,
            order=tuple(params["order"]),
        )

        fitted = model.fit()

        forecast = fitted.forecast(horizon)

        return forecast.tolist()

    # =========================
    # SARIMA
    # =========================
    elif model_name == "SARIMA":

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            ts,
            order=tuple(params["order"]),
            seasonal_order=tuple(params["seasonal_order"]),
        )

        fitted = model.fit(disp=False)

        forecast = fitted.forecast(horizon)

        return forecast.tolist()

    else:
        raise ValueError(f"Unsupported model: {model_name}")