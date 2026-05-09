from statsmodels.tsa.holtwinters import ExponentialSmoothing


def hw_forecast(train, horizon, trend, seasonal, seasonal_periods):

    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    ).fit()

    forecast = model.forecast(horizon)

    return forecast.values