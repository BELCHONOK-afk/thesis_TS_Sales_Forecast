import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error as mape,
    mean_absolute_error as mae,
    root_mean_squared_error as rmse
)


def smape(y, y_forecast):
    denominator = (np.abs(y) + np.abs(y_forecast)) / 2
    diff = np.abs(y - y_forecast) / denominator
    diff[denominator == 0] = 0
    return np.mean(diff) * 100


def evaluate(y, y_hat):
    return {
        "MAE": float(mae(y, y_hat)),
        "RMSE": float(rmse(y, y_hat)),
        "MAPE": float(mape(y, y_hat)),
        "SMAPE": float(smape(y, y_hat)),
    }