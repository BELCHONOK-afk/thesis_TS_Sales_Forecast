import pandas as pd

from app.models.holt_winters import hw_forecast
from app.metrics import evaluate


def ts_split(series, horizon):
    return series.iloc[:-horizon], series.iloc[-horizon:]


def prepare_groups(df, group_by):

    if group_by == "total":
        return {"total": df.groupby("date")["value"].sum()}

    elif group_by == "product":
        return {
            k: v.groupby("date")["value"].sum()
            for k, v in df.groupby("product")
        }

    elif group_by == "market":
        return {
            k: v.groupby("date")["value"].sum()
            for k, v in df.groupby("market")
        }

    elif group_by == "market_product":
        return {
            (m, p): v.groupby("date")["value"].sum()
            for (m, p), v in df.groupby(["market", "product"])
        }


def run_forecast(df, horizons, group_by, trend, seasonal, seasonal_periods):

    results = []

    groups = prepare_groups(df, group_by)

    for group_key, series in groups.items():

        series = series.sort_index()
        series = series.asfreq("MS").fillna(0)

        # защита от mul модели с нулями
        if (series <= 0).any():
            seasonal = "add"

        for h in horizons:

            if len(series) < 2 * seasonal_periods or len(series) <= h:
                continue

            train, test = ts_split(series, h)

            try:
                forecast = hw_forecast(
                    train,
                    h,
                    trend,
                    seasonal,
                    seasonal_periods
                )

                metrics = evaluate(test.values, forecast)

                result = {
                    "horizon": h,
                    "model": "holt_winters",
                    **metrics
                }

                # раскладываем ключ
                if isinstance(group_key, tuple):
                    result["market"], result["product"] = group_key
                elif group_by == "product":
                    result["product"] = group_key
                elif group_by == "market":
                    result["market"] = group_key

                results.append(result)

            except Exception:
                continue

    return pd.DataFrame(results)