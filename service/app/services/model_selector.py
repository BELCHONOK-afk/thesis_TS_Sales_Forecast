from app.services.data_loader import load_best_models


def get_available_series():
    df = load_best_models()

    series = (
        df[["market", "product"]]
        .drop_duplicates()
        .sort_values(["market", "product"])
        .to_dict(orient="records")
    )

    return series


def get_best_model(
    market: str,
    product: str,
    horizon: int | None = None,
):
    df = load_best_models()

    filtered = df[
        (df["market"] == market)
        & (df["product"] == product)
    ]

    if horizon is not None:
        filtered = filtered[filtered["horizon"] == horizon]

    if filtered.empty:
        return None

    best_row = (
        filtered
        .sort_values("smape")
        .iloc[0]
        .to_dict()
    )

    return best_row