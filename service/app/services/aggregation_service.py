import pandas as pd


VALID_AGGREGATION_LEVELS = {
    "total",
    "market",
    "product",
    "market_product",
}


def aggregate_sales_history(
    df: pd.DataFrame,
    aggregation_level: str,
    market: str | None = None,
    product: str | None = None,
) -> pd.DataFrame:
    if aggregation_level not in VALID_AGGREGATION_LEVELS:
        raise ValueError(
            f"Unknown aggregation_level: {aggregation_level}. "
            f"Available: {VALID_AGGREGATION_LEVELS}"
        )

    df = df.copy()

    if aggregation_level == "total":
        result = (
            df.groupby("date", as_index=False)["sales"]
            .sum()
            .sort_values("date")
        )
        result["market"] = "total"
        result["product"] = "total"
        return result

    if aggregation_level == "market":
        if market is None:
            raise ValueError("market is required for aggregation_level='market'")

        result = df[df["market"] == market]

        result = (
            result.groupby("date", as_index=False)["sales"]
            .sum()
            .sort_values("date")
        )
        result["market"] = market
        result["product"] = "all_products"
        return result

    if aggregation_level == "product":
        if product is None:
            raise ValueError("product is required for aggregation_level='product'")

        result = df[df["product"] == product]

        result = (
            result.groupby("date", as_index=False)["sales"]
            .sum()
            .sort_values("date")
        )
        result["market"] = "all_markets"
        result["product"] = product
        return result

    if aggregation_level == "market_product":
        if market is None or product is None:
            raise ValueError(
                "market and product are required for aggregation_level='market_product'"
            )

        result = df[
            (df["market"] == market)
            & (df["product"] == product)
        ].sort_values("date")

        return result

def aggregate_business_metrics(
    df: pd.DataFrame,
    aggregation_level: str,
    market: str | None = None,
    product: str | None = None,
) -> pd.DataFrame:
    if aggregation_level not in VALID_AGGREGATION_LEVELS:
        raise ValueError(f"Unknown aggregation_level: {aggregation_level}")

    df = df.copy()

    sum_cols = ["revenue", "volume"]
    mean_cols = [
        "avg_price",
        "penetration",
        "frequency",
        "spend_per_trip",
        "volume_per_trip",
    ]

    if aggregation_level == "total":
        group_cols = ["date"]
        result = df.groupby(group_cols, as_index=False).agg(
            {**{c: "sum" for c in sum_cols}, **{c: "mean" for c in mean_cols}}
        )
        result["market"] = "total"
        result["product"] = "total"
        return result.sort_values("date")

    if aggregation_level == "market":
        if market is None:
            raise ValueError("market is required for aggregation_level='market'")

        df = df[df["market"] == market]
        result = df.groupby("date", as_index=False).agg(
            {**{c: "sum" for c in sum_cols}, **{c: "mean" for c in mean_cols}}
        )
        result["market"] = market
        result["product"] = "all_products"
        return result.sort_values("date")

    if aggregation_level == "product":
        if product is None:
            raise ValueError("product is required for aggregation_level='product'")

        df = df[df["product"] == product]
        result = df.groupby("date", as_index=False).agg(
            {**{c: "sum" for c in sum_cols}, **{c: "mean" for c in mean_cols}}
        )
        result["market"] = "all_markets"
        result["product"] = product
        return result.sort_values("date")

    if aggregation_level == "market_product":
        if market is None or product is None:
            raise ValueError(
                "market and product are required for aggregation_level='market_product'"
            )

        return df[
            (df["market"] == market)
            & (df["product"] == product)
        ].sort_values("date")