import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Mars Forecasting Dashboard",
    layout="wide",
)

st.title("Mars Sales Forecasting Dashboard")


# ==========================================
# LOAD SERIES
# ==========================================

series_response = requests.get(f"{API_URL}/series")

series = series_response.json()

series_df = pd.DataFrame(series)

markets = sorted(series_df["market"].unique())
products = sorted(series_df["product"].unique())


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Filters")


selected_aggregation = st.sidebar.selectbox(
    "Aggregation Level",
    ["total", "market", "product", "market_product"],
)

selected_market = None
selected_product = None

if selected_aggregation in ["market", "market_product"]:
    selected_market = st.sidebar.selectbox(
        "Market",
        markets,
    )

if selected_aggregation in ["product", "market_product"]:
    selected_product = st.sidebar.selectbox(
        "Product",
        products,
    )

selected_horizon = st.sidebar.select_slider(
    "Forecast Horizon",
    options=[3, 6, 12],
    value=6,
)


# ==========================================
# GET FORECAST
# ==========================================

payload = {
    "aggregation_level": selected_aggregation,
    "market": selected_market,
    "product": selected_product,
    "horizon": selected_horizon,
}

forecast_response = requests.post(
    f"{API_URL}/forecast",
    json=payload,
)

if forecast_response.status_code != 200:
    st.error(forecast_response.text)
    st.stop()

result = forecast_response.json()

if selected_aggregation == "market_product":
    metrics_response = requests.get(
        f"{API_URL}/metrics",
        params={
            "market": selected_market,
            "product": selected_product,
            "horizon": selected_horizon,
        },
    )

    if metrics_response.status_code != 200:
        metrics_df = pd.DataFrame()
    else:
        metrics_df = pd.DataFrame(metrics_response.json())
else:
    metrics_df = pd.DataFrame()



# ==========================================
# MODEL INFO
# ==========================================

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model",
    result["model_name"],
)

col2.metric(
    "Market",
    result["market"],
)

col3.metric(
    "Product",
    result["product"],
)

st.subheader("Forecast Quality Metrics")

if selected_aggregation != "market_product":
    st.info("Metrics are currently available only for market_product level")
elif metrics_df.empty:
    st.info("No metrics available for selected filters")
else:
    best_metric_row = metrics_df.sort_values("smape").iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAPE", f"{best_metric_row['mape']:.2f}%")
    col2.metric("SMAPE", f"{best_metric_row['smape']:.2f}%")
    col3.metric("RMSE", f"{best_metric_row['rmse']:.2f}")
    col4.metric("MAE", f"{best_metric_row['mae']:.2f}")

    st.subheader("Metrics by Horizon")

    horizon_metrics_response = requests.get(
        f"{API_URL}/metrics",
        params={
            "market": selected_market,
            "product": selected_product,
        },
    )

    if horizon_metrics_response.status_code == 200:
        horizon_metrics_df = pd.DataFrame(
            horizon_metrics_response.json()
        )

        horizon_metrics_df = horizon_metrics_df.sort_values("horizon")

        fig_metrics = go.Figure()

        fig_metrics.add_trace(
            go.Scatter(
                x=horizon_metrics_df["horizon"],
                y=horizon_metrics_df["mape"],
                mode="lines+markers",
                name="MAPE",
            )
        )

        fig_metrics.add_trace(
            go.Scatter(
                x=horizon_metrics_df["horizon"],
                y=horizon_metrics_df["smape"],
                mode="lines+markers",
                name="SMAPE",
            )
        )

        fig_metrics.update_layout(
            height=400,
            xaxis_title="Forecast Horizon",
            yaxis_title="Error, %",
        )

        st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.info("No horizon metrics available")

st.subheader("Business Metrics")

business_response = requests.get(
    f"{API_URL}/business-metrics",
    params={
        "aggregation_level": selected_aggregation,
        "market": selected_market,
        "product": selected_product,
    },
)

if business_response.status_code != 200:
    st.info("Business metrics not found")
else:
    business_df = pd.DataFrame(business_response.json())
    business_df["date"] = pd.to_datetime(business_df["date"])

    latest = business_df.sort_values("date").iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Revenue", f"{latest['revenue']:,.0f}")
    col2.metric("Volume", f"{latest['volume']:,.0f}")
    col3.metric("Avg Price", f"{latest['avg_price']:,.2f}")
    col4.metric("Penetration", f"{latest['penetration']:.2%}")

    selected_business_metric = st.selectbox(
        "Business metric",
        [
            "revenue",
            "volume",
            "avg_price",
            "penetration",
            "frequency",
            "spend_per_trip",
            "volume_per_trip",
        ],
    )

    fig_business = go.Figure()

    fig_business.add_trace(
        go.Scatter(
            x=business_df["date"],
            y=business_df[selected_business_metric],
            mode="lines+markers",
            name=selected_business_metric,
        )
    )

    fig_business.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title=selected_business_metric,
    )

    st.plotly_chart(fig_business, use_container_width=True)

    with st.expander("Business Metrics Table"):
        st.dataframe(business_df, use_container_width=True)
# ==========================================
# HISTORY
# ==========================================

history_df = pd.DataFrame(result["history"])

history_df["date"] = pd.to_datetime(history_df["date"])

forecast_values = result["forecast"]

last_date = history_df["date"].max()

forecast_dates = pd.date_range(
    start=last_date,
    periods=selected_horizon + 1,
    freq="MS",
)[1:]


forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "forecast": forecast_values,
})


# ==========================================
# PLOT
# ==========================================

st.subheader("Actual vs Forecast")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=history_df["date"],
        y=history_df["sales"],
        mode="lines+markers",
        name="Actual",
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["forecast"],
        mode="lines+markers",
        name="Forecast",
    )
)

fig.update_layout(
    height=600,
    xaxis_title="Date",
    yaxis_title="Sales",
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================
# FORECAST TABLE
# ==========================================

st.subheader("Forecast Values")

st.dataframe(forecast_df)


# ==========================================
# RAW HISTORY
# ==========================================

with st.expander("Historical Data"):
    st.dataframe(history_df)

with st.expander("Metrics Table"):
    if not metrics_df.empty:
        st.dataframe(metrics_df)