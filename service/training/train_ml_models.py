import os
import json
import joblib
from pathlib import Path

import boto3
import mlflow
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from training.metrics import evaluate


# =========================
# CONFIG
# =========================

DATA_PATH = "data/processed/data_4_forecast.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET = "sales"

HORIZONS = [3, 6, 12]

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5050",
)

MLFLOW_EXPERIMENT_NAME = "mars_ml_training"

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mars-forecasting")


# =========================
# PARAM GRIDS
# =========================

lgbm_param_grid = {
    "n_estimators": [100, 300],
    "learning_rate": [0.03, 0.05],
    "num_leaves": [15, 31],
    "max_depth": [-1, 5],
}

xgb_param_grid = {
    "n_estimators": [100, 300],
    "learning_rate": [0.03, 0.05],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
}


# =========================
# S3 UTILS
# =========================

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def ensure_bucket_exists():
    client = get_s3_client()

    try:
        client.head_bucket(Bucket=S3_BUCKET_NAME)
    except Exception:
        client.create_bucket(Bucket=S3_BUCKET_NAME)


def upload_model_to_s3(local_path: Path, s3_key: str):
    ensure_bucket_exists()

    client = get_s3_client()

    client.upload_file(
        Filename=str(local_path),
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
    )

    return f"s3://{S3_BUCKET_NAME}/{s3_key}"
def preprocess(df):
    from sklearn.preprocessing import OrdinalEncoder

    df_ml = df.copy()

    df_ml['date'] = pd.to_datetime(df_ml['date'])

    df_ml = df_ml.sort_values(['product', 'market', 'date']).reset_index(drop=True)

    # календарные признаки
    df_ml['year'] = df_ml['date'].dt.year
    df_ml['month'] = df_ml['date'].dt.month
    df_ml['quarter'] = df_ml['date'].dt.quarter

    df_ml['month_sin'] = np.sin(2 * np.pi * df_ml['month'] / 12)
    df_ml['month_cos'] = np.cos(2 * np.pi * df_ml['month'] / 12)

    # кодирование категориальных признаков
    cat_cols = ['product', 'market']

    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    df_ml[cat_cols] = encoder.fit_transform(df_ml[cat_cols])
    return df_ml, encoder


# =========================
# TRAINING
# =========================

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df_ml, encoder = preprocess(pd.read_csv(DATA_PATH))

    features = [
        'product', 'market', 'sales_lag1', 'month_sin',
        'month_cos', 'year', 'month', 'quarter',
        'sales_lag3', 'sales_lag6', 'sales_lag12', 'sales_rolling3_mean',
        'sales_rolling3_std', 'sales_rolling6_mean', 'sales_rolling6_std',
        'sales_rolling12_mean', 'sales_rolling12_std', 'Frequency',
        'Penetration', 'Spend per Trip'
    ]

    experiment_results = []

    for horizon in HORIZONS:
        print(f"===== Horizon: {horizon} =====")

        max_date = df_ml["date"].max()
        test_start = max_date - pd.DateOffset(months=horizon - 1)

        train_df = df_ml[df_ml["date"] < test_start].copy()
        test_df = df_ml[df_ml["date"] >= test_start].copy()

        X_train = train_df[features]
        y_train = train_df[TARGET]

        X_test = test_df[features]
        y_test = test_df[TARGET]

        tscv = TimeSeriesSplit(n_splits=3)

        models = [
            (
                "LightGBM",
                LGBMRegressor(
                    random_state=42,
                    objective="regression",
                    verbose=-1,
                ),
                lgbm_param_grid,
            ),
            (
                "XGBoost",
                XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                ),
                xgb_param_grid,
            ),
        ]

        for model_name, model, param_grid in models:
            print(f"----- {model_name} -----")

            with mlflow.start_run(
                run_name=f"{model_name}_horizon_{horizon}"
            ):
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="neg_mean_absolute_error",
                    cv=tscv,
                    n_jobs=-1,
                    verbose=1,
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

                pred = best_model.predict(X_test)
                pred = np.maximum(pred, 0)

                global_metrics = evaluate(y_test, pred)

                model_file_name = f"{model_name}_horizon_{horizon}.pkl"
                model_path = MODELS_DIR / model_file_name

                artifact = {
                    "model": best_model,
                    "features": features,
                    "target": TARGET,
                    "horizon": horizon,
                    "model_name": model_name,
                    "best_params": grid.best_params_,
                }

                joblib.dump(artifact, model_path)

                s3_key = f"models/{model_file_name}"
                s3_uri = upload_model_to_s3(model_path, s3_key)

                mlflow.log_param("model_name", model_name)
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("target", TARGET)
                mlflow.log_param("features", json.dumps(features))
                mlflow.log_param("s3_model_uri", s3_uri)
                mlflow.log_param("cv_score", grid.best_score_)

                for param_name, param_value in grid.best_params_.items():
                    mlflow.log_param(param_name, param_value)

                for metric_name, metric_value in global_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                mlflow.log_artifact(str(model_path), artifact_path="models")

                temp = test_df.copy()
                temp["prediction"] = pred

                series_metrics = (
                    temp
                    .groupby(["product", "market"])
                    .apply(
                        lambda x: pd.Series(
                            evaluate(
                                x[TARGET],
                                x["prediction"],
                            )
                        )
                    )
                    .reset_index()
                )

                series_metrics["horizon"] = horizon
                series_metrics["model_name"] = model_name
                series_metrics["model_params"] = json.dumps(grid.best_params_)
                series_metrics["cv_score"] = grid.best_score_
                series_metrics["model_artifact_s3_key"] = s3_key

                experiment_results.append(series_metrics)

                print(f"Saved model locally: {model_path}")
                print(f"Uploaded model to S3: {s3_uri}")
                print(f"Metrics: {global_metrics}")

    results_df = pd.concat(experiment_results, ignore_index=True)
    decoded = encoder.inverse_transform(
        results_df[['product', 'market']]
    )

    results_df[
        ['product', 'market']
    ] = decoded

    output_path = "data/processed/ml_experiment_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"Saved experiment results: {output_path}")


if __name__ == "__main__":
    main()