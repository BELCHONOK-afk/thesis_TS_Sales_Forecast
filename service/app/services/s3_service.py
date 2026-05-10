import os
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError


S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mars-forecasting")


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
    except ClientError:
        client.create_bucket(Bucket=S3_BUCKET_NAME)


def upload_file_to_s3(
    local_path: str | Path,
    s3_key: str,
):
    ensure_bucket_exists()

    client = get_s3_client()

    client.upload_file(
        Filename=str(local_path),
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
    )

    return {
        "bucket": S3_BUCKET_NAME,
        "key": s3_key,
    }


def read_csv_from_s3(s3_key: str) -> pd.DataFrame:
    client = get_s3_client()

    response = client.get_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
    )

    body = response["Body"].read()

    return pd.read_csv(BytesIO(body))


def object_exists(s3_key: str) -> bool:
    client = get_s3_client()

    try:
        client.head_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
        )
        return True
    except ClientError:
        return False