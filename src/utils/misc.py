import os
import logging
from typing import Optional
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT")
    if not endpoint:
        raise ValueError("MINIO_ENDPOINT environment variable is not set")
    
    clean_endpoint = endpoint.replace("https://", "").replace("http://", "")
    
    return Minio(
        endpoint=clean_endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=endpoint.startswith("https"),
        region=os.getenv("MINIO_REGION", "us-east-1")
    )


def upload_to_minio(filename: str) -> str:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET")
    minio_url = os.getenv("MINIO_URL")
    prefix = os.getenv("MINIO_PREFIX", "generated-videos/")
    
    if not bucket or not minio_url:
        raise ValueError("MINIO_BUCKET or MINIO_URL not configured")
    
    s3_key = f"{prefix}{filename}"
    
    try:
        with open(filename, "rb") as f:
            stat = os.stat(filename)
            client.put_object(
                bucket_name=bucket,
                object_name=s3_key,
                data=f,
                length=stat.st_size,
                content_type="video/mp4"
            )
        logger.info(f"Uploaded {filename} to MinIO")
        return f"https://{minio_url}/{bucket}/{s3_key}"
    except S3Error as e:
        logger.error(f"MinIO upload error: {str(e)}")
        raise


def cleanup_temp_files(filename: Optional[str]) -> None:
    if filename and os.path.exists(filename):
        try:
            os.remove(filename)
            logger.debug(f"Cleaned up temp file: {filename}")
        except OSError as e:
            logger.warning(f"Failed to cleanup file {filename}: {str(e)}")


def check_minio_connection() -> bool:
    try:
        client = get_minio_client()
        bucket = os.getenv("MINIO_BUCKET")
        if bucket:
            return client.bucket_exists(bucket)
        return False
    except Exception as e:
        logger.error(f"MinIO connection check failed: {str(e)}")
        return False
