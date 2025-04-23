import os
from minio import Minio
from dotenv import load_dotenv

load_dotenv()

def get_minio_client():
    return Minio(
        endpoint=os.getenv("MINIO_ENDPOINT").replace("https://", "").replace("http://", ""),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=True,
        region="us-east-1"
    )

def upload_to_minio(filename):
    client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET")
    minio_url = os.getenv("MINIO_URL")
    s3_key = f"generated-videos/{filename}"
    with open(filename, "rb") as f:
        stat = os.stat(filename)
        client.put_object(
            bucket_name=bucket,
            object_name=s3_key,
            data=f,
            length=stat.st_size,
            content_type="video/mp4"
        )
    return f"https://{minio_url}/{bucket}/{s3_key}"

def cleanup_temp_files(filename):
    if filename and os.path.exists(filename):
        os.remove(filename)