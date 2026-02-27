import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MINIO_ENDPOINT: Optional[str] = os.getenv("MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: Optional[str] = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: Optional[str] = os.getenv("MINIO_SECRET_KEY")
    MINIO_BUCKET: Optional[str] = os.getenv("MINIO_BUCKET")
    MINIO_URL: Optional[str] = os.getenv("MINIO_URL")
    MINIO_PREFIX: str = os.getenv("MINIO_PREFIX", "generated-videos/")
    MODEL_ID: str = os.getenv("MODEL_ID", "Lightricks/LTX-Video")
    DEFAULT_WIDTH: int = int(os.getenv("DEFAULT_WIDTH", "704"))
    DEFAULT_HEIGHT: int = int(os.getenv("DEFAULT_HEIGHT", "480"))
    DEFAULT_NUM_FRAMES: int = int(os.getenv("DEFAULT_NUM_FRAMES", "161"))
    DEFAULT_FPS: int = int(os.getenv("DEFAULT_FPS", "24"))
    DEFAULT_INFERENCE_STEPS: int = int(os.getenv("DEFAULT_INFERENCE_STEPS", "30"))
    DEFAULT_GUIDANCE_SCALE: float = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "7.5"))
    
    @classmethod
    def validate(cls) -> bool:
        required = ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET", "MINIO_URL"]
        missing = [field for field in required if not getattr(cls, field)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        return True


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


settings = Settings()