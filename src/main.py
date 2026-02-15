import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.lightricks_txt2video import router as lightricks_txt2video_router
from src.api.lightricks_img2video import router as lightricks_img2video_router
from src.config.settings import setup_logging, settings
from src.utils.misc import check_minio_connection
from src.models.model_manager import get_model_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting Image/Video Generation API")
    
    try:
        settings.validate()
        logger.info("Environment variables validated")
    except ValueError as e:
        logger.warning(f"Environment validation failed: {str(e)}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available, using CPU")
    
    if check_minio_connection():
        logger.info("MinIO connection successful")
    else:
        logger.warning("MinIO connection failed or bucket not accessible")
    
    yield
    
    logger.info("Shutting down Image/Video Generation API")
    manager = get_model_manager()
    manager.clear_cache()


app = FastAPI(
    title="Image/Video Generator",
    description="API for generating videos from text and images using Lightricks LTX-Video model",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(lightricks_txt2video_router)
app.include_router(lightricks_img2video_router)


@app.get("/health")
def health_check():
    manager = get_model_manager()
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": manager.device,
        "txt2video_loaded": manager.is_model_loaded("txt2video"),
        "img2video_loaded": manager.is_model_loaded("img2video"),
    }


@app.get("/health/minio")
def minio_health():
    connected = check_minio_connection()
    return {
        "status": "ok" if connected else "error",
        "minio_connected": connected
    }


@app.get("/")
def root():
    return {
        "message": "Image/Video Generation API",
        "docs": "/docs",
        "endpoints": {
            "txt2video": "/lightricks-api/txt2video",
            "img2video": "/lightricks-api/img2video",
            "health": "/health"
        }
    }
