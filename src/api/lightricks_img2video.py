import os
import uuid
import torch
import logging
import gc
from fastapi import APIRouter, HTTPException
from datetime import datetime
from diffusers import LTXImageToVideoPipeline
from minio.error import S3Error
from diffusers.utils import export_to_video, load_image
from src.utils.misc import upload_to_minio, cleanup_temp_files
from src.models.request_model import GenerationRequestImage
from src.models.model_manager import get_model_manager

router = APIRouter(prefix="/lightricks-api")
logger = logging.getLogger(__name__)


@router.post("/img2video")
def generate_img2video(req: GenerationRequestImage):
    manager = get_model_manager()
    filename = None
    
    try:
        if not torch.cuda.is_available() and manager.device == "cuda":
            raise HTTPException(status_code=503, detail="CUDA not available")
        
        logger.info(f"Generating video from image: {req.image_url[:50]}...")
        pipe = manager.get_img2video_pipe()
        
        image = load_image(req.image_url)
        
        generator = None
        if req.seed and req.seed > 0:
            generator = torch.Generator(device=manager.device).manual_seed(req.seed)
        
        result = pipe(
            image=image,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
            num_videos_per_prompt=req.num_videos_per_prompt,
        )

        frames = result.frames[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
        export_to_video(frames, filename, fps=req.fps)

        url = upload_to_minio(filename)
        return {"message": "Video generated and uploaded (img2video)", "url": url}

    except S3Error as e:
        logger.error(f"MinIO error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MinIO error: {str(e)}")
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory")
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory")
    except Exception as e:
        logger.error(f"Unhandled exception occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_files(filename)
        gc.collect()
        torch.cuda.empty_cache()
