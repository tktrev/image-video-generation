import os
import uuid
import torch
import logging
import gc
from fastapi import APIRouter, HTTPException
from datetime import datetime
from diffusers import LTXPipeline
from minio.error import S3Error
from diffusers.utils import export_to_video
from src.utils.misc import upload_to_minio, cleanup_temp_files
from src.models.request_model import GenerationRequest
from src.models.model_manager import get_model_manager

router = APIRouter(prefix="/lightricks-api")
logger = logging.getLogger(__name__)


@router.post("/txt2video")
def generate_txt2video(req: GenerationRequest):
    manager = get_model_manager()
    filename = None
    
    try:
        if not torch.cuda.is_available() and manager.device == "cuda":
            raise HTTPException(status_code=503, detail="CUDA not available")
        
        logger.info(f"Generating video for prompt: {req.prompt[:50]}...")
        pipe = manager.get_txt2video_pipe()
        
        generator = None
        if req.seed and req.seed > 0:
            generator = torch.Generator(device=manager.device).manual_seed(req.seed)

        result = pipe(
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
        return {"message": "Video generated and uploaded (txt2video)", "url": url}

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


@router.post("/txt2video/batch")
def generate_txt2video_batch(req: GenerationRequest):
    manager = get_model_manager()
    filenames_to_cleanup = []
    
    try:
        if not torch.cuda.is_available() and manager.device == "cuda":
            raise HTTPException(status_code=503, detail="CUDA not available")
        
        logger.info(f"Generating batch video for prompt: {req.prompt[:50]}...")
        pipe = manager.get_txt2video_pipe()
        
        generator = None
        if req.seed and req.seed > 0:
            generator = torch.Generator(device=manager.device).manual_seed(req.seed)

        result = pipe(
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

        urls = []
        for idx, frames in enumerate(result.frames):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}_{uuid.uuid4().hex[:8]}_batch_{idx}.mp4"
            filenames_to_cleanup.append(filename)
            export_to_video(frames, filename, fps=req.fps)
            url = upload_to_minio(filename)
            urls.append(url)
            cleanup_temp_files(filename)

        return {"message": "Batch videos generated and uploaded", "urls": urls}

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
        for f in filenames_to_cleanup:
            cleanup_temp_files(f)
        gc.collect()
        torch.cuda.empty_cache()
