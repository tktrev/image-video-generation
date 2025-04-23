import os
import uuid
import torch
import logging
from fastapi import APIRouter, HTTPException
from datetime import datetime
from diffusers import LTXPipeline
from minio.error import S3Error
from diffusers.utils import export_to_video
from src.utils.misc import upload_to_minio, cleanup_temp_files
from src.models.request_model import GenerationRequest

router = APIRouter(prefix="/lightricks-api")
logger = logging.getLogger(__name__)

@router.post("/txt2video")
def generate_txt2video(req: GenerationRequest):
    global pipe

    logger.info("Loading LTX Pipeline...")
    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    logger.info("Model loaded successfully.")

    filename = None
    result = None
    frames = None
    generator = None

    try:
        logger.info(f"Generating video for prompt: {req.prompt}")
        if req.seed and req.seed > 0:
            generator = torch.Generator(device="cuda").manual_seed(req.seed)

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
        raise HTTPException(status_code=500, detail=f"MinIO error: {str(e)}")
    except Exception as e:
        logger.error(f"Unhandled exception occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_files(filename)
        del result
        del frames
        del generator
        del pipe
        import gc
        gc.collect()
        torch.cuda.empty_cache()