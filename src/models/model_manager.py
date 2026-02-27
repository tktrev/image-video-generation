import os
import gc
import torch
import logging
from functools import lru_cache
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from typing import Optional, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._txt2video_pipe = None
                    cls._instance._img2video_pipe = None
                    cls._instance._device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance
    
    @property
    def device(self) -> str:
        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self._device = "cpu"
        return self._device
    
    def get_txt2video_pipe(self, model_id: str = "Lightricks/LTX-Video") -> LTXPipeline:
        if self._txt2video_pipe is None:
            with self._lock:
                if self._txt2video_pipe is None:
                    logger.info(f"Loading LTX Text-to-Video Pipeline on {self.device}...")
                    self._txt2video_pipe = LTXPipeline.from_pretrained(
                        model_id, 
                        torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                    )
                    self._txt2video_pipe.to(self.device)
                    logger.info("Text-to-Video model loaded successfully.")
        return self._txt2video_pipe
    
    def get_img2video_pipe(self, model_id: str = "Lightricks/LTX-Video") -> LTXImageToVideoPipeline:
        if self._img2video_pipe is None:
            with self._lock:
                if self._img2video_pipe is None:
                    logger.info(f"Loading LTX Image-to-Video Pipeline on {self.device}...")
                    self._img2video_pipe = LTXImageToVideoPipeline.from_pretrained(
                        model_id, 
                        torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                    )
                    self._img2video_pipe.to(self.device)
                    logger.info("Image-to-Video model loaded successfully.")
        return self._img2video_pipe
    
    def clear_cache(self):
        if self._txt2video_pipe:
            self._txt2video_pipe = None
        if self._img2video_pipe:
            self._img2video_pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model cache cleared.")
    
    def is_model_loaded(self, model_type: str = "txt2video") -> bool:
        if model_type == "txt2video":
            return self._txt2video_pipe is not None
        return self._img2video_pipe is not None


model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return model_manager
