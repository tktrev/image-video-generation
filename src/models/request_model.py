from pydantic import BaseModel, Field
from typing import Optional

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the video to generate.")
    negative_prompt: Optional[str] = Field(None, description="Things to avoid in the generation.")
    width: Optional[int] = Field(704, ge=64, le=1280, description="Video width (multiple of 8).")
    height: Optional[int] = Field(480, ge=64, le=720, description="Video height (multiple of 8).")
    num_frames: Optional[int] = Field(161, ge=1, le=300, description="Number of frames to generate.")
    fps: Optional[int] = Field(24, description="Frames per second for output video.")
    num_inference_steps: Optional[int] = Field(30, ge=1, le=100, description="Diffusion steps.")
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0, description="Prompt guidance strength.")
    num_videos_per_prompt: Optional[int] = Field(1, ge=1, le=4, description="Number of videos to generate.")
    seed: Optional[int] = Field(None, description="Random seed for reproducible output.")
    
class GenerationRequestImage(BaseModel):
    image_url: str = Field(..., description="Input image usead as base for the video.")
    prompt: str = Field(..., description="Text prompt describing the video to generate.")
    negative_prompt: Optional[str] = Field(None, description="Things to avoid in the generation.")
    image_url: Optional[str] = Field(..., description="Things to avoid in the generation.")
    width: Optional[int] = Field(704, ge=64, le=1280, description="Video width (multiple of 8).")
    height: Optional[int] = Field(480, ge=64, le=720, description="Video height (multiple of 8).")
    num_frames: Optional[int] = Field(161, ge=1, le=300, description="Number of frames to generate.")
    fps: Optional[int] = Field(24, description="Frames per second for output video.")
    num_inference_steps: Optional[int] = Field(30, ge=1, le=100, description="Diffusion steps.")
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0, description="Prompt guidance strength.")
    num_videos_per_prompt: Optional[int] = Field(1, ge=1, le=4, description="Number of videos to generate.")
    seed: Optional[int] = Field(None, description="Random seed for reproducible output.")