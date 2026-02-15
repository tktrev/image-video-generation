# Image/Video Generation API

A FastAPI-based service for generating videos from text and images using the [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video) model.

## Features

- **Text-to-Video Generation**: Generate videos from text prompts
- **Image-to-Video Generation**: Animate static images into videos
- **Batch Processing**: Generate multiple videos in a single request
- **GPU Acceleration**: Optimized for CUDA-based GPU inference
- **Cloud Storage**: Automatic upload to MinIO/S3-compatible storage
- **Health Monitoring**: Built-in health checks for GPU and storage

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- MinIO or S3-compatible storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image-video-generation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Configure the following environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `MINIO_ENDPOINT` | MinIO/S3 endpoint URL | Required |
| `MINIO_ACCESS_KEY` | MinIO access key | Required |
| `MINIO_SECRET_KEY` | MinIO secret key | Required |
| `MINIO_BUCKET` | Storage bucket name | Required |
| `MINIO_URL` | Public MinIO URL | Required |
| `MINIO_REGION` | MinIO region | `us-east-1` |
| `MODEL_ID` | HuggingFace model ID | `Lightricks/LTX-Video` |
| `DEFAULT_WIDTH` | Default video width | `704` |
| `DEFAULT_HEIGHT` | Default video height | `480` |
| `DEFAULT_NUM_FRAMES` | Default frame count | `161` |
| `DEFAULT_FPS` | Default frames per second | `24` |
| `DEFAULT_INFERENCE_STEPS` | Default diffusion steps | `30` |
| `DEFAULT_GUIDANCE_SCALE` | Default guidance scale | `7.5` |

## Usage

### Starting the Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "cuda_available": true,
  "device": "cuda",
  "txt2video_loaded": false,
  "img2video_loaded": false
}
```

#### Text-to-Video Generation
```bash
POST /lightricks-api/txt2video
```

Request body:
```json
{
  "prompt": "A beautiful sunset over the ocean",
  "negative_prompt": "blurry, low quality",
  "width": 704,
  "height": 480,
  "num_frames": 161,
  "fps": 24,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "num_videos_per_prompt": 1,
  "seed": 42
}
```

#### Image-to-Video Generation
```bash
POST /lightricks-api/img2video
```

Request body:
```json
{
  "image_url": "https://example.com/image.jpg",
  "prompt": "The image is moving gently",
  "negative_prompt": "blurry, distorted",
  "width": 704,
  "height": 480,
  "num_frames": 161,
  "fps": 24,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "num_videos_per_prompt": 1,
  "seed": 42
}
```

#### Batch Text-to-Video Generation
```bash
POST /lightricks-api/txt2video/batch
```

Request body: Same as text-to-video (generates multiple videos)

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
image-video-generation/
├── src/
│   ├── api/
│   │   ├── lightricks_txt2video.py   # Text-to-video endpoints
│   │   └── lightricks_img2video.py   # Image-to-video endpoints
│   ├── config/
│   │   └── settings.py               # Configuration management
│   ├── models/
│   │   ├── request_model.py         # Pydantic request models
│   │   └── model_manager.py         # Singleton model manager
│   ├── utils/
│   │   └── misc.py                   # Utility functions
│   └── main.py                      # FastAPI application
├── .env.example                     # Environment template
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Parameter Reference

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `prompt` | string | Required | Text description of the video |
| `negative_prompt` | string | Optional | Things to avoid |
| `width` | integer | 64-1280 | Video width (multiple of 8) |
| `height` | integer | 64-720 | Video height (multiple of 8) |
| `num_frames` | integer | 1-300 | Number of frames |
| `fps` | integer | 1-60 | Frames per second |
| `num_inference_steps` | integer | 1-100 | Diffusion steps |
| `guidance_scale` | float | 1.0-20.0 | Prompt guidance strength |
| `num_videos_per_prompt` | integer | 1-4 | Videos per prompt |
| `seed` | integer | Optional | Random seed for reproducibility |

## Issues Fixed & Improvements

### Critical Bug Fixes
- **Fixed**: Duplicate `image_url` field in `GenerationRequestImage` model

### Architecture Improvements
- **Singleton Pattern**: Implemented `ModelManager` for efficient model loading (models now load once)
- **Thread Safety**: Added proper locking for model management
- **Resource Management**: Improved GPU memory handling with better cleanup
- **Environment Validation**: Added startup validation for required environment variables

### Error Handling
- CUDA availability checking with graceful fallback
- GPU out-of-memory error handling
- MinIO connection validation
- Comprehensive logging with structured output

### New Features
- Batch video generation endpoint (`/txt2video/batch`)
- Extended health checks (GPU, MinIO)
- Root endpoint with API information
- Lifespan context manager for startup/shutdown

### Code Quality
- Removed code duplication between endpoints
- Added proper type hints
- Improved logging structure
- Cleaner resource cleanup

## Performance Tips

1. **GPU Memory**: Monitor GPU memory usage; reduce `num_frames` if out of memory
2. **Model Caching**: Models are cached in memory after first use for faster subsequent requests
3. **Batch Processing**: Use batch endpoint for generating multiple variants
4. **Seed Usage**: Use fixed seeds for reproducible outputs during testing

## License

MIT License
