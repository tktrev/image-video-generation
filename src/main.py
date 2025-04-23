from fastapi import FastAPI
from src.api.lightricks_txt2video import router as lightricks_txt2video_router
from src.api.lightricks_img2video import router as lightricks_img2video_router
from src.config.settings import setup_logging

app = FastAPI(title="Image/Video Generator", version="1.0")
setup_logging()

app.include_router(lightricks_txt2video_router)
app.include_router(lightricks_img2video_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}