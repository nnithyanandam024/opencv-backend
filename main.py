"""
FastAPI main application entry point.
Run with: uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import (
    image_router,
    video_router,
    draw_router,
    text_router,
    resize_router,
    color_router,
    blur_router,
    edges_router,
    threshold_router,
    contours_router,
    face_router,
    detect_router,
    webcam_ws,
)

app = FastAPI(
    title="OpenCV Learning API",
    description="Real-time image processing using OpenCV for all 13 learning modules.",
    version="1.0.0",
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST Routes ────────────────────────────────────────────
app.include_router(image_router.router,     prefix="/api/image",     tags=["01 - Image"])
app.include_router(video_router.router,     prefix="/api/video",     tags=["02 - Video"])
app.include_router(draw_router.router,      prefix="/api/draw",      tags=["04 - Drawing"])
app.include_router(text_router.router,      prefix="/api/text",      tags=["05 - Text"])
app.include_router(resize_router.router,    prefix="/api/resize",    tags=["06 - Resize"])
app.include_router(color_router.router,     prefix="/api/color",     tags=["07 - Color"])
app.include_router(blur_router.router,      prefix="/api/blur",      tags=["08 - Blur"])
app.include_router(edges_router.router,     prefix="/api/edges",     tags=["09 - Edges"])
app.include_router(threshold_router.router, prefix="/api/threshold", tags=["10 - Threshold"])
app.include_router(contours_router.router,  prefix="/api/contours",  tags=["11 - Contours"])
app.include_router(face_router.router,      prefix="/api/face",      tags=["12 - Face"])
app.include_router(detect_router.router,    prefix="/api/detect",    tags=["13 - YOLO"])

# ── WebSocket Route ────────────────────────────────────────
app.include_router(webcam_ws.router, tags=["03 - Webcam / Live"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "OpenCV Learning API is running 🟢"}
