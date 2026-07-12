"""
Module 02 – Video
Endpoint: POST /api/video/info
Returns FPS, frame count, duration, resolution, and 6 sample frames.
OpenCV: cv2.VideoCapture, cap.get(), cap.read()
"""
import cv2
import tempfile
import os
from fastapi import APIRouter, UploadFile, File
from utils import encode_image

router = APIRouter()


@router.post("/info")
async def video_info(file: UploadFile = File(...)):
    data = await file.read()

    # Save to a temp file because VideoCapture needs a file path
    suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = round(frame_count / fps, 2) if fps > 0 else 0

    # Extract 6 evenly-spaced sample frames
    sample_frames = []
    sample_positions = [int(frame_count * i / 6) for i in range(6)]
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            thumb = cv2.resize(frame, (320, 180))
            sample_frames.append(encode_image(thumb))

    cap.release()
    os.unlink(tmp_path)

    return {
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": duration,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "sample_frames": sample_frames,
    }
