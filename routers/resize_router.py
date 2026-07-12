"""
Module 06 – Resize & Crop
Endpoints:
  POST /api/resize/resize  → cv2.resize
  POST /api/resize/crop    → numpy slicing
OpenCV: cv2.resize
"""
import cv2
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image

router = APIRouter()


@router.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: int = Form(640),
    height: int = Form(480),
    keep_aspect: bool = Form(False),
):
    data = await file.read()
    img = decode_image(data)
    orig_h, orig_w = img.shape[:2]

    if keep_aspect:
        ratio = min(width / orig_w, height / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
    else:
        new_w, new_h = width, height

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return {
        "original_size": f"{orig_w}x{orig_h}",
        "new_size": f"{new_w}x{new_h}",
        "image_b64": encode_image(resized),
    }


@router.post("/crop")
async def crop_image(
    file: UploadFile = File(...),
    x1: int = Form(0),
    y1: int = Form(0),
    x2: int = Form(400),
    y2: int = Form(300),
):
    data = await file.read()
    img = decode_image(data)
    h, w = img.shape[:2]

    # Clamp coordinates to image bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))

    # NumPy slicing — the core of cropping in OpenCV
    cropped = img[y1:y2, x1:x2]

    return {
        "crop_region": f"({x1},{y1}) → ({x2},{y2})",
        "cropped_size": f"{x2-x1}x{y2-y1}",
        "image_b64": encode_image(cropped),
    }
