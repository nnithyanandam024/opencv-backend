"""
Module 01 – Image Basics
Endpoint: POST /api/image/properties
Returns image shape, dtype, size, and a base64 thumbnail.
OpenCV: cv2.imread, img.shape, img.dtype
"""
import cv2
from fastapi import APIRouter, UploadFile, File
from utils import decode_image, encode_image

router = APIRouter()


@router.post("/properties")
async def image_properties(file: UploadFile = File(...)):
    data = await file.read()
    img = decode_image(data)

    if img is None:
        return {"error": "Could not decode image"}

    height, width, channels = img.shape
    total_pixels = height * width
    file_size_kb = round(len(data) / 1024, 2)

    # Create a smaller thumbnail for display
    thumb_h = 300
    ratio = thumb_h / height
    thumb = cv2.resize(img, (int(width * ratio), thumb_h))

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(img.dtype),
        "total_pixels": total_pixels,
        "file_size_kb": file_size_kb,
        "color_mode": "BGR (OpenCV default)",
        "image_b64": encode_image(thumb),
    }
