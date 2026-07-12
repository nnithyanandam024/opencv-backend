"""
Module 07 – Color Spaces
Endpoint: POST /api/color/convert
Converts an image between color spaces.
OpenCV: cv2.cvtColor with BGR2GRAY, BGR2HSV, BGR2RGB, etc.
"""
import cv2
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image

router = APIRouter()

COLOR_CONVERSIONS = {
    "grayscale": cv2.COLOR_BGR2GRAY,
    "hsv":       cv2.COLOR_BGR2HSV,
    "rgb":       cv2.COLOR_BGR2RGB,
    "lab":       cv2.COLOR_BGR2LAB,
    "ycrcb":     cv2.COLOR_BGR2YCrCb,
    "hls":       cv2.COLOR_BGR2HLS,
}


@router.post("/convert")
async def convert_color(
    file: UploadFile = File(...),
    space: str = Form("grayscale"),
):
    data = await file.read()
    img = decode_image(data)

    code = COLOR_CONVERSIONS.get(space)
    if code is None:
        return {"error": f"Unknown color space: {space}"}

    converted = cv2.cvtColor(img, code)

    # Grayscale → needs to be converted back to BGR for JPEG encoding
    if space == "grayscale":
        converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)

    return {
        "original_b64": encode_image(img),
        "converted_b64": encode_image(converted),
        "space": space,
        "cv2_code": str(code),
    }
