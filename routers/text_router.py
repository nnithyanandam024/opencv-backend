"""
Module 05 – Text Overlay
Endpoint: POST /api/text/overlay
Adds cv2.putText on an uploaded image.
OpenCV: cv2.putText, cv2.FONT_HERSHEY_*
"""
import cv2
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image

router = APIRouter()

FONTS = {
    "simplex":    cv2.FONT_HERSHEY_SIMPLEX,
    "plain":      cv2.FONT_HERSHEY_PLAIN,
    "duplex":     cv2.FONT_HERSHEY_DUPLEX,
    "complex":    cv2.FONT_HERSHEY_COMPLEX,
    "triplex":    cv2.FONT_HERSHEY_TRIPLEX,
    "script":     cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
}


@router.post("/overlay")
async def text_overlay(
    file: UploadFile = File(...),
    text: str = Form("Hello OpenCV!"),
    font: str = Form("simplex"),
    font_scale: float = Form(1.5),
    color_r: int = Form(0),
    color_g: int = Form(255),
    color_b: int = Form(0),
    thickness: int = Form(2),
    x: int = Form(50),
    y: int = Form(50),
):
    data = await file.read()
    img = decode_image(data)

    cv_font = FONTS.get(font, cv2.FONT_HERSHEY_SIMPLEX)
    color = (color_b, color_g, color_r)  # OpenCV uses BGR

    cv2.putText(img, text, (x, y), cv_font, font_scale, color, thickness, cv2.LINE_AA)

    return {"image_b64": encode_image(img)}
