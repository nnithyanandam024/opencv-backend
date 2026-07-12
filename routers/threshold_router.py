"""
Module 10 – Thresholding
Endpoint: POST /api/threshold/apply
Applies binary/adaptive/Otsu thresholding.
OpenCV: cv2.threshold, cv2.adaptiveThreshold, cv2.THRESH_OTSU
"""
import cv2
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image

router = APIRouter()


@router.post("/apply")
async def apply_threshold(
    file: UploadFile = File(...),
    method: str = Form("binary"),
    thresh_value: int = Form(127),
    max_value: int = Form(255),
    block_size: int = Form(11),
    C: int = Form(2),
):
    data = await file.read()
    img = decode_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "binary":
        _, result = cv2.threshold(gray, thresh_value, max_value, cv2.THRESH_BINARY)

    elif method == "binary_inv":
        _, result = cv2.threshold(gray, thresh_value, max_value, cv2.THRESH_BINARY_INV)

    elif method == "otsu":
        _, result = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive_mean":
        bs = block_size if block_size % 2 == 1 else block_size + 1
        result = cv2.adaptiveThreshold(
            gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, C
        )

    elif method == "adaptive_gaussian":
        bs = block_size if block_size % 2 == 1 else block_size + 1
        result = cv2.adaptiveThreshold(
            gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, C
        )

    else:
        result = gray

    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return {
        "original_b64": encode_image(img),
        "threshold_b64": encode_image(result_bgr),
        "method": method,
    }
