"""
Module 08 – Blur / Smoothing
Endpoint: POST /api/blur/apply
Applies one of four blur types to an image.
OpenCV: cv2.GaussianBlur, cv2.blur, cv2.medianBlur, cv2.bilateralFilter
"""
import cv2
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image

router = APIRouter()


@router.post("/apply")
async def apply_blur(
    file: UploadFile = File(...),
    blur_type: str = Form("gaussian"),
    kernel_size: int = Form(15),
    sigma: float = Form(0),
    diameter: int = Form(9),
    sigma_color: float = Form(75),
    sigma_space: float = Form(75),
):
    data = await file.read()
    img = decode_image(data)

    # Kernel size must be odd
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if blur_type == "gaussian":
        result = cv2.GaussianBlur(img, (k, k), sigma)
    elif blur_type == "average":
        result = cv2.blur(img, (k, k))
    elif blur_type == "median":
        result = cv2.medianBlur(img, k)
    elif blur_type == "bilateral":
        result = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    else:
        result = img

    return {
        "original_b64": encode_image(img),
        "blurred_b64": encode_image(result),
        "blur_type": blur_type,
        "kernel_size": k,
    }
