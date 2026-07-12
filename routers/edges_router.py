"""
Module 09 – Edge Detection
Endpoint: POST /api/edges/detect
Detects edges using Canny, Sobel, or Laplacian.
OpenCV: cv2.Canny, cv2.Sobel, cv2.Laplacian
"""
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image

router = APIRouter()


@router.post("/detect")
async def detect_edges(
    file: UploadFile = File(...),
    algorithm: str = Form("canny"),
    threshold1: int = Form(100),
    threshold2: int = Form(200),
    ksize: int = Form(3),
):
    data = await file.read()
    img = decode_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if algorithm == "canny":
        edges = cv2.Canny(gray, threshold1, threshold2)

    elif algorithm == "sobel":
        k = ksize if ksize % 2 == 1 else ksize + 1
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.convertScaleAbs(magnitude)

    elif algorithm == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize if ksize % 2 == 1 else ksize + 1)
        edges = cv2.convertScaleAbs(lap)

    else:
        edges = gray

    # Convert single-channel back to BGR for display
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return {
        "original_b64": encode_image(img),
        "edges_b64": encode_image(edges_bgr),
        "algorithm": algorithm,
    }
