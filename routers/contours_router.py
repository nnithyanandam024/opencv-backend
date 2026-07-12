"""
Module 11 – Contours
Endpoint: POST /api/contours/find
Finds and draws contours on an image.
OpenCV: cv2.findContours, cv2.drawContours, cv2.boundingRect,
        cv2.contourArea, cv2.arcLength
"""
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image

router = APIRouter()


@router.post("/find")
async def find_contours(
    file: UploadFile = File(...),
    min_area: float = Form(100.0),
    draw_bounding: bool = Form(True),
):
    data = await file.read()
    img = decode_image(data)
    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

    cv2.drawContours(output, filtered, -1, (0, 255, 0), 2)

    contour_data = []
    for i, c in enumerate(filtered):
        area = round(cv2.contourArea(c), 2)
        perimeter = round(cv2.arcLength(c, True), 2)
        x, y, w, h = cv2.boundingRect(c)

        if draw_bounding:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(output, str(i + 1), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        contour_data.append({
            "id": i + 1,
            "area": area,
            "perimeter": perimeter,
            "bounding_box": {"x": x, "y": y, "w": w, "h": h},
        })

    return {
        "original_b64": encode_image(img),
        "contours_b64": encode_image(output),
        "total_contours": len(filtered),
        "contours": contour_data[:20],  # return first 20 for display
    }
