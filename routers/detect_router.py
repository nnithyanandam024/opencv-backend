"""
Module 13 – YOLO Object Detection
Endpoint: POST /api/detect/yolo
Runs YOLOv8n on an uploaded image.
Uses: ultralytics YOLO (model auto-downloads yolov8n.pt ~6MB on first use)
"""
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from utils import decode_image, encode_image

router = APIRouter()

# Lazy-load YOLO to avoid slowing down server startup
_yolo_model = None

# Check once at import time whether ultralytics is available
try:
    import ultralytics as _ult  # noqa: F401
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")  # nano model — fast & small
    return _yolo_model


@router.post("/yolo")
async def yolo_detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.4),
):
    if not YOLO_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "error": "ultralytics not installed",
                "fix": "Run: pip install ultralytics --no-cache-dir",
            },
        )

    data = await file.read()
    img = decode_image(data)

    model = get_yolo()
    results = model(img, conf=confidence, verbose=False)

    # annotated frame with bounding boxes drawn by YOLO
    annotated = results[0].plot()

    # Extract detection metadata
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = round(float(box.conf[0]), 3)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return {
        "original_b64": encode_image(img),
        "detected_b64": encode_image(annotated),
        "total_objects": len(detections),
        "detections": detections,
    }
