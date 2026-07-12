"""
Module 13 – YOLO Object Detection
Endpoint: POST /api/detect/yolo
Runs YOLOv8n on an uploaded image using lightweight cv2.dnn.
"""
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image, run_yolo_inference

router = APIRouter()


@router.post("/yolo")
async def yolo_detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.4),
):
    data = await file.read()
    img = decode_image(data)

    # Run lightweight ONNX inference
    annotated, detections = run_yolo_inference(img, confidence)

    return {
        "original_b64": encode_image(img),
        "detected_b64": encode_image(annotated),
        "total_objects": len(detections),
        "detections": detections,
    }

