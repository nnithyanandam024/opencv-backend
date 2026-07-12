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
    try:
        data = await file.read()
        img = decode_image(data)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

        # Run lightweight ONNX inference
        annotated, detections = run_yolo_inference(img, confidence)

        return {
            "original_b64": encode_image(img),
            "detected_b64": encode_image(annotated),
            "total_objects": len(detections),
            "detections": detections,
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"YOLO Detection Error:\n{tb}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb}
        )


