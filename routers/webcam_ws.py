"""
Module 03 / Live Processing – WebSocket handler
Endpoint: WS /ws/webcam/{module}
Receives base64-encoded frames from the browser webcam,
processes them with OpenCV, and sends back processed frames.

Supported modules: webcam, blur, edges, colorspace, face, detect
"""
import cv2
import numpy as np
import base64
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from utils import decode_image, encode_image

router = APIRouter()

# Cache YOLO model so it isn't reloaded every frame
_yolo_model = None

# Check if ultralytics is available
try:
    import ultralytics as _ult  # noqa: F401
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def process_frame(img: np.ndarray, module: str, params: dict) -> np.ndarray:
    """Apply OpenCV processing based on the selected module."""

    if module == "webcam":
        return img  # raw feed, no processing

    elif module == "blur":
        k = int(params.get("ksize", 15))
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    elif module == "edges":
        t1 = int(params.get("threshold1", 100))
        t2 = int(params.get("threshold2", 200))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, t1, t2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif module == "colorspace":
        space = params.get("space", "grayscale")
        if space == "grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif space == "hsv":
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif space == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    elif module == "face":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = float(params.get("scale_factor", 1.1))
        neighbors = int(params.get("min_neighbors", 5))
        faces = face_cascade.detectMultiScale(gray, scale, neighbors, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img

    elif module == "detect":
        if not YOLO_AVAILABLE:
            cv2.putText(img, "ultralytics not installed", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(img, "pip install ultralytics --no-cache-dir", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            return img
        try:
            conf = float(params.get("confidence", 0.4))
            model = get_yolo()
            results = model(img, conf=conf, verbose=False)
            return results[0].plot()
        except Exception:
            return img

    return img


@router.websocket("/ws/webcam/{module}")
async def webcam_ws(websocket: WebSocket, module: str):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            frame_b64: str = msg.get("frame", "")
            params: dict = msg.get("params", {})

            if not frame_b64:
                continue

            # Decode incoming base64 JPEG frame
            img_bytes = base64.b64decode(frame_b64)
            img = decode_image(img_bytes)

            if img is None:
                continue

            processed = process_frame(img, module, params)
            result_b64 = encode_image(processed)

            await websocket.send_text(json.dumps({"frame": result_b64}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))
