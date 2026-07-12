"""
Shared utility helpers for image encoding/decoding between
OpenCV (NumPy arrays) and the HTTP API (base64 strings).
"""
import cv2
import numpy as np
import base64


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw file bytes into an OpenCV BGR image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def encode_image(img: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode an OpenCV BGR image to a base64 string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92] if fmt == ".jpg" else []
    _, buffer = cv2.imencode(fmt, img, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


def b64_to_image(b64_str: str) -> np.ndarray:
    """Decode a base64 string back into an OpenCV BGR image."""
    img_bytes = base64.b64decode(b64_str)
    return decode_image(img_bytes)


# ── Lightweight cv2.dnn YOLOv8 ONNX Engine ───────────────────────
import os
import urllib.request

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

_yolo_net = None

def download_model_if_needed():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLOv8n ONNX model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

def get_yolo_net():
    global _yolo_net
    if _yolo_net is None:
        download_model_if_needed()
        _yolo_net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    return _yolo_net

def run_yolo_inference(img: np.ndarray, confidence_threshold: float = 0.4):
    """Run YOLOv8 object detection on a frame using OpenCV DNN."""
    net = get_yolo_net()
    h_img, w_img = img.shape[:2]

    # Preprocess image to blob (640x640)
    blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward() # shape: (1, 84, 8400)

    # Postprocess output
    # Transpose to (8400, 84)
    preds = np.transpose(preds[0], (1, 0))

    boxes = []
    confidences = []
    class_ids = []

    for pred in preds:
        scores = pred[4:]
        class_id = np.argmax(scores)
        conf = scores[class_id]

        if conf >= confidence_threshold:
            cx, cy, w, h = pred[:4]
            # Convert normalized 640x640 coords to input size coords
            x = int((cx - w/2) / 640.0 * w_img)
            y = int((cy - h/2) / 640.0 * h_img)
            width = int(w / 640.0 * w_img)
            height = int(h / 640.0 * h_img)

            boxes.append([x, y, width, height])
            confidences.append(float(conf))
            class_ids.append(int(class_id))

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.45)

    detections = []
    output_img = img.copy()

    if len(indices) > 0:
        indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
        for i in indices:
            x, y, w, h = boxes[i]
            conf = confidences[i]
            class_id = class_ids[i]
            label = COCO_CLASSES[class_id]

            # Keep box within image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)

            # Draw bounding box (Monochrome styling)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            label_str = f"{label} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_img, (x, y - text_h - 4), (x + text_w, y), (0, 0, 0), -1)
            cv2.putText(output_img, label_str, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
            })

    return output_img, detections

