"""
Module 12 – Face Detection
Endpoint: POST /api/face/detect
Detects faces using Haar Cascade Classifier.
OpenCV: cv2.CascadeClassifier, detectMultiScale
"""
import cv2
from fastapi import APIRouter, UploadFile, File, Form
from utils import decode_image, encode_image

router = APIRouter()

# Haar cascade is bundled with opencv-python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


@router.post("/detect")
async def detect_faces(
    file: UploadFile = File(...),
    scale_factor: float = Form(1.1),
    min_neighbors: int = Form(5),
    detect_eyes: bool = Form(False),
):
    data = await file.read()
    img = decode_image(data)
    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
    )

    face_list = []
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Face {i+1}"
        cv2.putText(output, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        face_list.append({"id": i + 1, "x": int(x), "y": int(y),
                           "w": int(w), "h": int(h)})

        if detect_eyes:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = output[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.circle(roi_color,
                           (ex + ew // 2, ey + eh // 2),
                           ew // 2, (255, 0, 0), 2)

    return {
        "original_b64": encode_image(img),
        "detected_b64": encode_image(output),
        "face_count": len(faces),
        "faces": face_list,
    }
