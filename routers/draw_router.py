"""
Module 04 – Drawing
Endpoint: POST /api/draw/shapes
Draws shapes on an uploaded image (or a blank canvas).
OpenCV: cv2.line, cv2.rectangle, cv2.circle, cv2.ellipse, cv2.polylines
"""
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from utils import decode_image, encode_image
import json

router = APIRouter()


@router.post("/shapes")
async def draw_shapes(
    file: Optional[UploadFile] = File(None),
    shapes: str = Form("[]"),
):
    """
    shapes: JSON array of shape objects, e.g.:
    [
      {"type": "rectangle", "x1":50,"y1":50,"x2":200,"y2":150,
       "color":[0,255,0], "thickness":2},
      {"type": "circle",    "cx":300,"cy":200,"r":80,
       "color":[255,0,0],   "thickness":3},
      {"type": "line",      "x1":10,"y1":10,"x2":400,"y2":400,
       "color":[0,0,255],   "thickness":2}
    ]
    """
    if file is not None:
        data = await file.read()
        img = decode_image(data)
    else:
        img = np.ones((480, 640, 3), dtype=np.uint8) * 30  # dark blank canvas

    shape_list = json.loads(shapes)

    for s in shape_list:
        color = tuple(s.get("color", [0, 255, 0]))
        thickness = int(s.get("thickness", 2))
        t = s.get("type", "")

        if t == "line":
            cv2.line(img, (s["x1"], s["y1"]), (s["x2"], s["y2"]), color, thickness)
        elif t == "rectangle":
            cv2.rectangle(img, (s["x1"], s["y1"]), (s["x2"], s["y2"]), color, thickness)
        elif t == "circle":
            cv2.circle(img, (s["cx"], s["cy"]), s["r"], color, thickness)
        elif t == "ellipse":
            cv2.ellipse(img,
                        (s["cx"], s["cy"]),
                        (s["axes_x"], s["axes_y"]),
                        s.get("angle", 0), 0, 360,
                        color, thickness)
        elif t == "polygon":
            pts = np.array(s["points"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    return {"image_b64": encode_image(img)}
