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
