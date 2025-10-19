import os
import cv2
from ultralytics import YOLO


def detect_image(input_path, output_dir, conf=0.5, model_path="weights/best.pt"):
    # --- Check paths ---
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Load model ---
    model = YOLO(model_path)

    # --- Read image ---
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")

    # --- Run inference ---
    results = model(img, conf=conf)
    out_img = results[0].plot()  # image with boxes (BGR)

    # --- Save output ---
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_detected{ext}")

    cv2.imwrite(output_path, out_img)

    print(f"✅ Detection done: {output_path}")
    return output_path

import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO

def detect_image_bytes(image_bytes: bytes, conf=0.5, model_path="Image/weights/best.pt") -> bytes:
    """
    Runs YOLO detection on an image from bytes and returns the result as PNG bytes.
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Failed to decode image bytes")

    # Load YOLO model
    model = YOLO(model_path)

    # Run inference
    results = model(img, conf=conf)
    out_img = results[0].plot()  # image with boxes

    # Encode image to bytes (PNG)
    success, buffer = cv2.imencode('.png', out_img)
    if not success:
        raise ValueError("Failed to encode output image")
    return buffer.tobytes()

# Example usage
if __name__ == "__main__":
    input_image = r"inputs/example4.jpg"   
    output_directory = r"outputs/"      
    detect_image(input_image, output_directory, conf=0.5)
