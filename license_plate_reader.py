"""
license_plate_reader.py
"""

import argparse
import os
import re
import sys
import warnings

# ── Suppress all third-party warnings before any imports ─────────────────────
warnings.filterwarnings("ignore")
os.environ["FLAGS_enable_pir_in_executor"]   = "0"
os.environ["FLAGS_use_mkldnn"]               = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]           = "3"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"

import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.disable(logging.WARNING)   # silence all WARNING-level logs globally

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR


# -----------------------------------------------------------------------------
# Indian plate correction
# -----------------------------------------------------------------------------

INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CH","CG","DD","DL","DN",
    "GA","GJ","HR","HP","JK","JH","KA","KL","LA","LD",
    "MP","MH","MN","ML","MZ","NL","OD","PY","PB","RJ",
    "SK","TN","TS","TR","UK","UP","WB",
}

PLATE_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$")


def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def correct_plate(text):
    text  = clean_text(text)
    match = PLATE_RE.match(text)
    if not match:
        return text
    state, district, series, number = match.groups()
    if state not in INDIAN_STATES:
        return text
    return f"{state} {district} {series} {number}"


# -----------------------------------------------------------------------------
# Image preprocessing
# -----------------------------------------------------------------------------

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# -----------------------------------------------------------------------------
# OCR
# -----------------------------------------------------------------------------

ocr = PaddleOCR(
    use_textline_orientation=False,
    lang="en",
    enable_mkldnn=False,
    cpu_threads=4,
    # Use mobile det (~5 MB) instead of server_det (~85 MB) — keeps total OCR under 15 MB
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    # Disable heavyweight sub-models not needed for cropped plate images
    use_doc_orientation_classify=False,  # saves ~6 MB PP-LCNet model
    use_doc_unwarping=False,             # saves ~31 MB UVDoc model
)


def read_plate(crop):
    processed = preprocess_plate(crop)

    try:
        results = ocr.predict(processed)
    except Exception:
        return "<unreadable>"

    if not results:
        return "<unreadable>"

    texts = []
    try:
        for text, conf in zip(results[0]["rec_texts"], results[0]["rec_scores"]):
            if conf > 0.30:
                texts.append(text)
    except Exception:
        return "<unreadable>"

    if not texts:
        return "<unreadable>"

    final_text = clean_text("".join(texts))
    if len(final_text) < 4:
        return "<unreadable>"

    return correct_plate(final_text)


# -----------------------------------------------------------------------------
# Main extraction
# -----------------------------------------------------------------------------

def extract_plates(image_path, model_path, conf_thresh):
    model  = YOLO(model_path, verbose=False)
    image  = cv2.imread(image_path)

    if image is None:
        sys.exit("Could not read image")

    results = model(image, conf=conf_thresh, verbose=True)[0]
    output  = image.copy()
    found   = []

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plate_text = read_plate(crop)
        found.append(plate_text)

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{plate_text} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2,
        )

        print(f"[Plate {idx+1}] {plate_text}")

    cv2.imwrite("output.jpg", output)
    print("\nSaved annotated image -> output.jpg")
    return found


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image",   nargs="?",         help="Input image path")
    parser.add_argument("--image", dest="image_flag", help="Input image path (named)")
    parser.add_argument("--model", default="models/plate_model.pt")
    parser.add_argument("--conf",  type=float, default=0.15)
    args = parser.parse_args()

    image_path = args.image or args.image_flag
    if not image_path:
        parser.error("provide an image path")

    extract_plates(image_path, args.model, args.conf)


if __name__ == "__main__":
    main()