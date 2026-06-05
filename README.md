# Traffic Rule Violation Detection — AID 728
**Roll Number:** BT2024220

## Overview
A computer vision pipeline that detects traffic rule violations involving two-wheelers from street images.

## Violations Detected
- More than 2 riders on a single motorcycle
- One or more riders not wearing a helmet
- Both of the above combined

## Pipeline
1. **Detection** — YOLO11l-seg detects motorcycles and persons (two passes, imgsz=1536)
2. **Enclosed-person filter** — Persons inside cars/buses/trucks (≥25% mask overlap) are removed
3. **Assignment** — Two-phase greedy scoring assigns persons to motorcycles using 6 weighted geometric signals
4. **Helmet check** — Custom YOLO helmet model classifies riders on a padded bike crop
5. **Plate reading** — YOLO plate detector + PaddleOCR (PP-OCRv5 mobile) reads the license plate from the full image
6. **Violation aggregation** — Results combined into the required output dict

## Models (`models/`)
| File | Purpose | Size |
|------|---------|------|
| `yolo11l-seg.pt` | Person + vehicle segmentation | ~54 MB |
| `helmet_model.pt` | Helmet / no-helmet classification | ~49 MB |
| `plate_model.pt` | License plate region detection | ~39 MB |
| `paddle_ocr/PP-OCRv5_mobile_det/` | PaddleOCR text detection | ~5 MB |
| `paddle_ocr/en_PP-OCRv5_mobile_rec/` | PaddleOCR text recognition | ~8 MB |

**Total: ~153 MB** (well under the 250 MB limit)

## Setup

### 1. Download model weights

The `models/` folder is hosted on Google Drive (too large for direct submission):

**Download link:** https://drive.google.com/drive/folders/11axiqNj0geUpuyz7Rh42xB_cLKp2CQgH?usp=sharing

Download the zip, extract it, and place the contents so the folder structure looks exactly like this:

```
BT2024220/
├── solution.py
├── requirements.txt
├── README.md
└── models/
    ├── yolo11l-seg.pt
    ├── helmet_model.pt
    ├── plate_model.pt
    └── paddle_ocr/
        ├── PP-OCRv5_mobile_det/
        │   ├── inference.pdiparams
        │   ├── inference.json
        │   ├── inference.yml
        │   └── config.json
        └── en_PP-OCRv5_mobile_rec/
            ├── inference.pdiparams
            ├── inference.json
            ├── inference.yml
            └── config.json
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage
```python
from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="./models")
result = detector.predict("image.jpg")
print(result)
```
