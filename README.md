#  Golf Scorecard Scanner

This solution incorporates **YOLOv8 OBB (Oriented Bounding Box)** for flexible field detection, **VietOCR** for text recognition, and **Flash API by Google** for data labeling. Additionally, we'll discuss handling complexities like symbols, multiple users, and varying templates.

---

## **Project Overview**
This solution:
1. Detects handwritten fields using **YOLO OBB**.
2. Recognizes text using **VietOCR** tailored for Vietnamese handwriting.
3. Handles multiple layouts/templates dynamically.
4. Extracts structured data and exports it to a JSON file.
5. Uses **Google's Flash API** for large-scale annotation of images.

---

## **Project Constraints**
1. The scorecards primarily contain handwritten information that needs to be accurately interpreted.

2. Symbols, such as a triangle (△) or circle (〇), have specific meanings: an empty triangle represents a value of -1, while a triangle containing a number (e.g., 2 or -2) reflects the explicitly stated value. Circles indicate positive values regardless of whether the number is inside or outside.

3. Slots that are completely blackened, deleted, or scratched out should be treated as empty fields.

4. Numbers that have been hand-corrected, such as changing a 1 to a 7, should be detected and interpreted correctly.

5. Each golf course or location may have distinct scorecard formats, affecting the layout of fields and requiring adaptable contour detection and grid labeling.

6. Some locations have multiple courses, such as "Course A", "Course B", or names like "West", "East", "South", or "North". The system must identify which course corresponds to the entered values for each user.

7. Scorecards typically list multiple users, and the system must detect and assign names and corresponding scores accurately for each individual.

8. Additional fields may exist on certain scorecards, such as "Caddie Name" or "HDC (Handicap)", and column headers might vary (e.g., "Total" could be labeled as "Net"). These variations must be accounted for during processing.

9. The text on the scorecards is written in Vietnamese, necessitating the use of the VietOCR model for accurate text recognition.

---

## **1. Setting Up the Environment**

### Install Required Libraries
```bash
# Core Libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python pillow numpy pandas pytesseract vietocr

# Flash API and Labeling
pip install google-cloud-storage google-cloud-vision label-studio albumentations flask flask-restful

# Additional Dependencies for YOLOv8 OBB
pip install shapely matplotlib
```

---

## **2. Directory Structure**
Organize the project as follows:
```plaintext
GolfScorecardScanner/
├── data/                # Input images and labeled datasets
│   ├── images/          # Scorecard images
│   ├── labels/          # YOLO OBB annotations (e.g., bounding boxes)
├── models/              # Trained models
├── output/              # JSON output files
├── scripts/             # Scripts for detection, recognition, JSON generation
│   ├── detect_fields.py
│   ├── recognize_text.py
│   ├── generate_json.py
│   ├── parse_symbols.py
├── train/               # Training scripts and configurations
│   ├── yolo_train.py
├── templates/           # Template-matching logic for dynamic field mapping
├── utils.py             # Utility functions
├── main.py              # Prediction BE
└── app.py               # Flask application
```

---

## **3. Handling Constraints**

### **Handwritten Text (Constraint 1)**

- **VietOCR** is used for recognizing Vietnamese handwriting.
- Preprocessing involves:
  1. Converting images to grayscale.
  2. Binarization using adaptive thresholding.

---

### **Symbols Parsing (Constraint 2)**

Symbols like `△` (triangle) and `〇` (circle) require custom parsing:
1. **△ (Triangle)**: Represents a negative score unless a number is inside.
2. **〇 (Circle)**: Represents a positive score with or without numbers inside.

**Code Example**:
```python
def parse_symbols(text):
    if "△" in text:
        return -1 if text == "△" else int(text.strip("△")) * -1
    elif "〇" in text:
        return int(text.strip("〇"))
    elif text.isdigit():
        return int(text)
    return None
```

---

### **Uncertainties & Corrections (Constraints 3 & 4)**

- Detect **scratches/blackened slots** using a confidence threshold from YOLO's detection or additional blob detection using OpenCV.
- **Corrected numbers**: Use contour analysis to detect overlayed characters and apply probabilistic matching.

---

### **Dynamic Templates (Constraints 5, 6, 8)**

- **YOLO OBB** handles varying layouts by training on diverse templates and using oriented bounding boxes.
- **Dynamic Mapping**: A dictionary maps detected labels to standard fields (e.g., "Net" ↔ "Total").

**Dynamic Template Mapping Example**:
```python
TEMPLATE_MAP = {
    "course_a": {"PlayerName": "User", "Total": "Net"},
    "course_b": {"HDC": "Handicap", "Score": "Points"}
}

def map_fields(fields, template):
    mapped_fields = {}
    for field in fields:
        mapped_label = TEMPLATE_MAP.get(template, {}).get(field["label"], field["label"])
        mapped_fields[mapped_label] = field
    return mapped_fields
```

This currently handled in utils.py.  

---

### **Multiple Users (Constraint 7)**

Each user's name and data are extracted separately. YOLO detects user names by assigning a "PlayerName" label. A dictionary organizes data for each user.

---

### **Vietnamese OCR (Constraint 9)**

VietOCR recognizes Vietnamese handwriting effectively, especially when fine-tuned on scorecards.

---

## **4. Field Detection with YOLO OBB**

### Training YOLOv8 OBB
1. Prepare data with oriented bounding boxes (YOLO OBB format).
2. Train the YOLO OBB model:
   ```bash
   yolo task=detect mode=train data=data.yaml model=yolov8x-obb.pt epochs=50 imgsz=640
   ```
3. YOLO OBB data format (`data.yaml`):
   ```yaml
   train: data/train/images
   val: data/val/images
   nc: 10  # Number of classes
   names: ["PlayerName", "CourseName", "Score", "Total", "CaddieName", ...]
   ```

### Field Detection Script: `detect_fields.py`
```python
from ultralytics import YOLO

def detect_fields(image_path, model_path="models/yolo_obb.pt"):
    model = YOLO(model_path)
    results = model(image_path)
    fields = [
        {"label": model.names[int(box[-1])], "bbox": box[:4].tolist()}
        for box in results[0].boxes.data
    ]
    return fields
```

---

## **5. Text Recognition with VietOCR**

### Training VietOCR
1. Annotate text fields with tools like **Label Studio**.
2. Train VietOCR:
   ```python
   from vietocr.tool.config import Cfg
   from vietocr.model.trainer import Trainer

   config = Cfg.load_config_from_name('vgg_transformer')
   config['dataset']['train_annotation'] = 'train_annotations.txt'
   config['dataset']['valid_annotation'] = 'valid_annotations.txt'
   config['trainer']['epochs'] = 30
   Trainer(config).train()
   ```

### Text Recognition Script: `recognize_text.py`
```python
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

def recognize_text(image, model):
    return model.predict(Image.fromarray(image))

if __name__ == "__main__":
    image_path = "path/to/field.jpg"
    vietocr_model = Predictor(Cfg.load_config_from_name("vgg_transformer"))
    text = recognize_text(cv2.imread(image_path), vietocr_model)
    print(text)
```

---

## **6. Generate JSON Output**
### Script: `generate_json.py`
```python
import json

def write_json(fields, recognized_texts, output_path="output/scorecard.json"):
    result = {}
    for field, text in zip(fields, recognized_texts):
        result[field["label"]] = text

    with open(output_path, "w") as json_file:
        json.dump(result, json_file, indent=4)

if __name__ == "__main__":
    fields = [{"label": "PlayerName", "bbox": [0, 0, 100, 50]}]
    recognized_texts = ["John Doe"]
    write_json(fields, recognized_texts)
```

---

## **7. Google Flash API for Annotation**
### Setting Up Flash API
1. Install **Google Cloud Vision**:
   ```bash
   pip install google-cloud-vision
   ```
2. Use Flash API to label data:
   ```python
   from google.cloud import vision

   client = vision.ImageAnnotatorClient()

   def label_image(image_path):
       with open(image_path, "rb") as img_file:
           content = img_file.read()
       image = vision.Image(content=content)
       response = client.text_detection(image=image)
       return response.text_annotations
   ```

---

## **8. Flask Deployment**
Deploy with Flask for scalable usage:
```python
from flask import Flask, request, jsonify
from detect_fields import detect_fields
from recognize_text import recognize_text

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_scorecard():
    image = request.files['file']
    fields = detect_fields(image)
    texts = [recognize_text(field) for field in fields]
    return jsonify({"fields": fields, "texts": texts})

if __name__ == "__main__":
    app.run(debug=True)
```

