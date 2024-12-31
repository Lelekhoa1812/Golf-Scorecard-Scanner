# Golf Scorecard Scanner

This solution uses **YOLOv8 and YOLOv11** for field detection, **VietOCR** for text recognition, and **LabelMe** for annotation. It processes complex golf scorecard layouts and structures them into a JSON file for further analysis. Optionally, it incorporate Flash Vision API for direct parsing of data reading in case data cannot be accurate or sustainable.

---

## **Project Overview**
This solution:
1. Detects fields (e.g., PlayerName, CourseName, Score ...) using **YOLOv8** and **YOLOv11** large models.
2. Recognizes text using **VietOCR** tailored for Vietnamese handwriting.
3. Handles multiple layouts/templates dynamically.
4. Extracts structured data and exports it to a JSON file.
5. Uses **LabelMe** for annotating golf scorecard fields.  
**Optionally:** Use Flash Vision API with provided script and instruction in this project for direct data extraction.

---

## **Project Constraints**
1. The scorecards primarily contain handwritten information that needs to be accurately interpreted.

2. Symbols, such as a triangle (△) or circle (〇), have specific meanings: an empty triangle represents a value of -1, while a triangle containing a number (e.g., 2 or -2) reflects the explicitly stated value. Circles indicate positive values regardless of whether the number is inside or outside.

3. Slots that are completely blackened, deleted, or scratched out should be treated as empty fields.

4. Numbers that have been hand-corrected, such as changing a 1 to a 7, should be detected and interpreted correctly.

5. Each golf course or location may have distinct scorecard formats, affecting the layout of fields and requiring adaptable contour detection and grid labeling.

6. Some locations have multiple courses, such as "Course A", "Course B", or names like "West", "East", "South", or "North". The system must identify which course corresponds to the entered values for each user.

7. Scorecards typically list multiple users, and the system must detect and assign names and corresponding scores accurately for each individual.

8. Additional fields may exist on certain scorecards, such as "Hole" or "HDC (Handicap)", and column headers might vary (e.g., "Total", "Net"). These variations must be accounted for during processing.

9. The text on the scorecards is written in Vietnamese, necessitating the use of the VietOCR model for accurate text recognition.

---

## **1. Setting Up the Environment**

### Install Required Libraries
Install all required libraries with:
```bash
pip install -r requirements.txt
```

---

## **2. Directory Structure**
Organize the project as follows:
```plaintext
GolfScorecardScanner/
├── dataset/
│   ├── crop/
│       ├── PlayerName/
│       ├── CourseName/
│       ├── Score/
│       ├── ... (other field labels that are cropped)
│   ├── labels/          # LabelMe JSON files
│       ├── train/
│       ├── val/
│   ├── images/          # Source images
│       ├── train/
│       ├── val/
│   ├── config.yaml      # YOLO configuration file
├── models/              # Pre-trained models
│   ├── yolov8l.pt       # YOLOv8 large model
│   ├── yolov11l.pt      # YOLOv11 large model
│   ├── vietocr_weights.pth  # VietOCR weights
├── output/              # JSON output files
├── scripts/             # Detection, recognition, and export scripts (UI)
│   ├── detect_fields.py
│   ├── recognize_text.py
│   ├── generate_json.py
│   ├── utils.py
├── train/               # YOLO OBB Training scripts and configurations
│   ├── yolo_train.py
│   ├── hyp.yaml
├── labelling/           # Scripts aid in Flash Vision API detection
│   ├── thicken_grid.py  # Script that thicken the grid for better data extraction
│   ├── auto_label.py    # Script to extract data from input images into JSON using Flash Vision API
│   ├── convert_json_to_yolo.py # Script directly convert necessitated fields from JSON file into txt format for YOLO training
│   ├── visualize_labels_from_json.py # Script visualize debug image detected from Flash Vision JSON file
│   ├── visualize_labels_from_json.py # Script visualize debug image detected from YOLO txt conversion
├── key/                 # Your credential key from Google Flash Vision
├── data/                # Input and output directory from using Flash Vision
│   ├── train/           # All be used for training
│       ├── debug_label/ # Visualized images
│       ├── images/      # Source of input
│       ├── labels/      # Output labels in JSON
│       ├── yolo_labels/ # Output labels in txt
│       ├── processed_images/ # Output img with grid thickened
├── main.py              # Main pipeline script
```

---

## **3. Annotating Data with LabelMe**
1. Use **LabelMe** for annotating fields in golf scorecards:
   ```bash
   labelme dataset/src/
   ```
2. Save annotations in `dataset/labels/` as JSON files.

3. Convert LabelMe annotations to YOLO format:
   ```bash
   python scripts/convert_labelme_to_yolo.py
   ```

---

## **4. Training YOLO for Field Detection**

### Data Preparation
1. Organize your dataset:
```plaintext
├── dataset/
│   ├── crop/
│       ├── PlayerName/
│       ├── CourseName/
│       ├── Score/
│       ├── ... (other field labels that are cropped)
│   ├── labels/          
│       ├── train/
│       ├── val/
│   ├── images/          
│       ├── train/
│       ├── val/
```

### Training
Example script training the YOLO model:
```bash
yolo task=detect mode=train data=dataset/config.yaml model=yolov8l.pt epochs=100 imgsz=640
```
You can change configs from the pre-configured options below:

### **YOLOv8l:**  
#### a. Configuration:  
'''plaintext
batch: 32                 # Specific batch size
cache: "ram"              # Use RAM for caching
device: "0"               # Use the first GPU
epochs: 300               # Train for more epochs
imgsize: 640              
patience: 30              # Early stopping after 30 epochs of no improvement (optional)
lr0: 0.001                # Learning rate for stability
optimizer: "AdamW"        # Use AdamW optimizer
augment: True             # Enable advanced augmentations
weights: "yolov11l.pt"     # Start with pretrained weights
'''

#### b. Model Evaluation:  
<img src="imgsrc/YOLOv8l.png" alt="YOLO v8 Large Model Evaluation" style="width: 70%; max-width: 1000px;">    

---

### **YOLOv11l**
#### a. Configuration:  
'''plaintext
batch: 32                 # Specific batch size
cache: "ram"              # Use RAM for caching
device: "0"               # Use the first GPU
epochs: 300               # Train for more epochs
imgsize: 640              
patience: 30              # Early stopping after 30 epochs of no improvement (optional)
lr0: 0.001                # Learning rate for stability
optimizer: "AdamW"        # Use AdamW optimizer
augment: True             # Enable advanced augmentations
weights: "yolov11l.pt"     # Start with pretrained weights
'''

#### b. Model Evaluation:  
<img src="imgsrc/YOLOv11l.png" alt="YOLO v11 Large Model Evaluation" style="width: 70%; max-width: 1000px;">   

---

### Reflection on the Evaluation Graphs for YOLOv8l and YOLOv11l

#### a. **Comparison of mAP50 and mAP50-95**
   - **YOLOv8l:**
     - The mAP50 (blue line) shows a steady increase and stabilizes near 0.9 after 50 epochs. 
     - mAP50-95 (orange line) has a slower growth curve and plateaus around 0.65 after 90 epochs. This reflects moderate performance across varying IoU thresholds.
   - **YOLOv11l:**
     - The mAP50 reaches a higher value (~0.93) and stabilizes earlier compared to YOLOv8l, indicating stronger performance for IoU=0.5.
     - mAP50-95 grows more steadily and achieves a final value of ~0.8, significantly outperforming YOLOv8l. This suggests YOLOv11l generalizes better across different IoU thresholds.

#### b. **Precision and Recall**
   - **YOLOv8l:**
     - Precision (cyan line) increases steadily but fluctuates after epoch 30, stabilizing near 0.85. 
     - Recall (magenta line) grows steadily and stabilizes close to 0.87, indicating a balanced performance between finding true positives and avoiding false negatives.
   - **YOLOv11l:**
     - Precision stabilizes at a higher level (~0.9) compared to YOLOv8l. The consistent high precision shows YOLOv11l is more confident in its predictions.
     - Recall is slightly lower (~0.88), but it consistently improves over the epochs, indicating the model captures most relevant instances effectively.

#### c. **Training Epochs and Stability**
   - **YOLOv8l:**
     - Stabilizes after 70 epochs, but precision and mAP50-95 plateau at lower values than YOLOv11l, suggesting potential underfitting for complex data structures.
   - **YOLOv11l:**
     - Requires more training epochs (~150-200) to stabilize. However, the final metrics (mAP50, mAP50-95, precision, recall) are significantly higher, demonstrating better learning and performance.

#### d. **Model Complexity and Generalization**
   - YOLOv8l is less complex, requiring fewer epochs to train but sacrifices generalization (lower mAP50-95 and precision).
   - YOLOv11l, with higher complexity, learns more intricate features, evident in higher precision and mAP50-95, making it more suitable for nuanced and varied datasets like golf scorecards.

#### e. **Recommendations**
   - **YOLOv8l:** Suitable for faster inference with acceptable accuracy. Use for real-time applications or less complex layouts.
   - **YOLOv11l:** Better for high-accuracy requirements, especially when handling multiple templates or challenging layouts, but may require more computational resources and training time.

In conclusion, the evaluation graphs indicate that while YOLOv8l provides faster convergence and decent accuracy, YOLOv11l outperforms it in precision, recall, and generalization, making it the preferred model for complex field detection tasks.

---

## **5. Field Detection Script**
`detect_fields.py`:
```python
from ultralytics import YOLO

def detect_fields(image_path, model_path="models/yolov8l.pt"):
    model = YOLO(model_path)
    results = model(image_path)
    fields = [
        {"label": model.names[int(box[-1])], "bbox": box[:4].tolist()}
        for box in results[0].boxes.data
    ]
    return fields
```

---

## **6. Text Recognition with VietOCR**
1. Use **VietOCR** to recognize Vietnamese text in each detected field.

### Script
`recognize_text.py`:
```python
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def recognize_text(image_path):
    config = Cfg.load_config_from_name("vgg_transformer")
    model = Predictor(config)
    return model.predict(image_path)
```

---

## **7. Handling Constraints
### **Symbols Parsing (Constraint 2)**

Symbols like `△` (triangle) and `〇` (circle) require custom parsing:
1. **△ (Triangle)**: Represents a negative score unless a number is inside.
2. **〇 (Circle)**: Represents a positive score with or without numbers inside.

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

## **8. JSON Output Generation**
Combine detection and recognition results into a structured JSON file:
`generate_json.py`:
```python
import json

def generate_json(fields, texts, output_file="output/result.json"):
    result = {field["label"]: text for field, text in zip(fields, texts)}
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
```

---

## **9. Google Flash API for Annotation**
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

## **10. Flask Deployment**
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
