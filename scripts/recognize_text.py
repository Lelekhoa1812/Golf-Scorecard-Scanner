from ultralytics import YOLO
import cv2
import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import pytesseract
import numpy as np

'''
This script is inherited from:
1. scripts/detect_fields.py 
2. labelling/thicken_grid.py

This script integrates Tesseract OCR in numeric only mode for Score, Net and Total fields.
'''

# Load VietOCR model
def load_vietocr_model(vietocr_model_path):
    """Load VietOCR model for text recognition."""
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = vietocr_model_path
    config['device'] = 'cpu'  # Change to 'cuda' if GPU is available
    config['predictor']['beamsearch'] = False
    return Predictor(config)

# Recognize text with VietOCR
def recognize_text(image, model):
    """Recognize text from an image."""
    pil_image = Image.fromarray(image)
    return model.predict(pil_image)

# Recognize numeric text using Tesseract (deploy on Google Collab if cannot download pytesseract)
def recognize_numeric_text(image):
    """Recognize numeric text using Tesseract OCR in numeric-only mode."""
    config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

# Process image for better grid detection
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # Increase contrast
    contrasted = cv2.convertScaleAbs(image, alpha=0.8, beta=0)
    gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    horizontal_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.bitwise_or(horizontal_lines, vertical_lines)
    grid = cv2.dilate(grid, kernel, iterations=2)
    grid_inverted = cv2.bitwise_not(grid)
    grid_bgr = cv2.cvtColor(grid_inverted, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.8, grid_bgr, 0.3, 0)

# Detect fields and recognize text
def detect_and_recognize_fields(processed_image, image_path, yolo_model_path, vietocr_model):
    yolo_model = YOLO(yolo_model_path)
    results = yolo_model(processed_image)
    image = processed_image  # Visualize processed image

    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.tolist()
        label = yolo_model.names[int(cls)]
        field_image = image[int(y1):int(y2), int(x1):int(x2)]

        if field_image.size == 0:
            text = "[Empty Field]"
        else:
            # Use Tesseract for numeric-only fields
            if label in ["Score", "Net", "Total"]:
                text = recognize_text(field_image, vietocr_model)
                retry_count = 0
                while not text.isdigit() and retry_count < 5:  # Retry if not numeric
                    text = recognize_numeric_text(field_image)
                    retry_count += 1
                print(f"Retrying numeric OCR for {label}: Attempt {retry_count}, Text: {text}")
            else:
                # Use VietOCR for other fields
                text = recognize_text(field_image, vietocr_model)
                print(f"Detected field: {label}, Confidence: {conf:.2f}, Text: {text}")

        # Draw a rectangle and annotate the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({conf:.2f})", (int(x1), int(y1) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"{text}", (int(x1), int(y2) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Detected Fields with Text", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    base_dir = "/Users/khoale/Downloads/GolfScoreCardScanner"
    image_path = os.path.join(base_dir, "dataset/images/train/IMG_9475.JPG")
    yolo_model_path = os.path.join(base_dir, "models/yolov11l.pt")
    vietocr_model_path = os.path.join(base_dir, "models/vgg_transformer.pth")

    # Paths
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at {image_path}")
    elif not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model file does not exist at {yolo_model_path}")
    elif not os.path.exists(vietocr_model_path):
        print(f"Error: VietOCR weights file does not exist at {vietocr_model_path}")
    # Processor
    else:
        vietocr_model = load_vietocr_model(vietocr_model_path)
        processed_image = process_image(image_path)
        detect_and_recognize_fields(processed_image, image_path, yolo_model_path, vietocr_model)
