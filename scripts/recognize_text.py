from ultralytics import YOLO
import cv2
import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import numpy as np

'''
This script is inherited from:
1. scripts/detect_fields.py 
2. labelling/thicken_grid.py
'''

def load_vietocr_model(vietocr_model_path):
    """Load VietOCR model for text recognition."""
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = vietocr_model_path
    config['device'] = 'cpu'  # Change to 'cuda' if GPU is available
    config['predictor']['beamsearch'] = False
    return Predictor(config)


def recognize_text(image, model):
    """Recognize text from an image."""
    pil_image = Image.fromarray(image)
    return model.predict(pil_image)

# From detect_fields.py 
def detect_and_recognize_fields(process_image, image_path, yolo_model_path, vietocr_model):
    """Detect fields in the scorecard and recognize text using VietOCR."""
    # Load the YOLO model
    yolo_model = YOLO(yolo_model_path)
    # Run YOLO model on the input processed image
    results = yolo_model(process_image)
    # Load the original image for visualization
    # image = cv2.imread(image_path) # In case visualizing non-processed/original image
    image = process_image            # In case visualizing processed image

    # Iterate through the detected fields
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.tolist()
        label = yolo_model.names[int(cls)]
        # Crop the detected field from the image
        field_image = image[int(y1):int(y2), int(x1):int(x2)]
        # Recognize text in the cropped field
        if field_image.size != 0:  # Ensure the cropped image is not empty
            text = recognize_text(field_image, vietocr_model)
        else:
            text = "[Empty Field]"

        # Draw a rectangle around the detected field (can be commented to reduce runtime)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put the label, confidence, and recognized text above the rectangle
        cv2.putText(image, f"{label} ({conf:.2f})", (int(x1), int(y1) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"{text}", (int(x1), int(y2) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"Detected field: {label}, Confidence: {conf:.2f}, Text: {text}")

    # Display the image with the detections and recognized text
    cv2.imshow("Detected Fields with Text", image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


# From thicken_grid.py
def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Can't find img
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Increase contrast for better grid detection
    contrasted = cv2.convertScaleAbs(image, alpha=0.8, beta=0)  # Increase contrast (dark to darker and bright to brighter) by alpha factor, by opposite, beta decrease it

    # Convert to grayscale
    gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold to emphasize grid lines
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    # Use morphological operations to isolate grid lines
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3)  # Increased iterations for thicker lines

    # Find horizontal and vertical lines separately
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1)) # Larger kernel
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))   # Larger kernel

    horizontal_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)

    # Combine the lines
    grid = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Dilate the combined grid further for intensity
    grid = cv2.dilate(grid, kernel, iterations=2)

    # Invert the grid to make lines black on a white background
    grid_inverted = cv2.bitwise_not(grid)

    # Overlay the black grid on the original image
    grid_bgr = cv2.cvtColor(grid_inverted, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(image, 0.8, grid_bgr, 0.3, 0)  # Increased blending for visibility
    return result


if __name__ == "__main__":
    # Define paths
    base_dir = "/Users/khoale/Downloads/GolfScoreCardScanner"
    image_path = os.path.join(base_dir, "dataset/images/train/IMG_9475.JPG")  # Input image
    yolo_model_path = os.path.join(base_dir, "models/yolov11l.pt")            # YOLO model v8l or v11l
    vietocr_model_path = os.path.join(base_dir, "models/vgg_transformer.pth") # VietOCR transformer weights

    # Check if paths are valid
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at {image_path}")
    elif not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model file does not exist at {yolo_model_path}")
    elif not os.path.exists(vietocr_model_path):
        print(f"Error: VietOCR weights file does not exist at {vietocr_model_path}")
    else:
        # Load VietOCR model
        vietocr_model = load_vietocr_model(vietocr_model_path)

        # Detect fields and recognize text
        detect_and_recognize_fields(process_image(image_path), image_path, yolo_model_path, vietocr_model)
