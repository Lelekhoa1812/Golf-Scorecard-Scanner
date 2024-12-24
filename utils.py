import cv2
import numpy as np

# Template Mapping for Different Courses
TEMPLATE_MAP = {
    "course_a": {"PlayerName": "User", "Total": "Net"},
    "course_b": {"HDC": "Handicap", "Score": "Points"},
    "default": {"PlayerName": "Player", "Total": "Total", "Score": "Score"}
}

def map_fields(fields, template="default"):
    """Map field labels based on the template."""
    mapped_fields = {}
    for field in fields:
        mapped_label = TEMPLATE_MAP.get(template, {}).get(field["label"], field["label"])
        mapped_fields[mapped_label] = field
    return mapped_fields

def preprocess_image(image_path):
    """Convert image to grayscale and apply thresholding."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary

def crop_field(image, bbox):
    """Crop the field from the image using bounding box."""
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    return image[y1:y2, x1:x2]
