import cv2
from detect_fields import detect_fields
from recognize_text import load_vietocr_model, recognize_text
from generate_json import write_json
from utils import crop_field, map_fields

def main(image_path, template="default"):
    # Load models
    vietocr_model = load_vietocr_model()
    detected_fields = detect_fields(image_path)

    # Map fields to standard labels based on template
    mapped_fields = map_fields(detected_fields, template)

    # Read the image
    image = cv2.imread(image_path)

    # Recognize text for each field
    recognized_texts = []
    for field in mapped_fields.values():
        cropped = crop_field(image, field["bbox"])
        text = recognize_text(cropped, vietocr_model)
        recognized_texts.append(text)

    # Generate JSON
    write_json(mapped_fields.values(), recognized_texts)

if __name__ == "__main__":
    # Specify the template dynamically, e.g., based on the course
    main("data/images/scorecard.jpg", template="course_a")
