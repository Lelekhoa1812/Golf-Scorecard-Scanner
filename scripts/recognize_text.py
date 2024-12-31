from ultralytics import YOLO
import cv2
import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image


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


def detect_and_recognize_fields(image_path, yolo_model_path, vietocr_model):
    """Detect fields in the scorecard and recognize text using VietOCR."""
    # Load the YOLO model
    yolo_model = YOLO(yolo_model_path)

    # Run YOLO model on the input image
    results = yolo_model(image_path)

    # Load the image for visualization
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

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


if __name__ == "__main__":
    # Define paths
    base_dir = "/Users/khoale/Downloads/GolfScoreCardScanner"
    image_path = os.path.join(base_dir, "dataset/images/train/IMG_9475.JPG")         # Input image
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
        detect_and_recognize_fields(image_path, yolo_model_path, vietocr_model)
