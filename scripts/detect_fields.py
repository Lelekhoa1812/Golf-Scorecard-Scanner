from ultralytics import YOLO
import cv2

def detect_fields(image_path, model_path="models/yolo_obb.pt"):
    """Detect fields in the scorecard using YOLO OBB."""
    model = YOLO(model_path)
    results = model(image_path)

    fields = []
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.tolist()
        fields.append({
            "label": model.names[int(cls)],
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })
    return fields

if __name__ == "__main__":
    image_path = "data/images/scorecard.jpg"
    detected_fields = detect_fields(image_path)
    print(detected_fields)
