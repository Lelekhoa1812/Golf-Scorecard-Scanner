from ultralytics import YOLO
import cv2
import os

def detect_fields(image_path, model_path):
    """Detect fields in the scorecard using YOLO."""
    # Load the YOLO model
    model = YOLO(model_path)

    # Run the model on the input image
    results = model(image_path)

    # Load the image for visualization
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Iterate through the detected fields
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.tolist()
        label = model.names[int(cls)]
        
        # Draw a rectangle around the detected field
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Put the label and confidence above the rectangle
        cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with the detections
    cv2.imshow("Detected Fields", image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define absolute paths (modify with your own)
    base_dir = "/Users/khoale/Downloads/GolfScoreCardScanner"
    image_path = os.path.join(base_dir, "dataset/images/train/IMG_9475.JPG")  
    model_path = os.path.join(base_dir, "models/yolov11l.pt") 

    # Check if paths are valid
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at {image_path}")
    elif not os.path.exists(model_path):
        print(f"Error: Model file does not exist at {model_path}")
    else:
        # Call the function to detect and visualize fields
        detect_fields(image_path, model_path)
