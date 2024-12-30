import os
import cv2
import json
import numpy as np

def draw_polygon(image, bbox, label, color=(0, 255, 0), thickness=2):
    """
    Draw a polygon on the image based on the bounding box and label it.
    :param image: The image on which to draw.
    :param bbox: Bounding box vertices in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    :param label: The text label to display.
    :param color: Color of the polygon and text.
    :param thickness: Thickness of the polygon lines.
    """
    # Convert the bounding box into a numpy array
    points = np.array(bbox, dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    # Draw the polygon on the image
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

    # Calculate the position for the label
    x, y = bbox[0]
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def visualize_json_labels(json_file, image_file, output_dir):
    """
    Visualize bounding boxes from a JSON file on the corresponding image.
    :param json_file: Path to the JSON file containing the annotations.
    :param image_file: Path to the image file.
    :param output_dir: Directory to save the output image.
    """
    # Load the JSON data
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    # Load the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error loading image: {image_file}")
        return

    # Draw each bounding box
    for annotation in annotations:
        description = annotation.get("description", "")
        bbox = annotation.get("bounding_box", [])

        if len(bbox) == 4:
            draw_polygon(image, bbox, description)

    # Prepare the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the output image
    base_name = os.path.basename(image_file)
    output_path = os.path.join(output_dir, f"debug_json_{base_name}")
    cv2.imwrite(output_path, image)
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    # Paths
        # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up paths relative to the script's location
    json_dir = os.path.join(script_dir, "../data/train/labels")
    image_dir = os.path.join(script_dir, "../data/train/processed_images")
    output_dir = os.path.join(script_dir, "../data/train/debug_label")


    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSON file
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            base_name = os.path.splitext(json_file)[0]
            image_file = os.path.join(image_dir, f"{base_name}.jpg")

            if os.path.exists(image_file):
                visualize_json_labels(
                    json_file=os.path.join(json_dir, json_file),
                    image_file=image_file,
                    output_dir=output_dir
                )
            else:
                print(f"Image file not found for JSON: {json_file}")
