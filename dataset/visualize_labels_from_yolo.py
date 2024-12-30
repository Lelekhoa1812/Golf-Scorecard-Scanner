import os
import cv2
import math
import numpy as np

# Mapping class IDs to human-readable labels
CLASS_NAMES = {
    0: "PlayerName",
    1: "CourseName",
    2: "Score",
    3: "Total",
    4: "CaddieName",
    5: "HoleNumber",
    6: "HCP",
    7: "NET",
    8: "Card"
}

# Draw the bbox rectangle visualizing each label and their class text
def draw_polygon(image, bbox, class_id, color=(0, 255, 0), thickness=2):
    """
    Draw a polygon on the image based on the bounding box.
    :param image: The image on which to draw.
    :param bbox: Bounding box in YOLO OBB format (center_x, center_y, width, height, theta).
    :param color: Color of the polygon (green).
    :param thickness: Thickness of the polygon lines.
    """
    center_x, center_y, width, height, theta = bbox
    h, w = image.shape[:2]

    # Convert normalized values back to absolute values
    center_x *= w
    center_y *= h
    width *= w
    height *= h

    # Calculate the four corners of the bounding box
    angle = -theta  # YOLO OBB format uses clockwise rotation, invert for OpenCV
    c, s = math.cos(angle), math.sin(angle)

    dx = width / 2
    dy = height / 2

    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy]
    ])

    rotation_matrix = np.array([[c, -s], [s, c]])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    translated_corners = rotated_corners + [center_x, center_y]

    points = translated_corners.astype(np.int32)
    points = points.reshape((-1, 1, 2))

# Draw the polygon on the image
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

    # Add the class label text
    label = CLASS_NAMES.get(int(class_id), "Unknown")
    text_position = (int(center_x - width / 2), int(center_y - height / 2) - 10)  # Position near the top-left corner of the bbox
    cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
def process_labels_and_images(label_dir, image_dir, output_dir):
    """
    Process all label files and corresponding images to overlay bounding boxes.
    :param label_dir: Directory containing YOLO OBB label files.
    :param image_dir: Directory containing the corresponding images.
    :param output_dir: Directory to save the output images with overlays.
    """
    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            # Get corresponding image file
            base_name = os.path.splitext(label_file)[0]
            image_file = os.path.join(image_dir, f"{base_name}.jpg")

            if not os.path.exists(image_file):
                print(f"Image file not found for label: {label_file}")
                continue

            # Read image
            image = cv2.imread(image_file)
            if image is None:
                print(f"Failed to load image: {image_file}")
                continue

            # Read label file
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                # Ensure label format of class_id, center_x, center_y, width, height, theta 
                if len(parts) != 7:
                    print(f"Invalid label format in file: {label_file}")
                    continue

                class_id, center_x, center_y, width, height, theta = map(float, parts)

                # Draw polygon for the bounding box
                draw_polygon(image, (center_x, center_y, width, height, theta), class_id)

            # Save the output image
            output_path = os.path.join(output_dir, f"{base_name}_debug.jpg")
            cv2.imwrite(output_path, image)
            print(f"Saved debug image: {output_path}")

if __name__ == "__main__":
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up paths relative to the script's location
    label_dir = os.path.join(script_dir, "../dataset/yolo_labels")
    image_dir = os.path.join(script_dir, "../dataset/images/train")
    output_dir = os.path.join(script_dir, "../dataset/images/debug_label")

    process_labels_and_images(label_dir, image_dir, output_dir)