import os
import json
import cv2
import numpy as np

# Directory setup
root_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.join(root_dir, "train")
labels_dir = os.path.join(root_dir, "labels")
src_dir = os.path.join(root_dir, "src")

# Ensure label subdirectories exist
os.makedirs(train_dir, exist_ok=True)

for label_name in ["PlayerName", "CourseName", "Score", "Total", "HoleNumber"]:
    label_dir = os.path.join(train_dir, label_name)
    os.makedirs(label_dir, exist_ok=True)

def crop_and_save(image, points, label, image_name, iteration):
    # Convert points to integer tuples
    points = [(int(point[0]), int(point[1])) for point in points]

    # Get bounding rectangle from polygon
    x, y, w, h = cv2.boundingRect(np.array(points))

    # Crop the region of interest (ROI)
    cropped = image[y:y+h, x:x+w]

    # Save the cropped image
    output_dir = os.path.join(train_dir, label)
    output_path = os.path.join(output_dir, f"{image_name}_{label}_{iteration}.jpg")
    cv2.imwrite(output_path, cropped)
    print(f"Saved: {output_path}")

def process_labels(json_path, image_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Process each shape in the JSON file
    label_counts = {}
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points")

        if not label or not points:
            continue

        # Increment label count for naming
        label_counts[label] = label_counts.get(label, 0) + 1
        crop_and_save(image, points, label, image_name, label_counts[label])

if __name__ == "__main__":
    for json_file in os.listdir(labels_dir):
        if json_file.endswith(".json"):
            # Get corresponding image file
            base_name = os.path.splitext(json_file)[0]
            json_path = os.path.join(labels_dir, json_file)
            image_path = os.path.join(src_dir, f"{base_name}.jpg")

            if not os.path.exists(image_path):
                print(f"Image file not found for {json_file}, skipping.")
                continue

            process_labels(json_path, image_path)
