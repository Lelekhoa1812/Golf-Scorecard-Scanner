import os
import json

# Define class names and their corresponding class IDs
CLASS_NAMES = {
    "fixed": 0,
    "positive": 1,
    "negative": 2,
    "boggey": 3
}

def convert_labelme_to_yolo(json_file, image_width, image_height, output_txt_file):
    """
    Convert a Labelme JSON file into YOLO format.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    yolo_annotations = []

    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]

        # Check if the label is in the class names
        if label not in CLASS_NAMES:
            print(f"Skipping unknown label: {label}")
            continue

        class_id = CLASS_NAMES[label]

        # Find the bounding box from the points
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Convert to YOLO format (x_center, y_center, width, height)
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write the YOLO annotations to the output file
    with open(output_txt_file, 'w') as f:
        f.write("\n".join(yolo_annotations))
    print(f"Converted {json_file} to {output_txt_file}")


def process_all_json_files(label_dir, src_dir, output_dir):
    """
    Process all Labelme JSON files in the label directory and generate YOLO labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(label_dir):
        if json_file.endswith(".json"):
            base_name = os.path.splitext(json_file)[0]
            json_path = os.path.join(label_dir, json_file)
            image_path = os.path.join(src_dir, f"{base_name}.jpg")

            # Validate image existence
            if not os.path.exists(image_path):
                print(f"Image file not found for {json_file}, skipping.")
                continue

            # Get image dimensions
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}, skipping.")
                continue

            image_height, image_width, _ = image.shape

            # Define the output YOLO label file
            output_txt_file = os.path.join(output_dir, f"{base_name}.txt")

            # Convert JSON to YOLO format
            convert_labelme_to_yolo(json_path, image_width, image_height, output_txt_file)


if __name__ == "__main__":
    # Directory setup
    script_dir = "/Users/khoale/Downloads/GolfScoreCardScanner/symbol/"
    label_dir = os.path.join(script_dir, "labelme/") # Path to JSON labels
    src_dir = os.path.join(script_dir, "images/train/")    # Path to images
    output_dir = os.path.join(script_dir, "labels/train/")  # Output YOLO label files

    # Ensure the output directory is writable and exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    process_all_json_files(label_dir, src_dir, output_dir)
