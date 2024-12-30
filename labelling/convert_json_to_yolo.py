import json
import math
import os
from PIL import Image  # For getting image dimensions

# Define class mappings with more robust identification
CLASS_LABELS = {
    "PlayerName": 0,
    "CourseName": 1,
    "Score": 2,
    "Total": 3,
    "CaddieName": 4,
    "HoleNumber": 5,
    "HCP": 6,
    "Card": 7
}


def preprocess_description(description):
    """
    Preprocess and infer the class label from description text.
    """
    description = description.strip().lower()
    try: # Implement try and catch excepting the value error 
        if description in ["course", "san"]:
            return "CourseName"
        if description.startswith("a ") or description.startswith("a."): #or description.isalpha():
            return "PlayerName"
        if description.isdigit() and len(description) <= 2:  # Ensure score is two digits or fewer
            return "Score"
        if description in ["total", "net"]:
            return "Total"
        if description in ["hcp", "hdcp", "handicap", "hcap"]:
            return "HCP"
        if description.isnumeric(): #and int(description) in range(1, 10): # Hole number can only be from 1 to 9
            return "HoleNumber"
    except ValueError as e:
        print(f"Value error for {description}, will be skipped")
        return None
    return None


def calculate_area(bbox):
    """
    Calculate the area of a bounding box given its four corners.
    This is mainly used to state the biggest label from JSON to be the scorecard itself
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x3, y3 = bbox[2]
    x4, y4 = bbox[3]

    # Calculate width and height using the first two points and the last two points
    width = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    height = math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)
    return width * height


def adjust_course_name_bbox(bbox, image_width, image_height):
    """
    Adjust the CourseName bounding box based on its orientation.
    If horizontal, expand width; if vertical, expand height (by 300px).
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x3, y3 = bbox[2]
    x4, y4 = bbox[3]

    # Calculate width and height
    width = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    height = math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)

    # Determine the center
    x_center = (x1 + x2 + x3 + x4) / 4
    y_center = (y1 + y2 + y3 + y4) / 4

    # Adjust the bounding box
    if width > height:  # Horizontal orientation
        width += 300  # Expand width
    elif width > height:  # Vertical orientation
        height += 300  # Expand height
    else:
        print("This tag has equal size of width and height, cannot determine orientation.")


def convert_bounding_box(bbox, image_width, image_height):
    """
    Convert a bounding box to YOLO OBB format.
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x3, y3 = bbox[2]
    x4, y4 = bbox[3]

    x_center = (x1 + x2 + x3 + x4) / 4
    y_center = (y1 + y2 + y3 + y4) / 4

    width = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    height = math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)

    theta = math.atan2(y2 - y1, x2 - x1)

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height, theta


def json_to_yolo_obb(json_file, image_file, output_dir):
    """
    Convert Flash Vision API JSON labels to YOLO OBB format.
    """
    # Get image dimensions
    with Image.open(image_file) as img:
        image_width, image_height = img.size

    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(json_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    yolo_obb_annotations = []
    largest_bbox = None
    largest_area = 0

    for annotation in data:
        description = annotation.get("description", "").strip()
        bbox = annotation.get("bounding_box", [])

        # Validate the bbox that cannot be null or having less than 4 components, else skip
        if not bbox or len(bbox) != 4 or not description:
            continue

        area = calculate_area(bbox)
        if area > largest_area:
            largest_area = area
            largest_bbox = annotation

        # Label the largest bbox is the score card itself
        class_label = "Card" if annotation == largest_bbox else preprocess_description(description)
        class_id = CLASS_LABELS.get(class_label, -1) # Get class label by id and null to be -1

        # Class id of null is skipped
        if class_id == -1:
            continue

# Adjust CourseName bbox if applicable
        if class_label == "CourseName":
            bbox = adjust_course_name_bbox(bbox, image_width, image_height)

        # Obtaining the extraction of data from JSON bbox and convert to YOLO x,y,w,h,θ format for usage
        try: # Implement a try-catch to avoid bbox mismatching
            x_center, y_center, width, height, theta = convert_bounding_box(
                bbox, image_width, image_height
            )
        except Exception as e:
            print(f"Error converting {base_name} with {description} bounding box: {bbox}, error: {e}")
            continue

        # Append the formatted YOLO label file
        yolo_obb_annotations.append(
            f"{class_id} {x_center} {y_center} {width} {height} {theta}"
        )

    with open(output_file, 'w') as f:
        f.write("\n".join(yolo_obb_annotations))

    print(f"YOLO OBB annotations saved to {output_file}")


def process_all_json_files(input_dir, image_dir, output_dir):
    for json_file in os.listdir(input_dir):
        if json_file.endswith(".json"):
            base_name = os.path.splitext(json_file)[0]
            image_file = os.path.join(image_dir, f"{base_name}.jpg")

            if not os.path.exists(image_file):
                print(f"Image file not found for {json_file}, skipping.")
                continue

            json_path = os.path.join(input_dir, json_file)
            json_to_yolo_obb(json_path, image_file, output_dir)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "../data/train/labels")
    image_dir = os.path.join(script_dir, "../data/train/processed_images")
    output_dir = os.path.join(script_dir, "../data/train/yolo_labels")

    process_all_json_files(input_dir, image_dir, output_dir)
 