import json

def convert_to_yolo_format(labels, image_width, image_height):
    """
    Convert labeled bounding boxes to YOLO format.
    """
    yolo_labels = []
    for label in labels:
        description = label["description"]
        bbox = label["bounding_box"]

        # Calculate bounding box center, width, and height
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[2]
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Add YOLO format label
        yolo_labels.append(f"{description} {x_center} {y_center} {width} {height}")
    
    return yolo_labels
