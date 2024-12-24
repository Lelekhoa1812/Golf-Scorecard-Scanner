from google.cloud import vision
import os
import json

# Set the Google application credentials environment variable
def label_scorecard(image_path):
    """
    Use Google Vision API to detect text and generate field labels.
    """
    # Set the Google application credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key/your_google_credentials.json"
    
    # Initialize the Vision API client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Detect text in the image
    response = client.text_detection(image=image)
    annotations = response.text_annotations

    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")

    # Process detected text
    labels = []
    for annotation in annotations:
        bounding_poly = annotation.bounding_poly
        vertices = [(vertex.x, vertex.y) for vertex in bounding_poly.vertices]
        labels.append({
            "description": annotation.description,
            "bounding_box": vertices
        })

    return labels


def save_labels_to_json(labels, output_path):
    """
    Save the labeled data to a JSON file.
    """
    with open(output_path, "w") as json_file:
        json.dump(labels, json_file, indent=4)

def process_images(input_dir, output_dir):
    """
    Process all images in the input directory and save labeled data.
    """
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.json")

        # Label the scorecard and save the output
        print(f"Processing {image_name}...")
        labels = label_scorecard(image_path)
        save_labels_to_json(labels, output_path)
        print(f"Labels saved to {output_path}")

if __name__ == "__main__":
    # Input and output directories
    input_dir = "data/images"
    output_dir = "data/labels" # Automatically labelled

    os.makedirs(output_dir, exist_ok=True)
    process_images(input_dir, output_dir)
