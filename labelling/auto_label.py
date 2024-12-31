from google.cloud import vision
import os
import json

# Set the Google application credentials environment variable
def label_scorecard(image_path):
    """
    Use Google Vision API to detect text and generate field labels.
    """
    # Set the Google application credentials
    script_dir = os.path.dirname(os.path.abspath(__file__))
    API_KEY = os.path.join(script_dir, "../key/yourapikey.json")  # Make sure to remove this when commit to GitHub
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY
    
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
    Only supports .jpg, .jpeg, .png file formats.
    """
    # Supported file extensions
    supported_extensions = (".jpg", ".jpeg", ".png", ".PNG", ".JPG")

    for image_name in os.listdir(input_dir):
        # Check file extension
        if image_name.endswith(supported_extensions):
            image_path = os.path.join(input_dir, image_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.json")

            # Label the scorecard and save the output
            print(f"Processing {image_name}...")
            labels = label_scorecard(image_path)
            save_labels_to_json(labels, output_path)
            print(f"Labels saved to {output_path}")
        else:
            print(f"File Type Not Supported: {image_name}")


if __name__ == "__main__":


    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up paths relative to the script's location
    input_dir_train = os.path.join(script_dir, "../data/train/processed_images")  # Directory containing input processed images of train set
    output_dir_train = os.path.join(script_dir, "../data/train/labels")           # Directory to save labeled JSON files
    # input_dir_test = os.path.join(script_dir, "../data/test/processed_images")  # Directory containing input processed images of test set
    # output_dir_test = os.path.join(script_dir, "../data/test/labels")           # Directory to save labeled JSON files

    # Create output directory if it doesn't exist
    os.makedirs(output_dir_train, exist_ok=True)
    # os.makedirs(output_dir_test, exist_ok=True)

    # Process images in the input directory
    process_images(input_dir_train, output_dir_train)
    # process_images(input_dir_test, output_dir_test)
