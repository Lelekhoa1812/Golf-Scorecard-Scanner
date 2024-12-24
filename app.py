from google.cloud import vision
import os

def label_image(image_path):
    """Use Google Vision API to label an image."""
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as img_file:
        content = img_file.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")
    return response.text_annotations

if __name__ == "__main__":
    image_path = "data/images/scorecard.jpg"
    labels = label_image(image_path)
    for label in labels:
        print(label.description)
