# Acknowledgement from: https://github.com/dredwardhyde/crnn-ctc-loss-pytorch/blob/main/mnist_sequence_recognition.py
import cv2
import numpy as np
import random
import csv
import os

# Preprocessing Function
def preprocess(image, imgSize):
    """
    Resize, transpose, and normalize the image to match MNIST-like expectations.
    """
    widthTarget, heightTarget = imgSize
    height, width = image.shape
    # Scaling factor to maintain aspect ratio
    factor = max(width / widthTarget, height / heightTarget)
    newSize = (int(width / factor), int(height / factor))
    # Resize the image while maintaining aspect ratio
    image = cv2.resize(image, newSize, interpolation=cv2.INTER_AREA)
    # Center the resized image in a white canvas
    target = np.ones((heightTarget, widthTarget), dtype='uint8') * 255
    start_x = (widthTarget - newSize[0]) // 2
    start_y = (heightTarget - newSize[1]) // 2
    target[start_y:start_y + newSize[1], start_x:start_x + newSize[0]] = image
    # Normalize the image to MNIST-like format
    target = (255 - target)  # Invert colors (MNIST is white on black)
    return target

# Data Augmentation
def augmentation(image):
    """
    Add realistic noise, blobs, and distortions.
    """
    height, width = image.shape
    num_blob = max(1, int(random.gauss(5, 2)))  # Fewer blobs for realism
    for _ in range(num_blob):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(1, 2)  # Smaller blobs
        cv2.circle(image, (x, y), radius=radius, color=(0), thickness=-1)
    # Random erosion or dilation
    kernel = np.ones((2, 2), np.uint8)
    if random.random() > 0.5:
        image = cv2.erode(image, kernel, iterations=1)
    else:
        image = cv2.dilate(image, kernel, iterations=1)
    return image

# Directory to save synthetic digit images
input_dir = '../synthetic_digits/'
output_dir = '../processed_digits/'
os.makedirs(output_dir, exist_ok=True)

# Initialize CSV file
csv_file_path = os.path.join(output_dir, 'aug_digit_data.csv')
header = ['label', 'image']
with open(csv_file_path, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# Load images and labels
images = []
labels = []
for file_name in os.listdir(input_dir):
    if file_name.endswith('.png'):
        # Read image
        image_path = os.path.join(input_dir, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        # Extract label from file name (e.g., "1234567.png" -> "1234567")
        label = os.path.splitext(file_name)[0]
        images.append(image)
        labels.append(label)
# Image size for preprocessing
imgSize = (128, 32)

# Save Flattened Data and Labels
for i, image in enumerate(images):
    label = labels[i]  # Label as string
    image = 255 - image  # Invert the image
    image = augmentation(image)
    processed_image = preprocess(image, imgSize)
    # Save processed image
    file_name = f"{label}.png"
    file_path = os.path.join(output_dir, file_name)
    cv2.imwrite(file_path, processed_image)
    # Flatten and save in CSV
    flattened_image = processed_image.flatten()
    flattened_image_str = " ".join(map(str, flattened_image))
    with open(csv_file_path, 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([label, flattened_image_str])
print(f"Data preparation completed. Processed images saved to {output_dir} and CSV file at {csv_file_path}.")