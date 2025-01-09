# Read CSV and Reconstruct Images
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../processed_digits/aug_digit_data.csv')
labels = data['label'].tolist()
images = data['image'].tolist()

# Reconstruct Images
reconstructed_images = []
for image_str in images:
    image_array = np.array(list(map(int, image_str.split())), dtype=np.uint8)
    image_reshaped = image_array.reshape(32, 128)  # Shape as (H, W)
    reconstructed_images.append(image_reshaped)

# Visualize Images
plt.figure(figsize=(10, 10))
for i in range(6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(reconstructed_images[i], cmap='gray')
    plt.title(labels[i])
    plt.axis('off')
plt.tight_layout()
plt.show()