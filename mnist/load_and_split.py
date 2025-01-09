# Imports
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import subprocess
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from collections import Counter

# Loader
data_dir = Path("../processed_digits")
# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
# path to image 
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)
print("Number of images found: ", len(images)) # Should be around 10k+ number of image
print("Number of labels found: ", len(labels)) # Must match the length
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)
# Batch size for training and validation
batch_size = 16
# Desired image dimensions
img_width = 128
img_height = 32
# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4
# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

# Split train test
train_size = int(0.8 * len(labels))
valid_size= int(len(labels) - train_size)
# 80% data is for training. The rest of them is for validation
print ('\ntrain_size',train_size,'  valid_size',valid_size)
# Init 
train_x = []
valid_x = []
i=0
for image in images:
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image, (128,32)) 
    image = image/255.
    if i < train_size:
        train_x.append(image)
    else:
        valid_x.append(image)
    i = i+1
# Shaping
train_x = np.array(train_x).reshape(-1, 128, 32, 1)
valid_x = np.array(valid_x).reshape(-1, 128, 32, 1)
print ('\n train_x.shape',train_x.shape)
print ('\n valid_x.shape',valid_x.shape)
# Labelling
label_train = labels[0:train_size]
label_valid = labels[train_size:len(labels)]
# They usually slit images into 2 folders, but I just have 1 so I have to label it again.
#print ('\n label_train',label_train)
print('\n Example of label_valid',label_valid[3])
# Simple plotting for verificaiton
plt.figure(num='multi digit',figsize=(9,18))
for i in range(3):
    plt.subplot(3,3,i+1) 
    plt.title(label_valid[i])
    plt.imshow(np.squeeze(valid_x[i,:,:,]))
plt.show()