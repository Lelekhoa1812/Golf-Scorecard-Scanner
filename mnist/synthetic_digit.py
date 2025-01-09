# Create synthetic data
# Acknowledgement of resources from: https://github.com/mrzaizai2k/Multi-digit-images-generator-MNIST-/blob/main/prepare_multi_digit.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import os
import subprocess
import matplotlib.pyplot as plt
import random
import numpy as np 
import pandas as pd 
import csv

# Originally there are 2 dataset
print(os.listdir("../digit-recognizer"))
train_data = pd.read_csv('/content/drive/My Drive/GolfScorecard/digit-recognizer/train.csv')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# We can combine them to enlarge the data dataset
train_data1 = np.concatenate([x_train, x_test], axis=0)
labels1 = np.concatenate([y_train, y_test], axis=0)
labels2 = train_data.label
labels = np.concatenate([labels1, labels2], axis=0)
train_data2 = train_data.drop(columns='label')
images = np.concatenate([train_data1.reshape([-1,28*28]), train_data2.values], axis=0)
print(images.shape)
print(labels.shape)

# Processing steps, you can change the sequence as follow
digits_per_sequence = 7
number_of_sequences = 100
dataset_sequences = []
dataset_labels = []

# Loop
for i in range(number_of_sequences):
    random_indices = np.random.randint(len(images), size=(digits_per_sequence,)) # Take 7 random indices
    random_digits_images = images[random_indices] # Take 7 images from 7 random indices
    transformed_random_digits_images = []
    # Rotate image for visibility
    for img in random_digits_images:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)
        transformed_random_digits_images.append(img)

    random_digits_images = np.array(transformed_random_digits_images) # code of transformed images: ahdjagj
    random_digits_labels = labels[random_indices] # e.g., [9 9 9 2 9 8 6]

    random_sequence = np.hstack(random_digits_images.reshape((digits_per_sequence, 28, 28))) # set of image
    random_labels = np.hstack(random_digits_labels.reshape(digits_per_sequence, 1)) # label for set of image
    
    dataset_sequences.append(random_sequence) # set of transformed images ahdjagj
    dataset_labels.append(random_labels) # their labels 9,1,2,7,4,6,7


labels = np.array(dataset_labels)
images = np.array(dataset_sequences).reshape([-1, 28,28*digits_per_sequence,1])

#plt.figure(num='multi digit',figsize=(9,9))
#for i in range(9):
#    plt.subplot(3,3,i+1) 
#    plt.title(np.array(dataset_labels)[i])
#    plt.imshow(np.squeeze(images[i,:,:,]))
#plt.show()

for i in range (len (images)):
    label = ( "".join( str(e) for e in labels[i] ) ) # can modify with parentheness
    images[i] = 255 - images[i]
    cv2.imwrite('/content/drive/My Drive/GolfScorecard/synthetic_digits/'+str(label)+'.png',images[i])



cv2.waitKey(0)
