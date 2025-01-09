# MNIST Digits Recognition and Processing Pipeline
This project implements a pipeline for recognizing multi-digit handwritten sequences using the MNIST dataset. It covers synthetic data generation, preprocessing, augmentation, and building a recognition model with CTC loss.

---

## Steps Overview
### 1. **Create Synthetic Data**
This step combines the MNIST dataset with an external digit dataset to create synthetic multi-digit sequences.
- Randomly select digits and transform them.
- Create sequences of digits with labels.
- Save images and labels for further processing.

[Script](https://github.com/Lelekhoa1812/Golf-Scorecard-Scanner/blob/main/mnist/synthetic_digit.py)

---

### 2. **Preprocessing and Augmentation**

Preprocessing ensures that all images conform to the input size of the model, and augmentation adds realistic distortions to simulate real-world conditions.

- Resize and normalize images.
- Add random blobs and noise for augmentation.
- Save processed images and flattened CSV data for training.

[Script](https://github.com/Lelekhoa1812/Golf-Scorecard-Scanner/blob/main/mnist/preprocess_augmentation.py)

---

### 3. **Visualization (Optional)**

Visualize the processed images to ensure proper preprocessing and labeling.

- Reconstruct images from flattened data.
- Display images with corresponding labels using Matplotlib.

Example:    
<img src="../imgsrc/mnist_visualization_post_process.png" alt="Post Process Visualization" style="width: 80%; max-width: 1000px;">   

[Script](https://github.com/Lelekhoa1812/Golf-Scorecard-Scanner/blob/main/mnist/visualize_preprocess.py)

---

### 4. **Load and Split Train/Test Set**

Split the processed dataset into training and validation sets.

- Read processed images and labels.
- Resize images to model-compatible dimensions.
- Normalize image data and split into train/test sets.

Example Output:
```
Number of images found:  11004
Number of labels found:  11004
Number of unique characters:  10
Characters present:  {'1', '9', '3', '0', '6', '2', '5', '7', '4', '8'}

train_size 8803   valid_size 2201
```  
[Script](https://github.com/Lelekhoa1812/Golf-Scorecard-Scanner/blob/main/mnist/load_and_split.py)

---

### 5. **Prepare Labels for CTC Loss**

Convert labels to numeric form and prepare them for training using the CTC loss function.

- Define an alphabet for numeric labels.
- Convert string labels to numeric sequences.
- Configure `train_y`, `train_label_len`, `train_input_len`, and other inputs.

[Script](https://github.com/Lelekhoa1812/Golf-Scorecard-Scanner/blob/main/mnist/prepare_label.py)

---

### 6. **Build and Train MNIST RCNN Model**

Build an RCNN model for sequence recognition using convolutional layers, LSTMs, and CTC loss.

- Define the architecture with CNNs for feature extraction and RNNs for sequential learning.
- Use CTC loss for alignment between predictions and ground truth.
- Train the model with early stopping.

[Script](https://github.com/Lelekhoa1812/Golf-Scorecard-Scanner/blob/main/mnist/build_and_train_model.py)

---

## Directory Structure

```
GolfScoreCardScanner/
├── mnist/                         # Scripts used for digit training with MNIST RCNN
│   ├── synthetic_digit.py         # Create synthetic mnist data
│   ├── preprocess_augmentation.py # preprocess and apply augmentation 
│   ├── visualize_preprocess.py    # Visualization
│   ├── load_and_split.py          # Load data and split to train/test sets
│   ├── prepare_label.py           # Prepare labelling for CTC loss
│   ├── build_and_train_model.py   # Build and config training model for MNIST RCNN
│   ├── validation_test.py         # Some validation of model prediction on test set
├── synthetic_digits/              # Dataset of CSV, image and labels synthetically created
└── README.md                      # Project documentation
```

---

## How to Run the Project

1. Clone this repository.
2. Install required dependencies:  
   ```bash
    pip install -r requirements.txt
   ```
   If you are working in a Colab environment, ensure that the system-level dependencies (like tesseract-ocr) are installed separately:  
    ```bash
    !apt-get update
    !apt-get install -y tesseract-ocr
    ```
3. Run each script in the order outlined above:
   ```bash
   python mnist/synthetic_digit.py
   python mnist/preprocess_augmentation.py
   python mnist/visualize_preprocess.py
   python mnist/load_and_split.py
   python mnist/prepare_label.py
   python mnist/build_and_train_model.py
   python mnist/validation_test.py
   ```

---

## Acknowledgments

- Synthetic data generation was inspired by [Multi-digit MNIST Generator](https://github.com/mrzaizai2k/Multi-digit-images-generator-MNIST-/blob/main/prepare_multi_digit.py).  
- Preprocessing techniques were adapted from [CRNN-CTC PyTorch](https://github.com/dredwardhyde/crnn-ctc-loss-pytorch).  
- MNIST CRNN sample data is from [Kaggle CRNN for Mnist](https://www.kaggle.com/code/duansm/crnn-for-mnist/data)