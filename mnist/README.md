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

#### Statistics on setup of the MNIST model:  
<img src="../imgsrc/mnist_ocr_model_stats.png" alt="MNIST model setup" style="width: 80%; max-width: 1000px;">   

#### Statistics on Training:  
```
Epoch 1/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 10s 78ms/step - loss: 54.2124 - val_loss: 52.9344 - learning_rate: 5.0000e-06
Epoch 2/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 66ms/step - loss: 50.3430 - val_loss: 49.8045 - learning_rate: 5.0000e-06
Epoch 3/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 65ms/step - loss: 43.8993 - val_loss: 43.7889 - learning_rate: 5.0000e-06
Epoch 4/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 67ms/step - loss: 36.3857 - val_loss: 35.3692 - learning_rate: 5.0000e-06
Epoch 5/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 67ms/step - loss: 29.8556 - val_loss: 27.8266 - learning_rate: 5.0000e-06
Epoch 6/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 64ms/step - loss: 25.2242 - val_loss: 23.0234 - learning_rate: 5.0000e-06
Epoch 7/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 65ms/step - loss: 22.3933 - val_loss: 20.6264 - learning_rate: 5.0000e-06
Epoch 8/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 63ms/step - loss: 20.8601 - val_loss: 19.6484 - learning_rate: 5.0000e-06
Epoch 9/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 65ms/step - loss: 20.1012 - val_loss: 19.2620 - learning_rate: 5.0000e-06
Epoch 10/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 65ms/step - loss: 19.6729 - val_loss: 19.1036 - learning_rate: 5.0000e-06
Epoch 11/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 66ms/step - loss: 19.4437 - val_loss: 19.0325 - learning_rate: 5.0000e-06
Epoch 12/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 65ms/step - loss: 19.2698 - val_loss: 18.9529 - learning_rate: 5.0000e-06
Epoch 13/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 69ms/step - loss: 19.1440 - val_loss: 18.9133 - learning_rate: 5.0000e-06
Epoch 14/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 66ms/step - loss: 19.0561 - val_loss: 18.8634 - learning_rate: 5.0000e-06
Epoch 15/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 65ms/step - loss: 18.9711 - val_loss: 18.8086 - learning_rate: 5.0000e-06
Epoch 16/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 67ms/step - loss: 18.8604 - val_loss: 18.7693 - learning_rate: 5.0000e-06
Epoch 17/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 64ms/step - loss: 18.8112 - val_loss: 18.7421 - learning_rate: 5.0000e-06
Epoch 18/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 64ms/step - loss: 18.7408 - val_loss: 18.7311 - learning_rate: 5.0000e-06
Epoch 19/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 5s 66ms/step - loss: 18.6467 - val_loss: 18.6894 - learning_rate: 5.0000e-06
Epoch 20/60
69/69 ━━━━━━━━━━━━━━━━━━━━ 4s 64ms/step - loss: 18.6356 - val_loss: 18.6690 - learning_rate: 5.0000e-06
...
```
---

#### Evaluation on training:
##### **Key Observations from Training**
1. **Initial Loss**:
   - The initial training loss starts at 54.21, which indicates that the model begins with random weights and makes incorrect predictions.
   - The validation loss also starts similarly high at 52.93.

2. **Progression of Loss**:
   - By Epoch 6, the loss drops to 25.22, and the validation loss decreases to 23.02, showing that the model is learning effectively.
   - By Epoch 20, the training and validation losses stabilize at around **18.5**, which might represent a performance plateau.

3. **Learning Rate Scheduling**:
   - The **ReduceLROnPlateau** callback reduced the learning rate after loss improvement stagnated at Epoch 43, helping stabilize training and avoid overfitting.
   - The final learning rate reaches `1e-6`, which ensures the model does not over-correct as it converges.

4. **Early Stopping**:
   - Early stopping was appropriately used with a patience of 20, allowing the model enough epochs to converge while preventing unnecessary overfitting or wasted computation.
   - The model converged at Epoch 60, which matches the training patience strategy.

5. **CTC Loss Behavior**:
   - The loss plateaued at ~18.5, which may indicate either:
     - A limitation in the current architecture or the capacity of the model.
     - Potential noise in the labels or data requiring further tuning or preprocessing.

---

##### **Strengths of the Setup**
1. **Appropriate Architecture**:
   - The model architecture balances convolutional layers (for spatial feature extraction) with recurrent layers (for sequence modeling). This is ideal for OCR tasks where the input is sequential.
   - Use of **Batch Normalization** helps improve gradient flow and stability.

2. **CTC Loss**:
   - Connectionist Temporal Classification (CTC) loss is correctly used for handling variable-length sequences without explicit alignment between inputs and labels.

3. **Callbacks**:
   - The combination of **early stopping** and **learning rate scheduling** optimizes training efficiency.

---

##### **Potential Issues**
1. **Plateauing Loss**:
   - The training loss and validation loss plateau at ~18.5, which may indicate:
     - Suboptimal data preprocessing (e.g., noisy or inconsistent labels).
     - Insufficient model capacity for this specific task.
     - A learning rate that could be further optimized for fine-tuning.

2. **Validation Gap**:
   - The slight gap between training and validation losses suggests minor overfitting. Techniques such as data augmentation or dropout regularization could mitigate this.

3. **Limited Epochs**:
   - The model converged within the given 60 epochs, but further training (with adjustments) might improve results if the learning rate remains low.

---

##### **Suggestions for Improvement**
1. **Data Augmentation**:
   - To improve generalization, apply data augmentation techniques like:
     - Random rotations, scaling, or shifting of the input images.
     - Adding random noise to simulate real-world variations.

2. **Regularization**:
   - Add **L2 regularization** to Conv2D and Dense layers to prevent overfitting.

3. **Hyperparameter Tuning**:
   - Experiment with:
     - Higher learning rates at the beginning of training.
     - A larger model capacity (e.g., more filters in Conv2D or higher LSTM dimensions).

4. **More Training Data**:
   - If possible, increase the training dataset size or use transfer learning to pretrain the convolutional layers on a larger OCR-related dataset.

---

##### **Evaluation Metrics**
After training, evaluate the model on a held-out test set using metrics such as:
1. **Character Error Rate (CER)**:
   - Measures the average number of character-level errors.
   - Lower CER indicates better performance.

2. **Word Error Rate (WER)**:
   - Measures the average number of word-level errors.
   - Especially useful if the output consists of words or numerical sequences.


Find more at:  
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