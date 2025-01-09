#y_true: tensor (samples, max_string_length) containing the truth labels.
#y_pred: tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.
#input_length: tensor (samples, 1) containing the sequence length of slices coming out from RNN for each batch item in y_pred.
#label_length: tensor (samples, 1) containing the sequence length of label for each batch item in y_true.

# Build and Train Model
from tensorflow.keras import layers, Model
from tensorflow import keras
import tensorflow as tf

def build_model(img_width=128, img_height=32, max_str_len=10, num_classes=len(alphabets) + 1):
    # Input layer
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    # First Convolutional Block
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    # Second Convolutional Block
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    # Reshape layer
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    # Output layer
    y_pred = layers.Dense(num_classes, activation="softmax", name="dense2")(x)
    model = Model(inputs=input_img, outputs=y_pred, name="ocr_model")
    # CTC Loss Layer
    labels = layers.Input(name="gtruth_labels", shape=[max_str_len], dtype="float32")
    input_length = layers.Input(name="input_length", shape=[1], dtype="int64")
    label_length = layers.Input(name="label_length", shape=[1], dtype="int64")
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]  # Skip first 2 outputs
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )
    # Final model
    model_final = Model(
        inputs=[input_img, labels, input_length, label_length], outputs=ctc_loss, name="ocr_model_final"
    )
    return model, model_final
# Build Model
model, model_final = build_model()
model.summary()
model_final.summary()
# Compile Model
model_final.compile(
    loss={"ctc": lambda y_true, y_pred: y_pred},
    optimizer = keras.optimizers.Adam(learning_rate=0.000005), # Lower learning rate for more stable train in case it stop unexpectably
)
# Early Stopping and Learning Rate Schedule
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)
lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)
# Train Model
model_final.fit(
    x=[train_x, train_y, train_input_len, train_label_len],
    y=train_output,
    validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
    epochs=60,
    batch_size=128,
    callbacks=[early_stopping, lr_schedule],
)
# Save Model
model_final.save("../models/mnistrcnn_m1.keras", save_format="keras")