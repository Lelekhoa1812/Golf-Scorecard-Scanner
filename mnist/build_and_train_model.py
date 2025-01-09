# Build Model
#y_true: tensor (samples, max_string_length) containing the truth labels.
#y_pred: tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.
#input_length: tensor (samples, 1) containing the sequence length of slices coming out from RNN for each batch item in y_pred.
#label_length: tensor (samples, 1) containing the sequence length of label for each batch item in y_true.
def build_model(img_width = 128,img_height = 32, max_str_len = 10):
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    ) 
    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    # Output layer
    y_pred = layers.Dense(10 + 1, activation="softmax", name="dense2")(x) # y pred
    model = keras.models.Model(inputs=input_img, outputs=y_pred, name="functional_1")
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage
        y_pred = y_pred[:, 2:, :]
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
    # Config label and input shape
    labels = layers.Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    # Construct CTC loss
    ctc_loss = keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model_final = keras.models.Model(inputs=[input_img, labels, input_length, label_length], outputs=ctc_loss, name = "ocr_model_v1")
    # Finalise
    return model, model_final
# Demo
model, model_final = build_model()
model.summary()
model_final.summary()

# Train Model (modify setup as your preference)
opt = keras.optimizers.Adam()
early_stopping_patience = 5
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)
# Compile model with optimizer (Adam)
model_final.compile(
    loss={'ctc': lambda y_true, y_pred: y_pred}, 
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
)
# Fit model epochs=200, batch_size=128
model_final.fit(
    x=[train_x, train_y, train_input_len, train_label_len],
    y=train_output, 
    validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
    epochs=200, 
    batch_size=128,
    callbacks=[early_stopping]  # Include early stopping to monitor val_loss
)
model.save('../models/mnistrcnn_m1.h5')
