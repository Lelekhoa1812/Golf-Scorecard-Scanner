import random
plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    fold_dir = '../input/dataset/multi_digit_images_10k/multi_digit_images/'
    filename = random.sample((os.listdir(fold_dir)),1)
    filename = ( "".join( str(e) for e in filename ) ) # bỏ ngoặc
    print ('\n filename',filename)
    img_dir = fold_dir+str(filename) 
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    image = preprocess(image,(128,32))
    image = image/255.
    pred = model.predict(image.reshape(1, 128, 32, 1))
    decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])

    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
