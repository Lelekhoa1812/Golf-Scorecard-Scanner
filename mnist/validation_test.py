#### Validation
preds = model.predict(valid_x)
#print('\n preds',preds)
decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])
print ('\n decoded',decoded)
prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))
print ('\n predict',num_to_label(decoded[0]))
y_true = label_valid
correct_char = 0
total_char = 0
correct = 0
for i in range(valid_size):
    pr = prediction[i] 
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))
####


#### Testing
import random
plt.figure(figsize=(15, 10))
# Define the alphabets
alphabets = "0123456789' "
for i in range(6):
    ax = plt.subplot(2, 3, i + 1)
    fold_dir = '../synthetic_digits/'
    filename = random.sample((os.listdir(fold_dir)), 1)
    filename = ("".join(str(e) for e in filename))  # Remove brackets
    print('\n filename', filename)
    img_dir = fold_dir + str(filename)
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    image = preprocess(image, (128, 32))
    image = image / 255.0
    pred = model.predict(image.reshape(1, 128, 32, 1))
    decoded = tf.keras.backend.get_value(
        tf.keras.backend.ctc_decode(
            pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True
        )[0][0]
    )
    # Plot
    plt.title(num_to_label(decoded[0], alphabets), fontsize=12)  # Pass alphabets
    plt.axis('off')
####