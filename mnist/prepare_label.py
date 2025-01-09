# The labels have to be converted to numbers which represent each character in the training set. 
# The 'alphabets' consist of A-Z and three special characters (- ' and space).
alphabets = u"0123456789' "
max_str_len = 10 # max length of input labels
# My project have 7 digits per image, but as long as the max_str_len > the number of digit per image, 
#it's just fine
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 32 # max length of predicted labels
# Converter of type
def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
    return np.array(label_num)
def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

name = '39816931'
print(name, '\n',label_to_num(name))

# Prepare labelling of train set for output
train_y = np.ones([train_size, max_str_len]) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])
# Label from img name
for i in range(train_size):
    train_label_len[i] = len(label_train[i])
    train_y[i, 0:len(label_train[i])]= label_to_num(label_train[i])  
# Prepare labelling of valid (test) set for output
valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])
# Label from img name
for i in range(valid_size):
    valid_label_len[i] = len(label_valid[i])
    valid_y[i, 0:len(label_valid[i])]= label_to_num(label_valid[i])   
# Debugs
print('\n True label_train  : ',label_train[10] , '\ntrain_y : ',train_y[10],
      '\ntrain_label_len : ',train_label_len[10], '\ntrain_input_len : ', train_input_len[10])
print('\n True label_valid : ',label_valid[10] , '\ntrain_y : ',valid_y[10],
      '\ntrain_label_len : ',valid_label_len[10], '\ntrain_input_len : ', valid_input_len[10])