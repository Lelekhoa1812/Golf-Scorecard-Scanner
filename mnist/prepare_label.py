# Prepare Labels for CTC Loss
import numpy as np

def label_to_num(label, alphabets):
    label_num = []
    for ch in label:
        if alphabets.find(ch) == -1:
            print(f"Warning: Character '{ch}' not found in alphabets.")
        label_num.append(alphabets.find(ch))
    return np.array(label_num)

def num_to_label(num, alphabets):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabets[ch]
    return ret

def prepare_labels(labels, size, max_str_len, num_of_timestamps, alphabets):
    y = np.ones([size, max_str_len]) * -1
    label_len = np.zeros([size, 1])
    input_len = np.ones([size, 1]) * (num_of_timestamps - 2)
    for i in range(size):
        label_len[i] = len(labels[i])
        y[i, 0:len(labels[i])] = label_to_num(labels[i], alphabets)
    return y, label_len, input_len

# Configuration
alphabets = u"0123456789' "
max_str_len = 10  # Maximum length of input labels
num_of_timestamps = 32  # Maximum length of predicted labels
train_size = len(label_train)
valid_size = len(label_valid)

# Prepare Training and Validation Labels
train_y, train_label_len, train_input_len = prepare_labels(label_train, train_size, max_str_len, num_of_timestamps, alphabets)
valid_y, valid_label_len, valid_input_len = prepare_labels(label_valid, valid_size, max_str_len, num_of_timestamps, alphabets)
train_output = np.zeros([train_size])
valid_output = np.zeros([valid_size])

# Debug Outputs
print(f"Example Train Label: {label_train[10]}")
print(f"Train Label Encoded: {train_y[10]}")
print(f"Train Label Length: {train_label_len[10]}")
print(f"Train Input Length: {train_input_len[10]}")

print(f"Example Valid Label: {label_valid[10]}")
print(f"Valid Label Encoded: {valid_y[10]}")
print(f"Valid Label Length: {valid_label_len[10]}")
print(f"Valid Input Length: {valid_input_len[10]}")