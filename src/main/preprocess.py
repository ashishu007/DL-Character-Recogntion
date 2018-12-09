import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import cv2

DATA_PATH = "./Train/"

def getCV2(path, IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    return img

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


def save_data_to_array(IMG_SIZE, path=DATA_PATH):
    labels, _, _ = get_labels(path)

    for label in labels:
        cv2_vectors = []
        images = [path + label + '/' + image for image in os.listdir(path + '/' + label)]
        for image in tqdm(images, "Saving vectors of label - '{}'".format(label)):
            cv2V = getCV2(image, IMG_SIZE)
            cv2_vectors.append(cv2V)
        np.save("./npy/" + label + '.npy', cv2_vectors)


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load("./npy/" + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load("./npy/" + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)