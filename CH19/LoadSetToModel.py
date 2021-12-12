import numpy as np
import cv2
import os

# Load minst dataset
def load_mnist_dataset(dataset, path):

    #scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # create list of samples and labels
    X = []
    y = []

    # for each lable folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # read the image
            image = cv2.imread(os.path.join(
                path, dataset, label, file
            ), cv2.IMREAD_UNCHANGED)

            # And append the data and labels to lists
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype(np.uint8)

# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test
np.set_printoptions(linewidth=200)
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')


X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
             127.5) / 127.5

print(X[:1])

