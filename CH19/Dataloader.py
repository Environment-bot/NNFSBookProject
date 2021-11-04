import os
import cv2

labels = os.listdir('fashion_mnist_images/train')
print(labels)
# data

X = []
y = []

for label in labels:
    # open that folder
    for file in os.listdir(os.path.join(
                            'fashion_mnist_images/train', label 
    )):
        # read images
        image = cv2.imread(os.path.join(
                    'fashion_mnist_images/train', label, file
                ), cv2.IMREAD_UNCHANGED)

        X.append(image)
        y.append(label)


print(X[:1])
print(y[:5])
# test = '0'
# print(os.listdir(os.path.join('fashion_mnist_images/train', test)))