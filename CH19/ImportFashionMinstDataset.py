import os
import urllib
import urllib.request
from zipfile import ZipFile
import ssl

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'



if not os.path.isfile(FILE):
    with urllib.request.urlopen(URL, context=ssl.SSLContext()) as u, \
        open(FILE, 'wb') as f:
            f.write(u.read())
print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)
print('Done!')