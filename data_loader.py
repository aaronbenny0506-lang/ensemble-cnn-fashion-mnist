"""
data_loader.py
Loads the Fashion-MNIST dataset from local IDX-format files.

Note: `keras.datasets.fashion_mnist.load_data()` normally downloads the
dataset from `storage.googleapis.com`, which is not reachable from this
sandboxed environment's network allow-list. Instead, the same four IDX
files Keras would download are fetched ahead of time from the official
Zalando Research GitHub repository (github.com is reachable) and parsed
here with a small IDX reader. The returned data has the exact same shape
and dtype as `keras.datasets.fashion_mnist.load_data()`:

    (x_train, y_train), (x_test, y_test)
    x_*: uint8 arrays of shape (N, 28, 28)
    y_*: uint8 arrays of shape (N,)
"""

import gzip
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    # Header: magic(4) num_images(4) rows(4) cols(4), big-endian
    num_images = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    images = np.frombuffer(data[16:], dtype=np.uint8)
    images = images.reshape(num_images, rows, cols)
    return images


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels


def load_data():
    """Mirrors keras.datasets.fashion_mnist.load_data()'s return signature."""
    x_train = _read_idx_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
    y_train = _read_idx_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
    x_test = _read_idx_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
    y_test = _read_idx_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print("x_train:", x_train.shape, x_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("x_test:", x_test.shape, x_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)
