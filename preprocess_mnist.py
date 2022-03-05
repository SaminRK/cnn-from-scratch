import gzip
import numpy as np

# inspired by: https://github.com/hsjeong5/MNIST-for-Numpy


def get_mnist(one_hot=False):

    IMAGES_FILENAME = {
        "training_images": "train-images-idx3-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
    }
    LABELS_FILENAME = {
        "training_labels": "train-labels-idx1-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    DATASET_ROOT_DIR = "./datasets/"

    mnist = {}
    for image_data in IMAGES_FILENAME:
        with gzip.open(f"{DATASET_ROOT_DIR}{IMAGES_FILENAME[image_data]}", 'rb') as f:
            mnist[image_data] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1)
        print(mnist[image_data].shape)
    
    for label_data in LABELS_FILENAME:
        with gzip.open(f"{DATASET_ROOT_DIR}{LABELS_FILENAME[label_data]}", 'rb') as f:
            mnist[label_data] = np.frombuffer(f.read(), np.uint8, offset=8)
        print(mnist[label_data].shape)
    
    if one_hot:
        from sklearn.preprocessing import OneHotEncoder
        for label_data in LABELS_FILENAME:
            enc = OneHotEncoder()
            one_hot_encoded = enc.fit_transform(mnist[label_data].reshape(-1, 1)).toarray()
            mnist[f"{label_data}_one_hot"] = one_hot_encoded

    return mnist

def show_image(img_index=0):
    import matplotlib.pyplot as plt
    mnist = get_mnist()
    img = mnist["training_images"][img_index]
    print(mnist["training_labels"][img_index])
    plt.imshow(img, cmap='gray')
    plt.show()

# show_image(7)
get_mnist(one_hot=True)