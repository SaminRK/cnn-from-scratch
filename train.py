import numpy as np
import sys

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax, CrossCatergoricalEntropy
from model import Model
import preprocess_mnist

def main():
    mnist = preprocess_mnist.get_mnist(one_hot=True)
    X = mnist["training_images"]
    Y = mnist["training_labels_one_hot"]
    val_X = mnist["test_images"][:5000]
    val_Y = mnist["test_labels_one_hot"][:5000]
    test_X = mnist["test_images"][5000:]
    test_Y = mnist["test_labels_one_hot"][5000:]


    nn_model = Model()
    nn_model.add_layer(Conv2D(filter_size=5, n_filters=6, stride=1, padding=2))
    nn_model.add_layer(ReLU())
    nn_model.add_layer(MaxPool2D(filter_size=2, stride=2))
    nn_model.add_layer(Conv2D(filter_size=5, n_filters=12, stride=1, padding=0))
    nn_model.add_layer(ReLU())
    nn_model.add_layer(MaxPool2D(filter_size=2, stride=2))
    nn_model.add_layer(Conv2D(filter_size=5, n_filters=100, stride=1, padding=0))
    nn_model.add_layer(ReLU())
    nn_model.add_layer(Flatten())
    nn_model.add_layer(Dense(units=10))
    nn_model.add_layer(Softmax())

    nn_model.train(X, Y, val_X, val_Y, batch_size=30, n_epochs=10, learning_rate=5e-5, output_file='log.txt')
    nn_model.evaluate(test_X, test_Y, output_file='log.txt')


if __name__ == "__main__":
    main()