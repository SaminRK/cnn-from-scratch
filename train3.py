import numpy as np
import sys

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax, StableSoftmax
from model import Model
import preprocess_toy

def main():
    toy = preprocess_toy.get_toy(one_hot=True)
    X = toy["train_data"]
    Y = toy["train_one_hot"]
    test_X = toy["test_data"]
    test_Y = toy["test_one_hot"]


    nn_model = Model()
    nn_model.add_layer(Dense(units=20))
    nn_model.add_layer(ReLU())
    nn_model.add_layer(Dense(units=100))
    nn_model.add_layer(ReLU())
    nn_model.add_layer(Dense(units=4))
    nn_model.add_layer(Softmax())

    nn_model.train(X, Y, test_X, test_Y, batch_size=30, n_epochs=10000, learning_rate=1e-4)
    


if __name__ == "__main__":
    main()