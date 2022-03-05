import numpy as np
import sys


class Layer:
    def __init__(self) -> None:
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Conv2D(Layer):
    def __init__(self, filter_size, n_filters, stride, padding) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding
        self.w = None
        self.b = None
        self.height = None
        self.width = None
        self.n_input_channels = None
        self.input = None
        self.output = None
        self.padded_input = None

    def forward(self, input: np.ndarray, verbose=False) -> np.ndarray:
        self.input = input
        batch_size = input.shape[0]

        if self.w is None:
            self.height = input.shape[1]
            self.width = input.shape[2]
            self.n_input_channels = input.shape[3]
            self.w = np.random.randn(self.n_filters, self.filter_size, self.filter_size,
                                     self.n_input_channels) / (self.filter_size * self.filter_size)
            self.b = np.random.randn(
                self.n_filters) / (self.filter_size * self.filter_size)

        if self.padding > 0:
            padded_input = np.zeros(
                (batch_size, 1, self.height+2*self.padding, self.width+2*self.padding, self.n_input_channels))
            padded_input[:, 0, self.padding:-self.padding,
                         self.padding:-self.padding, :] = input
        else:
            padded_input = input.reshape(
                (batch_size, 1, self.height, self.width, self.n_input_channels))

        # for now consider input to be square
        out_dim = (self.height - self.filter_size +
                   2 * self.padding) // self.stride + 1
        output = np.zeros((batch_size, out_dim, out_dim, self.n_filters))

        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                ii = i * self.stride
                jj = j * self.stride
                output[:, i, j, :] = np.sum(np.multiply(
                    padded_input[:, :, ii:ii+self.filter_size, jj:jj+self.filter_size, :], self.w), axis=(2, 3, 4))

        if verbose:
            print_pos_of_batch = 0
            print(f"input\n{padded_input[print_pos_of_batch, 0, :, :, 0]}")
            print(f"w:\n{self.w[0, :, :, 0]}")
            print(f"b:\n{self.b}")
            print(f"out before bias:\n{output[print_pos_of_batch, :, :, 0]}")

        output += self.b
        
        if verbose:
            print(f"out after bias:\n{output[print_pos_of_batch, :, :, 0]}")
            
        self.output = output
        self.padded_input = np.reshape(
            padded_input, (batch_size, self.height+2*self.padding, self.width+2*self.padding, self.n_input_channels))

        return output

    def backward(self, prev_d, lr, verbose=False):
        batch_size = self.input.shape[0]
        padded_d = np.zeros((batch_size, self.height+2*self.padding,
                            self.width+2*self.padding, self.n_input_channels))
        dw = np.zeros(self.w.shape)
        db = np.zeros(self.b.shape)

        reshaped_w = np.transpose(self.w, (1, 2, 0, 3))
        for i in range(self.output.shape[1]):
            for j in range(self.output.shape[2]):
                ii = i * self.stride
                jj = j * self.stride

                t1 = prev_d[:, i, j, :].T
                t2 = np.transpose(
                    self.padded_input[:, ii:ii+self.filter_size, jj:jj+self.filter_size, :], (1, 2, 0, 3))
                influence_dw = np.dot(t1, t2)
                dw += influence_dw

                influence = np.dot(prev_d[:, i, j, :], reshaped_w)
                padded_d[:, ii:ii+self.filter_size,
                         jj:jj+self.filter_size, :] += influence

        db = np.sum(prev_d, (0, 1, 2))

        dw /= (batch_size)
        db /= (batch_size)

        self.w -= dw * lr
        self.b -= db * lr

        if self.padding > 0:
            d = padded_d[:, -self.padding, self.padding, :]
        else:
            d = padded_d
        if verbose:
            print(f"prev_d:\n{prev_d}")
            print(prev_d.shape)

            print(f"padded_d:\n{padded_d}")
            print(padded_d.shape)

            print(f"d:\n{d}")
            print(d.shape)

            print(f"dw\n{dw}")
            print(dw.shape)

            print(f"db:\n{db}")
            print(db.shape)

        return d


class Dense(Layer):
    def __init__(self, units) -> None:
        super().__init__()
        self.units = units
        self.input_dim = None
        self.w = None
        self.b = None

        self.input = None
        self.output = None

    def forward(self, input: np.ndarray, verbose=False) -> np.ndarray:
        self.input = input
        batch_size = input.shape[0]

        if self.w is None:
            self.input_dim = input.shape[1]
            self.w = np.random.randn(
                self.units, self.input_dim) / (self.units * self.input_dim)
            self.b = np.random.randn(self.units) / \
                (self.units * self.input_dim)

        output = np.dot(input, self.w.T)

        if verbose:
            # print(np.multiply(input_reshaped, self.w).shape)

            # print(f"input:\n{input}")
            print(f"input shape: {input.shape}")
            print(f"w:\n{np.sum(self.w[4])}")
            # print(f"w shape: {self.w.shape}")
            print(f"b:\n{self.b}")
            # print(self.b.shape)
            print(f"out:\n{output}")

        output += self.b
        self.output = output

        return output

    def backward(self, prev_d, lr, verbose=False):
        batch_size = self.input.shape[0]

        dw = np.dot(prev_d.T, self.input)
        db = np.sum(prev_d, axis=0)

        dw /= batch_size
        db /= batch_size

        self.w -= dw * lr
        self.b -= db * lr

        d = np.dot(prev_d, self.w)

        if verbose:
            print(f"input:\n{self.input}")
            print(self.input.shape)

            print(f"prev_d:\n{prev_d}")
            print(prev_d.shape)

            print(f"w:\n{self.w}")
            print(self.w.shape)

            print(f"dw:\n{dw}")
            print(dw.shape)

            print(f"db:\n{db}")
            print(db.shape)

            print(f"d:\n{d}")
            print(d.shape)
            print(self.input.shape)

        return d


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, input: np.ndarray, verbose=False) -> np.ndarray:
        self.input = input
        output = input.clip(min=0)
        self.output = output
        return output

    def backward(self, prev_d, verbose=False):
        mask = (self.output > 0).astype(int)
        d = mask * prev_d
        if verbose:
            print(f"mask:\n{mask}")
            print(mask.shape)

            print(f"prev_d:\n{prev_d}")
            print(prev_d.shape)

            print(f"d:\n{d}")
            print(d.shape)

        return d


class MaxPool2D(Layer):
    def __init__(self, filter_size, stride) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.height = None

        self.input = None
        self.output = None
        self.mask = None

    def forward(self, input: np.ndarray, verbose=False) -> np.ndarray:
        self.input = input
        batch_size = input.shape[0]
        if not self.height:
            self.height = input.shape[1]
            self.width = input.shape[2]
            self.n_input_channels = input.shape[3]

        out_dim = (self.height - self.filter_size) // self.stride + 1
        output = np.zeros(
            (batch_size, out_dim, out_dim, self.n_input_channels))
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                ii = i * self.stride
                jj = j * self.stride
                output[:, i, j, :] = np.amax(
                    input[:, ii:ii+self.filter_size, jj:jj+self.filter_size, :], axis=(1, 2))

        if verbose:
            print_pos_of_batch = 0
            print(f"input\n{input[print_pos_of_batch, :, :, 0]}")
            print(f"max out:\n{output[print_pos_of_batch, :, :, 0]}\n")

        self.output = output
        return output

    def backward(self, prev_d, verbose=False) -> np.ndarray:
        d = np.zeros(self.input.shape)
        for i in range(self.output.shape[1]):
            for j in range(self.output.shape[2]):
                ii = i * self.stride
                jj = j * self.stride
                maxes = np.amax(
                    self.input[:, ii:ii+self.filter_size, jj:jj+self.filter_size, :], axis=(1, 2), keepdims=True)
                mask = np.equal(
                    maxes, self.input[:, ii:ii+self.filter_size, jj:jj+self.filter_size, :]).astype(int)
                # mask will be 1 in places where it is equal to 1, this can result in multiple 1's, ignored for now
                influence = np.transpose(
                    mask, (1, 2, 0, 3)) * prev_d[:, i, j, :]
                influence = np.transpose(influence, (2, 0, 1, 3))

                if verbose:
                    print(mask[0, :, :, 0])
                    print(influence[0, :, :, 0])
                    print(prev_d[0, i, j, 0])
                    print(mask.shape)
                    print(influence.shape)

                d[:, ii:ii+self.filter_size, jj:jj +
                    self.filter_size, :] += influence

        if verbose:
            print(f"prev:\n{prev_d[0, :, :, 0]}")
            print(prev_d.shape)

            print_pos_of_batch = 0
            print(f"input\n{self.input[print_pos_of_batch, :, :, 0]}")
            print(f"max out:\n{self.output[print_pos_of_batch, :, :, 0]}\n")

            print(f"d:\n{d[print_pos_of_batch, :, :, 0]}")
            print(d.shape)

        return d


class Flatten(Layer):
    def __init__(self) -> None:
        self.input = None
        self.output = None
        super().__init__()

    def forward(self, input: np.ndarray, verbose=False):
        self.input = input
        batch_size = input.shape[0]
        output = input.reshape(batch_size, -1)

        if verbose:
            print(f"input:\n{input}")
            print(input.shape)
            print(f"flattened out:\n{output}")
            print(output.shape)

        self.output = output
        return output

    def backward(self, prev_d, verbose=False):
        d = prev_d.reshape(
            self.input.shape[0], self.input.shape[1], self.input.shape[2], self.input.shape[3])

        if verbose:
            print(f"prev_d:\n{prev_d}")
            print(prev_d.shape)

            print(self.input.shape)
            print(self.output.shape)

            output_reshaped = self.output.reshape(
                self.input.shape[0], self.input.shape[1], self.input.shape[2], self.input.shape[3])
            if np.array_equal(self.input, output_reshaped):
                print("equality")
            else:
                print("not equal")
            print(f"d:\n{d}")
            print(d.shape)

        return d


class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, verbose=False):
        exp_input = np.exp(input)
        sum_exp_input = np.sum(exp_input, axis=1)

        if verbose:
            print(f"input:\n{input}")
            print(f"sum:\n{sum_exp_input}")
            print(f"out:\n{exp_input}")

        exp_input = exp_input.T / sum_exp_input

        if verbose:
            print(f"out:\n{exp_input.T}\n\n")

        return exp_input.T

    def backward(self, y_hat, y, verbose=False):
        diff_mat = y_hat - y
        if verbose:
            print(f"softmax backprop {diff_mat}")
        return diff_mat


class StableSoftmax(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, verbose=False):
        max_exp_input = np.max(input, axis=1)
        exp_input = np.exp(input.T - max_exp_input)
        exp_input = exp_input.T
        sum_exp_input = np.sum(exp_input, axis=1)

        if verbose:
            print(f"input:\n{input}")
            print(f"sum:\n{sum_exp_input}")
            print(f"out:\n{exp_input}")

        exp_input = exp_input.T / sum_exp_input

        if verbose:
            print(f"out:\n{exp_input.T}\n\n")

        return exp_input.T

    def backward(self, y_hat, y, verbose=False):
        diff_mat = y_hat - y
        if verbose:
            print(f"softmax backprop {diff_mat}")
        return diff_mat


class CrossCatergoricalEntropy():
    def __init__(self) -> None:
        pass

    def calculate_loss(self, y, y_hat):
        ln_y_hat = np.log(y_hat)
        out = - np.sum(np.multiply(y, ln_y_hat), axis=1)
        return out


def main():
    np.random.seed(0)
    conv2d1 = Conv2D(filter_size=3, n_filters=5, stride=1, padding=0)
    relu = ReLU()
    max2d1 = MaxPool2D(filter_size=2, stride=1)
    flatten = Flatten()
    dense = Dense(units=10)
    softmax = Softmax()
    crossCategoricalEntropy = CrossCatergoricalEntropy()

    input = np.array(
        [[0, 5, 1, 8, -7],
         [0, -5, 2, -8, 6],
         [7, 5, 1, 8, -7],
         [3, 5, 1, 8, -7],
         [11, 5, 1, 8, -7]]
    )
    input = input.reshape((1, 5, 5, 1))
    input = np.broadcast_to(input, (2, 5, 5, 1))
    print(input.shape)
    out1 = conv2d1.forward(input)
    out2 = relu.forward(out1)
    out3 = max2d1.forward(out2)
    out4 = flatten.forward(out3)
    out5 = dense.forward(out4)
    y_hat = softmax.forward(out5)

    y = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    loss = crossCategoricalEntropy.calculate_loss(y, y_hat)

    learning_rate = 0.1

    back1 = softmax.backward(y, y_hat)
    back2 = dense.backward(back1, learning_rate)
    back3 = flatten.backward(back2)
    back4 = max2d1.backward(back3)
    back5 = relu.backward(back4)
    back6 = conv2d1.backward(back5, learning_rate)

    # print(loss)


if __name__ == "__main__":
    main()
