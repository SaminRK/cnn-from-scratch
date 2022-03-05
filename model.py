import numpy as np

from typing import List
from layers import Conv2D, CrossCatergoricalEntropy, Dense, Layer, Softmax, StableSoftmax
from sklearn.metrics import f1_score


class Model:
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_fn = CrossCatergoricalEntropy()  # This model only works for soft-max with crosscategoricalentropy

    
    
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
    
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        inter_input = input
        
        for layer in self.layers:
            inter_input = layer.forward(inter_input)
        
        return inter_input
    
    def backward(self, input: np.ndarray, y: np.ndarray, learning_rate) -> np.ndarray:
        inter_input = input
        
        for layer in reversed(self.layers):
            if isinstance(layer, Softmax) or isinstance(layer, StableSoftmax):
                inter_input = layer.backward(input, y)
            elif isinstance(layer, Conv2D) or isinstance(layer, Dense): 
                inter_input = layer.backward(inter_input, learning_rate)
            else:
                inter_input = layer.backward(inter_input)
        
        return inter_input
    

    def _shuffle(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        p = np.random.permutation(X.shape[0])
        return X[p], Y[p]

    def train(self, train_X, train_Y, val_X, val_Y, batch_size, n_epochs, learning_rate, seed=0, output_file=None):
        np.random.seed(seed)

        n_batches = (train_Y.shape[0] + batch_size - 1) // batch_size

        out_f = None
        if output_file:
            out_f = open(output_file, 'a')

        for epoch in range(n_epochs):
            print(f"epoch {epoch+1}")
            if out_f:
                print(f"epoch {epoch+1}", file=out_f)
            total_corr = 0
            total_inputs = 0
            total_loss = 0
            
            shuffled_X, shuffled_Y = self._shuffle(train_X, train_Y)
            
            for batch_num in range(n_batches):
                x = shuffled_X[batch_num*batch_size:(batch_num+1)*batch_size]
                y = shuffled_Y[batch_num*batch_size:(batch_num+1)*batch_size]

                y_hat = self.forward(x)
                losses = self.loss_fn.calculate_loss(y, y_hat)
                self.backward(y_hat, y, learning_rate)
                
                # metrics
                batch_corr = np.sum(np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1)).astype(int))
                total_corr += batch_corr
                total_loss += np.sum(losses)
                total_inputs += losses.shape[0]
                f1_train = f1_score(np.argmax(y, axis=1), np.argmax(y_hat, axis=1), average='macro')

                if total_inputs % 300 == 0: 
                    total_acc = total_corr * 100 / total_inputs
                    print(f"[{total_inputs}] loss: {total_loss / total_inputs}    \taccuracy: {total_acc}%     \tmacro-f1: {f1_train}")
                    if out_f:
                        print(f"[{total_inputs}] loss: {total_loss / total_inputs}    \taccuracy: {total_acc}%     \tmacro-f1: {f1_train}", file=out_f)
            
            val_Y_hat = self.forward(val_X)
            test_corr = np.sum(np.equal(np.argmax(val_Y_hat, axis=1), np.argmax(val_Y, axis=1)).astype(int))
            f1_val = f1_score(np.argmax(val_Y, axis=1), np.argmax(val_Y_hat, axis=1), average='macro')
            val_loss = np.sum(self.loss_fn.calculate_loss(val_Y, val_Y_hat))
            total_vals = val_Y.shape[0]
            val_acc = test_corr * 100 / total_vals

            print(f"VALIDATION [{total_vals}] loss: {val_loss / total_vals}    \taccuracy: {val_acc}%     \tmacro-f1: {f1_val}")
            if out_f:
                print(f"VALIDATION [{total_vals}] loss: {val_loss / total_vals}    \taccuracy: {val_acc}%     \tmacro-f1: {f1_val}", file=out_f, flush=True)

    
    def evaluate(self, test_X, test_Y, output_file):
        test_Y_hat = self.forward(test_X)
        test_corr = np.sum(np.equal(np.argmax(test_Y_hat, axis=1), np.argmax(test_Y, axis=1)).astype(int))
        f1_test = f1_score(np.argmax(test_Y, axis=1), np.argmax(test_Y_hat, axis=1), average='macro')
        test_loss = np.sum(self.loss_fn.calculate_loss(test_Y, test_Y_hat))
        total_tests = test_Y.shape[0]
        test_acc = test_corr * 100 / total_tests

        print(f"TEST [{total_tests}] loss: {test_loss / total_tests}    \taccuracy: {test_acc}%     \tmacro-f1: {f1_test}")
        with open(output_file, 'a') as f:
            print(f"TEST [{total_tests}] loss: {test_loss / total_tests}    \taccuracy: {test_acc}%     \tmacro-f1: {f1_test}", file=f)
