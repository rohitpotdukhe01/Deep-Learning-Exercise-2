import numpy as np


class SoftMax:
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.output = None
        self.softmax_gradient = None

    def forward(self, input_tensor):
        exps = np.exp(input_tensor - np.max(input_tensor))
        self.output = exps / exps.sum(axis=1, keepdims=True)
        self.softmax_gradient = self.output.copy()
        return self.output

    def backward(self, error_tensor):
        self.output = self.softmax_gradient * (error_tensor - (error_tensor * self.softmax_gradient).sum(axis=1)[:, None])
        return self.output
