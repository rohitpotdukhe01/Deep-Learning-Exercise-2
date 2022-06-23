import numpy as np
from .Base import BaseLayer


class Flatten(BaseLayer):

    def __init__(self):
        super().__init__()
        self.output = None
        self.trainable = False
        self.forward_output = None
        self.backward_output = None

    def forward(self, input_tensor):
        self.forward_output = np.shape(input_tensor)
        self.output = input_tensor.reshape(self.forward_output[0],-1)
        return self.output.copy()

    def backward(self, error_tensor):
        self.output = error_tensor.reshape(self.forward_output)
        return self.output.copy()


