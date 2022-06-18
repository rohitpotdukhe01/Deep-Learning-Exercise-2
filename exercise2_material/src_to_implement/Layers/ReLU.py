import numpy as np
from .Base import BaseLayer


class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.output = None
        self.relu_gradient = None

    def forward(self, input_tensor):
        self.output=np.clip(input_tensor, a_min=0, a_max=None)
        self.relu_gradient=self.output.copy()
        return self.output

    def backward(self, error_tensor):
        self.relu_gradient[self.relu_gradient > 0] = 1
        self.output = np.multiply(error_tensor, self.relu_gradient)
        return self.output
