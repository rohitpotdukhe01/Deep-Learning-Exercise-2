import numpy as np

class Flatten():

    def __init__(self):
        self.output = None
        self.trainable = False

    def forward(self, input_tensor):
        self.output = np.arange(input_tensor).reshape(input_tensor)
        return self.output.copy()

    def backward(self,error_tensor):
        self.output = np.reshape(np.array(error_tensor))
        return self.output.copy()

