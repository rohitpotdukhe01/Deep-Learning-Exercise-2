import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.output = None
        self.log_output = None
        self.eps = None
        self.label_array = None
        self.prediction_array = None

    def forward(self, prediction_tensor, label_tensor):
        self.eps = np.finfo(float).eps
        self.prediction_array = np.array(prediction_tensor)
        self.log_output = np.log(np.array(prediction_tensor) + self.eps)
        self.label_array = np.array(label_tensor)
        self.output = np.where(self.label_array == 1, -self.log_output, 0).sum()
        return self.output

    def backward(self, label_tensor):
        error_tensor = np.where(label_tensor == 1, -1 / self.prediction_array, 0)
        return error_tensor

