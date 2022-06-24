from .Base import BaseLayer
import numpy as np


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.convolution_shape= (0,self.num_kernels)
        self.weights = np.random.uniform(0., 1., convolution_shape)
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_tensor):

        pass

    def backward(self, error_tensor):
        pass

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    def gradient_weights(self, val):
        self.__gradient_weights = val

    @property
    def gradient_bias(self):
        return self.__gradient_bias

    def gradient_bias(self, val):
        self.__gradient_bias = val
