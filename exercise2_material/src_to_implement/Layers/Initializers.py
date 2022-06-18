import numpy as np


class Constant:
    def __init__(self, constant_val):
        self.weights_const = constant_val

    def initialize(self, weights_shape, fan_in, fan_out):
        weight_tensor = np.ones(fan_in) * self.weights_const
        output = np.reshape(weight_tensor, weights_shape)
        return output


class UniformRandom:

    def initialize(self, weights_shape, fan_in, fan_out):
        weight_tensor = np.random.uniform(low=0, high=1, size=(fan_in, fan_out))
        output = np.reshape(weight_tensor, weights_shape)
        return output


class Xavier:

    def initialize(self, weights_shape, fan_in, fan_out):
        sqrt_val = np.sqrt(2 / float(fan_in + fan_out))
        weight_tensor = np.random.normal(0.0, sqrt_val, size=(fan_in, fan_out))
        output = np.reshape(weight_tensor, weights_shape)
        return output


class He:

    def initialize(self, weights_shape, fan_in, fan_out):
        sqrt_val = np.sqrt(2 / float(fan_in))
        weight_tensor = np.random.normal(0.0, sqrt_val, size=(fan_in, fan_out))
        output = np.reshape(weight_tensor, weights_shape)
        return output
