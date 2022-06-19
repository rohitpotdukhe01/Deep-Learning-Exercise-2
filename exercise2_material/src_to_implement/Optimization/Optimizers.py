import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        assert (type(learning_rate) == float or type(learning_rate) == int)
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """ Stochastic gradient descent """
        updated_weights = weight_tensor - (self.learning_rate * gradient_tensor)
        return updated_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.previous_momentum = 0
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        # self.output = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        current_momentum = np.dot(self.momentum_rate, self.previous_momentum) - np.dot(self.learning_rate, gradient_tensor)
        self.previous_momentum = current_momentum
        new_weights = weight_tensor + current_momentum
        return new_weights


class Adam:

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.vk_prev = 0
        self.rk_prev = 0
        self.k = 1
        self.output = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        vk = np.dot(self.mu, self.vk_prev) + np.dot((1 - self.mu), gk)
        rk = np.dot(self.rho, self.rk_prev) + np.dot(np.dot((1 - self.rho), gk), gk)
        self.vk_prev = vk
        self.rk_prev = rk

        # Bias Correction
        vk_cap = vk / (1 - self.mu ** self.k)
        rk_cap = rk / (1 - self.rho ** self.k)

        self.output = weight_tensor - np.dot(self.learning_rate, (vk_cap / (np.sqrt(rk_cap) + np.finfo(float).eps)))
        self.k = self.k + 1
        return self.output
