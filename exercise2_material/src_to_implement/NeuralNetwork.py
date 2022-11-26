#bye bye birdie
import copy as cp


class NeuralNetwork():
    def __init__(self, optimizer, weight_initializers, bias_initializers):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.next_input = None
        self.labels_input = None
        self.weight_initializers = weight_initializers
        self.bias_initializers = bias_initializers
        self.output = None

    def forward(self):
        self.next_input, self.labels_input = self.data_layer.next()
        input_copy = self.next_input.copy()
        for layer in self.layers:
            input_copy = layer.forward(input_copy)
        self.output = self.loss_layer.forward(input_copy, self.labels_input)
        return self.output

    def backward(self):
        loss_gradiant = self.loss_layer.backward(self.labels_input)
        for layer in self.layers[::-1]:
            loss_gradiant = layer.backward(loss_gradiant)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = cp.deepcopy(self.optimizer)
            layer.initialize(self.weight_initializers, self.bias_initializers)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        inp = input_tensor
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp
