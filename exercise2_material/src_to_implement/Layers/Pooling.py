import numpy as np
from .Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape=(2, 2), pooling_shape=(2, 2)):
        super().__init__()
        self.back_input = None
        self.stride = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.back_input = input_tensor
        batch_size, num_channels, input_height, input_width = input_tensor.shape
        output_height = int((input_height - self.pooling_shape[1]) / self.stride[0]) + 1
        output_width = int((input_width - self.pooling_shape[0]) / self.stride[1]) + 1

        down_sampling = np.zeros((batch_size, num_channels, output_height, output_width))
        for i in range(batch_size):
            for j in range(num_channels):
                current_y = output_y = 0
                while current_y + self.pooling_shape[1] <= input_height:
                    current_x = output_x = 0
                    while current_x + self.pooling_shape[0] <= input_width:
                        patch = input_tensor[i, j, current_y:current_y + self.pooling_shape[1],
                                current_x:current_x + self.pooling_shape[0]]
                        down_sampling[i, j, output_y, output_x] = np.max(patch)
                        current_x += self.stride[1]
                        output_x += 1
                    current_y += self.stride[0]
                    output_y += 1

        return down_sampling

    def backward(self, error_tensor):
        batch_size, num_channels, input_height, input_width = self.back_input.shape

        output = np.zeros(self.back_input.shape)
        for i in range(batch_size):
            for j in range(num_channels):
                temp_y = output_y = 0
                while temp_y + self.pooling_shape[1] <= input_height:
                    temp_x = output_x = 0
                    while temp_x + self.pooling_shape[0] <= input_width:
                        patch = self.back_input[i, j, temp_y:temp_y + self.pooling_shape[1],
                                temp_x:temp_x + self.pooling_shape[0]]
                        (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)
                        output[i, j, temp_y + x, temp_x + y] += error_tensor[i, j, output_y, output_x]
                        temp_x += self.stride[1]
                        output_x += 1
                    temp_y += self.stride[0]
                    output_y += 1

        return output
