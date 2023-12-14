import numpy as np

def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    N = input_matrix_shape[0]
    C_out = out_channels
    w_out = (input_matrix_shape[2] - (kernel_size + 2 * padding)) / stride + 1
    h_out = (input_matrix_shape[3] - (kernel_size + 2 * padding)) / stride + 1
    out_shape = [N, C_out, w_out, h_out]
    return out_shape


print(np.array_equal(
calc_out_shape(input_matrix_shape=[2, 3, 10, 10],
out_channels=10,
kernel_size=3,
stride=1,
padding=0),
[2, 10, 8, 8]))