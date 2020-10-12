import matplotlib.pyplot as plt
import numpy as np
import pyximport; pyximport.install()
import lines as lin
from shapes import generate_mesh
from scipy.ndimage import convolve, generate_binary_structure


def find_perimeter(shape_mat):
    kernel = generate_binary_structure(2, 2)
    convolution = convolve(shape_mat, kernel, mode="wrap")
    unsorted_perimeter = np.where(np.logical_and(convolution < 8, shape_mat == 1))
    index_mat = np.indices(shape_mat.shape) #- np.reshape(np.array(shape_mat.shape), (2, 1, 1)) / 2
    unsorted_perimeter_pos = index_mat[:, unsorted_perimeter[0], unsorted_perimeter[1]]
    unsorted_perimeter_angle = np.arctan2(unsorted_perimeter_pos[0], unsorted_perimeter_pos[1])
    sorted_perimeter = unsorted_perimeter_pos[:, np.argsort(unsorted_perimeter_angle)].transpose()
    # print(sorted_perimeter)
    return sorted_perimeter


def generate_print_path(printer, shape, pattern, show_plots=False):
    boundaries = shape[1]
    shape_function = shape[0]
    x, y = generate_mesh(boundaries, printer)
    x_field, y_field = pattern(x, y)
    desired_shape = np.array(shape_function(x, y), dtype=np.int32)
    perimeter = find_perimeter(desired_shape)
    perimeter = np.array(perimeter, dtype=np.int32)

    lin.generate_lines(x_field, y_field, desired_shape, perimeter, printer)

    if show_plots:
        plt.figure(figsize=(8, 8))
        plt.quiver(x, y, x_field, y_field)
        plt.show()

        plt.imshow(desired_shape)
        plt.show()
    # return lines
