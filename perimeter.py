import matplotlib.pyplot as plt
import numpy as np
import pyximport; pyximport.install()
import line_generation.generator as lin
from shapes import generate_mesh
from scipy.ndimage import convolve, generate_binary_structure


def find_perimeter(shape_mat):
    kernel = generate_binary_structure(2, 2)
    sum_of_neighbours = convolve(shape_mat, kernel, mode="wrap")

    index_mat = np.indices(shape_mat.shape) - np.reshape(np.array(shape_mat.shape), (2, 1, 1)) / 2

    perimeter_unsorted_index = np.where(np.logical_and(sum_of_neighbours < 8, shape_mat == 1))
    perimeter_unsorted_pos = index_mat[:, perimeter_unsorted_index[0], perimeter_unsorted_index[1]]
    perimeter_unsorted_angle = np.arctan2(perimeter_unsorted_pos[0], perimeter_unsorted_pos[1])

    sorted_perimeter = perimeter_unsorted_pos[:, np.argsort(perimeter_unsorted_angle)].transpose() + \
        np.reshape(np.array(shape_mat.shape), (1, 2)) / 2
    return sorted_perimeter


def generate_print_path(printer, shape, pattern, show_plots=False, min_line_separation=0.9, sorting="consecutive"):
    boundaries = shape[1]
    shape_function = shape[0]
    x, y = generate_mesh(boundaries, printer)
    x_field, y_field = pattern(x, y)
    desired_shape = np.array(shape_function(x, y), dtype=np.int32)
    perimeter = find_perimeter(desired_shape)

    perimeter = np.array(perimeter, dtype=np.int32)
    filled = lin.generate_lines_class(x_field, y_field, desired_shape, perimeter, printer, min_line_separation, sorting)

    plt.figure(dpi=300)
    plt.imshow(filled)
    plt.show()
    if show_plots:
        plt.figure(figsize=(8, 8))
        plt.quiver(x, y, x_field, y_field)
        plt.show()

        # plt.imshow(desired_shape)
        # plt.show()
    return 0
