import numpy as np
from printer import Printer


def circle(r):
    def shape(x: np.ndarray, y: np.ndarray):
        mat = np.zeros_like(x)
        mat[np.where(x ** 2 + y ** 2 <= r ** 2)] = 1
        return mat

    boundaries = [-r, r, -r, r]
    return shape, boundaries


def rectangle(a, b):
    def shape(x, y):
        mat = np.zeros_like(x)
        mat[np.where(np.logical_and(np.logical_and(x > -a / 2, x < a / 2), np.logical_and(y > -b / 2, y < b / 2)))] = 1
        return mat

    boundaries = [-a / 2, a / 2, -b / 2, b / 2]
    return shape, boundaries


def square(a):
    return rectangle(a, a)


def generate_mesh(boundaries, printer: Printer):
    x_axis = np.arange(boundaries[0] - 2 * printer.accuracy, boundaries[1] + 2 * printer.accuracy, printer.accuracy)
    y_axis = np.arange(boundaries[2] - 2 * printer.accuracy, boundaries[3] + 2 * printer.accuracy, printer.accuracy)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    return x_grid, y_grid

# TODO add more shapes
