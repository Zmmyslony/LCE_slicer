import numpy as np


def spiral(angle):
    def field(x: np.ndarray, y: np.ndarray):
        x_field = np.cos(angle) * x - np.sin(angle) * y
        y_field = np.sin(angle) * x + np.cos(angle) * y
        return x_field, y_field
    return field


radial = spiral(0)
concentric = spiral(np.pi / 2)


def normalized_spiral(angle):
    def field(x: np.ndarray, y: np.ndarray):
        x_field = np.cos(angle) * x - np.sin(angle) * y
        y_field = np.sin(angle) * x + np.cos(angle) * y
        norm = np.sqrt(x_field ** 2 + y_field ** 2)
        norm[np.where(norm == 0)] = 1

        return x_field / norm, y_field / norm
    return field


normalized_radial = normalized_spiral(0)
normalized_concentric = normalized_spiral(np.pi / 2)


def lines(angle):
    def field(x: np.ndarray, y: np.ndarray):
        x_field = np.zeros_like(x)
        y_field = np.zeros_like(y)
        x_field += np.cos(angle)
        y_field += np.sin(angle)
        return x_field, y_field
    return field


horizontal_lines = lines(0)
vertical_lines = lines(np.pi / 2)


def archimedean_spiral(line_width):
    return spiral(np.arctan(2 * np.pi / line_width))


def normalized_archimedean_spiral(line_width):
    return normalized_spiral(np.arctan(2 * np.pi / line_width))


# TODO add more fields
