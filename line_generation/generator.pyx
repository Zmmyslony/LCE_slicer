#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport floor, ceil, round, sqrt
from printer import Printer
import matplotlib.pyplot as plt

# cimport line_generation.auxiliary as aux


cdef int MIN_SEGMENT_NUMBER = 5
cdef double ANGLE_THRESHOLD = 0.02
cdef double MIN_DISTANCE_COEFFICIENT = 0.5
cdef double NEW_LINE_SEPARATION = 1
cdef double MIN_LINE_SEPARATION = 1
cdef int NEIGHBOUR_THRESHOLD = 1

cdef double norm(double a, double b):
    cdef double result = sqrt(a ** 2 + b ** 2)
    return result


cdef (double, double) normalize(double a, double b):
    return a / norm(a, b), b / norm(a, b)


cdef int check_index(int x_size, int y_size, int x, int y):
    if 0 <= x < x_size and 0 <= y < y_size:
        return 1
    else:
        return 0


cdef int check_proximity(int[:, :] filled_elements, double distance, int x, int y, double sign=1, double threshold=0):
    cdef int limit = <int>ceil(distance)
    cdef int i = 0
    cdef int j = 0
    for i in range(-limit, limit):
        for j in range(-limit, limit):
            if check_index(filled_elements.shape[0], filled_elements.shape[1], x + i, y + j) and i ** 2 + j ** 2 < distance ** 2:
                if sign * filled_elements[x + i, y + j] > threshold:
                    return 0
    return 1


cdef int check_nearest_neighbours(int x_lower, int y_lower, int[:, :] empty_elements):
    cdef int up_left = empty_elements[x_lower, y_lower + 1]
    cdef int up_right = empty_elements[x_lower + 1, y_lower + 1]
    cdef int down_left = empty_elements[x_lower, y_lower]
    cdef int down_right = empty_elements[x_lower + 1, y_lower]
    return up_left + up_right + down_left + down_right


cdef int normalize_matrix(int[:, :] matrix):
    cdef int i = 0
    cdef int j = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    return 0

    
cdef (int, int) find_next_perimeter(int[:, :] perimeter, int[:, :] empty_elements, int[:, :] filled_elements,
                                    double[:, :] x_field, double[:, :] y_field, double line_width):
    cdef int i = 0
    cdef double distance_from_filled = 0
    cdef double vx = x_field[perimeter[0, 0], perimeter[0, 1]]
    cdef double vy = y_field[perimeter[0, 0], perimeter[0, 1]]
    cdef int x_filled = 0
    cdef int y_filled = 0

    for i in range(perimeter.shape[0]):
        if empty_elements[perimeter[i, 0], perimeter[i, 1]]:
            if check_proximity(filled_elements, line_width * NEW_LINE_SEPARATION, perimeter[i, 0], perimeter[i, 1]):
                return perimeter[i, 0], perimeter[i, 1]
        else:
            vx = x_field[perimeter[i, 0], perimeter[i, 1]]
            vy = y_field[perimeter[i, 0], perimeter[i, 1]]
            vx, vy = normalize(vx, vy)
            x_filled, y_filled = perimeter[i, 0], perimeter[i, 1]
            distance_from_filled = 0
    return 0, 0


cdef double get_average_vector(double[:, :] field, double x, double y):
    cdef int x_int = int(floor(x))
    cdef int y_int = int(floor(y))
    cdef double v = 0

    v += field[x_int, y_int] * (1 + x_int - x) * (1 + y_int - y)
    v += field[x_int, y_int + 1] * (1 + x_int - x) * (y - y_int)
    v += field[x_int + 1, y_int + 1] * (x - x_int) * (y - y_int)
    v += field[x_int + 1, y_int] * (x - x_int) * (1 + y_int - y)
    return v


#TODO make this function pretty
cdef (double, double) calculate_next_move(double x, double y, double vx_previous, double vy_previous,
                        double[:, :] x_field, double[:, :] y_field, int[:, :] empty_elements):
    cdef int x_int = int(floor(x))
    cdef int y_int = int(floor(y))

    if check_nearest_neighbours(x_int, y_int, empty_elements) < NEIGHBOUR_THRESHOLD:
        return 10, 0

    cdef double vx = get_average_vector(x_field, x, y)
    cdef double vy = get_average_vector(y_field, x, y)
    vx, vy = normalize(vx, vy)

    cdef double scalar_product = vx_previous * vx + vy_previous * vy
    if scalar_product < 0:
        return -vx, -vy
    else:
        return vx, vy


cdef int fill_nearest_neighbours(double x_current, double y_current, int[:, :] filled_elements,
                         double line_width, int x_size, int y_size):
    cdef int x_pos = <int>round(x_current)
    cdef int y_pos = <int>round(y_current)
    cdef int limit = <int>ceil(line_width)

    cdef int i = 0
    cdef int j = 0
    for i in range(-limit, limit):
        for j in range(-limit, limit):
            if (x_pos + i - x_current) ** 2 + (y_pos + j - y_current) ** 2 < line_width ** 2 and \
                    check_index(x_size, y_size, x_pos + i, y_pos + j):
                filled_elements[x_pos + i, y_pos + j] += 1
    return 0


cdef int check_moving_forward(double distance, double scalar_product, double line_width, double distance_coefficient,
                              int[:, :] filled_elements, double x_current, double y_current):
    cdef int x_pos = <int>round(x_current)
    cdef int y_pos = <int>round(y_current)

    if distance > 0 and ((scalar_product < 1 - ANGLE_THRESHOLD) or (distance >
        MIN_SEGMENT_NUMBER * line_width and distance_coefficient > MIN_DISTANCE_COEFFICIENT * line_width)):

        return 0
    elif not check_proximity(filled_elements, line_width * MIN_LINE_SEPARATION, x_pos, y_pos):
        return 0
    else:
        return 1

cdef (double, double) add_line(double x_current, double y_current, double x_start, double y_start, double vx, double vy,
                               double line_width, int[:, :] previously_filled_elements):

    cdef double distance_from_start = norm(x_current - x_start, y_current - y_start)
    cdef double vx_dir = (x_current - x_start) / distance_from_start
    cdef double vy_dir = (y_current - y_start) / distance_from_start

    cdef double scalar_product = vx * vx_dir + vy * vy_dir
    cdef double distance_from_nearest_grid = norm(round(x_current) - x_current, round(y_current) - y_current)
    cdef double distance_coefficient = distance_from_nearest_grid / distance_from_start

    if check_moving_forward(distance_from_start, scalar_product, line_width, distance_coefficient,
                            previously_filled_elements, x_current, y_current):
        return x_current + vx, y_current + vy
    else:
        return x_current, y_current



cdef int update_empty_spots(int[:, :] empty_elements, int[:, :] filled_elements):
    cdef int i = 0
    cdef int j = 0
    for i in range(empty_elements.shape[0]):
        for j in range(empty_elements.shape[1]):
            empty_elements[i, j] -= filled_elements[i, j]
    normalize_matrix(empty_elements)
    return 0


def generate_lines(np.ndarray[double, ndim=2] x_field, np.ndarray[double, ndim=2] y_field,
                   np.ndarray[int, ndim=2] desired_shape, np.ndarray[int, ndim=2] perimeter,
                   printer: Printer):
    cdef int[:, :] cperimeter = perimeter

    cdef double[:, :] cx_field = x_field
    cdef double[:, :] cy_field = y_field
    cdef int[:, :] empty_elements = desired_shape
    cdef int[:, :] filled_elements = np.zeros_like(desired_shape)
    cdef int[:, :] previously_filled_elements = np.zeros_like(desired_shape)

    # TODO write line creation algorithm starting from perimeter
    # TODO write empty spot finding algorithm that goes in both directions
    # TODO return list of lines
    cdef double line_width = printer.nozzle.line_width / printer.accuracy
    cdef int x_size = x_field.shape[0]
    cdef int y_size = y_field.shape[0]

    cdef int x_start = 0
    cdef int y_start = 0
    cdef double x_pos = 0
    cdef double y_pos = 0
    cdef double x_new = 0
    cdef double y_new = 0
    cdef double vx = 0
    cdef double vy = 0
    cdef int iterator = 0

    while True:
        x_start, y_start = find_next_perimeter(cperimeter, empty_elements, filled_elements, cx_field, cy_field, line_width)
        if x_start == 0 and y_start == 0:
            break
        x_pos = x_start
        y_pos = y_start
        x_new = 0
        y_new = 0
        # vx = cx_field[x_start, y_start]
        # vy = cy_field[x_start, y_start]
        vx = empty_elements.shape[0] / 2 - x_pos
        vy = empty_elements.shape[1] / 2 - y_pos

        while empty_elements[<int>x_pos, <int>y_pos]:
            vx, vy = calculate_next_move(x_pos, y_pos, vx, vy, cx_field, cy_field, empty_elements)
            if vx == 10:
                fill_nearest_neighbours(x_pos, y_pos, filled_elements, line_width, x_size, y_size)
                break

            x_new, y_new = add_line(x_pos, y_pos, x_start, y_start, vx, vy, line_width, previously_filled_elements)
            fill_nearest_neighbours(x_new, y_new, filled_elements, line_width, x_size, y_size)

            if x_new != x_pos or y_new != y_pos:
                x_pos = x_new
                y_pos = y_new
            else:
                break

        update_empty_spots(empty_elements, filled_elements)
        iterator += 1
        previously_filled_elements = filled_elements
    plt.imshow(filled_elements)
    plt.show()


    return 0

# TODO write line sorter based on closest starting or ending line