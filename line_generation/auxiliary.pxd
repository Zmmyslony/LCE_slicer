#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport floor, ceil, round, sqrt

cdef int MIN_SEGMENT_NUMBER = 5
cdef double ANGLE_THRESHOLD = 0.02
cdef double MIN_DISTANCE_COEFFICIENT = 0.5
cdef double MIN_LINE_SEPARATION = 0.9


cdef inline double norm(double a, double b):
    cdef double result = sqrt(a ** 2 + b ** 2)
    return result


cdef inline (double, double) normalize(double a, double b):
    return a / norm(a, b), b / norm(a, b)


cdef inline int check_index(int x_size, int y_size, int x, int y):
    if 0 <= x < x_size and 0 <= y < y_size:
        return 1
    else:
        return 0


cdef inline int check_proximity(int[:, :] filled_elements, double distance, int x, int y):
    cdef int limit = <int>ceil(distance)
    cdef int i = 0
    cdef int j = 0
    for i in range(-limit, limit):
        for j in range(-limit, limit):
            if check_index(filled_elements.shape[0], filled_elements.shape[1], x + i, y + j) and i ** 2 + j ** 2 < distance ** 2:
                if filled_elements[x + i, y + j] > 0:
                    return 0
    return 1


cdef inline int check_nearest_neighbours(int x_lower, int y_lower, int[:, :] empty_elements):
    cdef int up_left = empty_elements[x_lower, y_lower + 1]
    cdef int up_right = empty_elements[x_lower + 1, y_lower + 1]
    cdef int down_left = empty_elements[x_lower, y_lower]
    cdef int down_right = empty_elements[x_lower + 1, y_lower]
    return up_left + up_right + down_left + down_right


cdef inline int normalize_matrix(int[:, :] matrix):
    cdef int i = 0
    cdef int j = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    return 0


cdef inline int check_moving_forward(double distance, double scalar_product, double line_width, double distance_coefficient,
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