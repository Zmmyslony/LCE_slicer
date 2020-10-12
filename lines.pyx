from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport floor, ceil, round, sqrt
from printer import Printer
import matplotlib.pyplot as plt


cdef int MIN_SEGMENT_NUMBER = 5
cdef double ANGLE_THRESHOLD = 0.02
cdef double MIN_DISTANCE_COEFFICIENT = 0.5
cdef double LINE_SEPARATION = 0.8


@cython.boundscheck(False)
@cython.wraparound(False)
cdef (int, int) find_next_perimeter(int[:, :] perimeter, int[:, :] empty_elements, double[:, :] x_field,
                                    double[:, :] y_field, float line_width):
    cdef int i = 0
    cdef int max_i = perimeter.shape[0]
    cdef float distance_from_filled = 0
    cdef float vx = 0
    cdef float vy = 0
    cdef int x_filled = 0
    cdef int y_filled = 0

    for i in range(max_i):
        if empty_elements[perimeter[i, 0], perimeter[i, 1]]:
            distance_from_filled += 1
            if abs(distance_from_filled) > LINE_SEPARATION * line_width:
                return perimeter[i, 0], perimeter[i, 1]
        else:
            vx = x_field[perimeter[i, 0], perimeter[i, 1]]
            vy = y_field[perimeter[i, 0], perimeter[i, 1]]
            x_filled, y_filled = perimeter[i, 0], perimeter[i, 1]
    return 0, 0


cdef double norm(double a, double b):
    result = sqrt(a ** 2 + b ** 2)
    return result

@cython.cdivision(True)
cdef (double, double) normalize(double a, double b):
    return a / norm(a, b), b / norm(a, b)


@cython.boundscheck(False)
cdef int check_index(int x_size, int y_size, int x, int y):
    if 0 <= x < x_size and 0 <= y < y_size:
        return 1
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
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
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef (double, double) calculate_next_move(double x, double y, double vx_previous, double vy_previous,
                        double[:, :] x_field, double[:, :] y_field, int[:, :] empty_elements):
    cdef int x_int = int(floor(x))
    cdef int y_int = int(floor(y))

    if check_neighbours(x_int, y_int, empty_elements) < 1:
        return 10, 0

    cdef double vx = get_average_vector(x_field, x, y)
    cdef double vy = get_average_vector(y_field, x, y)
    cdef double scalar_product = vx_previous * vx + vy_previous * vy
    vx, vy = normalize(vx, vy)

    if scalar_product < 0:
        return -vx, -vy
    else:
        return vx, vy


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef fill_nearest_neighbours(double x_current, double y_current, int[:, :] filled_elements,
                         double line_width, int x_size, int y_size):
    cdef int i = 0
    cdef int j = 0

    cdef int x_pos = <int>round(x_current)
    cdef int y_pos = <int>round(y_current)
    cdef int limit = <int>ceil(line_width)
    for i in range(-limit - 1, limit + 2):
        for j in range(- limit - 1, limit + 2):
            if (x_pos + i - x_current) ** 2 + (y_pos + j - y_current) ** 2 < line_width ** 2 and \
                    check_index(x_size, y_size, x_pos + i, y_pos + j):
                filled_elements[x_pos + i, y_pos + j] += 1



@cython.boundscheck(False)
cdef int check_neighbours(int x_lower, int y_lower, int[:, :] empty_elements):
    cdef int up_left = empty_elements[x_lower, y_lower + 1]
    cdef int up_right = empty_elements[x_lower + 1, y_lower + 1]
    cdef int down_left = empty_elements[x_lower, y_lower]
    cdef int down_right = empty_elements[x_lower + 1, y_lower]
    return up_left + up_right + down_left + down_right


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef (double, double) add_line(double x_current, double y_current, double x_start, double y_start, double vx, double vy,
                               double line_width):

    cdef double distance_from_start = norm(x_current - x_start, y_current - y_start)
    cdef double vx_tot = (x_current - x_start) / distance_from_start
    cdef double vy_tot = (y_current - y_start) / distance_from_start
    cdef double scalar_product = vx * vx_tot + vy * vy_tot
    cdef double distance_from_nearest_grid = norm(round(x_current) - x_current, round(y_current) - y_current)
    cdef double distance_coefficient = distance_from_nearest_grid / distance_from_start

    if distance_from_start > 0 and ((scalar_product < 1 - ANGLE_THRESHOLD) or (distance_from_start >
        MIN_SEGMENT_NUMBER * line_width and distance_coefficient > MIN_DISTANCE_COEFFICIENT * line_width)):
        return x_current, y_current
    else:
        return x_current + vx, y_current + vy


@cython.boundscheck(False)
@cython.wraparound(False)
cdef normalize_matrix(int[:, :] matrix):
    cdef int i = 0
    cdef int j = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef update_empty_spots(int[:, :] empty_elements, int[:, :] filled_elements):
    cdef int i = 0
    cdef int j = 0
    for i in range(empty_elements.shape[0]):
        for j in range(empty_elements.shape[1]):
            empty_elements[i, j] -= filled_elements[i, j]
    normalize_matrix(empty_elements)


@cython.boundscheck(False)
@cython.wraparound(False)
def generate_lines(np.ndarray[double, ndim=2] x_field, np.ndarray[double, ndim=2] y_field,
                   np.ndarray[int, ndim=2] desired_shape, np.ndarray[int, ndim=2] perimeter,
                   printer: Printer):
    cdef int[:, :] cperimeter = perimeter

    cdef double[:, :] cx_field = x_field
    cdef double[:, :] cy_field = y_field
    cdef int[:, :] empty_elements = desired_shape
    cdef int[:, :] filled_elements = np.zeros_like(desired_shape)

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
    while True:
        x_start, y_start = find_next_perimeter(cperimeter, empty_elements, cx_field, cy_field, line_width)
        if x_start == 0 and y_start == 0:
            break
        x_pos = x_start
        y_pos = y_start
        x_new = 0
        y_new = 0
        vx = cx_field[x_start, y_start]
        vy = cy_field[x_start, y_start]

        while empty_elements[<int>x_pos, <int>y_pos]:
            vx, vy = calculate_next_move(x_pos, y_pos, vx, vy, cx_field, cy_field, empty_elements)

            x_new, y_new = add_line(x_pos, y_pos, x_start, y_start, vx, vy, line_width)
            if x_new != x_pos or y_new != y_pos:
                fill_nearest_neighbours(x_new, y_new, filled_elements, line_width, x_size, y_size)
                x_pos = x_new
                y_pos = y_new
            else:
                break

        update_empty_spots(empty_elements, filled_elements)
    plt.imshow(filled_elements)
    plt.show()


    return 0

# TODO write line sorter based on closest starting or ending line