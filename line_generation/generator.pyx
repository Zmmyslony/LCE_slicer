#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from __future__ import print_function
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil, round, sqrt, log
from printer import Printer
import matplotlib.pyplot as plt


#TODO make the algorithm work irregardless of the starting angle
#TODO fix random empty lines in between lines
cdef int MIN_SEGMENT_NUMBER = 5
cdef double ANGLE_THRESHOLD = 0.02
cdef double MIN_DISTANCE_COEFFICIENT = 0.5
cdef double NEW_LINE_SEPARATION = 1
cdef int NEIGHBOUR_THRESHOLD = 1

cdef int LOG_BASES_SIZE = 7
cdef int LOG_BASES[7]
LOG_BASES[:] = [2, 3, 5, 7, 11, 13, 17]

cdef class Slicer:
    cdef int min_segment_number
    cdef int neighbour_threshold
    cdef double angle_threshold
    cdef double min_distance_coefficient
    cdef double new_line_separation

    def __init__(self, int min_segment_number=5, int neighbour_threshold=1, double angle_threshold=0.02,
             double min_distance_coefficient=0.5, double new_line_separation=1):
        self.min_segment_number = min_segment_number
        self.neighbour_threshold = neighbour_threshold
        self.angle_threshold = angle_threshold
        self.min_distance_coefficient = min_distance_coefficient
        self.new_line_separation = new_line_separation


cdef double norm(double a, double b):
    cdef double result = sqrt(a ** 2 + b ** 2)
    return result


cdef (double, double) normalize(double a, double b):
    cdef double normalization_factor = norm(a, b)
    cdef double a_normalized = a / normalization_factor
    cdef double b_normalized = b / normalization_factor
    return a_normalized, b_normalized


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


cdef int make_binary(int[:, :] matrix, int low=0, int high=1, double threshold=0):
    cdef int i = 0
    cdef int j = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > threshold:
                matrix[i, j] = high
            else:
                matrix[i, j] = low
    return 0


cdef double decimal_part(double val):
    return val - <int>val


cdef int index_generator(int i, int size):
    cdef int base = 0
    cdef int exponent = 0
    cdef double closeness = 1
    cdef int j = 0
    cdef int current_base = 0
    size -= 1

    #TODO optimize the python out of here
    for j in range(LOG_BASES_SIZE): #[2, 3, 5, 7, 11, 13, 17]:
        current_base = LOG_BASES[j]
        if decimal_part(log(size) / log(current_base)) < closeness:
            closeness = decimal_part(log(size) / log(current_base))
            base = current_base
    base = 2
    if i <= base:
        return <int>(size / base * i)
    else:
        exponent = <int>(log(i) / log(base))
        return <int>(size / base ** exponent * (i - base ** exponent) + size / base ** exponent)


cdef (bint, int, int) find_empty_spot(int[:, :] perimeter, int[:, :] empty_elements, int[:, :] filled_elements,
                                           double line_width, int starting_index, int ending_index, int step=1):
    cdef int i = 0
    cdef int j = 0
    cdef int rng = <int>(abs(starting_index - ending_index) / step)
    for i in range(rng):
        j = starting_index + step * i
        if empty_elements[perimeter[j, 0], perimeter[j, 1]]:
            if check_proximity(filled_elements, line_width * NEW_LINE_SEPARATION, perimeter[j, 0], perimeter[j, 1]):
                return True, perimeter[j, 0], perimeter[j, 1]

    return False, 0, 0


cdef (int, int) find_next_perimeter(int[:, :] perimeter, int[:, :] empty_elements, int[:, :] filled_elements,
                                    double[:, :] x_field, double[:, :] y_field, double line_width, int* last_index,
                                    str sorting):
    cdef int i = 0
    cdef int x = 0
    cdef int y = 0
    cdef bint found_spot = False

    if sorting == "opposite":
        i = index_generator(last_index[0], perimeter.shape[0])
        last_index[0] += 1

        found_spot, x, y = find_empty_spot(perimeter, empty_elements, filled_elements, line_width, i, perimeter.shape[0])
        if not found_spot:
            found_spot, x, y = find_empty_spot(perimeter, empty_elements, filled_elements, line_width, i, 0, step=-1)
        elif not found_spot:
            found_spot, x, y = find_empty_spot(perimeter, empty_elements, filled_elements, line_width, 0,
                                               perimeter.shape[0])
    elif sorting == "consecutive":
        found_spot, x, y = find_empty_spot(perimeter, empty_elements, filled_elements, line_width, 0,
                                               perimeter.shape[0])
    return x, y


cdef double bilinear_interpolation(double[:, :] field, double x, double y):
    cdef int x_1 = int(floor(x))
    cdef int x_2 = x_1 + 1
    cdef int y_1 = int(floor(y))
    cdef int y_2 = y_1 + 1
    cdef double v = 0

    v += field[x_1, y_1] * (x_2 - x) * (y_2 - y)
    v += field[x_1, y_2] * (x_2 - x) * (y - y_1)
    v += field[x_2, y_1] * (x - x_1) * (y_2 - y)
    v += field[x_2, y_2] * (x - x_1) * (y - y_1)
    return v


cdef (double, double) calculate_next_move(double x, double y, double vx_previous, double vy_previous,
                        double[:, :] x_field, double[:, :] y_field, int[:, :] empty_elements):

    if check_nearest_neighbours(<int>x, <int>y, empty_elements) < NEIGHBOUR_THRESHOLD:
        return 10, 0

    cdef double vx = bilinear_interpolation(x_field, x, y)
    cdef double vy = bilinear_interpolation(y_field, x, y)
    # TODO investigate why this has to be swapped for it to work
    vy, vx = normalize(vx, vy)

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
                              int[:, :] filled_elements, double x_current, double y_current, double min_line_separation):
    cdef int x_pos = <int>round(x_current)
    cdef int y_pos = <int>round(y_current)

    # if distance > 0 and ((scalar_product < 1 - ANGLE_THRESHOLD) or (distance >
    #     MIN_SEGMENT_NUMBER * line_width and distance_coefficient > MIN_DISTANCE_COEFFICIENT * line_width)):
    #
    #     return 0
    # elif not check_proximity(filled_elements, line_width * min_line_separation, x_pos, y_pos):
    if not check_proximity(filled_elements, line_width * min_line_separation, x_pos, y_pos):
        return 0
    else:
        return 1


cdef (double, double) add_line(double x_current, double y_current, double x_start, double y_start, double vx, double vy,
                               double line_width, int[:, :] previously_filled_elements, double min_line_separation):

    cdef double distance_from_start = norm(x_current - x_start, y_current - y_start)
    cdef double vx_dir = (x_current - x_start) / distance_from_start
    cdef double vy_dir = (y_current - y_start) / distance_from_start

    cdef double scalar_product = vx * vx_dir + vy * vy_dir
    cdef double distance_from_nearest_grid = norm(round(x_current) - x_current, round(y_current) - y_current)
    cdef double distance_coefficient = distance_from_nearest_grid / distance_from_start

    if check_moving_forward(distance_from_start, scalar_product, line_width, distance_coefficient,
                            previously_filled_elements, x_current, y_current, min_line_separation):
        return x_current + vx, y_current + vy
    else:
        return x_current, y_current



cdef int update_empty_elements(int[:, :] empty_elements, int[:, :] filled_elements):
    cdef int i = 0
    cdef int j = 0
    for i in range(empty_elements.shape[0]):
        for j in range(empty_elements.shape[1]):
            empty_elements[i, j] -= filled_elements[i, j]
    make_binary(empty_elements)
    return 0


def generate_lines(np.ndarray[double, ndim=2] x_field, np.ndarray[double, ndim=2] y_field,
                   np.ndarray[int, ndim=2] desired_shape, np.ndarray[int, ndim=2] perimeter,
                   printer: Printer, double min_line_separation, str sorting):
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

    cdef int last_index = 0

    while True:
        x_start, y_start = find_next_perimeter(cperimeter, empty_elements, filled_elements, cx_field, cy_field,
                                               line_width, &last_index, sorting)
        if x_start == 0 and y_start == 0:
            break
        x_pos = x_start
        y_pos = y_start
        x_new = 0
        y_new = 0

        vx = empty_elements.shape[0] / 2 - x_pos
        vy = empty_elements.shape[1] / 2 - y_pos
        previously_filled_elements[:] = filled_elements


        while empty_elements[<int>x_pos, <int>y_pos]:
            vx, vy = calculate_next_move(x_pos, y_pos, vx, vy, cx_field, cy_field, empty_elements)
            if vx == 10:
                fill_nearest_neighbours(x_pos, y_pos, filled_elements, line_width, x_size, y_size)
                break

            x_new, y_new = add_line(x_pos, y_pos, x_start, y_start, vx, vy, line_width, previously_filled_elements,
                                    min_line_separation)
            fill_nearest_neighbours(x_new, y_new, filled_elements, line_width, x_size, y_size)

            if x_new != x_pos or y_new != y_pos:
                x_pos = x_new
                y_pos = y_new
            else:
                break

        update_empty_elements(empty_elements, filled_elements)

        # plt.imshow(previously_filled_elements)
        # plt.show()
    plt.imshow(filled_elements)
    plt.show()

    return 0

# TODO write line sorter based on closest starting or ending line