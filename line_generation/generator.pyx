#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
from __future__ import print_function
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil, round, sqrt, log
from printer import Printer


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
    cdef int x_size
    cdef int y_size

    cdef double min_line_separation
    cdef double angle_threshold
    cdef double min_distance_coefficient
    cdef double new_line_separation
    cdef double line_width

    cdef int[:, :] perimeter
    cdef int[:, :] empty_elements
    cdef int[:, :] filled_elements
    cdef int[:, :] previously_filled_elements

    cdef double[:, :] x_field
    cdef double[:, :] y_field

    cdef int x_start
    cdef int y_start
    cdef int last_index
    cdef double x_pos
    cdef double y_pos
    cdef double x_new
    cdef double y_new
    cdef double vx
    cdef double vy
    cdef str sorting
    cdef int base

    def __init__(self, np.ndarray[int, ndim=2] perimeter, np.ndarray[double, ndim=2] x_field,
                 np.ndarray[double, ndim=2]  y_field, np.ndarray[int, ndim=2] desired_shape, printer, sorting,
                 int min_segment_number=5, int neighbour_threshold=1, double angle_threshold=0.02,
                 double new_line_separation=1, double min_line_separation=1):

        self.min_segment_number = min_segment_number
        self.neighbour_threshold = neighbour_threshold
        self.angle_threshold = angle_threshold
        self.line_width = printer.nozzle.line_width / printer.accuracy

        self.new_line_separation = new_line_separation
        self.min_line_separation = min_line_separation
        self.perimeter = perimeter
        # self.perimeter = perimeter[::(<int>self.line_width)]
        self.x_field = x_field
        self.y_field = y_field

        self.empty_elements = desired_shape
        self.filled_elements = np.zeros_like(desired_shape)
        self.previously_filled_elements = np.zeros_like(desired_shape)

        self.sorting = sorting

        self.x_size = x_field.shape[0]
        self.y_size = y_field.shape[1]
        self.base = find_base(self.perimeter.shape[0])

        self.x_start = 0
        self.y_start = 0
        self.last_index = 0
        self.x_pos = 0
        self.y_pos = 0
        self.x_new = 0
        self.y_new = 0
        self.vx = 0
        self.vy = 0

    cdef (int, int) find_next_perimeter(self):
        cdef int i = 0
        cdef int x = 0
        cdef int y = 0
        cdef bint found_spot = False

        if self.sorting == "opposite":
            i = self.generate_index()
            self.last_index += 1
            x, y = self.find_empty_spot(i)
        elif self.sorting == "consecutive":
            x, y = self.find_empty_spot(0)
        return x, y


    cdef (double, double) calculate_next_move(self):
        if check_nearest_neighbours(<int>self.x_pos, <int>self.y_pos, self.empty_elements) < NEIGHBOUR_THRESHOLD:
            return 10, 0

        cdef double vx = bilinear_interpolation(self.x_field, self.x_pos, self.y_pos)
        cdef double vy = bilinear_interpolation(self.y_field, self.x_pos, self.y_pos)
        # TODO investigate why this has to be swapped for it to work
        vy, vx = normalize(vx, vy)

        cdef double scalar_product = self.vx * vx + self.vy * vy
        if scalar_product < 0:
            return -vx, -vy
        else:
            return vx, vy


    cdef (double, double) add_line(self):
        cdef double distance_from_start = norm(self.x_pos - self.x_start, self.y_pos - self.y_start)
        cdef double vx_dir = (self.x_pos - self.x_start) / distance_from_start
        cdef double vy_dir = (self.y_pos - self.y_start) / distance_from_start

        cdef double scalar_product = self.vx * vx_dir + self.vy * vy_dir
        cdef double distance_from_nearest_grid = norm(round(self.x_pos) - self.x_pos, round(self.y_pos) - self.y_pos)
        cdef double distance_coefficient = distance_from_nearest_grid / distance_from_start

        if check_moving_forward(distance_from_start, scalar_product, self.line_width, distance_coefficient,
                                self.previously_filled_elements, self.x_pos, self.y_pos, self.min_line_separation):
            return self.x_pos + self.vx, self.y_pos + self.vy
        else:
            return self.x_pos, self.y_pos


    cdef void update_empty_elements(self):
        cdef int i
        cdef int j
        for i in range(self.x_size):
            for j in range(self.y_size):
                self.empty_elements[i, j] -= self.filled_elements[i, j]
        make_binary(self.empty_elements)


    cdef int generate_index(self):
        cdef int exponent = 0
        if self.last_index < self.base:
            return <int>(self.perimeter.shape[0] / self.base * self.last_index)
        else:
            exponent = <int>(log(self.last_index / self.base) / log(2))
            return <int>(self.perimeter.shape[0] / 2 ** exponent * (self.last_index - self.base * 2 ** exponent))


    cdef (int, int) find_empty_spot(self, int starting_index):
        cdef int j
        cdef int i
        for i in range(self.perimeter.shape[0]):
            j = symmetric_index(starting_index, self.perimeter.shape[0], i)
            if self.empty_elements[self.perimeter[j, 0], self.perimeter[j, 1]]:
                if check_proximity(self.filled_elements, self.line_width * self.new_line_separation, self.perimeter[j, 0],
                                   self.perimeter[j, 1]):
                    return self.perimeter[j, 0], self.perimeter[j, 1]
        return 0, 0


cdef int symmetric_index(int starting_index, int size, int i):
    return (starting_index + <int>(i / 2) * (-1) ** (i % 2) + size) % size


cdef double norm(double a, double b):
    cdef double result = sqrt(a ** 2 + b ** 2)
    return result


cdef (double, double) normalize(double a, double b):
    cdef double normalization_factor = norm(a, b)
    cdef double a_normalized = a / normalization_factor
    cdef double b_normalized = b / normalization_factor
    return a_normalized, b_normalized


cdef bint check_index(int x_size, int y_size, int x, int y):
    if 0 <= x < x_size and 0 <= y < y_size:
        return 1
    else:
        return 0


cdef bint check_proximity(int[:, :] filled_elements, double distance, int x, int y, double sign=1, double threshold=0):
    cdef int limit = <int>ceil(distance)
    cdef int i = 0
    cdef int j = 0
    if distance > 0:
        for i in range(-limit, limit):
            for j in range(-limit, limit):
                if check_index(filled_elements.shape[0], filled_elements.shape[1], x + i, y + j) and \
                        i ** 2 + j ** 2 < distance ** 2:
                    if sign * filled_elements[x + i, y + j] > sign * threshold:
                        return 0
    return 1


cdef int check_nearest_neighbours(int x_lower, int y_lower, int[:, :] empty_elements):
    cdef int up_left = empty_elements[x_lower, y_lower + 1]
    cdef int up_right = empty_elements[x_lower + 1, y_lower + 1]
    cdef int down_left = empty_elements[x_lower, y_lower]
    cdef int down_right = empty_elements[x_lower + 1, y_lower]
    return up_left + up_right + down_left + down_right


cdef void make_binary(int[:, :] matrix, int low=0, int high=1, double threshold=0):
    cdef int i = 0
    cdef int j = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > threshold:
                matrix[i, j] = high
            else:
                matrix[i, j] = low


cdef (double, int) get_closeness(int size, int power):
    cdef int exponent = <int>(log(size) / log(2))
    cdef int base = <int>(size / 2 ** (exponent - power))
    cdef double closeness = size / <double>(base * 2 ** (exponent - power)) - 1
    return closeness, base


cdef int find_base(int size):
    cdef int exponent = <int>(log(size) / log(2))
    cdef double closeness = 1
    cdef double closeness_temp
    cdef int base = 1
    cdef int base_temp = 1
    cdef int j = 0
    for j in range(1, 2):
        closeness_temp, base_temp = get_closeness(size, j)
        if closeness_temp < closeness:
            closeness = closeness_temp
            base = base_temp
    return base


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


cdef int check_moving_forward(double distance, double scalar_product, double line_width, double distance_coefficient,
                              int[:, :] filled_elements, double x_current, double y_current, double min_line_separation):
    cdef int x_pos = <int>round(x_current)
    cdef int y_pos = <int>round(y_current)
    if not check_proximity(filled_elements, line_width * min_line_separation, x_pos, y_pos):
        return 0
    else:
        return 1


cdef bint fill_nearest_neighbours(double x_current, double y_current, int[:, :] filled_elements,
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


def generate_lines_class(np.ndarray[double, ndim=2] x_field, np.ndarray[double, ndim=2] y_field,
                   np.ndarray[int, ndim=2] desired_shape, np.ndarray[int, ndim=2] perimeter,
                   printer: Printer, double min_line_separation, str sorting):
    cdef Slicer slicer = Slicer(perimeter, x_field, y_field, desired_shape, printer, sorting,
                                min_line_separation=min_line_separation)

    while True:
        slicer.x_start, slicer.y_start = slicer.find_next_perimeter()

        if slicer.x_start == 0 and slicer.y_start == 0:
            break
        slicer.x_pos = slicer.x_start
        slicer.y_pos = slicer.y_start
        slicer.x_new = 0
        slicer.y_new = 0

        slicer.vx = slicer.x_size / 2 - slicer.x_pos
        slicer.vy = slicer.y_size / 2 - slicer.y_pos
        slicer.previously_filled_elements[:] = slicer.filled_elements

        while True:
            slicer.vx, slicer.vy = slicer.calculate_next_move()
            if slicer.vx == 10:
                fill_nearest_neighbours(slicer.x_pos, slicer.y_pos, slicer.filled_elements, slicer.line_width,
                                        slicer.x_size, slicer.y_size)
                break

            slicer.x_new, slicer.y_new = slicer.add_line()
            fill_nearest_neighbours(slicer.x_new, slicer.y_new, slicer.filled_elements, slicer.line_width,
                                    slicer.x_size, slicer.y_size)

            if slicer.x_new != slicer.x_pos or slicer.y_new != slicer.y_pos:
                slicer.x_pos = slicer.x_new
                slicer.y_pos = slicer.y_new
            else:
                break
            if norm(slicer.x_pos - slicer.x_size / 2, slicer.y_pos - slicer.y_size / 2) < slicer.line_width:
                break
        slicer.update_empty_elements()
    return slicer.filled_elements

# TODO write line sorter based on closest starting or ending line