import fields as fld
from perimeter import generate_print_path
from printer import standard_print
import shapes as shp
import math
import time


if True:
    for j in range(10):
        for i in [0.04]:
            standard_print.accuracy = i
            start_time = time.time()
            generate_print_path(standard_print, shp.circle(15), fld.normalized_spiral(math.pi * j/20),
                                min_line_separation=0, sorting="consecutive")
            times = time.time() - start_time
            print(f"Accuracy: {i} \tTime: {times:.2f}s")
