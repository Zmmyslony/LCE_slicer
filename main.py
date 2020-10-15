import fields as fld
from perimeter import generate_print_path
from printer import standard_print
import shapes as shp
import math

for i in [0.04]:
    standard_print.accuracy = i
    generate_print_path(standard_print, shp.circle(25), fld.normalized_radial,
                        min_line_separation=0, sorting="opposite")
# generate_print_path(standard_print, shp.circle(25), fld.normalized_radial)
# generate_print_path(standard_print, shp.circle(25), fld.horizontal_lines)
# generate_print_path(standard_print, shp.circle(25), fld.lines(math.pi * 1/4), min_line_separation=0.8)
