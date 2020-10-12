import fields as fld
from perimeter import generate_print_path
from printer import standard_print
import shapes as shp
import math

generate_print_path(standard_print, shp.circle(25), fld.vertical_lines, show_plots=False)
