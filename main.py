import fields as fld
from perimeter import generate_print_path
from printer import standard_print
import shapes as shp
import math

generate_print_path(standard_print, shp.circle(25), fld.normalized_radial, show_plots=False,
                    min_line_separation=0.0, sorting="opposite")
generate_print_path(standard_print, shp.circle(25), fld.normalized_radial, show_plots=False)
generate_print_path(standard_print, shp.circle(25), fld.horizontal_lines, show_plots=False)

