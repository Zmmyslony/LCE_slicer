import fields as fld
from perimeter import generate_print_path
from printer import standard_print
import shapes as shp

# generate_print_path(standard_print, shp.circle(40), fld.lines(math.pi * 1 / 4), show_plots=False)
generate_print_path(standard_print, shp.circle(25), fld.horizontal_lines, show_plots=False)