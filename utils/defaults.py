from utils.types import (
    type_math_complex,
    type_math_float,
    type_math_int,
    type_enum_int,
)
from fractal.colors import Normalization_Mode, Palette_Mode
from fractal.fractal import Fractal_Mode
from fractal.palette import palettes_definitions

# fractal variables
xmin = type_math_float(-2.5)
xmax = type_math_float(1.5)
ymin = type_math_float(-1.5)
ymax = type_math_float(1.5)
xcenter = type_math_float(-0.5)
ycenter = type_math_float(0)
yheight = type_math_float(3)
max_iterations = type_math_int(1000)
power = type_math_int(2)
escape_radius = type_math_int(4)
epsilon = type_math_float(0.001)
fractal_mode = type_enum_int(Fractal_Mode.MANDELBROT)
juliaxy = type_math_complex(0 + 0j)

# color variables
palette_mode = type_enum_int(Palette_Mode.HUE)
normalization_mode = type_enum_int(Normalization_Mode.ITER_NORMALIZED)
palette_width = type_math_float(1.0)
palette_shift = type_math_float(0.0)
custom_palette_name = list(palettes_definitions.keys())[0]

# UI variables
show_info = True
