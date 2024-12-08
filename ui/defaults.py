from utils.types import (
    type_math_complex,
    type_math_float,
    type_math_int,
    type_enum_int,
)
from fractal.colors import K_Mode, Palette_Mode
from fractal.fractal import Fractal_Mode
from fractal.palette import palettes_definitions

# Fractal
default_xcenter = type_math_float(-0.5)
default_ycenter = type_math_float(0)
default_yheight = type_math_float(3)
default_max_iterations = type_math_int(1000)
default_power = type_math_int(2)
default_escape_radius = type_math_int(4)
default_epsilon = type_math_float(0.001)
default_fractal_mode = type_enum_int(Fractal_Mode.MANDELBROT)
default_juliaxy = type_math_complex(0 + 0j)

# color variables
default_palette_mode = type_enum_int(Palette_Mode.HUE)
default_k_mode = type_enum_int(K_Mode.ITER_WAVES)
default_color_waves = type_math_int(2)
default_custom_palette_name = list(palettes_definitions.keys())[0]
default_palette_shift=type_math_int(0)

# UI variables
default_show_info = True

# Const
default_ZOOM_RATE = 2
default_PAN_SPEED = 0.3  # ratio of xmax-xmin
default_DISPLAY_HEIGTH = 1024
default_DISPLAY_RATIO = 4 / 3