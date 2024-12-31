import math
from dataclasses import dataclass
from utils.types import (
    type_math_complex,
    type_math_float,
    type_math_int,
    type_enum_int,
)
from fractal.colors import K_Mode, Palette_Mode
from fractal.fractal import Fractal_Mode
from fractal.palette import palettes_definitions
from utils import defaults
from utils import const


@dataclass
class AppState:
    def __init__(self):
        # uses default values from defaults.py
        # fractal variables
        self.xcenter = defaults.xcenter
        self.ycenter = defaults.ycenter
        self.yheight = defaults.yheight
        self.max_iterations = defaults.max_iterations
        self.power = defaults.power
        self.escape_radius = defaults.escape_radius
        self.epsilon = defaults.epsilon
        self.fractal_mode = defaults.fractal_mode
        self.juliaxy = defaults.juliaxy

        # color variables
        self.palette_mode = defaults.palette_mode
        self.k_mode = defaults.k_mode
        self.color_waves = defaults.color_waves
        self.custom_palette_name = defaults.custom_palette_name
        self.palette_shift = defaults.palette_shift

        # UI variables
        self.show_info = defaults.show_info

        # Const
        self.ZOOM_RATE = const.ZOOM_RATE
        self.PAN_SPEED = const.PAN_SPEED
        self.DISPLAY_HEIGTH = const.DISPLAY_HEIGTH
        self.DISPLAY_RATIO = const.DISPLAY_RATIO
        self.DISPLAY_WIDTH = math.floor(self.DISPLAY_HEIGTH * self.DISPLAY_RATIO)
        self.WINDOW_SIZE = self.DISPLAY_WIDTH, self.DISPLAY_HEIGTH

    def reset(self):
        print("Reset ")
        self.__init__()

    def zoom_in(self, mousePos=None):
        self._zoom(self.ZOOM_RATE, mousePos)

    def zoom_out(self, mousePos=None):
        self._zoom(1 / self.ZOOM_RATE, mousePos)

    def _zoom(self, zoom_rate, mousePos):
        if mousePos is not None:
            (mouseX, mouseY) = mousePos
        else:
            (mouseX, mouseY) = (self.DISPLAY_WIDTH / 2, self.DISPLAY_HEIGTH / 2)
        self.xcenter = self.xmin + mouseX * (self.xmax - self.xmin) / self.DISPLAY_WIDTH
        self.ycenter = (
            self.ymin
            + (self.DISPLAY_HEIGTH - mouseY)
            * (self.ymax - self.ymin)
            / self.DISPLAY_HEIGTH
        )
        self.yheight /= zoom_rate
        print(
            f"Zoom {mouseX},{mouseY}, ({self.xcenter},{self.ycenter}), factor {zoom_rate}"
        )

    def change_k_mode(self):
        self.k_mode = (self.k_mode + 1) % len(K_Mode)
        print(f"K mode: {self.k_mode}:  {K_Mode(self.k_mode).name}")

    def change_color_palette_mode(self):
        self.palette_mode = (self.palette_mode + 1) % len(Palette_Mode)
        print(
            f"Palette mode: {self.palette_mode}: {Palette_Mode(self.palette_mode).name}"
        )

    def change_color_palette_name(self):
        # Cycle names from palettes_definitions
        palette_names = list(palettes_definitions.keys())
        current_id = palette_names.index(self.custom_palette_name)
        new_id = (current_id + 1) % len(palette_names)
        self.custom_palette_name = palette_names[new_id]
        print(f"Custom palette: ({self.custom_palette_name})")

    def change_palette_shift(self, plusminus):
        self.palette_shift += type_math_int(plusminus)
        print(f"Palette shift: {self.palette_shift}")

    def reset_palette_shift(self):
        self.palette_shift = type_math_int(0)
        print(f"Palette shift: {self.palette_shift}")

    def change_color_waves(self, plusminus):
        self.color_waves = self.color_waves + plusminus
        if self.color_waves == 0:
            self.color_waves = 1
        print(f"Color waves: {self.color_waves}")

    def change_max_iterations(self, factor):
        self.max_iterations = int(self.max_iterations * factor)
        print(f"Max iterations: {self.max_iterations}")

    def change_power(self, plusminus):
        self.power = (self.power + plusminus) % 16
        print(f"Power: {self.power}")

    def change_escape_radius(self, factor):
        self.escape_radius = int(self.escape_radius * factor)
        print(f"Escape radius: {self.escape_radius}")

    def change_epsilon(self, factor):
        if factor == self.epsilon == 0:
            self.epsilon = 0.001
        else:
            self.epsilon *= factor
        print(f"Epsilon: {self.epsilon}")

    def change_fractal_mode(self, pos):
        (mouseX, mouseY) = pos
        self.fractal_mode = (self.fractal_mode + 1) % len(Fractal_Mode)
        juliax = self.xmin + mouseX * (self.xmax - self.xmin) / self.DISPLAY_WIDTH
        juliay = (
            self.ymin
            + (self.DISPLAY_HEIGTH - mouseY)
            * (self.ymax - self.ymin)
            / self.DISPLAY_HEIGTH
        )
        self.juliaxy = type_math_complex(juliax + juliay * 1j)
        print(f"Fractal mode: {Fractal_Mode(self.fractal_mode).name}")

    def recalc_size(self):
        xwidth = self.yheight * self.DISPLAY_WIDTH / self.DISPLAY_HEIGTH
        self.xmin = self.xcenter - xwidth / 2
        self.xmax = self.xcenter + xwidth / 2
        self.ymin = self.ycenter - self.yheight / 2
        self.ymax = self.ycenter + self.yheight / 2

    def pan(self, x, y):
        self.xcenter += x * self.PAN_SPEED * (self.xmax - self.xmin)
        self.ycenter += y * self.PAN_SPEED * (self.ymax - self.ymin)

    def toggle_info(self):
        self.show_info = not self.show_info

    def get_info(self):
        info_list = []
        info_list.append(f"fractal mode: {Fractal_Mode(self.fractal_mode).name}")
        info_list.append(f"x: {self.xmin} - {self.xmax}")
        info_list.append(f"y: {self.ymin} - {self.ymax}")
        info_list.append(f"K mode: {K_Mode(self.k_mode).name}")
        info_list.append(f"palette mode: {Palette_Mode(self.palette_mode).name}")
        if self.palette_mode == Palette_Mode.CUSTOM:
            info_list.append(f"palette name: {self.custom_palette_name}")
            info_list.append(f"palette shift: {self.palette_shift}")
        info_list.append(f"color waves: {self.color_waves}")
        info_list.append(f"max iterations: {self.max_iterations}")
        info_list.append(f"power: {self.power}")
        info_list.append(f"escape radius: {self.escape_radius}")
        info_list.append(f"epsilon: {self.epsilon}")
        return info_list

    def get_info_table(self):
        info_table = {}
        info_table["fractal_mode"] = self.fractal_mode
        info_table["xmin"] = self.xmin
        info_table["xmax"] = self.xmax
        info_table["ymin"] = self.ymin
        info_table["ymax"] = self.ymax
        info_table["k_mode"] = self.k_mode
        info_table["palette_mode"] = self.palette_mode
        info_table["custom_palette_name"] = self.custom_palette_name
        info_table["palette_shift"] = self.palette_shift
        info_table["color_waves"] = self.color_waves
        info_table["max_iterations"] = self.max_iterations
        info_table["power"] = self.power
        info_table["escape_radius"] = self.escape_radius
        info_table["epsilon"] = self.epsilon
        return info_table

    def get_info_table_value(self, info_table, key, default):
        if key in info_table:
            return info_table[key]
        else:
            print(f"Key {key} not found in info table, using default value {default}")
            return default

    def set_from_info_table(self, info_table):
        # use default values from defaults.py if info is missing
        self.fractal_mode = type_enum_int(
            self.get_info_table_value(info_table, "fractal_mode", defaults.fractal_mode)
        )
        self.xmin = type_math_float(
            self.get_info_table_value(info_table, "xmin", defaults.xmin)
        )
        self.xmax = type_math_float(
            self.get_info_table_value(info_table, "xmax", defaults.xmax)
        )
        self.ymin = type_math_float(
            self.get_info_table_value(info_table, "ymin", defaults.ymin)
        )
        self.ymax = type_math_float(
            self.get_info_table_value(info_table, "ymax", defaults.ymax)
        )
        self.k_mode = type_enum_int(
            self.get_info_table_value(info_table, "k_mode", defaults.k_mode)
        )
        self.palette_mode = type_enum_int(
            self.get_info_table_value(info_table, "palette_mode", defaults.palette_mode)
        )
        self.custom_palette_name = self.get_info_table_value(
            info_table, "custom_palette_name", defaults.custom_palette_name
        )
        self.palette_shift = type_math_int(
            self.get_info_table_value(
                info_table, "palette_shift", defaults.palette_shift
            )
        )
        self.color_waves = type_math_int(
            self.get_info_table_value(info_table, "color_waves", defaults.color_waves)
        )
        self.max_iterations = type_math_int(
            self.get_info_table_value(
                info_table, "max_iterations", defaults.max_iterations
            )
        )
        self.power = type_math_int(
            self.get_info_table_value(info_table, "power", defaults.power)
        )
        self.escape_radius = type_math_int(
            self.get_info_table_value(
                info_table, "escape_radius", defaults.escape_radius
            )
        )
        self.epsilon = type_math_float(
            self.get_info_table_value(info_table, "epsilon", defaults.epsilon)
        )
        # recalc derived variables
        self.xcenter = type_math_float(self.xmin + (self.xmax - self.xmin) / 2)
        self.ycenter = type_math_float(self.ymin + (self.ymax - self.ymin) / 2)
        self.yheight = type_math_float(self.ymax - self.ymin)
