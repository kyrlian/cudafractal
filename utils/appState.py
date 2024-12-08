import math
from dataclasses import dataclass
from utils.types import (
    type_math_complex ,
    type_math_float ,
    type_math_int,
    type_enum_int ,
)
from fractal.colors import K_Mode, Palette_Mode
from fractal.fractal import Fractal_Mode
from fractal.palette import palletes_definitions

@dataclass
class AppState:
    def __init__(self):

        # fractal variables
        self.xcenter = type_math_float(-0.5)
        self.ycenter = type_math_float(0)
        self.yheight = type_math_float(3)
        self.max_iterations = type_math_int(1000)
        self.power = type_math_int(2)
        self.escape_radius = type_math_int(4)
        self.epsilon = type_math_float(0.001)
        self.fractal_mode = type_enum_int(Fractal_Mode.MANDELBROT)
        self.juliaxy = type_math_complex(0 + 0j)

        # color variables
        self.palette_mode = type_enum_int(Palette_Mode.HUE)
        self.k_mode = type_enum_int(K_Mode.ITER_WAVES)
        self.color_waves = type_math_int(2)
        self.custom_palette_name = list(palletes_definitions.keys())[0]  # TODO use first key from palletes_definitions
        # UI variables
        self.show_info = True

        # Const
        self.ZOOM_RATE = 2
        self.PAN_SPEED = 0.3  # ratio of xmax-xmin
        self.DISPLAY_HEIGTH = 1024
        self.DISPLAY_RATIO = 4 / 3
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
        # TODO cycle names from palletes_definitions
        # self.custom_palette_name
        print(f"Custom palette: ({self.custom_palette_name})")

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
        return [
            f"fractal mode: {Fractal_Mode(self.fractal_mode).name}",
            f"x: {self.xmin} - {self.xmax}",
            f"y: {self.ymin} - {self.ymax}",
            f"K mode: {K_Mode(self.k_mode).name}",
            f"palette mode: {Palette_Mode(self.palette_mode).name}",
            f"palette name: {self.custom_palette_name}",#TODO only display if self.palette_mode = Palette_Mode.CUSTOM"
            f"color waves: {self.color_waves}",
            f"max iterations: {self.max_iterations}",
            f"power: {self.power}",
            f"escape radius: {self.escape_radius}",
            f"epsilon: {self.epsilon}",
        ]

    def get_info_table(self):
        info_table = {}
        info_table["fractal_mode"] = self.fractal_mode
        info_table["xmin"] = self.xmin
        info_table["xmax"] = self.xmax
        info_table["ymin"] = self.ymin
        info_table["ymax"] = self.ymax
        info_table["k_mode"] = self.k_mode
        info_table["palette_mode"] = self.palette_mode
        # TODO add appstate.custom_palette_name 
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
        self.fractal_mode = type_enum_int(self.get_info_table_value(info_table,"fractal_mode",0))
        self.xmin = type_math_float(self.get_info_table_value(info_table,"xmin",-2.5))
        self.xmax = type_math_float(self.get_info_table_value(info_table,"xmax",1.5))
        self.ymin = type_math_float(self.get_info_table_value(info_table,"ymin",-1.5))
        self.ymax = type_math_float(self.get_info_table_value(info_table,"ymax",1.5))
        self.k_mode = type_enum_int(self.get_info_table_value(info_table,"k_mode",0))
        self.palette_mode = type_enum_int(self.get_info_table_value(info_table,"palette_mode",0))
        # TODO add appstate.custom_palette_name 
        self.color_waves = type_math_int(self.get_info_table_value(info_table,"color_waves",2))
        self.max_iterations = type_math_int(self.get_info_table_value(info_table,"max_iterations",1000))
        self.power = type_math_int(self.get_info_table_value(info_table,"power",2))
        self.escape_radius = type_math_int(self.get_info_table_value(info_table,"escape_radius",4))
        self.epsilon = type_math_float(self.get_info_table_value(info_table,"epsilon",0.001))
        self.xcenter = type_math_float(self.xmin + (self.xmax - self.xmin)/2)
        self.ycenter = type_math_float(self.ymin + (self.ymax - self.ymin)/2)
        self.yheight = type_math_float(self.ymax - self.ymin)
