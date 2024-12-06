import math
from numpy import complex128
from dataclasses import dataclass
from fractal.colors import K_Mode, Palette_Mode
from fractal.fractal import Fractal_Mode
from numpy import float64


@dataclass
class AppState:
    def __init__(self):

        # fractal variables
        self.xcenter = -0.5
        self.ycenter = 0
        self.yheight = 3
        self.max_iterations = 1000
        self.power = 2
        self.escape_radius = 4
        self.epsilon = 0.001
        self.fractal_mode = Fractal_Mode.MANDELBROT
        self.palette_mode = Palette_Mode.HUE
        self.k_mode = K_Mode.ITER_WAVES
        self.color_waves = 2
        self.juliaxy = complex128(0 + 0j)
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
        self.juliaxy = complex128(juliax + juliay * 1j)
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
            f"Fractal mode: {Fractal_Mode(self.fractal_mode).name}",
            f"x: {self.xmin} - {self.xmax}",
            f"y: {self.ymin} - {self.ymax}",
            f"K mode: {K_Mode(self.k_mode).name}",
            f"Palette mode: {Palette_Mode(self.palette_mode).name}",
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
        info_table["ymaw"] = self.ymax
        info_table["k_mode"] = self.k_mode
        info_table["palette_mode"] = self.palette_mode
        info_table["color_waves"] = self.color_waves
        info_table["max_iterations"] = self.max_iterations
        info_table["power"] = self.power
        info_table["escape_radius"] = self.escape_radius
        info_table["epsilon"] = self.epsilon
        return info_table

    def set_from_info_table(self, info_table):
        self.fractal_mode = int(info_table["fractal_mode"])
        self.xmin = float64(info_table["xmin"])
        self.xmax = float64(info_table["xmax"])
        self.ymin = float64(info_table["ymin"])
        self.ymax = float64(info_table["ymaw"])
        self.k_mode = int(info_table["k_mode"])
        self.palette_mode = int(info_table["palette_mode"])
        self.color_waves = int(info_table["color_waves"])
        self.max_iterations = int(info_table["max_iterations"])
        self.power = int(info_table["power"])
        self.escape_radius = float64(info_table["escape_radius"])
        self.epsilon = float64(info_table["epsilon"])
        self.xcenter = self.xmin + (self.xmax - self.xmin)/2
        self.ycenter = self.ymin + (self.ymax - self.ymin)/2
        self.yheight = self.ymax - self.ymin
