import math
from numba import float64, complex128
from dataclasses import dataclass

from fractal_cuda.fractal_cuda import FRACTAL_MODES
from fractal_cuda.colors_cuda import ColorMode, Palette, NB_COLOR_MODES, NB_PALETTES

@dataclass
class AppState:
    # variables
    xcenter: float64 = -0.5
    ycenter: float64 = 0
    yheight: float64 = 3
    max_iterations: int = 1000
    power: int = 2
    escape_radius: int = 4
    epsilon: float64 = 0.001
    fractal_mode: int = 0
    color_mode: int = ColorMode.ITER_WAVES
    palette: int = Palette.HUE
    color_waves: int = 2
    juliaxy = complex128(0 + 0j)

    # Const
    ZOOM_RATE = 2
    PAN_SPEED = 0.3  # ratio of xmax-xmin
    DISPLAY_HEIGTH = 1024
    DISPLAY_RATIO = 4 / 3
    DISPLAY_WIDTH = math.floor(DISPLAY_HEIGTH * DISPLAY_RATIO)
    WINDOW_SIZE = DISPLAY_WIDTH, DISPLAY_HEIGTH

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

    def change_color_mode(self):
        self.color_mode = (self.color_mode + 1) % NB_COLOR_MODES
        print(f"Color mode: {self.color_mode}")

    def change_color_palette(self):
        self.palette = (self.palette + 1) % NB_PALETTES
        print(f"Palette: {self.palette}")

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
        self.fractal_mode = (self.fractal_mode + 1) % len(FRACTAL_MODES)
        juliax = self.xmin + mouseX * (self.xmax - self.xmin) / self.DISPLAY_WIDTH
        juliay = (
            self.ymin
            + (self.DISPLAY_HEIGTH - mouseY)
            * (self.ymax - self.ymin)
            / self.DISPLAY_HEIGTH
        )
        self.juliaxy = complex128(juliax + juliay * 1j)
        print(
            f"Fractal mode: {FRACTAL_MODES[self.fractal_mode].__name__}"
        )

    def recalc_size(self):
        xwidth = self.yheight * self.DISPLAY_WIDTH / self.DISPLAY_HEIGTH
        self.xmin = self.xcenter - xwidth / 2
        self.xmax = self.xcenter + xwidth / 2
        self.ymin = self.ycenter - self.yheight / 2
        self.ymax = self.ycenter + self.yheight / 2

    def pan(self, x, y):
        self.xcenter += x * self.PAN_SPEED * (self.xmax - self.xmin)
        self.ycenter += y * self.PAN_SPEED * (self.ymax - self.ymin)
