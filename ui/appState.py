import math
from numba import float64, complex128
from dataclasses import dataclass

from fractal_cuda.fractal_cuda import FRACTAL_MODES
from fractal_cuda.colors_cuda import NB_COLOR_MODES, NB_PALETTES

@dataclass
class AppState:
    # variables
    xcenter: float64 = -0.5
    ycenter: float64 = 0
    yheight: float64 = 3
    maxiterations: int = 1000
    power: int = 2
    escaper: int = 4
    epsilon: float64 = 0.001
    fractalmode: int = 0
    colormode: int = 0
    palette: int = 0
    colorwaves: int = 1
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

    def _zoom(self, zoomrate, mousePos):
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
        self.yheight /= zoomrate
        print(
            f"Zoom {mouseX},{mouseY}, ({self.xcenter},{self.ycenter}), factor {zoomrate}"
        )

    def changecolormode(self):
        self.colormode = (self.colormode + 1) % NB_COLOR_MODES
        print(f"Color mode: {self.colormode}")

    def changecolorpalette(self):
        self.palette = (self.palette + 1) % NB_PALETTES
        print(f"Palette: {self.palette}")

    def changecolor_waves(self, plusminus):
        self.colorwaves = self.colorwaves + plusminus
        if self.colorwaves == 0:
            self.colorwaves = 1
        print(f"Color waves: {self.colorwaves}")

    def changemaxiterations(self, factor):
        self.maxiterations = int(self.maxiterations * factor)
        print(f"Max iterations: {self.maxiterations}")

    def changepower(self, plusminus):
        self.power = (self.power + plusminus) % 16
        print(f"Power: {self.power}")

    def changeescaper(self, factor):
        self.escaper = int(self.escaper * factor)
        print(f"Escape R: {self.escaper}")

    def changeepsilon(self, factor):
        if factor == self.epsilon == 0:
            self.epsilon = 0.001
        else:
            self.epsilon *= factor
        print(f"Epsilon: {self.epsilon}")

    def changefractalmode(self, pos):
        (mouseX, mouseY) = pos
        self.fractalmode = (self.fractalmode + 1) % len(FRACTAL_MODES)
        juliax = self.xmin + mouseX * (self.xmax - self.xmin) / self.DISPLAY_WIDTH
        juliay = (
            self.ymin
            + (self.DISPLAY_HEIGTH - mouseY)
            * (self.ymax - self.ymin)
            / self.DISPLAY_HEIGTH
        )
        self.juliaxy = complex128(juliax + juliay * 1j)
        print(
            f"Fractal mode: {FRACTAL_MODES[self.fractalmode].__name__}"
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
