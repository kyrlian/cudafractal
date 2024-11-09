from math import log
from numba import cuda, float64
from enum import IntEnum, auto


class ColorMode(IntEnum):
    ITER_WAVES = auto()
    ITER = auto()
    LOG_ITER = auto()
    R_Z2 = auto()
    LOG_R_Z2 = auto()
    INV_Z2 = auto()


NB_COLOR_MODES = len(ColorMode)


class Palette(IntEnum):
    HUE = auto()
    GRAYSCALE = auto()
    CUSTOM = auto()


NB_PALETTES = len(Palette)


@cuda.jit("void(uint32[:,:], int32, int32, int32, int32, int32)", device=True)
def set_color_rgb(device_array, x, y, r, g, b):
    # r, g, b should be [0:255]
    packed = (r * 256 + g) * 256 + b
    device_array[x, y] = packed


@cuda.jit("void(uint32[:,:], int32, int32, float64, float64, float64)", device=True)
def set_color_hsv(device_array, x, y, h, s, v):
    # h,s,v should be [0:1]
    r, g, b = 0, 0, 0
    if s > 0:
        if h == 1.0:
            h = 0.0
        i = int(h * 6.0)
        f = h * 6.0 - i

        w = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        if i == 0:
            r, g, b = v, t, w
        if i == 1:
            r, g, b = q, v, w
        if i == 2:
            r, g, b = w, v, t
        if i == 3:
            r, g, b = w, q, v
        if i == 4:
            r, g, b = t, w, v
        if i == 5:
            r, g, b = v, w, q
    else:
        r, g, b = v, v, v
    set_color_rgb(device_array, x, y, int(r * 255), int(g * 255), int(b * 255))


@cuda.jit(
    "void(uint32[:,:], int32, int32, int32, int32, float64, float64, float64, int32, int32, int32)",
    device=True,
)
def set_pixel_color(
    device_array, x, y, nbi, max_iter, z2, r, der2, cmode, palette, color_waves
):
    if z2 > r:
        # first calculate k[0-1] based on color mode
        match cmode:
            case ColorMode.ITER_WAVES:
                mic = max_iter / color_waves
                k = float64(nbi % mic / mic)
            case ColorMode.ITER:
                k = float64(nbi / max_iter)
            case ColorMode.LOG_ITER:
                k = log(float64(nbi)) / log(float64(max_iter))
            case ColorMode.R_Z2:
                # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                # TODO : z2 is slightly bigger than r, so k doesnt cover 0-1
                k = float64(r) / float64(z2)
            case ColorMode.LOG_R_Z2:
                k = log(float64(r)) / log(float64(z2))
            case ColorMode.INV_Z2:
                # k = math.sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                # k = sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                k = 1 / z2
        # then calculate color from k
        match palette:
            case Palette.HUE:
                set_color_hsv(device_array, x, y, k, 1, 1)
            case Palette.GRAYSCALE:
                kk = int(k * 255)
                set_color_rgb(device_array, x, y, kk, kk, kk)
            case Palette.CUSTOM:  # custom palette k to rgb
                colors = ((0,255,255,255),(1,255,0,0))
                r=g=b=0
                for c in colors:
                    (position,pr,pg,pb)=c
                    r += 1-abs(k-position)*pr
                    g += 1-abs(k-position)*pg
                    b += 1-abs(k-position)*pb
                set_color_rgb(device_array, x, y, r,g,b)

    else:
        device_array[x, y] = 0
