from enum import IntEnum, auto
from numba import cuda


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
def set_color_rgb(device_array_rgb, x, y, r, g, b):
    # r, g, b should be [0:255]
    packed = (r * 256 + g) * 256 + b
    device_array_rgb[x, y] = packed


@cuda.jit("void(uint32[:,:], int32, int32, float64, float64, float64)", device=True)
def set_color_hsv(device_array_rgb, x, y, h, s, v):
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
    set_color_rgb(device_array_rgb, x, y, int(r * 255), int(g * 255), int(b * 255))


@cuda.jit(
    "void(uint32[:,:], int32, int32, int32, int32)",
    device=True,
)
def set_pixel_color(device_array_rgb, x, y, k, palette):
    # calculate color from k
    match palette:
        case Palette.HUE:
            set_color_hsv(device_array_rgb, x, y, k, 1, 1)
        case Palette.GRAYSCALE:
            kk = int(k * 255)
            set_color_rgb(device_array_rgb, x, y, kk, kk, kk)
        case Palette.CUSTOM:  # custom palette k to rgb
            colors = ((0.0, 255, 255, 255), (0.5, 0, 255, 0), (1.0, 255, 0, 0))
            for i in range(len(colors) - 1):
                pa_k, pa_r, pa_g, pa_b = colors[i]
                pb_k, pb_r, pb_g, pb_b = colors[i + 1]
                if k >= pa_k and k < pb_k:
                    d_ab = pb_k - pa_k
                    d_ak = 1 - (k - pa_k) / d_ab
                    d_bk = 1 - (pb_k - k) / d_ab
                    r = pa_r * d_ak + pb_r * d_bk
                    g = pa_g * d_ak + pb_g * d_bk
                    b = pa_b * d_ak + pb_b * d_bk
                    set_color_rgb(device_array_rgb, x, y, r, g, b)
    set_color_rgb(device_array_rgb, x, y, r, g, b)
