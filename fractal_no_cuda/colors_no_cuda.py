from math import log
from enum import IntEnum
from numpy import float64

class ColorMode(IntEnum):
    ITER_WAVES = 0
    ITER = 1
    LOG_ITER = 2
    R_Z2 = 3
    LOG_R_Z2 = 4
    INV_Z2 = 5

class Palette(IntEnum):
    HUE = 0
    GRAYSCALE = 1
    CUSTOM = 2


def set_color_rgb(device_array_rgb, x, y, r, g, b):
    # r, g, b should be [0:255]
    packed = (r * 256 + g) * 256 + b
    device_array_rgb[x, y] = packed



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



def set_pixel_color(device_array_rgb, device_array_k, x, y,  palette):
    # calculate color from k
    k = device_array_k[x,y]
    match palette:
        case Palette.HUE:
            set_color_hsv(device_array_rgb, x, y, k, 1, 1)
        case Palette.GRAYSCALE:
            kk = int(k * 255)
            set_color_rgb(device_array_rgb, x, y, kk, kk, kk)
        case Palette.CUSTOM:  # custom palette k to rgb
            colors = ((0.0, 255, 0,0), (0.5, 0, 255, 0), (1.0, 255, 0, 0))
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
        case _:
            set_color_rgb(device_array_rgb, x, y, 0, 255, 0)


def set_pixel_k(
    device_array_k,
    x,
    y,
    nb_iter,
    max_iterations,
    z2,
    escape_radius,
    der2,
    color_mode,
    color_waves,
):
    # calculate k[0-1] based on color mode
    k = 0.0
    if z2 > escape_radius:
        match color_mode:
            case ColorMode.ITER_WAVES:
                mic = max_iterations / color_waves
                k = float64(nb_iter % mic / mic)
            case ColorMode.ITER:
                k = float64(nb_iter / max_iterations)
            case ColorMode.LOG_ITER:
                k = log(float64(nb_iter)) / log(float64(max_iterations))
            case ColorMode.R_Z2:
                # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                # TODO : z2 is slightly bigger than r, so k doesnt cover 0-1
                k = float64(escape_radius) / float64(z2)
            case ColorMode.LOG_R_Z2:
                k = log(float64(escape_radius)) / log(float64(z2))
            case ColorMode.INV_Z2:
                # k = math.sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                # k = sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                k = 1 / z2
    device_array_k[x, y] = k