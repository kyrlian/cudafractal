from math import log
from enum import IntEnum
from numpy import float64
from fractal.utils_cuda import cuda_jit
from pygame import Color
from typing import List


class K_Mode(IntEnum):
    ITER_WAVES = 0
    ITER = 1
    LOG_ITER = 2
    R_Z2 = 3
    LOG_R_Z2 = 4
    INV_Z2 = 5


class Palette_Mode(IntEnum):
    HUE = 0
    GRAYSCALE = 1
    CUSTOM = 2


@cuda_jit(device=True)
def set_color_rgb(device_array_rgb, x, y, r, g, b):
    # r, g, b should be [0:255]
    packed = (r * 256 + g) * 256 + b
    device_array_rgb[x, y] = packed


@cuda_jit(device=True)
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

@cuda_jit(device=True)
def set_color_custom(device_array_rgb, x, y, k:float64):
    colors = ((0.0, 0, 0, 0) , (0.5, 255, 0,0),(1.0, 255, 255, 255))
    for i in range(len(colors) - 1):
        color_a_k, color_a_red, color_a_green, color_a_blue = colors[i]
        color_b_k, color_b_red, color_b_green, color_b_blue = colors[i + 1]
        if k >= color_a_k and k < color_b_k:
            ratio_a = 1.0 - (k - color_a_k) / (color_b_k - color_a_k)
            ratio_b = 1.0 - ratio_a
            r = int(color_a_red * ratio_a + color_b_red * ratio_b)
            g = int(color_a_green * ratio_a + color_b_green * ratio_b)
            b = int(color_a_blue * ratio_a + color_b_blue * ratio_b)
            set_color_rgb(device_array_rgb, x, y, r, g, b)

@cuda_jit(device=True)
def set_pixel_color(device_array_rgb, device_array_k, x, y, palette_mode):
    # calculate color from k
    k = device_array_k[x, y]
    match palette_mode:
        case Palette_Mode.HUE:
            if k == 0.0:
                set_color_rgb(device_array_rgb, x, y, 0, 0, 0)
            else:
                set_color_hsv(device_array_rgb, x, y, k, 1, 1)
        case Palette_Mode.GRAYSCALE:
            kk = int(k * 255)
            set_color_rgb(device_array_rgb, x, y, kk, kk, kk)
        case Palette_Mode.CUSTOM:  # custom palette_mode k to rgb
            set_color_custom(device_array_rgb, x, y, k)
        case _:
            set_color_rgb(device_array_rgb, x, y, 255, 0, 0)

@cuda_jit(device=True)
def set_pixel_k(
    device_array_k,
    x,
    y,
    nb_iter,
    max_iterations,
    z2,
    escape_radius,
    der2,
    k_mode,
    color_waves,
):
    # calculate k[0-1] based on k mode
    k = 0.0
    if z2 > escape_radius:
        match k_mode:
            case K_Mode.ITER_WAVES:
                iter_by_wave = max_iterations / color_waves
                k = float64((nb_iter % iter_by_wave) / iter_by_wave)
            case K_Mode.ITER:
                k = float64(nb_iter / max_iterations)
            case K_Mode.LOG_ITER:
                k = log(float64(nb_iter)) / log(float64(max_iterations))
            case K_Mode.R_Z2:
                # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                # TODO : z2 is slightly bigger than r, so k doesnt cover 0-1
                k = float64(escape_radius) / float64(z2)
            case K_Mode.LOG_R_Z2:
                k = log(float64(escape_radius)) / log(float64(z2))
            case K_Mode.INV_Z2:
                # k = math.sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                # k = sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                k = 1 / z2
    device_array_k[x, y] = k


def build_custom_palette_mode(color_list: List[Color], steps):
    if len(color_list) == 0:
        color_list.append(Color("white"))
    if len(color_list) == 1:
        color_list.append(Color("black"))
    steps_by_color = int(steps / (len(color_list) - 1))
    palette_mode = []
    for i in range(len(color_list) - 1):
        ColorA = color_list[i]
        ColorB = color_list[i + 1]
        for j in range(steps_by_color):
            InterimColor = ColorA.lerp(ColorB, j / steps_by_color)
            palette_mode.append(InterimColor)
    return palette_mode


@cuda_jit(device=True)
def get_custom_palette_mode_color(palette_mode, x):
    return palette_mode[x % len(palette_mode)]
