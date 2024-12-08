from math import log
from enum import IntEnum
from typing import Tuple
from utils.cuda import cuda_jit, cuda_grid
from utils.types import (
    type_math_float ,
    type_math_int,
    type_enum_int ,
    type_color_float ,
    type_color_int ,
    type_color_int_small ,
)


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


@cuda_jit("uint32(uint8, uint8, uint8)", device=True)
def rgb_to_packed(r: type_color_int_small, g: type_color_int_small, b: type_color_int_small) -> type_color_int:
    # r, g, b should be [0:255]
    # Cant assert in device function
    # assert r >= 0 and r <= 255, f"r should be in [0:255], got {r}"
    # assert g >=  0 and g  <= 255, f"g should be in [0:255], got {g}"
    # assert b >=  0 and b  <= 255, f"b should be in [0:255], got {b}"
    packed = ((type_color_int(r) * 256) + g) * 256 + b
    return packed


@cuda_jit("uint32(float64, float64, float64)", device=True)
def hsv_to_rgb(h: type_color_float, s: type_color_float, v: type_color_float) -> type_color_int:
    # h,s,v should be [0:1]
    r = g = b = type_color_float(0)
    if s > 0:
        if h == 1.0:
            h = type_color_float(0)
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
    return rgb_to_packed(type_color_int_small(r * 255), type_color_int_small(g * 255), type_color_int_small(b * 255))


@cuda_jit("uint32(float64)", device=True)
def compute_color_custom(k: type_math_float) -> type_color_int:
    colors = ((0.0, 0, 0, 0), (0.5, 255, 0, 0), (1.0, 255, 255, 255))
    for i in range(len(colors) - 1):
        color_a_k, color_a_red, color_a_green, color_a_blue = colors[i]
        color_b_k, color_b_red, color_b_green, color_b_blue = colors[i + 1]
        if k >= color_a_k and k < color_b_k:
            ratio_a = 1.0 - (k - color_a_k) / (color_b_k - color_a_k)
            ratio_b = 1.0 - ratio_a
            r = int(color_a_red * ratio_a + color_b_red * ratio_b)
            g = int(color_a_green * ratio_a + color_b_green * ratio_b)
            b = int(color_a_blue * ratio_a + color_b_blue * ratio_b)
            return rgb_to_packed(r, g, b)
    return type_color_int(0)


@cuda_jit("uint32(float64, uint8)", device=True)
def compute_pixel_color(k: type_math_float, palette_mode: type_enum_int) -> type_color_int:
    # calculate color from k
    match palette_mode:
        case Palette_Mode.HUE:
            if k == float(0.0):
                return rgb_to_packed(0, 0, 0)
            else:
                return hsv_to_rgb(k, 1, 1)
        case Palette_Mode.GRAYSCALE:
            kk = int(k * 255)
            return rgb_to_packed(kk, kk, kk)
        case Palette_Mode.CUSTOM:  # custom palette_mode k to rgb
            return compute_color_custom(k)
        case _:
            return rgb_to_packed(255, 0, 0)


@cuda_jit(
    "float64(int32, int32, float64, int32, uint8, int32)",
    device=True,
)
def compute_pixel_k(
    nb_iter: type_math_int,
    max_iterations: type_math_int,
    z2: type_math_float,
    escape_radius: type_math_int,
    k_mode: type_enum_int,
    color_waves: type_math_int,
) -> type_math_float:
    # calculate k[0-1] based on k mode
    k = type_math_float(0.0)
    if z2 > escape_radius:
        match k_mode:
            case K_Mode.ITER_WAVES:
                iter_by_wave = max_iterations / color_waves
                k = type_math_float((nb_iter % iter_by_wave) / iter_by_wave)
            case K_Mode.ITER:
                k = type_math_float(nb_iter / max_iterations)
            case K_Mode.LOG_ITER:
                k = log(type_math_float(nb_iter)) / log(type_math_float(max_iterations))
            case K_Mode.R_Z2:
                # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                # TODO : z2 is slightly bigger than r, so k doesnt cover 0-1
                k = type_math_float(escape_radius) / type_math_float(z2)
            case K_Mode.LOG_R_Z2:
                k = log(type_math_float(escape_radius)) / log(type_math_float(z2))
            case K_Mode.INV_Z2:
                # k = math.sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                # k = sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                k = 1 / z2
    return k

@cuda_jit(
    "(int32, int32, int32, int32, float64, int32, uint8, uint8, int32)",
    device=True,
)
def color_xy(
    x: type_math_int,
    y: type_math_int,
    nb_iter: type_math_int,
    max_iterations: type_math_int,
    z2: type_math_float,
    escape_radius: type_math_int,
    k_mode: type_enum_int,
    palette_mode: type_enum_int,
    color_waves: type_math_int,
) -> Tuple[type_math_float, type_color_int]:
    k = compute_pixel_k(
        nb_iter,
        max_iterations,
        z2,
        escape_radius,
        k_mode,
        color_waves,
    )
    packedrgb = compute_pixel_color(k, palette_mode)
    return k, packedrgb


@cuda_jit(
    "(int32[:,:], float64[:,:], float64[:,:], int32[:,:], int32, int32, uint8, uint8, int32)"
)
def color_kernel(
    device_array_niter,
    device_array_z2,
    device_array_k,
    device_array_rgb,
    max_iterations: type_math_int,
    escape_radius: type_math_int,
    k_mode: type_enum_int,
    palette_mode: type_enum_int,
    color_waves: type_math_int,
) -> None:
    x, y = cuda_grid(2)
    if x < device_array_niter.shape[0] and y < device_array_niter.shape[1]:
        nb_iter = device_array_niter[x, y]
        z2 = device_array_z2[x, y]
        k, packedrgb = color_xy(
            x,
            y,
            nb_iter,
            max_iterations,
            z2,
            escape_radius,
            k_mode,
            palette_mode,
            color_waves,
        )
        device_array_k[x, y] = k
        device_array_rgb[x, y] = packedrgb

# def build_custom_palette(color_list: List[Color], steps):
#     if len(color_list) == 0:
#         color_list.append(Color("white"))
#     if len(color_list) == 1:
#         color_list.append(Color("black"))
#     steps_by_color = int(steps / (len(color_list) - 1))
#     custom_palette = []
#     for i in range(len(color_list) - 1):
#         ColorA = color_list[i]
#         ColorB = color_list[i + 1]
#         for j in range(steps_by_color):
#             InterimColor = ColorA.lerp(ColorB, j / steps_by_color)
#             packed = (InterimColor.r * 256 + InterimColor.g) * 256 + InterimColor.b
#             custom_palette.append(packed)
#     return custom_palette


# def init_custom_palette_sample():
#     return build_custom_palette([Color("Black"), Color("Red"), Color("White")], 1000)


# def get_custom_palette_mode_color(custom_palette, x):
#     return custom_palette[x % len(custom_palette)]
