# from timeit import default_timer
from enum import IntEnum
from typing import List
from fractal.colors import Palette_Mode, Normalization_Mode
from fractal.fractal_math import Fractal_Mode
from utils.types import (
    type_math_int,
    type_math_float,
    type_math_complex,
    type_enum_int,
    type_color_int,
)
from utils.cuda import (
    cuda_available,
    init_array,
    cuda_copy_to_host,
)
from utils.timer import timing_wrapper
from fractal.fractal_cuda import compute_fracta_cuda
from fractal.fractal_cpu import compute_fractal_cpu

@timing_wrapper
def init_arrays(WINDOW_SIZE):
    (screenw, screenh) = WINDOW_SIZE
    device_array_niter = init_array(screenw, screenh, type_math_int)
    device_array_z2 = init_array(screenw, screenh, type_math_float)
    device_array_der2 = init_array(screenw, screenh, type_math_float)
    device_array_k = init_array(screenw, screenh, type_math_float)
    device_array_rgb = init_array(screenw, screenh, type_math_int)
    host_array_niter = cuda_copy_to_host(device_array_niter)
    host_array_z2 = cuda_copy_to_host(device_array_z2)
    host_array_der2 = cuda_copy_to_host(device_array_der2)
    host_array_k = cuda_copy_to_host(device_array_k)
    host_array_rgb = cuda_copy_to_host(device_array_rgb)
    return (
        host_array_niter,
        host_array_z2,
        host_array_der2,
        host_array_k,
        host_array_rgb,
    )


# TODO read stuff from AppState
@timing_wrapper
def compute_fractal(
    host_array_niter,
    niter_min,
    niter_max,
    host_array_z2,
    z2_min,
    z2_max,
    host_array_der2,
    der2_min,
    der2_max,
    host_array_k,
    host_array_rgb,
    WINDOW_SIZE,
    xmax: type_math_float,
    xmin: type_math_float,
    ymin: type_math_float,
    ymax: type_math_float,
    fractalmode: Fractal_Mode,
    max_iterations: type_math_int,
    power: type_math_int,
    escape_radius: type_math_int,
    epsilon: type_math_float,
    juliaxy: type_math_complex,
    normalization_mode: Normalization_Mode,
    palette_mode: Palette_Mode,
    custom_palette: List[type_color_int],
    palette_width: type_math_float,
    palette_shift: type_math_float,
    recalc_fractal: bool = True,
    recalc_color: bool = False,
):
    # timerstart = default_timer()
    if cuda_available():
        compute_fractal = compute_fracta_cuda
    else:  # No cuda
        compute_fractal = compute_fractal_cpu

    return compute_fractal(
        host_array_niter,
        niter_min,
        niter_max,
        host_array_z2,
        z2_min,
        z2_max,
        host_array_der2,
        der2_min,
        der2_max,
        host_array_k,
        host_array_rgb,
        WINDOW_SIZE,
        xmax,
        xmin,
        ymin,
        ymax,
        fractalmode,
        max_iterations,
        power,
        escape_radius,
        epsilon,
        juliaxy,
        normalization_mode,
        palette_mode,
        custom_palette,
        palette_width,
        palette_shift,
        recalc_fractal,
        recalc_color,
    )
