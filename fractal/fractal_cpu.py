# from timeit import default_timer
from typing import List
from numpy import (
    vectorize as np_vectorize,
    minimum as np_minimum,
    maximum as np_maximum,
)
from utils.types import (
    type_math_int,
    type_math_float,
    type_math_complex,
    type_enum_int,
    type_color_int,
)
from utils.timer import timing_wrapper
from fractal.fractal_math import fractal_xy, Fractal_Mode
from fractal.colors import Normalization_Mode, Palette_Mode, color_cpu


@timing_wrapper
def fractal_cpu(
    host_array_niter,
    host_array_z2,
    host_array_der2,
    topleft: type_math_complex,
    xstep: type_math_float,
    ystep: type_math_float,
    fractalmode: Fractal_Mode,
    max_iterations: type_math_int,
    power: type_math_int,
    escape_radius: type_math_int,
    epsilon: type_math_float,
    juliaxy: type_math_complex,
):
    run_vectorized = True
    if run_vectorized:
        # vectorized version:
        vectorized_fractal_xy = np_vectorize(
            fractal_xy,
            otypes=[type_math_int, type_math_float, type_math_float],
        )  # fractal_xy returns nb_iter, z2, der2
        # vector_x and vector_y need to be same size, and represent all matrix cells:
        matrix_x = []
        matrix_y = []
        for x in range(host_array_niter.shape[0]):
            vector_x = []
            vector_y = []
            for y in range(host_array_niter.shape[1]):
                vector_x.append(x)
                vector_y.append(y)
            matrix_x.append(vector_x)
            matrix_y.append(vector_y)
        result_arrays = vectorized_fractal_xy(
            matrix_x,
            matrix_y,
            topleft,
            xstep,
            ystep,
            fractalmode,
            max_iterations,
            power,
            escape_radius,
            epsilon,
            juliaxy,
        )
        host_array_niter, host_array_z2, host_array_der2 = result_arrays
    else:
        # NON vectorized version:
        for x in range(host_array_niter.shape[0]):
            for y in range(host_array_niter.shape[1]):
                niter, z2, der2 = fractal_xy(
                    x,
                    y,
                    topleft,
                    xstep,
                    ystep,
                    fractalmode,
                    max_iterations,
                    power,
                    escape_radius,
                    epsilon,
                    juliaxy,
                )
                host_array_niter[x, y] = niter
                host_array_z2[x, y] = z2
                host_array_der2[x, y] = der2
    return host_array_niter, host_array_z2, host_array_der2


@timing_wrapper
def compute_min_max_cpu(host_array):
    flat_array = host_array.ravel()
    return np_minimum.reduce(flat_array, initial=0), np_maximum.reduce(
        flat_array, initial=0
    )


# TODO read stuff from AppState
@timing_wrapper
def compute_fractal_cpu(
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
    (screenw, screenh) = WINDOW_SIZE
    xstep = abs(xmax - xmin) / screenw
    ystep = abs(ymax - ymin) / screenh
    topleft = type_math_complex(xmin + 1j * ymax)

    # No cuda
    if recalc_fractal:
        host_array_niter, host_array_z2, host_array_der2 = fractal_cpu(
            host_array_niter,
            host_array_z2,
            host_array_der2,
            topleft,
            xstep,
            ystep,
            fractalmode,
            max_iterations,
            power,
            escape_radius,
            epsilon,
            juliaxy,
        )
        # compute min/max of niter and z2, so palette step can set k based on min/max niter of current image
        niter_min, niter_max = compute_min_max_cpu(host_array_niter)
        z2_min, z2_max = compute_min_max_cpu(host_array_z2)
        der2_min, der2_max = compute_min_max_cpu(host_array_der2)
        # TODO: store niter_min, niter_max, z2_min, z2_max, der2_min, der2_max in AppState
    if recalc_fractal or recalc_color:
        # color is calculated with fractal when it's called, but can be called by itself
        host_array_k, host_array_rgb = color_cpu(
            host_array_niter,
            host_array_z2,
            host_array_der2,
            host_array_k,
            host_array_rgb,
            niter_min,
            niter_max,
            z2_min,
            z2_max,
            der2_min,
            der2_max,
            max_iterations,
            escape_radius,
            normalization_mode,
            palette_mode,
            custom_palette,
            palette_width,
            palette_shift,
        )
    # TODO store stuff from AppState
    return (
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
    )
