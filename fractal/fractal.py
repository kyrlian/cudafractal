import timeit
import numpy
from math import ceil
from enum import IntEnum
from typing import Tuple
from utils.types import type_math_int, type_math_float, type_math_complex, type_enum_int, type_color_int
from utils.cuda import (
    cuda_jit,
    cuda_available,
    cuda_grid,
    compute_threadsperblock,
    init_array,
)
from fractal.colors import compute_pixel_color, compute_pixel_k


class Fractal_Mode(IntEnum):
    MANDELBROT = 0
    JULIA = 1


@cuda_jit(
    "(int32, int32, complex128, float64, float64, uint8, int32, int32, int32, float64, complex128, uint8, uint8, int32)",
    device=True,
)
def fractal_xy(
    x: type_math_int,
    y: type_math_int,
    topleft: type_math_complex,
    xstep: type_math_float,
    ystep: type_math_float,
    fractalmode: type_enum_int,
    max_iterations: type_math_int,
    power: type_math_int,
    escape_radius: type_math_int,
    epsilon: type_math_float,
    juliaxy: type_math_complex,
    k_mode: type_enum_int,
    palette_mode: type_enum_int,
    color_waves: type_math_int,
) -> Tuple[type_math_int, type_math_float, type_math_float, type_color_int]:
    z: type_math_complex = type_math_complex(
        topleft + type_math_float(x) * xstep - 1j * y * ystep
    )
    c: type_math_complex = z if fractalmode == Fractal_Mode.MANDELBROT else juliaxy
    nb_iter: type_math_int = type_math_int(0)
    z2: type_math_float = type_math_float(0)
    der: type_math_complex = type_math_complex(1 + 0j)
    der2: type_math_float = type_math_float(1)
    while nb_iter < max_iterations and z2 < escape_radius and der2 > epsilon:
        der = der * power * z
        z = z**power + c
        nb_iter += 1
        z2 = z.real**2 + z.imag**2
        der2 = der.real**2 + der.imag**2
    k = compute_pixel_k(
        nb_iter,
        max_iterations,
        z2,
        escape_radius,
        der2,
        k_mode,
        color_waves,
    )
    packedrgb = compute_pixel_color(k, palette_mode)
    return nb_iter, z2, k, packedrgb


@cuda_jit(
    "(int32[:,:], float64[:,:], float64[:,:], int32[:,:], complex128, float64, float64, uint8, int32, int32, int32, float64, complex128, uint8, uint8, int32)"
)
def fractal_kernel(
    device_array_niter,
    device_array_z2,
    device_array_k,
    device_array_rgb,
    topleft: type_math_complex,
    xstep: type_math_float,
    ystep: type_math_float,
    fractalmode: type_enum_int,
    max_iterations: type_math_int,
    power: type_math_int,
    escape_radius: type_math_int,
    epsilon: type_math_float,
    juliaxy: type_math_complex,
    k_mode: type_enum_int,
    palette_mode: type_enum_int,
    color_waves: type_math_int,
) -> None:
    x, y = cuda_grid(2)
    if x < device_array_niter.shape[0] and y < device_array_niter.shape[1]:
        nb_iter, z2, k, packedrgb = fractal_xy(
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
            k_mode,
            palette_mode,
            color_waves,
        )
        device_array_niter[x, y] = nb_iter
        device_array_z2[x, y] = z2
        device_array_k[x, y] = k
        device_array_rgb[x, y] = packedrgb


def compute_fractal(
    WINDOW_SIZE,
    xmax: type_math_float,
    xmin: type_math_float,
    ymin: type_math_float,
    ymax: type_math_float,
    fractalmode: type_enum_int,
    max_iterations: type_math_int,
    power: type_math_int,
    escape_radius: type_math_int,
    epsilon: type_math_float,
    juliaxy: type_math_complex,
    k_mode: type_enum_int,
    palette_mode: type_enum_int,
    color_waves: type_math_int,
):
    timerstart = timeit.default_timer()
    (screenw, screenh) = WINDOW_SIZE
    xstep = abs(xmax - xmin) / screenw
    ystep = abs(ymax - ymin) / screenh
    topleft = type_math_complex(xmin + 1j * ymax)
    device_array_niter = init_array(screenw, screenh, type_math_int)
    device_array_z2 = init_array(screenw, screenh, type_math_float)
    device_array_k = init_array(screenw, screenh, type_math_float)
    device_array_rgb = init_array(screenw, screenh, type_math_int)
    if cuda_available():
        threadsperblock = compute_threadsperblock(screenw, screenh)
        blockspergrid = (
            ceil(screenw / threadsperblock[0]),
            ceil(screenh / threadsperblock[1]),
        )
        fractal_kernel[blockspergrid, threadsperblock](
            device_array_niter,
            device_array_z2,
            device_array_k,
            device_array_rgb,
            topleft,
            xstep,
            ystep,
            fractalmode,
            max_iterations,
            power,
            escape_radius,
            epsilon,
            juliaxy,
            k_mode,
            palette_mode,
            color_waves,
        )
        output_array_niter = device_array_niter.copy_to_host()
        output_array_z2 = device_array_z2.copy_to_host()
        output_array_k = device_array_k.copy_to_host()
        output_array_rgb = device_array_rgb.copy_to_host()
    else:  # No cuda
        run_vectorized = True
        if run_vectorized:
            # vectorized version:
            vectorized_fractal_xy = numpy.vectorize(
                fractal_xy,
                otypes=[type_math_int, type_math_float, type_math_float, type_color_int],
            )  # fractal_xy returns nb_iter, z2, k, packedrgb
            # vector_x and vector_y need to be same size, and represent all matrix cells:
            matrix_x=[]
            matrix_y=[]
            for x in range(device_array_niter.shape[0]):
                vector_x=[]
                vector_y=[]
                for y in range(device_array_niter.shape[1]):
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
                k_mode,
                palette_mode,
                color_waves,
            )
            output_array_niter, output_array_z2, output_array_k, output_array_rgb = (
                result_arrays
            )
        else:
            # NON vectorized version:
            output_array_niter = init_array(screenw, screenh, type_math_int)
            output_array_z2 = init_array(screenw, screenh, type_math_float)
            output_array_k = init_array(screenw, screenh, type_math_float)
            output_array_rgb = init_array(screenw, screenh, type_math_int)
            for x in range(device_array_niter.shape[0]):
                for y in range(device_array_niter.shape[1]):
                    niter, z2, k, packedrgb = fractal_xy(
                        x,
                        y,            topleft,
                        xstep,
                        ystep,
                        fractalmode,
                        max_iterations,
                        power,
                        escape_radius,
                        epsilon,
                        juliaxy,
                        k_mode,
                        palette_mode,
                        color_waves,
                    )
                    output_array_niter[x,y]=niter
                    output_array_z2[x,y]=z2
                    output_array_k[x,y]=k
                    output_array_rgb[x,y]=packedrgb
    print(f"Frame calculated in {(timeit.default_timer() - timerstart)}s")
    return output_array_niter, output_array_z2, output_array_k, output_array_rgb
