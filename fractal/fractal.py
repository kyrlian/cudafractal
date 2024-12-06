import timeit
import numpy
from math import ceil
from numpy import int32, float64, complex128
from fractal.colors import set_pixel_color, set_pixel_k
from fractal.utils_cuda import (
    cuda_jit,
    cuda_available,
    cuda_grid,
    compute_threadsperblock,
    init_array,
)
from enum import IntEnum
from typing import Tuple

class Fractal_Mode(IntEnum):
    MANDELBROT = 0
    JULIA = 1


@cuda_jit(
    "(int32, int32, int32[:,:], float64[:,:], float64[:,:], int32[:,:], complex128, float64, float64, int32, int32, int32, int32, float64, complex128, int32, int32, int32)",
    device=True,
)
def fractal_xy(
    x: int32,
    y: int32,
    device_array_niter,
    device_array_z2,
    device_array_k,
    device_array_rgb,
    topleft: complex128,
    xstep: float64,
    ystep: float64,
    fractalmode: int32,
    max_iterations: int32,
    p: int32,
    escape_radius: int32,
    epsilon: float64,
    juliaxy: complex128,
    k_mode: int32,
    palette_mode: int32,
    color_waves: int32,
) ->  Tuple[int32, float64,float64,int32]:
    c: complex128 = (
        complex128(topleft + float64(x) * xstep - 1j * y * ystep)
        if fractalmode == Fractal_Mode.MANDELBROT
        else juliaxy
    )
    z: complex128 = (
        c
        if fractalmode == Fractal_Mode.MANDELBROT
        else complex128(topleft + float64(x) * xstep - 1j * y * ystep)
    )
    nb_iter: int32 = int32(0)
    z2: float64 = float64(0)
    der: complex128 = complex128(1 + 0j)
    der2: float64 = float64(1)
    while nb_iter < max_iterations and z2 < escape_radius and der2 > epsilon:
        der = der * p * z
        z = z**p + c
        nb_iter += 1
        z2 = z.real**2 + z.imag**2
        der2 = der.real**2 + der.imag**2
    device_array_niter[x, y] = nb_iter
    device_array_z2[x, y] = z2
    set_pixel_k(
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
    )
    k = device_array_k[x, y]
    packedrgb = set_pixel_color(device_array_rgb, x, y, k, palette_mode)
    return nb_iter, z2, k, packedrgb


@cuda_jit("(int32[:,:], float64[:,:], float64[:,:], int32[:,:], complex128, float64, float64, int32, int32, int32, int32, float64, complex128, int32, int32, int32)")
def fractal_kernel(
    device_array_niter,
    device_array_z2,
    device_array_k,
    device_array_rgb,
    topleft: complex128,
    xstep: float64,
    ystep: float64,
    fractalmode: int32,
    max_iterations: int32,
    p: int32,
    escape_radius: int32,
    epsilon: float64,
    juliaxy: complex128,
    k_mode: int32,
    palette_mode: int32,
    color_waves: int32,
) -> None:
    x, y = cuda_grid(2)
    if x < device_array_niter.shape[0] and y < device_array_niter.shape[1]:
        fractal_xy(
            x,
            y,
            device_array_niter,
            device_array_z2,
            device_array_k,
            device_array_rgb,
            topleft,
            xstep,
            ystep,
            fractalmode,
            max_iterations,
            p,
            escape_radius,
            epsilon,
            juliaxy,
            k_mode,
            palette_mode,
            color_waves,
        )


def compute_fractal(
    WINDOW_SIZE,
    xmax: float64,
    xmin: float64,
    ymin: float64,
    ymax: float64,
    fractalmode: int32,
    max_iterations: int32,
    power: int32,
    escape_radius: int32,
    epsilon: float64,
    juliaxy: complex128,
    k_mode: int32,
    palette_mode: int32,
    color_waves: int32,
):
    timerstart = timeit.default_timer()
    (screenw, screenh) = WINDOW_SIZE
    xstep = abs(xmax - xmin) / screenw
    ystep = abs(ymax - ymin) / screenh
    topleft = complex128(xmin + 1j * ymax)
    device_array_niter = init_array(screenw, screenh, numpy.int32)
    device_array_z2 = init_array(screenw, screenh, numpy.float64)
    device_array_k = init_array(screenw, screenh, numpy.float64)
    device_array_rgb = init_array(screenw, screenh, numpy.int32)
    if cuda_available():
        threadsperblock = compute_threadsperblock()
        blockspergrid = (
            ceil(screenw / threadsperblock[0]),
            ceil(screenh / threadsperblock[1]),
        )
        print("fractal_kernel types:")
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
        vectorized_fractal_xy = numpy.vectorize(fractal_xy, otypes=[int32, float64, float64, int32])# fractal_xy returns nb_iter, z2, k, packedrgb 
        vector_x = range(device_array_niter.shape[0])
        vector_y = range(device_array_niter.shape[1])
        result_arrays = vectorized_fractal_xy(vector_x,vector_y,
        # for x in range(device_array_niter.shape[0]):
        #     for y in range(device_array_niter.shape[1]):
        #         k, packedrgb = fractal_xy(
        #             x,
        #             y,
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
        output_array_niter, output_array_z2, output_array_k,  output_array_rgb = result_arrays
        # output_array_niter = device_array_niter
        # output_array_z2 = device_array_z2
        # output_array_k = device_array_k
        # output_array_rgb = device_array_rgb

    print(f"Frame calculated in {(timeit.default_timer() - timerstart)}s")
    return output_array_niter, output_array_z2, output_array_k, output_array_rgb
