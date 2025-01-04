from timeit import default_timer
from math import ceil
from enum import IntEnum
from typing import Tuple, List
from numpy import vectorize as np_vectorize
from utils.types import (
    type_math_int,
    type_math_float,
    type_math_complex,
    type_enum_int,
    type_color_int,
)
from utils.cuda import (
    cuda_jit,
    cuda_available,
    cuda_grid,
    compute_threadsperblock,
    init_array,
    cuda_copy_to_device,
    cuda_copy_to_host,
    cuda_reduce,
)
from fractal.colors import color_kernel, color_cpu
from utils.timer import timing_wrapper


class Fractal_Mode(IntEnum):
    MANDELBROT = 0
    JULIA = 1


@cuda_jit(
    "(int32, int32, complex128, float64, float64, uint8, int32, int32, int32, float64, complex128)",
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
) -> Tuple[type_math_int, type_math_float, type_math_float]:
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
    return nb_iter, z2, der2


@cuda_jit(
    "(int32[:,:], float64[:,:], float64[:,:], complex128, float64, float64, uint8, int32, int32, int32, float64, complex128)"
)
def fractal_kernel(
    device_array_niter,
    device_array_z2,
    device_array_der2,
    topleft: type_math_complex,
    xstep: type_math_float,
    ystep: type_math_float,
    fractalmode: type_enum_int,
    max_iterations: type_math_int,
    power: type_math_int,
    escape_radius: type_math_int,
    epsilon: type_math_float,
    juliaxy: type_math_complex,
) -> None:
    x, y = cuda_grid(2)
    if x < device_array_niter.shape[0] and y < device_array_niter.shape[1]:
        nb_iter, z2, der2 = fractal_xy(
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
        device_array_niter[x, y] = nb_iter
        device_array_z2[x, y] = z2
        device_array_der2[x, y] = der2


@cuda_reduce
def min_reduce(a, b):
    # https://numba.readthedocs.io/en/stable/cuda/reduction.html
    return min(a, b)

@cuda_reduce
def max_reduce(a, b):
    return max(a, b)

@timing_wrapper
def get_min_max(device_array):
    # https://numba.pydata.org/numba-doc/dev/cuda-reference/memory.html?highlight=ravel#numba.cuda.cudadrv.devicearray.DeviceNDArray.ravel
    # flatten the array as required by reduction - keeping it in device
    flat_array_niter = device_array.ravel()
    return min_reduce(flat_array_niter), max_reduce(flat_array_niter)

@timing_wrapper
def fractal_cpu(
    host_array_niter,
    host_array_z2,
    host_array_der2,
    topleft: type_math_complex,
    xstep: type_math_float,
    ystep: type_math_float,
    fractalmode: type_enum_int,
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
def compute_min_max_cpu(
    host_array_niter,
    host_array_z2,
    host_array_der2,
):
    # TODO: optimize this function using numpy
    niter_min = niter_max = host_array_niter[0][0]
    z2_min = z2_max = host_array_z2[0][0]
    der2_min = der2_max = host_array_der2[0][0]
    for x in range(host_array_niter.shape[0]):
        for y in range(host_array_niter.shape[1]):
            if niter_min > host_array_niter[x][y]:
                niter_min = host_array_niter[x][y]
            if niter_max < host_array_niter[x][y]:
                niter_max = host_array_niter[x][y]
            if z2_min > host_array_z2[x][y]:
                z2_min = host_array_z2[x][y]
            if z2_max < host_array_z2[x][y]:
                z2_max = host_array_z2[x][y]
            if der2_min > host_array_der2[x][y]:
                der2_min = host_array_der2[x][y]
            if der2_max < host_array_der2[x][y]:
                der2_max = host_array_der2[x][y]
    return niter_min, niter_max, z2_min, z2_max, der2_min, der2_max


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
    fractalmode: type_enum_int,
    max_iterations: type_math_int,
    power: type_math_int,
    escape_radius: type_math_int,
    epsilon: type_math_float,
    juliaxy: type_math_complex,
    normalization_mode: type_enum_int,
    palette_mode: type_enum_int,
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
    if cuda_available():
        # Copy host array to device
        device_array_niter = cuda_copy_to_device(host_array_niter)
        device_array_z2 = cuda_copy_to_device(host_array_z2)
        device_array_der2 = cuda_copy_to_device(host_array_der2)
        device_array_k = cuda_copy_to_device(host_array_k)
        device_array_rgb = cuda_copy_to_device(host_array_rgb)
        device_array_palette = cuda_copy_to_device(custom_palette)
        # Compute block and threads
        threadsperblock = compute_threadsperblock(screenw, screenh)
        blockspergrid = (
            ceil(screenw / threadsperblock[0]),
            ceil(screenh / threadsperblock[1]),
        )
        # Run kernels
        if recalc_fractal:
            fractal_kernel[blockspergrid, threadsperblock](
                device_array_niter,
                device_array_z2,
                device_array_der2,
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
            niter_min, niter_max = get_min_max(device_array_niter)
            z2_min, z2_max = get_min_max(device_array_z2)
            der2_min, der2_max = get_min_max(device_array_der2)
            # TODO: store niter_min, niter_max, z2_min, z2_max, der2_min, der2_max in AppState
        if recalc_fractal or recalc_color:
            # color is calculated with fractal when it's called, but can be called by itself
            color_kernel[blockspergrid, threadsperblock](
                device_array_niter,
                device_array_z2,
                device_array_der2,
                device_array_k,
                device_array_rgb,
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
                device_array_palette,
                palette_width,
                palette_shift,
            )
        # copy arrays back to host
        host_array_niter = cuda_copy_to_host(device_array_niter)
        host_array_z2 = cuda_copy_to_host(device_array_z2)
        host_array_der2 = cuda_copy_to_host(device_array_der2)
        host_array_k = cuda_copy_to_host(device_array_k)
        host_array_rgb = cuda_copy_to_host(device_array_rgb)
    else:  # No cuda
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
            niter_min, niter_max, z2_min, z2_max, der2_min, der2_max = (
                compute_min_max_cpu(host_array_niter, host_array_z2, host_array_der2)
            )
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
    # print(f"Frame calculated in {(default_timer() - timerstart)}s")
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
