# from timeit import default_timer
from math import ceil
from typing import List

from utils.types import (
    type_math_int,
    type_math_float,
    type_math_complex,
    type_enum_int,
    type_color_int,
)
from utils.cuda import (
    cuda_jit,
    cuda_grid,
    cuda_reduce,
)
from utils.cuda import (
    compute_threadsperblock,
    cuda_copy_to_device,
    cuda_copy_to_host,
)
from utils.timer import timing_wrapper

from fractal.colors import Normalization_Mode, Palette_Mode, color_kernel
from fractal.fractal_math import fractal_xy, Fractal_Mode


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
def compute_min_max_cuda(device_array):
    # https://numba.pydata.org/numba-doc/dev/cuda-reference/memory.html?highlight=ravel#numba.cuda.cudadrv.devicearray.DeviceNDArray.ravel
    # flatten the array as required by reduction - keeping it in device
    flat_array = device_array.ravel()
    return min_reduce(flat_array), max_reduce(flat_array)


# TODO read stuff from AppState
@timing_wrapper
def compute_fracta_cuda(
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
        niter_min, niter_max = compute_min_max_cuda(device_array_niter)
        z2_min, z2_max = compute_min_max_cuda(device_array_z2)
        der2_min, der2_max = compute_min_max_cuda(device_array_der2)
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
