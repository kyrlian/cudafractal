import timeit
import numpy
from math import ceil
from numpy import float64, complex128
from fractal.colors import set_pixel_color, set_pixel_k
from fractal.utils_cuda import cuda_jit, cuda_available, cuda_grid, compute_threadsperblock, init_array


@cuda_jit()
def mandelbrot_xy(
    x,
    y,
    device_array_niter,
    device_array_z2,
    device_array_k,
    device_array_rgb,
    topleft,
    xstep,
    ystep,
    max_iterations,
    p,
    escape_radius,
    eps,
    juliaxy,
    k_mode,
    palette_mode,
    color_waves,
) -> None:
    c: complex128 = complex128(topleft + x * xstep - 1j * y * ystep)
    z: complex128 = c
    nb_iter: int = 0
    z2: float64 = float64(0)
    der: complex128 = complex128(1 + 0j)
    der2: float64 = float64(1)
    while nb_iter < max_iterations and z2 < escape_radius and der2 > eps:
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
    set_pixel_color(device_array_rgb, device_array_k, x, y, palette_mode)


@cuda_jit()
def mandelbrot_kernel(
    device_array_niter,
    device_array_z2,
    device_array_k,
    device_array_rgb,
    topleft,
    xstep,
    ystep,
    max_iterations,
    p,
    escape_radius,
    eps,
    juliaxy,
    k_mode,
    palette_mode,
    color_waves,
) -> None:
    x, y = cuda_grid(2)
    if x < device_array_niter.shape[0] and y < device_array_niter.shape[1]:
        mandelbrot_xy(
            x,
            y,
            device_array_niter,
            device_array_z2,
            device_array_k,
            device_array_rgb,
            topleft,
            xstep,
            ystep,
            max_iterations,
            p,
            escape_radius,
            eps,
            juliaxy,
            k_mode,
            palette_mode,
            color_waves,
        )


@cuda_jit()
def julia_xy(
    x,
    y,
    device_array_niter,
    device_array_z2,
    device_array_k,
    device_array_rgb,
    topleft,
    xstep,
    ystep,
    max_iterations,
    p,
    escape_radius,
    eps,
    juliaxy,
    k_mode,
    palette_mode,
    color_waves,
) -> None:
    z = complex128(topleft + x * xstep - 1j * y * ystep)
    nb_iter = 0
    z2 = float64(0)
    der = complex128(1 + 0j)
    der2 = float64(1)
    while nb_iter < max_iterations and z2 < escape_radius and der2 > eps:
        # TODO test julia with/without der
        der = der * p * z
        z = z**p + juliaxy
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
    set_pixel_color(device_array_rgb, device_array_k, x, y, palette_mode)


@cuda_jit()
def julia_kernel(
    device_array_niter,
    device_array_z2,
    device_array_k,
    device_array_rgb,
    topleft,
    xstep,
    ystep,
    max_iterations,
    p,
    escape_radius,
    eps,
    juliaxy,
    k_mode,
    palette_mode,
    color_waves,
) -> None:
    x, y = cuda_grid(2)
    if x < device_array_niter.shape[0] and y < device_array_niter.shape[1]:
        julia_xy(
            x,
            y,
            device_array_niter,
            device_array_z2,
            device_array_k,
            device_array_rgb,
            topleft,
            xstep,
            ystep,
            max_iterations,
            p,
            escape_radius,
            eps,
            juliaxy,
            k_mode,
            palette_mode,
            color_waves,
        )


FRACTAL_NAMES = ["mandelbrot", "julia"]
FRACTAL_MODES = [mandelbrot_xy, julia_xy]
FRACTAL_KERNELS = [mandelbrot_kernel, julia_kernel]

def compute_fractal(
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
    k_mode,
    palette_mode,
    color_waves,
):
    timerstart = timeit.default_timer()
    (screenw, screenh) = WINDOW_SIZE
    xstep = abs(xmax - xmin) / screenw
    ystep = abs(ymax - ymin) / screenh
    topleft = complex128(xmin + 1j * ymax)
    device_array_niter = init_array(screenw, screenh, numpy.uint32)
    device_array_z2 = init_array(screenw, screenh, numpy.float64)
    device_array_k = init_array(screenw, screenh, numpy.float64)
    device_array_rgb = init_array(screenw, screenh, numpy.uint32)
    if cuda_available():
        threadsperblock = compute_threadsperblock()
        blockspergrid = (
            ceil(screenw / threadsperblock[0]),
            ceil(screenh / threadsperblock[1]),
        )
        fractal_kernel = FRACTAL_KERNELS[fractalmode]
        fractal_kernel[blockspergrid, threadsperblock](
            device_array_niter,
            device_array_z2,
            device_array_k,
            device_array_rgb,
            topleft,
            xstep,
            ystep,
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
        fractal_method = FRACTAL_MODES[fractalmode]
        for x in range(device_array_niter.shape[0]):
            for y in range(device_array_niter.shape[1]):
                fractal_method(
                    x,
                    y,
                    device_array_niter,
                    device_array_z2,
                    device_array_k,
                    device_array_rgb,
                    topleft,
                    xstep,
                    ystep,
                    max_iterations,
                    power,
                    escape_radius,
                    epsilon,
                    juliaxy,
                    k_mode,
                    palette_mode,
                    color_waves,
                )
        output_array_niter = device_array_niter
        output_array_z2 = device_array_z2
        output_array_k = device_array_k
        output_array_rgb = device_array_rgb

    print(f"Frame calculated in {(timeit.default_timer() - timerstart)}s")
    return output_array_niter, output_array_z2, output_array_k, output_array_rgb
