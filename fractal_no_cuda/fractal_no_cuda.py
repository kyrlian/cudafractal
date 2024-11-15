import timeit
import numpy

from numpy import float64, complex128
from fractal_no_cuda.colors_no_cuda import set_pixel_color, set_pixel_k


def mandelbrot(
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
    color_mode,
    palette,
    color_waves,
) -> None:
    for x in range(device_array_niter.shape[0]):
        for y in range(device_array_niter.shape[1]):
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
                color_mode,
                color_waves,
            )
            set_pixel_color(device_array_rgb, device_array_k, x, y, palette)


def julia(
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
    color_mode,
    palette,
    color_waves,
) -> None:
    for x in range(device_array_niter.shape[0]):
        for y in range(device_array_niter.shape[1]):
            z = complex128(topleft + x * xstep - 1j * y * ystep)
            nb_iter = 0
            z2: float64 = float64(0)
            der = complex128(1 + 0j)
            der2: float64 = float64(1)
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
                color_mode,
                color_waves,
            )
            set_pixel_color(device_array_rgb, device_array_k, x, y, palette)


FRACTAL_MODES = [mandelbrot, julia]


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
    color_mode,
    palette,
    color_waves,
):
    timerstart = timeit.default_timer()
    (screenw, screenh) = WINDOW_SIZE
    xstep = abs(xmax - xmin) / screenw
    ystep = abs(ymax - ymin) / screenh
    topleft = complex128(xmin + 1j * ymax)
    device_array_niter = numpy.zeros((screenw, screenw, 1), dtype=numpy.uint32)
    device_array_z2 = numpy.zeros((screenw, screenw, 1), dtype=numpy.float64)
    device_array_k = numpy.zeros((screenw, screenw, 1), dtype=numpy.float64)
    device_array_rgb = numpy.zeros((screenw, screenw, 1), dtype=numpy.uint32)
    fractalmethod = FRACTAL_MODES[fractalmode]
    fractalmethod(
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
        color_mode,
        palette,
        color_waves,
    )
    print(f"Frame calculated in {(timeit.default_timer() - timerstart)}s")
    return device_array_niter, device_array_z2, device_array_k, device_array_rgb
