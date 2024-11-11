import numpy
import timeit
from math import log
from numba import float64, complex128
from fractal_cuda.colors_cuda import set_pixel_color, ColorMode


def set_pixel_k(
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
):
    # calculate k[0-1] based on color mode
    k = 0.0
    if z2 > escape_radius:
        match color_mode:
            case ColorMode.ITER_WAVES:
                mic = max_iterations / color_waves
                k = float64(nb_iter % mic / mic)
            case ColorMode.ITER:
                k = float64(nb_iter / max_iterations)
            case ColorMode.LOG_ITER:
                k = log(float64(nb_iter)) / log(float64(max_iterations))
            case ColorMode.R_Z2:
                # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                # TODO : z2 is slightly bigger than r, so k doesnt cover 0-1
                k = float64(escape_radius) / float64(z2)
            case ColorMode.LOG_R_Z2:
                k = log(float64(escape_radius)) / log(float64(z2))
            case ColorMode.INV_Z2:
                # k = math.sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                # k = sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                k = 1 / z2
    device_array_k[x, y] = k


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
):
    for x in range(device_array_niter.shape[0]):
        for y in range(device_array_niter.shape[1]):
            c = complex128(topleft + x * xstep - 1j * y * ystep)
            z = c
            nb_iter = 0
            z2 = 0
            der = complex128(1 + 0j)
            der2 = 1
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
):
    for x in range(device_array_niter.shape[0]):
        for y in range(device_array_niter.shape[1]):
            z = complex128(topleft + x * xstep - 1j * y * ystep)
            nb_iter = 0
            z2 = 0
            der = complex128(1 + 0j)
            der2 = 1
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
    maxiter,
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
    device_array_niter = numpy.zeros((screenw, screenw, 1))
    device_array_z2 = numpy.zeros((screenw, screenw, 1))
    device_array_k = numpy.zeros((screenw, screenw, 1))
    device_array_rgb = numpy.zeros((screenw, screenw, 1))
    fractalmethod = FRACTAL_MODES[fractalmode]
    fractalmethod(
        device_array_niter,
        device_array_z2,
        device_array_k,
        device_array_rgb,
        topleft,
        xstep,
        ystep,
        maxiter,
        power,
        escape_radius,
        epsilon,
        juliaxy,
        color_mode,
        palette,
        color_waves,
    )
    print(f"Frame calculated in {(timeit.default_timer() - timerstart)}s")
    return (
        device_array_niter,
        device_array_z2,
        device_array_k,
        device_array_rgb,
    )
