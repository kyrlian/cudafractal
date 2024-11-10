from numba import complex128
import numpy
import timeit

def mandelbrot(
    device_array_niter,
    device_array_z2,
    topleft,
    xstep,
    ystep,
    maxiter,
    p,
    escape_radius,
    eps,
    juliaxy,
    cmode,
    palette,
    color_waves,
):
    for x in range(device_array_niter.shape[0]):
        for y in range(device_array_niter.shape[1]):
            c = complex128(topleft + x * xstep - 1j * y * ystep)
            z = c
            nbi = 0
            z2 = 0
            der = complex128(1 + 0j)
            der2 = 1
            while nbi < maxiter and z2 < escape_radius and der2 > eps:
                der = der * p * z
                z = z**p + c
                nbi += 1
                z2 = z.real**2 + z.imag**2
                der2 = der.real**2 + der.imag**2
            device_array_niter[x, y] = nbi
            device_array_z2[x, y] = z2


def julia(
    device_array_niter,
    device_array_z2,
    topleft,
    xstep,
    ystep,
    maxiter,
    p,
    escape_radius,
    eps,
    juliaxy,
    cmode,
    palette,
    color_waves,
):
    for x in range(device_array_niter.shape[0]):
        for y in range(device_array_niter.shape[1]):
            z = complex128(topleft + x * xstep - 1j * y * ystep)
            nbi = 0
            z2 = 0
            der = complex128(1 + 0j)
            der2 = 1
            while nbi < maxiter and z2 < escape_radius and der2 > eps:
                # TODO test julia with/without der
                der = der * p * z
                z = z**p + juliaxy
                nbi += 1
                z2 = z.real**2 + z.imag**2
                der2 = der.real**2 + der.imag**2
            device_array_niter[x, y] = nbi
            device_array_z2[x, y] = z2


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
    escaper,
    epsilon,
    juliaxy
):
    
    timerstart = timeit.default_timer()
    (screenw, screenh) = WINDOW_SIZE
    xstep = abs(xmax - xmin) / screenw
    ystep = abs(ymax - ymin) / screenh
    topleft = complex128(xmin + 1j * ymax)
    device_array = numpy.zeros((screenw, screenw, 1))
    fractalmethod = FRACTAL_MODES[fractalmode]
    fractalmethod(
        device_array,
        topleft,
        xstep,
        ystep,
        maxiter,
        power,
        escaper,
        epsilon,
        juliaxy
    )
    output_array = device_array
    print(f"Frame calculated in {(timeit.default_timer() - timerstart)}s")
    return output_array
