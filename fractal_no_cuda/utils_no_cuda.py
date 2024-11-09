import sys
import numpy
from numba import complex128
import timeit
from fractal_no_cuda.fractal_no_cuda import FRACTAL_MODES


def create_image(
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
    juliaxy,
    currentcolormode,
    currentpalette,
    currentcolor_waves,
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
        juliaxy,
        currentcolormode,
        currentpalette,
        currentcolor_waves,
    )
    output_array = device_array
    print(f"Frame calculated in {(timeit.default_timer() - timerstart)}s")
    return output_array
