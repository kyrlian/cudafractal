import math
import sys
import numpy
from numba import cuda
from numba import complex128
import timeit
from fractal_cuda.fractal_cuda import FRACTAL_MODES


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
    cuda.detect()

    timerstart = timeit.default_timer()
    (screenw, screenh) = WINDOW_SIZE
    xstep = abs(xmax - xmin) / screenw
    ystep = abs(ymax - ymin) / screenh
    topleft = complex128(xmin + 1j * ymax)
    device_array = cuda.device_array((screenw, screenh), dtype=numpy.uint32)
    threadsperblock = compute_threadsperblock()  # (32, 16) #real size = 32*16
    blockspergrid = (
        math.ceil(screenw / threadsperblock[0]),
        math.ceil(screenh / threadsperblock[1]),
    )
    fractalmethod = FRACTAL_MODES[fractalmode]
    fractalmethod[blockspergrid, threadsperblock](
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
    output_array = device_array.copy_to_host()
    sys.stdout.write(f"Frame calculated in {(timeit.default_timer() - timerstart)}s\n")
    return output_array


def compute_threadsperblock():
    gpu = cuda.get_current_device()
    # https://stackoverflow.com/questions/48654403/how-do-i-know-the-maximum-number-of-threads-per-block-in-python-code-with-either
    # https://numba.pydata.org/numba-doc/dev/cuda/kernels.html#choosing-the-block-size
    # Best is to have fewer blocks with max thread per block (see cuda.detect)
    # thread block size should always be a multiple of 32
    # need to chose block size x/y so x*y ~~ MAX_THREADS_PER_BLOCK
    # and x/y ~~ display_width / display_heigth
    # and x*y is a multiple of 32 (4*8)
    # mbx = math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK * display_width / display_heigth) / 8) * 8
    # mby = math.floor(gpu.MAX_THREADS_PER_BLOCK / mbx / 4) * 4
    # if we dont care about the ratio mbx/mby :
    mbx = 32
    mby = math.floor(gpu.MAX_THREADS_PER_BLOCK / mbx)
    return (mbx, mby)
