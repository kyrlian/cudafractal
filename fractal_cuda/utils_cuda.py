from math import floor
from numba import cuda

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
    mby = floor(gpu.MAX_THREADS_PER_BLOCK / mbx)
    return (mbx, mby)
