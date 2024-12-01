from math import floor
import numpy

try:
    from numba import cuda
    from numba.cuda import jit as myjit, detect as mydetect, grid as mygrid

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

    def init_array(dimx, dimy, dtype):
        # device_array_niter = cuda.device_array((screenw, screenh), dtype=numpy.uint32)
        return cuda.device_array((dimx, dimy), dtype=dtype)


except ImportError:
    print("numba cuda not installed")

    # If numba cuda is not installed, use a noop operations
    def myjit(func):
        return func

    def mydetect():
        return False

    def compute_threadsperblock():
        return (1, 1)

    def mygrid(n):
        return (1, 1)

    def init_array(dimx, dimy, dtype):
        # device_array_niter = numpy.zeros((screenw, screenw, 1), dtype=numpy.uint32)
        return numpy.zeros((dimx, dimy, 1), dtype=dtype)
