from math import floor
from numpy import zeros as np_zeros

try:
    from numba.cuda import (
        get_current_device as cuda_get_current_device,
        device_array as cuda_device_array,
        to_device as cuda_copy_to_device,
        jit as cuda_jit,
        detect as cuda_detect,
        is_available as cuda_available,
        grid as cuda_grid,
    )

    def compute_threadsperblock(screenw, screenh):
        gpu = cuda_get_current_device()
        # https://stackoverflow.com/questions/48654403/how-do-i-know-the-maximum-number-of-threads-per-block-in-python-code-with-either
        # https://numba.pydata.org/numba-doc/dev/cuda/kernels.html#choosing-the-block-size
        # Best is to have fewer blocks with max thread per block (see cuda.detect)
        # thread block size should always be a multiple of 32
        # need to chose block size x/y so x*y ~~ MAX_THREADS_PER_BLOCK
        # and x/y ~~ display_width / display_heigth
        # and x*y is a multiple of 32 (4*8)
        # mbx = floor(sqrt(gpu.MAX_THREADS_PER_BLOCK * screenw / screenh) / 8) * 8
        # mby = floor(gpu.MAX_THREADS_PER_BLOCK / mbx / 4) * 4
        # if we dont care about the ratio mbx/mby :
        mbx = 32
        mby = floor(gpu.MAX_THREADS_PER_BLOCK / mbx)
        print(
            f"compute_threadsperblock: MAX_THREADS_PER_BLOCK:{gpu.MAX_THREADS_PER_BLOCK}, mbx: {mbx}, mby: {mby}"
        )
        return (mbx, mby)

    def init_array(dimx, dimy, dtype):
        return cuda_device_array((dimx, dimy), dtype=dtype)

    def cuda_copy_to_host(device_array):
        return device_array.copy_to_host()

except ImportError:
    print("numba cuda not installed")

    # If numba cuda is not installed, use noop operations
    def cuda_jit(
        func_or_sig=None,
        device=False,
        inline=False,
        link=[],
        debug=None,
        opt=True,
        lineinfo=False,
        cache=False,
        **kws,
    ):
        def wrapper(func):
            return func

        return wrapper

    def cuda_detect():
        return False

    def cuda_available():
        return False

    def cuda_grid(n):
        return (1, 1)

    def compute_threadsperblock(screenw, screenh):
        return (1, 1)

    def init_array(dimx, dimy, dtype):
        return np_zeros((dimx, dimy), dtype=dtype)

    def cuda_copy_to_device(host_array):
        return host_array

    def cuda_copy_to_host(device_array):
        return device_array
