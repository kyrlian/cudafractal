import numpy
from ui.main_ui import create_image
from fractal_cuda.fractal_cuda import mandelbrot
from numba import complex128


def create_image_profiling():
    pixel_array = numpy.zeros((512, 512, 3))
    create_image(
        mandelbrot,
        pixel_array,
        -2.5,
        1.5,
        -1.5,
        1.5,
        255,
        2,
        4,
        0.001,
        complex128(0 + 0j),
        0,
        0,
        1,
    )
