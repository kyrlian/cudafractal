# Write python code using nvidia cuda to render a mandelbrot fractal.
# When the user left clicks, zoom in by a factor of 2 into the fractal at the cursor.
# When the user right clicks, zoom out by a factor of 2.

import math
import timeit
import sys
import numpy
from matplotlib import pyplot
from numba import cuda
from numba import float64, int8, complex128
from math import log


@cuda.jit("void(int8[:,:,:], int32, int32, int32, int32)", device=True)
def set_image_color(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:  # BLACK
        return
    k = 6.0 * log(float64(iterations)) / log(float64(max_iterations))
    fract = k - math.floor(k)
    r, g, b = 0, 0, 0
    if k < 1:  # RED to YELLOW
        r, g, b = 1, fract, 0
    elif k < 2:  # YELLOW to GREEN
        r, g, b = 1 - fract, 1, 0
    elif k < 3:  # GREEN to CYAN
        r, g, b = 0, 1, fract
    elif k < 4:  # CYAN to BLUE
        r, g, b = 0, 1 - fract, 1
    elif k < 5:  # BLUE to MAGENTA
        r, g, b = fract, 0, 1
    elif k < 6:  # MAGENTA to RED
        r, g, b = 1, 0, 1 - fract
    image_array[y, x, 0] = int8(r * 255)
    image_array[y, x, 1] = int8(g * 255)
    image_array[y, x, 2] = int8(b * 255)


@cuda.jit("void(int8[:,:,:], complex128, float64, float64, int32)")
def mandelbrot(image_array, topleft, xstride, ystride, max_iter):
    y, x = cuda.grid(2)
    if x < image_array.shape[1] and y < image_array.shape[0]:
        c = complex128(topleft + x * xstride - 1j * y * ystride)
        z = complex128(0 + 0j)
        nbi = 0
        while nbi < max_iter and z.real * z.real + z.imag * z.imag < 4:
            z = z * z + c
            nbi += 1
        set_image_color(image_array, x, y, nbi, max_iter)


def create_image(xmin, xmax, ymin, ymax, max_iter, base_accuracy):
    timerstart = timeit.default_timer()

    if abs(xmax - xmin) > abs(ymax - ymin):
        ny = base_accuracy
        nx = int((base_accuracy * abs(xmax - xmin) / abs(ymax - ymin)))
    else:
        nx = base_accuracy
        ny = int(base_accuracy * abs(ymax - ymin) / abs(xmax - xmin))
    xstride = abs(xmax - xmin) / nx
    ystride = abs(ymax - ymin) / ny
    topleft = complex128(xmin + 1j * ymax)
    image_array = numpy.zeros((ny, nx, 3), dtype=numpy.uint8)

    # device_array = cuda.to_device(array_of_random)
    # device_output_array = cuda.device_array(size)
    # kernel[512, 512](device_array,  .5, device_output_array)
    # output_array = device_output_array.copy_to_host()

    device_image = cuda.to_device(image_array)
    threadsperblock = (32, 16)
    blockspergrid = (
        math.ceil(image_array.shape[0] / threadsperblock[0]),
        math.ceil(image_array.shape[1] / threadsperblock[1]),
    )
    # blockspergrid = (image_array.shape[0] + threadsperblock[0] - 1) // threadsperblock[0], (image_array.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    mandelbrot[blockspergrid, threadsperblock](
        device_image, topleft, xstride, ystride, max_iter
    )
    image_array = device_image.copy_to_host()

    sys.stdout.write(
        "\r" + "Frame calculated in %f s" % (timeit.default_timer() - timerstart)
    )
    pyplot.imshow(image_array)
    pyplot.show()
    return image_array


# Initialize the variables for the fractal
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
base_accuracy = 1024
max_iterations = 255

image_array = create_image(xmin, xmax, ymin, ymax, max_iterations, base_accuracy)

# Handle user click events


def click(event):
    mouseX, mouseY = int(event.xdata), int(event.ydata)
    if event.button == "left":
        # Zoom in
        xmin = xmin + mouseX * (xmax - xmin) / image_array.shape[0] - (xmax - xmin) / 4
        ymin = ymin + mouseY * (ymax - ymin) / image_array.shape[1] - (ymax - ymin) / 4
        xmax = xmin + (xmax - xmin) / 2
        ymax = ymin + (ymax - ymin) / 2
    elif event.button == "right":
        # Zoom out
        xmin = xmin - (xmax - xmin) / 2
        ymin = ymin - (ymax - ymin) / 2
        xmax = xmin + (xmax - xmin) * 2
        ymax = ymin + (ymax - ymin) * 2
    else:
        return
    create_image(xmin, xmax, ymin, ymax, max_iterations, base_accuracy)
