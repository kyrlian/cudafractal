# Write python code using nvidia cuda to render a mandelbrot fractal. 
# When the user left clicks, zoom in by a factor of 2 into the fractal at the cursor. 
# When the user right clicks, zoom out by a factor of 2.

import math, timeit, sys
import numpy
from matplotlib import pyplot
from numba import cuda
from numba import float64, int8, complex128
from math import log
from cmath import isinf, sinh, cosh, exp, phase
import colorsys

@cuda.jit('void(int8[:,:,:], int32, int32, int32, int32)', device=True)
def set_log_color_rgb(image_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:
        return
    k = 3.0 * log(float64(iterations)) / log(float64(max_iterations))
    # Convert a single pixel from HSV to RGB
    #h, s, v = k, 0, 0
    #r, g, b = colorsys.hsv_to_rgb(h, s, v)
    if k < 1:
        image_array[y, x, 0] = int8(255 * k)
        image_array[y, x, 1] = 0
        image_array[y, x, 2] = 0
    elif k < 2:
        image_array[y, x, 0] = 255
        image_array[y, x, 1] = int8(255 * (k - 1))
        image_array[y, x, 2] = 0
    else:
        image_array[y, x, 0] = 255
        image_array[y, x, 1] = 255
        image_array[y, x, 2] = int8(255 * (k - 2))

@cuda.jit('void(int8[:,:,:], complex128, float64, float64, int32)')
def mandelbrot(image_array, topleft, xstride, ystride, max_iter):
    y, x = cuda.grid(2)
    if x < image_array.shape[1] and y < image_array.shape[0]:
        c = complex128(topleft + x * xstride - 1j * y * ystride)
        z = complex128(0+0j)
        nbi = 0
        while nbi < max_iter and z.real * z.real + z.imag * z.imag < 4:
            z = z * z + c
            nbi += 1
        set_log_color_rgb(image_array, x, y, nbi, max_iter)

def run_kernel(kernel, image_array, topleft, xstride, ystride, max_iter):
    start = timeit.default_timer()
    dimage = cuda.to_device(image_array)
    threadsperblock = (32, 16)
    blockspergrid = (math.ceil(image_array.shape[0] / threadsperblock[0]), math.ceil(image_array.shape[1] / threadsperblock[1]))
    kernel[blockspergrid, threadsperblock](dimage, topleft, xstride, ystride, max_iter)
    dimage.to_host()
    sys.stdout.write('\r' + "Fractal calculated on GPU in %f s" % (timeit.default_timer() - start))

def create_image_array(kernel, xmin, xmax, ymin, ymax, max_iter, base_accuracy):
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
    run_kernel(kernel, image_array, topleft, xstride, ystride, max_iter)
    return image_array

def display_image(xmin, xmax, ymin, ymax, max_iterations, base_accuracy):
    # Call the kernel to render the fractal
    image_array = create_image_array(mandelbrot, xmin, xmax, ymin, ymax, max_iterations, base_accuracy)
    pyplot.imshow(image_array)
    pyplot.show()
    return image_array

# Initialize the variables for the fractal
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
base_accuracy = 1024
max_iterations = 255

image_array = display_image(xmin, xmax, ymin, ymax, max_iterations, base_accuracy)

# Handle user click events
def click(event):
    mouseX, mouseY = int(event.xdata), int(event.ydata)
    if event.button == 'left':
        # Zoom in
        xmin=xmin+mouseX*(xmax-xmin)/image_array.shape[0]-(xmax-xmin)/4
        ymin=ymin+mouseY*(ymax-ymin)/image_array.shape[1]-(ymax-ymin)/4
        xmax=xmin+(xmax-xmin)/2
        ymax=ymin+(ymax-ymin)/2
    elif event.button == 'right':
        # Zoom out
        xmin=xmin-(xmax-xmin)/2
        ymin=ymin-(ymax-ymin)/2
        xmax=xmin+(xmax-xmin)*2
        ymax=ymin+(ymax-ymin)*2
    else:
        return
    display_image(xmin, xmax, ymin, ymax, max_iterations, base_accuracy)