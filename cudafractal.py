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
from scipy.misc import toimage
import tkinter
from PIL import Image, ImageTk

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


def create_image(xmin, xmax, ymin, ymax, max_iter, display_width, display_heigth):
    timerstart = timeit.default_timer()
    xstride = abs(xmax - xmin) / display_width
    ystride = abs(ymax - ymin) / display_heigth
    topleft = complex128(xmin + 1j * ymax)
    # Init image array on host, and send to device
    # image_array = numpy.zeros((display_width , display_heigth, 3), dtype=numpy.uint8)
    # device_array = cuda.to_device(image_array)
    # Init image array directly on device
    device_output_array = cuda.device_array(
        (display_width, display_heigth, 3), dtype=numpy.uint8
    )
    threadsperblock = (32, 16)
    blockspergrid = (
        math.ceil(display_width / threadsperblock[0]),
        math.ceil(display_heigth / threadsperblock[1]),
    )
    mandelbrot[blockspergrid, threadsperblock](
        device_output_array, topleft, xstride, ystride, max_iter
    )
    output_array = device_output_array.copy_to_host()
    sys.stdout.write(
        "\r" + "Frame calculated in %f s" % (timeit.default_timer() - timerstart)
    )
    # directly create and show a pylot plot (good for one shot)
    # pyplot.imshow(output_array)
    # pyplot.show()
    return output_array


# INITs
max_iterations = 255
display_heigth = 1440
display_ratio = 4 / 3
display_width = display_heigth * display_ratio
xcenter = 0  # xmin, xmax = -2.0, 1.0
ycenter = 0  # ymin, ymax = -1.5, 1.5
yheight = 3
xwidth = yheight * display_ratio  # 4
xmin = xcenter - xwidth / 2  # -2
xmax = xcenter + xwidth / 2  # 2
ymin = ycenter - yheight / 2  # -1.5
ymax = ycenter + yheight / 2  # +1.5
zoomrate = 2

# Handle user click events
def reset_size():
    global xcenter, ycenter, yheight
    xcenter = 0
    ycenter = 0
    yheight = 3
    recalc_size()

def recalc_size():
    global xwidth, xmin, xmax, ymin, ymax
    xwidth = yheight * display_ratio
    xmin = xcenter - xwidth / 2
    xmax = xcenter + xwidth / 2
    ymin = ycenter - yheight / 2
    ymax = ycenter + yheight / 2

def clickzoomin(event):
    mouseX, mouseY = int(event.x), int(event.y)
    # Zoom in
    sys.stdout.write("Zoom in")
    xcenter = xmin + mouseX * (xmax - xmin) / display_width
    ycenter = ymin + mouseY * (ymax - ymin) / display_heigth
    yheight /= zoomrate
    swapimage()

def clickzoomout(event):
    mouseX, mouseY = int(event.x), int(event.y)
    # Zoom out
    sys.stdout.write("Zoom out")
    yheight *= zoomrate
    swapimage()

def addimage(canvas):
    recalc_size()
    image_array = create_image(xmin, xmax, ymin, ymax, max_iterations, display_width, display_heigth)
    img = ImageTk.PhotoImage(image=Image.fromarray(image_array))
    canvas.create_image(0, 0, image=img, tags="fractal")

def swapimage():
    global canvas
    canvas.delete("fractal")
    addimage(canvas)

tkroot = tkinter.Tk()
canvas = tkinter.Canvas(tkroot, width=display_width, height=display_heigth)
canvas.pack()
addimage(canvas)
canvas.bind("<Button-1>", clickzoomin)
canvas.bind("<Button-2>", clickzoomout)
tkroot.mainloop()