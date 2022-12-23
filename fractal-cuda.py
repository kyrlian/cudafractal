# Write python code using nvidia cuda to render a mandelbrot fractal.
# When the user left clicks, zoom in by a factor of 2 into the fractal at the cursor.
# When the user right clicks, zoom out by a factor of 2.

import math
import timeit
import sys
from numba import cuda
from numba import float64, complex128
from math import log
import pygame

@cuda.jit #("void(int8[:,:,:], int32, int32, int32, int32)", device=True)
def set_image_color(pixel_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:  # BLACK
        pixel_array[x, y] = (0, 0, 0)
    else:
        k = 6.0 * log(float64(iterations)) / log(float64(max_iterations))
        # k = 6.0 * float64(iterations) / float64(max_iterations)
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
        pixel_array[x, y] = (r * 255, g * 255, b * 255)

@cuda.jit #("void(int8[:,:,:], complex128, float64, float64, int32)")
def mandelbrot_cuda(pixel_array, topleft, xstride, ystride, max_iter):
    x, y = cuda.grid(2)
    if x < pixel_array.shape[0] and y < pixel_array.shape[1]:
        c = complex128(topleft + x * xstride - 1j * y * ystride)
        z = complex128(0 + 0j)
        nbi = 0
        while nbi < max_iter and z.real * z.real + z.imag * z.imag < 4:
            z = z * z + c
            nbi += 1
        set_image_color(pixel_array, x, y, nbi, max_iter)

def create_image(pixel_array, xmin, xmax, ymin, ymax, max_iter):
    timerstart = timeit.default_timer()
    xstride = abs(xmax - xmin) / pixel_array.shape[0]
    ystride = abs(ymax - ymin) / pixel_array.shape[1]
    topleft = complex128(xmin + 1j * ymax)
    # Init image array on host, and send to device
    # image_array = numpy.zeros((display_width , display_heigth, 3), dtype=numpy.uint8)
    device_array = cuda.to_device(pixel_array)
    # Init image array directly on device
    # device_output_array = cuda.device_array((pixel_array.shape[0], pixel_array.shape[1], 3), dtype=numpy.uint8)
    threadsperblock = compute_threadsperblock() # (32, 16) #real size = 32*16
    blockspergrid = (
        math.ceil(pixel_array.shape[0] / threadsperblock[0]),
        math.ceil(pixel_array.shape[1] / threadsperblock[1]),
    )
    mandelbrot_cuda[blockspergrid, threadsperblock](device_array, topleft, xstride, ystride, max_iter)
    pixel_array = device_array.copy_to_host()
    sys.stdout.write(
        "\r" + "Frame calculated in %f s" % (timeit.default_timer() - timerstart)
    )

cuda.detect()

def compute_threadsperblock():
    gpu = cuda.get_current_device()
    # https://stackoverflow.com/questions/48654403/how-do-i-know-the-maximum-number-of-threads-per-block-in-python-code-with-either
    # https://numba.pydata.org/numba-doc/dev/cuda/kernels.html#choosing-the-block-size
    # Best is to have fewer blocks with max thread per block (see cuda.detect)
    # thread block size should always be a multiple of 32
    # need to chose block size x/y so x*y ~~ MAX_THREADS_PER_BLOCK
    # and x/y ~~ display_width / display_heigth
    # and x*y is a multiple of 32 (4*8)
    #mbx = math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK * display_width / display_heigth) / 8) * 8
    #mby = math.floor(gpu.MAX_THREADS_PER_BLOCK / mbx / 4) * 4
    # if we dont care about the ratio mbx/mby :
    mbx = 32
    mby = math.floor(gpu.MAX_THREADS_PER_BLOCK / mbx)
    threadsperblock = (mbx , mby)

# INITs
max_iterations = 255
display_heigth=256
display_ratio = 4/3
display_width = math.floor(display_heigth * display_ratio)
window_size = display_width, display_heigth
zoomrate = 2

# Handle user click events
def reset_size():
    sys.stdout.write("Reset \n")
    global xcenter, ycenter, yheight
    xcenter = -0.5
    ycenter = 0
    yheight = 3

def recalc_size():
    global xwidth, xmin, xmax, ymin, ymax
    xwidth = yheight * display_width / display_heigth
    xmin = xcenter - xwidth / 2
    xmax = xcenter + xwidth / 2
    ymin = ycenter - yheight / 2
    ymax = ycenter + yheight / 2


def zoom(pos, zoomrate):
    global xcenter, ycenter, yheight
    (mouseX, mouseY) = pos
    xcenter = xmin + mouseX * (xmax - xmin) / display_width
    ycenter = ymin + (display_heigth - mouseY) * (ymax - ymin) / display_heigth
    yheight /= zoomrate
    sys.stdout.write(
        "Zoom %f,%f, (%f,%f), factor %f \n"
        % (mouseX, mouseY, xcenter, ycenter, zoomrate)
    )


def redraw(pixel_array):
    recalc_size()
    create_image(pixel_array, xmin, xmax, ymin, ymax, max_iterations)
    pygame.display.flip()


# Initialize pygame
pygame.init()
screen = pygame.display.set_mode(window_size, pygame.HWSURFACE)
# Get the PixelArray object for the screen
screen_pixels = pygame.PixelArray(screen)
# Init the display
reset_size()
redraw(screen_pixels)
# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_r:
                reset_size()
                redraw(screen_pixels)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 1 - left click
            # 2 - middle click
            # 3 - right click
            # 4 - scroll up
            # 5 - scroll down
            if event.button == 1:
                zoom(pygame.mouse.get_pos(), zoomrate)  # zoom in
                redraw(screen_pixels)
            elif event.button == 3:
                zoom(pygame.mouse.get_pos(), 1 / zoomrate)  # zoom out
                redraw(screen_pixels)
        # NOTE - get_pressed() gives current state, not state of event
        # pygame.key.get_pressed()[pygame.K_q]
        # pygame.mouse.get_pressed()[0]
# Quit pygame
pygame.quit()
