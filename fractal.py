# Write python code using nvidia cuda to render a mandelbrot fractal.
# When the user left clicks, zoom in by a factor of 2 into the fractal at the cursor.
# When the user right clicks, zoom out by a factor of 2.

import math
import timeit
import sys
from numba import float64, complex128
from math import log
import pygame

def set_image_color(pixel_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:  # BLACK
        pixel_array[x, y] = (0, 0, 0)
    else:
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
        pixel_array[x, y] = (r * 255, g * 255, b * 255)


def mandelbrot_nocuda(pixel_array, topleft, xstride, ystride, max_iter):
    for x in range(pixel_array.shape[0]):
        for y in range(pixel_array.shape[1]):
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
    mandelbrot_nocuda(pixel_array, topleft, xstride, ystride, max_iter)
    sys.stdout.write(
        "Frame calculated in %f s \n" % (timeit.default_timer() - timerstart)
    )
    # directly create and show a pylot plot (good for one shot)


# INITs
max_iterations = 255
h=256
window_size = display_width, display_heigth = h * 4 / 3, h
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
