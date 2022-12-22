# Write python code using nvidia cuda to render a mandelbrot fractal.
# When the user left clicks, zoom in by a factor of 2 into the fractal at the cursor.
# When the user right clicks, zoom out by a factor of 2.

import math
from numba import float64
from math import log
import pygame


def set_image_color_k(pixel_array, x, y, k):
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


def set_image_color_log_hue(pixel_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:  # BLACK
        pixel_array[x, y] = (0, 0, 0)
    else:
        k = 6.0 * log(float64(iterations)) / log(float64(max_iterations))
        set_image_color_k(pixel_array, x, y, k)


def set_image_color_hue(pixel_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:  # BLACK
        pixel_array[x, y] = (0, 0, 0)
    else:
        k = 6.0 * float64(iterations) / float64(max_iterations)
        set_image_color_k(pixel_array, x, y, k)

def set_image_color_nb(pixel_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:  # BLACK
        pixel_array[x, y] = (0, 0, 0)
    else:
        r = g = b = iterations / max_iterations
        pixel_array[x, y] = (r * 255, g * 255, b * 255)

def set_image_color_sin(pixel_array, x, y, iterations, max_iterations):
    if iterations == max_iterations:  # BLACK
        pixel_array[x, y] = (0, 0, 0)
    else:
        r = math.sin(iterations / max_iterations) / 2 + 0.5
        g = math.sin(2 * iterations / max_iterations) / 2 + 0.5
        b = math.sin(4 * iterations / max_iterations) / 2 + 0.5
        pixel_array[x, y] = (r * 255, g * 255, b * 255)


colorfunctions = [set_image_color_log_hue, set_image_color_hue, set_image_color_sin, set_image_color_nb]
currentcolorfunction = 0


def set_image_color(pixel_array, x, y, nbi, max_iter):
    colorfunctions[currentcolorfunction](pixel_array, x, y, nbi, max_iter)


def nextpalette():
    global currentcolorfunction
    currentcolorfunction = (currentcolorfunction + 1) % len(colorfunctions)


def redraw(pixel_array):
    for x in range(pixel_array.shape[0]):
        for y in range(pixel_array.shape[1]):
            set_image_color(pixel_array, x, y, x + 1, pixel_array.shape[0])
    pygame.display.flip()


# INITs
display_heigth = 256
display_ratio = 4 / 3
display_width = math.floor(display_heigth * display_ratio)
window_size = display_width, display_heigth

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode(window_size, pygame.HWSURFACE)
# Get the PixelArray object for the screen
screen_pixels = pygame.PixelArray(screen)
# Init the display
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
            if event.key == pygame.K_p:
                nextpalette()
                redraw(screen_pixels)
# Quit pygame
pygame.quit()
