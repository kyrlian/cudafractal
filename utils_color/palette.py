import math
from numba import float64
from math import log
import pygame
import sys


def set_color_hue(pixel_array, x, y, k):
    # k should be [0:1]
    k6 = 6.0 * k
    fk = math.floor(k6)
    fract = k6 - fk
    r, g, b = 0, 0, 0
    match fk:
        case 0:  # RED to YELLOW
            r, g, b = 1, fract, 0
        case 1:  # YELLOW to GREEN
            r, g, b = 1 - fract, 1, 0
        case 2:  # GREEN to CYAN
            r, g, b = 0, 1, fract
        case 3:  # CYAN to BLUE
            r, g, b = 0, 1 - fract, 1
        case 4:  # BLUE to MAGENTA
            r, g, b = fract, 0, 1
        case 5:  # MAGENTA to RED
            r, g, b = 1, 0, 1 - fract
    pixel_array[x, y] = (r * 255, g * 255, b * 255)


def set_image_color(pixels, x, y, cmode, palette):
    nbi = x + 1
    max_iter = pixels.shape[0]
    match cmode:
        case 0:
            k = float64(nbi) / float64(max_iter)
        case 1:
            k = log(float64(nbi)) / log(float64(max_iter))
        case 2:
            # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
            k = 1 / nbi
        case 3:
            k = math.sin(nbi / max_iter) / 2 + 0.5
    match palette:
        case 0:  # hue
            set_color_hue(pixels, x, y, k)
        case 1:  # graysscale
            pixels[x, y] = (k * 255, k * 255, k * 255)
        case 2:
            r = math.sin(k) / 2 + 0.5
            g = math.sin(2 * k) / 2 + 0.5
            b = math.sin(4 * k) / 2 + 0.5
            pixels[x, y] = (r * 255, g * 255, b * 255)


currentcolormode = 0
nbcolormodes = 4
currentpalette = 0
nbpalettes = 3


def changecolormode():
    global currentcolormode
    currentcolormode = (currentcolormode + 1) % nbcolormodes
    sys.stdout.write("Color mode: %i \n" % currentcolormode)


def changecolorpalette():
    global currentpalette
    currentpalette = (currentpalette + 1) % nbpalettes
    sys.stdout.write("Palette: %i \n" % currentpalette)


def redraw(pixel_array):
    for x in range(pixel_array.shape[0]):
        set_image_color(pixel_array, x, 0, currentcolormode, currentpalette)
        for y in range(pixel_array.shape[1]):
            pixel_array[x, y] = pixel_array[x, 0]
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
            match event.key:
                case pygame.K_q:
                    running = False
                case pygame.K_k:
                    changecolormode()
                    redraw(screen_pixels)
                case pygame.K_c:
                    changecolorpalette()
                    redraw(screen_pixels)
# Quit pygame
pygame.quit()
