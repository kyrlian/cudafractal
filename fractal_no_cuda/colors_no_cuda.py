import math
from numba import float64
from math import log


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
    packed = (int(r * 255) * 256 + int(g * 255)) * 256 + int(b * 255)
    pixel_array[x, y] = packed


def set_image_color(pixels, x, y, nbi, max_iter, z2, r, der2, cmode, palette):
    if z2 > r:
        match cmode:
            case 0:
                k = float64(nbi) / float64(max_iter)
            case 1:
                k = log(float64(nbi)) / log(float64(max_iter))
            case 2:
                # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                k = float64(r) / float64(
                    z2
                )  # TODO : z2 is slightly bigger than r, so k doesnt cover 0-1
            case 3:
                k = log(float64(r)) / log(float64(z2))
            case 4:
                k = math.sin(log(z2)) / 2 + 0.5
        match palette:
            case 0:  # hue
                set_color_hue(pixels, x, y, k)
            case 1:  # graysscale
                pixels[x, y] = (k * 255, k * 255, k * 255)
    else:
        pixels[x, y] = (0, 0, 0)
