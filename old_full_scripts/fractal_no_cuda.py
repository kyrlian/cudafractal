#!python3

# Write python code using nvidia cuda to render a mandelbrot fractal.
# When the user left clicks, zoom in by a factor of 2 into the fractal at the cursor.
# When the user right clicks, zoom out by a factor of 2.

# without cuda
# pip install numba pygame numpy

import cProfile
import math
import timeit
import sys
from numba import float64, complex128
from math import log
import pygame
import numpy


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


currentcolormode = 0
nbcolormodes = 5
currentpalette = 0
nbpalettes = 2


def mandelbrot(
    pixels, topleft, xstep, ystep, maxiter, p, r, eps, jx, jy, cmode, palette
):
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            c = complex128(topleft + x * xstep - 1j * y * ystep)
            z = c  # complex128(0 + 0j)
            nbi = 0
            z2 = 0
            der = complex128(1 + 0j)
            der2 = 1
            while nbi < maxiter and z2 < r and der2 > eps:
                der = der * p * z
                z = z**p + c
                nbi += 1
                z2 = z.real**2 + z.imag**2
                der2 = der.real**2 + der.imag**2
            set_image_color(pixels, x, y, nbi, maxiter, z2, r, der2, cmode, palette)


def julia(pixels, topleft, xstep, ystep, maxiter, p, r, eps, jx, jy, cmode, palette):
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            c = complex128(jx + 1j * jy)
            z = complex128(topleft + x * xstep - 1j * y * ystep)
            nbi = 0
            z2 = 0
            der = complex128(1 + 0j)
            der2 = 1
            while nbi < maxiter and z2 < r and der2 > eps:
                # TODO test julia with/without der
                der = der * p * z
                z = z**p + c
                nbi += 1
                z2 = z.real**2 + z.imag**2
                der2 = der.real**2 + der.imag**2
            set_image_color(pixels, x, y, nbi, maxiter, z2, r, der2, cmode, palette)


fractalmodes = [mandelbrot, julia]
currentfractalmode = 0


def create_image(
    fractalmode,
    pixels,
    xmin,
    xmax,
    ymin,
    ymax,
    maxiter,
    p,
    r,
    eps,
    jx,
    jy,
    cmode,
    palette,
):
    timerstart = timeit.default_timer()
    xstep = abs(xmax - xmin) / pixels.shape[0]
    ystep = abs(ymax - ymin) / pixels.shape[1]
    topleft = complex128(xmin + 1j * ymax)
    fractalmode(
        pixels, topleft, xstep, ystep, maxiter, p, r, eps, jx, jy, cmode, palette
    )
    print("Frame calculated in %f s \n" % (timeit.default_timer() - timerstart))


def create_image_profiling():
    pixel_array = numpy.zeros((512, 512, 3))
    create_image(
        mandelbrot, pixel_array, -2.5, 1.5, -1.5, 1.5, 255, 2, 4, 0.001, 0, 0, 0, 0
    )


def pygamemain():
    # Initialize constants
    display_heigth = 512
    display_ratio = 4 / 3
    display_width = math.floor(display_heigth * display_ratio)
    window_size = display_width, display_heigth
    zoomrate = 2
    juliax, juliay = 0, 0
    panspeed = 0.3  # ratio of xmax-xmin

    def reset():
        print("Reset ")
        global \
            xcenter, \
            ycenter, \
            yheight, \
            maxiterations, \
            power, \
            escaper, \
            epsilon, \
            currentfractalmode, \
            currentcolormode, \
            currentpalette
        xcenter = -0.5
        ycenter = 0
        yheight = 3
        maxiterations = 100
        power = 2
        escaper = 4
        epsilon = 0.001
        currentfractalmode = 0
        currentcolormode = 0
        currentpalette = 0

    def recalc_size():
        global xwidth, xmin, xmax, ymin, ymax
        xwidth = yheight * display_width / display_heigth
        xmin = xcenter - xwidth / 2
        xmax = xcenter + xwidth / 2
        ymin = ycenter - yheight / 2
        ymax = ycenter + yheight / 2

    def zoom(mousePos, zoomrate):
        global xcenter, ycenter, yheight
        (mouseX, mouseY) = mousePos
        xcenter = xmin + mouseX * (xmax - xmin) / display_width
        ycenter = ymin + (display_heigth - mouseY) * (ymax - ymin) / display_heigth
        yheight /= zoomrate
        print(
            "Zoom %f,%f, (%f,%f), factor %f \n"
            % (mouseX, mouseY, xcenter, ycenter, zoomrate)
        )

    def pan(x, y):
        global xcenter, ycenter
        xcenter += x * panspeed * (xmax - xmin)
        ycenter += y * panspeed * (ymax - ymin)

    def redraw(pixel_array):
        recalc_size()
        create_image(
            fractalmodes[currentfractalmode],
            pixel_array,
            xmin,
            xmax,
            ymin,
            ymax,
            maxiterations,
            power,
            escaper,
            epsilon,
            juliax,
            juliay,
            currentcolormode,
            currentpalette,
        )
        pygame.display.flip()

    def changecolormode():
        global currentcolormode
        currentcolormode = (currentcolormode + 1) % nbcolormodes
        print("Color mode: %i \n" % currentcolormode)

    def changecolorpalette():
        global currentpalette
        currentpalette = (currentpalette + 1) % nbpalettes
        print("Palette: %i \n" % currentpalette)

    def changemaxiterations(factor):
        global maxiterations
        maxiterations *= maxiterations
        print("Max iterations: %f \n" % maxiterations)

    def changepower(plusminus):
        global power
        power = (power + plusminus) % 16
        print("Power: %f \n" % power)

    def changeescaper(factor):
        global escaper
        escaper *= factor
        print("Escape R: %f \n" % escaper)

    def changeepsilon(factor):
        global epsilon
        if factor == epsilon == 0:
            epsilon = 0.001
        else:
            epsilon *= factor
        print("Epsilon: %f \n" % epsilon)

    def changefractalmode(pos):
        global currentfractalmode, juliax, juliay
        (mouseX, mouseY) = pos
        currentfractalmode = (currentfractalmode + 1) % len(fractalmodes)
        juliax = xmin + mouseX * (xmax - xmin) / display_width
        juliay = ymin + (display_heigth - mouseY) * (ymax - ymin) / display_heigth
        print("Fractal mode: %s \n" % fractalmodes[currentfractalmode].__name__)

    def printhelp():
        print("Help: ")
        print("key, role, (default, current)")
        print("z, left click: zoom in")
        print("s, right click: zoom out ")
        print("up, down, left, right: pan ")
        print("k: color mode (0,%i)\n" % currentcolormode)
        print("c: color palette (0,%i)\n" % currentpalette)
        print("i, max_iterations (100,%i)\n" % maxiterations)
        print("p, power(2,%i)\n" % power)
        print("r, escape radius(4,%i)\n" % escaper)
        print("e, epsilon (0.001,%i)\n" % epsilon)
        print("a: epsilon=0 (0.001,%i)\n" % epsilon)
        print(
            "j, middle click: julia/mandel (mandel,%s)\n"
            % fractalmodes[currentfractalmode].__name__
        )
        print("backspace: reset ")
        print("q: quit ")
        print("current x,y,h: %f, %f, %f \n" % (xcenter, ycenter, yheight))

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode(window_size, pygame.HWSURFACE)
    # Get the PixelArray object for the screen
    screen_pixels = pygame.PixelArray(screen)
    # Init the display
    reset()
    printhelp()
    redraw(screen_pixels)
    # Run the game loop
    running = True
    while running:
        for event in pygame.event.get():
            doredraw = False
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                doredraw = True
                shift = (
                    pygame.key.get_pressed()[pygame.K_LSHIFT]
                    or pygame.key.get_pressed()[pygame.K_RSHIFT]
                )
                match event.key:
                    case pygame.K_q:
                        running = doredraw = False
                    case pygame.K_z:
                        zoom((display_width / 2, display_heigth / 2), zoomrate)  # in
                    case pygame.K_s:
                        zoom(
                            (display_width / 2, display_heigth / 2), 1 / zoomrate
                        )  # out
                    case pygame.K_UP:
                        pan(0, 1)
                    case pygame.K_DOWN:
                        pan(0, -1)
                    case pygame.K_LEFT:
                        pan(-1, 0)
                    case pygame.K_RIGHT:
                        pan(1, 0)
                    case pygame.K_i:
                        if shift:
                            changemaxiterations(0.5)
                        else:
                            changemaxiterations(1)
                    case pygame.K_r:
                        if shift:
                            changeescaper(0.5)
                        else:
                            changeescaper(2)
                    case pygame.K_a:
                        changeepsilon(0)
                    case pygame.K_e:
                        if shift:
                            changeepsilon(10)
                        else:
                            changeepsilon(0.1)
                    case pygame.K_p:
                        if shift:
                            changepower(-1)
                        else:
                            changepower(1)
                    case pygame.K_j:
                        changefractalmode(pygame.mouse.get_pos())
                    case pygame.K_k:
                        changecolormode()
                    case pygame.K_c:
                        changecolorpalette()
                    case pygame.K_BACKSPACE:
                        reset()
                    case pygame.K_h:
                        printhelp()
                        doredraw = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                doredraw = True
                # 1 - left click, 2 - middle click, 3 - right click, 4 - scroll up, 5 - scroll down
                if event.button == 1 or event.button == 4:
                    zoom(pygame.mouse.get_pos(), zoomrate)  # zoom in
                elif event.button == 3 or event.button == 5:
                    zoom(pygame.mouse.get_pos(), 1 / zoomrate)  # zoom out
                elif event.button == 2:
                    changefractalmode(pygame.mouse.get_pos())
            if doredraw:
                redraw(screen_pixels)
            # NOTE - get_pressed() gives current state, not state of event
            # pygame.key.get_pressed()[pygame.K_q]
            # pygame.mouse.get_pressed()[0]
    # Quit pygame
    pygame.quit()
    print("So Long, and Thanks for All the Fish!")


def main():
    args = sys.argv[1:]
    if len(args) > 0 and args[0] == "--profile":
        # https://docs.python.org/3.8/library/profile.html#module-cProfile
        cProfile.run("create_image_profiling()", sort="cumtime")
    else:
        pygamemain()


if __name__ == "__main__":
    main()
