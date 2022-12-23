# Write python code using nvidia cuda to render a mandelbrot fractal.
# When the user left clicks, zoom in by a factor of 2 into the fractal at the cursor.
# When the user right clicks, zoom out by a factor of 2.

import cProfile
import math
import timeit
import sys
from numba import float64, complex128
from math import log
import pygame
import numpy


def set_image_color_hue(pixel_array, x, y, k):
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


def set_image_color_log_lastr_to_hue(
    pixel_array, x, y, iterations, max_iterations, lastr
):
    # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
    if lastr < 1 or iterations == 0:
        pixel_array[x, y] = (0,0,0)
    else:
        # sys.stdout.write("%f, %f, %f\n"%(lastr, log(lastr), 1/lastr))
        # set_image_color_hue(pixel_array, x, y,log(lastr/escaper))
        set_image_color_hue(pixel_array, x, y,1/lastr)


def set_image_color_log_iter_to_hue(
    pixel_array, x, y, iterations, max_iterations, lastr
):
    set_image_color_hue(pixel_array, x, y, log(float64(iterations)) / log(float64(max_iterations)))

def set_image_color_iter_to_hue(pixel_array, x, y, iterations, max_iterations, lastr):
    set_image_color_hue(pixel_array, x, y,  float64(iterations) / float64(max_iterations))

def set_image_color(pixel_array, x, y, nbi, max_iter, lastr):
    if nbi == 0 or nbi == max_iter:
        pixel_array[x, y] = (0,0,0)
    else:
        colorfunctions[currentcolorfunction](pixel_array, x, y, nbi, max_iter, lastr)

colorfunctions = [
    set_image_color_log_iter_to_hue,
    set_image_color_iter_to_hue,
    set_image_color_log_lastr_to_hue,
]
currentcolorfunction = 0

def mandelbrot(pixel_array, topleft, xstride, ystride, max_iter, p, r, epsilon, juliax, juliay):
    for x in range(pixel_array.shape[0]):
        for y in range(pixel_array.shape[1]):
            c = complex128(topleft + x * xstride - 1j * y * ystride)
            z = c # complex128(0 + 0j)
            nbi = 0
            lastr = 0
            der = complex128(1+0j)
            while nbi < max_iter and lastr < r and (der.real**2 + der.imag**2) > epsilon:
                der =  der * p * z # p * z
                z = z**p + c
                nbi += 1
                lastr = z.real**2 + z.imag**2
            set_image_color(pixel_array, x, y, nbi, max_iter, lastr)


def julia(pixel_array, topleft, xstride, ystride, max_iter, p, r, epsilon, juliax, juliay):
    for x in range(pixel_array.shape[0]):
        for y in range(pixel_array.shape[1]):
            c = complex128(juliax + 1j * juliay)
            z = complex128(topleft + x * xstride - 1j * y * ystride)
            nbi = 0
            lastr = 0
            while nbi < max_iter and lastr < r:
                z = z**p + c
                nbi += 1
                lastr = z.real * z.real + z.imag * z.imag
            set_image_color(pixel_array, x, y, nbi, max_iter, lastr)


fractalmodes = [mandelbrot, julia]
currentfractalmode = 0


def create_image(
    fractalmode, pixel_array, xmin, xmax, ymin, ymax, max_iter, p, r, epsilon, juliax, juliay
):
    timerstart = timeit.default_timer()
    xstride = abs(xmax - xmin) / pixel_array.shape[0]
    ystride = abs(ymax - ymin) / pixel_array.shape[1]
    topleft = complex128(xmin + 1j * ymax)
    # mandelbrot(pixel_array, topleft, xstride, ystride, max_iter, p)
    # julia(pixel_array, topleft, xstride, ystride, max_iter, p, juliax, juliay)
    fractalmode(pixel_array, topleft, xstride, ystride, max_iter, p, r, epsilon, juliax, juliay)
    sys.stdout.write(
        "Frame calculated in %f s \n" % (timeit.default_timer() - timerstart)
    )
    # directly create and show a pylot plot (good for one shot)


def create_image_profiling():
    pixel_array = numpy.zeros((256, 256, 3))
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
        0,
        0,
    )


def pygamemain():
    # Initialize constants
    display_heigth = 256
    display_ratio = 4 / 3
    display_width = math.floor(display_heigth * display_ratio)
    window_size = display_width, display_heigth
    zoomrate = 2
    juliax, juliay = 0, 0
    panspeed = 0.01  # ratio of xmax-xmin

    def reset():
        sys.stdout.write("Reset \n")
        global xcenter, ycenter, yheight, maxiterations, power, escaper, epsilon, currentfractalmode, currentcolorfunction
        xcenter = -0.5
        ycenter = 0
        yheight = 3
        maxiterations = 100
        power = 2
        escaper = 4
        epsilon = 0.001
        currentfractalmode = 0
        currentcolorfunction = 0

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
        sys.stdout.write(
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
        )
        pygame.display.flip()

    def changecolorfunction():
        global currentcolorfunction
        currentcolorfunction = (currentcolorfunction + 1) % len(colorfunctions)
        sys.stdout.write(
            "Color function: %s \n" % colorfunctions[currentcolorfunction].__name__
        )

    def changemaxiterations(factor):
        global maxiterations
        maxiterations *= maxiterations
        sys.stdout.write("Max iterations: %f \n" % maxiterations)


    def changepower(plusminus):
        global power
        power = (power + plusminus) % 16
        sys.stdout.write("Power: %f \n" % power)

    def changeescaper(factor):
        global escaper
        escaper *= factor
        sys.stdout.write("Escape R: %f \n" % escaper)

    def changeepsilon(factor):
        global epsilon
        if factor == epsilon == 0:
            epsilon = .001
        else:
            epsilon *= factor
        sys.stdout.write("Epsilon: %f \n" % epsilon)

    def changefractalmode(pos):
        global currentfractalmode, juliax, juliay
        (mouseX, mouseY) = pos
        currentfractalmode = (currentfractalmode + 1) % len(fractalmodes)
        juliax = xmin + mouseX * (xmax - xmin) / display_width
        juliay = ymin + (display_heigth - mouseY) * (ymax - ymin) / display_heigth
        sys.stdout.write(
            "Fractal mode: %s \n" % fractalmodes[currentfractalmode].__name__
        )

    def printhelp():
        sys.stdout.write("Help: \n")
        sys.stdout.write("z, left click: zoom in\n")
        sys.stdout.write("s, right click: zoom out \n")
        sys.stdout.write("up, down, left, right: pan \n")
        sys.stdout.write("c: change color function - current: %s \n" % colorfunctions[currentcolorfunction].__name__)
        sys.stdout.write("i, j: max_iterations up/down (def:100) - current: %i \n" % maxiterations)
        sys.stdout.write("p, l: power up/down (def:2) - current: %i \n" % power)
        sys.stdout.write("r, t: escape radius up/down (def:4) - current: %i \n" % escaper)
        sys.stdout.write("e, f: epsilon up/down (def:.001) - current: %i \n" % epsilon)
        sys.stdout.write("a: epsilon=0 - current: %i \n" % epsilon)
        sys.stdout.write("j, middle click: julia/mandel - current: %s \n" % fractalmodes[currentfractalmode].__name__)
        sys.stdout.write("backspace: reset \n")
        sys.stdout.write("q: quit \n")

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
                if event.key == pygame.K_q:
                    running = doredraw = False
                if event.key == pygame.K_z:
                    zoom((display_width / 2, display_heigth / 2), zoomrate)  # zoom in
                if event.key == pygame.K_s:
                    zoom(
                        (display_width / 2, display_heigth / 2), 1 / zoomrate
                    )  # zoom out
                if event.key == pygame.K_UP:
                    pan(0, 1)
                if event.key == pygame.K_DOWN:
                    pan(0, -1)
                if event.key == pygame.K_LEFT:
                    pan(-1, 0)
                if event.key == pygame.K_RIGHT:
                    pan(1, 0)
                if event.key == pygame.K_i:
                    changemaxiterations(1)
                if event.key == pygame.K_j:
                    changemaxiterations(.5)
                if event.key == pygame.K_r:
                    changeescaper(2)
                if event.key == pygame.K_t:
                    changeescaper(.5)
                if event.key == pygame.K_a:
                    changeepsilon(0)
                if event.key == pygame.K_e:
                    changeepsilon(.1)
                if event.key == pygame.K_f:
                    changeepsilon(10)
                if event.key == pygame.K_BACKSPACE:
                    reset()
                if event.key == pygame.K_c:
                    changecolorfunction()
                if event.key == pygame.K_p:
                    changepower(1)
                if event.key == pygame.K_l:
                    changepower(-1)
                if event.key == pygame.K_j:
                    changefractalmode(pygame.mouse.get_pos())
                if event.key == pygame.K_h:
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
    sys.stdout.write("So Long, and Thanks for All the Fish!\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0 and args[0] == "--profile":
        # https://docs.python.org/3.8/library/profile.html#module-cProfile
        cProfile.run("create_image_profiling()", sort="cumtime")
    else:
        pygamemain()
