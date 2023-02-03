# Based on fractal-cuda-simple.py
# with more keyboard controls, and color mode choices

# pip install numba pygame numpy

import cProfile
import math
import timeit
import sys
from numba import float64, complex128, int32, uint32
from math import log, sin
import pygame
import numpy
from numba import cuda


@cuda.jit('void(uint32[:,:], int32, int32, float64)', device=True)
def set_color_hue(device_array, x, y, k):
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
    device_array[x, y] = (int(r * 255)*256 + int(g * 255))*256 + int(b * 255)


@cuda.jit('void(uint32[:,:], int32, int32, int32, int32, float64, float64, float64, int32, int32)', device=True)
def set_pixel_color(device_array, x, y, nbi, max_iter, z2, r, der2, cmode, palette):
    if z2 > r:
        match cmode:
            case 0:
                k = float64(nbi) / float64(max_iter)
            case 1:
                k = log(float64(nbi)) / log(float64(max_iter))
            case 2:
                # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                # TODO : z2 is slightly bigger than r, so k doesnt cover 0-1
                k = float64(r) / float64(z2)
            case 3:
                k = log(float64(r)) / log(float64(z2))
            case 4: 
                #k = math.sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
                #k = sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
                k = 1/z2
        match palette:
            case 0:  # hue
                set_color_hue(device_array, x, y, k)
            case 1:  # grayscale
                # pixels[x, y] = (k * 255, k * 255, k * 255)
                kk = int(k * 255)
                device_array[x, y] = (kk*256 + kk)*256 + kk
    else:
        device_array[x, y] = 0


currentcolormode = 0
nbcolormodes = 5
currentpalette = 0
nbpalettes = 2

@cuda.jit('void(uint32[:,:], complex128, float64, float64, int32, int32, int32, float64, complex128, int32, int32)')
def mandelbrot(device_array, topleft, xstep, ystep, maxiter, p, r, eps, juliaxy, cmode, palette):
    x, y = cuda.grid(2)
    if x < device_array.shape[0] and y < device_array.shape[1]:
        c = complex128(topleft + x * xstep - 1j * y * ystep)
        z = c 
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
        set_pixel_color(device_array, x, y, nbi, maxiter,
                        z2, r, der2, cmode, palette)

@cuda.jit('void(uint32[:,:], complex128, float64, float64, int32, int32, int32, float64, complex128, int32, int32)')
def julia(device_array, topleft, xstep, ystep, maxiter, p, r, eps, juliaxy, cmode, palette):
    x, y = cuda.grid(2)
    if x < device_array.shape[0] and y < device_array.shape[1]:
        z = complex128(topleft + x * xstep - 1j * y * ystep)
        nbi = 0
        z2 = 0
        der = complex128(1 + 0j)
        der2 = 1
        while nbi < maxiter and z2 < r and der2 > eps:
            # TODO test julia with/without der
            der = der * p * z
            z = z**p + juliaxy
            nbi += 1
            z2 = z.real**2 + z.imag**2
            der2 = der.real**2 + der.imag**2
        set_pixel_color(device_array, x, y, nbi, maxiter,
                        z2, r, der2, cmode, palette)

fractalmodes = [mandelbrot, julia]
currentfractalmode = 0


def create_image(
    fractalmode,
    window_size,
    xmin: float64,
    xmax: float64,
    ymin: float64,
    ymax: float64,
    maxiter: int32,
    p: int32,
    r: int32,
    eps: float64,
    juliaxy: complex128,
    cmode: int32,
    palette: int32,
):
    timerstart = timeit.default_timer()
    (screenw, screenh) = window_size
    xstep = abs(xmax - xmin) / screenw
    ystep = abs(ymax - ymin) / screenh
    topleft = complex128(xmin + 1j * ymax)
    device_array = cuda.device_array((screenw, screenh), dtype=numpy.uint32)
    threadsperblock = compute_threadsperblock()  # (32, 16) #real size = 32*16
    blockspergrid = (
        math.ceil(screenw / threadsperblock[0]),
        math.ceil(screenh / threadsperblock[1]),
    )
    fractalmode[blockspergrid, threadsperblock](device_array, topleft, xstep, ystep, maxiter, p, r, eps, juliaxy, cmode, palette)
    output_array = device_array.copy_to_host()
    sys.stdout.write(
        "Frame calculated in %f s \n" % (timeit.default_timer() - timerstart)
    )
    return output_array


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
    # mbx = math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK * display_width / display_heigth) / 8) * 8
    # mby = math.floor(gpu.MAX_THREADS_PER_BLOCK / mbx / 4) * 4
    # if we dont care about the ratio mbx/mby :
    mbx = 32
    mby = math.floor(gpu.MAX_THREADS_PER_BLOCK / mbx)
    return (mbx, mby)


def create_image_profiling():
    pixel_array = numpy.zeros((512, 512, 3))
    create_image(
        mandelbrot, pixel_array, -2.5, 1.5, -
        1.5, 1.5, 255, 2, 4, 0.001, complex128(0 + 0j), 0, 0
    )


def pygamemain():
    # Initialize constants
    display_heigth = 1024
    display_ratio = 4 / 3
    display_width = math.floor(display_heigth * display_ratio)
    window_size = display_width, display_heigth
    zoomrate = 2
    # juliax, juliay = 0, 0
    juliaxy = complex128(0 + 0j)
    panspeed = 0.3  # ratio of xmax-xmin

    def reset():
        sys.stdout.write("Reset \n")
        global xcenter, ycenter, yheight, maxiterations, power, escaper, epsilon, currentfractalmode, currentcolormode, currentpalette
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
        ycenter = ymin + (display_heigth - mouseY) * \
            (ymax - ymin) / display_heigth
        yheight /= zoomrate
        sys.stdout.write(
            "Zoom %f,%f, (%f,%f), factor %f \n"
            % (mouseX, mouseY, xcenter, ycenter, zoomrate)
        )

    def pan(x, y):
        global xcenter, ycenter
        xcenter += x * panspeed * (xmax - xmin)
        ycenter += y * panspeed * (ymax - ymin)

    def redraw(screen_surface):
        recalc_size()
        output_array = create_image(
            fractalmodes[currentfractalmode],
            window_size,
            xmin,
            xmax,
            ymin,
            ymax,
            maxiterations,
            power,
            escaper,
            epsilon,
            juliaxy,
            currentcolormode,
            currentpalette,
        )
        pygame.pixelcopy.array_to_surface(screen_surface, output_array)
        pygame.display.flip()

    def changecolormode():
        global currentcolormode
        currentcolormode = (currentcolormode + 1) % nbcolormodes
        sys.stdout.write("Color mode: %i \n" % currentcolormode)

    def changecolorpalette():
        global currentpalette
        currentpalette = (currentpalette + 1) % nbpalettes
        sys.stdout.write("Palette: %i \n" % currentpalette)

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
            epsilon = 0.001
        else:
            epsilon *= factor
        sys.stdout.write("Epsilon: %f \n" % epsilon)

    def changefractalmode(pos):
        global currentfractalmode, juliaxy
        (mouseX, mouseY) = pos
        currentfractalmode = (currentfractalmode + 1) % len(fractalmodes)
        juliax = xmin + mouseX * (xmax - xmin) / display_width
        juliay = ymin + (display_heigth - mouseY) * \
            (ymax - ymin) / display_heigth
        juliaxy = complex128(juliax + juliay*1j)
        sys.stdout.write(
            "Fractal mode: %s \n" % fractalmodes[currentfractalmode].__name__
        )

    def printhelp():
        sys.stdout.write("Help: \n")
        sys.stdout.write("key, role, (default, current)\n")
        sys.stdout.write("z, left click: zoom in\n")
        sys.stdout.write("s, right click: zoom out \n")
        sys.stdout.write("up, down, left, right: pan \n")
        sys.stdout.write("k: color mode (0,%i)\n" % currentcolormode)
        sys.stdout.write("c: color palette (0,%i)\n" % currentpalette)
        sys.stdout.write("i, max_iterations (100,%i)\n" % maxiterations)
        sys.stdout.write("p, power(2,%i)\n" % power)
        sys.stdout.write("r, escape radius(4,%i)\n" % escaper)
        sys.stdout.write("e, epsilon (0.001,%i)\n" % epsilon)
        sys.stdout.write("a: epsilon=0 (0.001,%i)\n" % epsilon)
        sys.stdout.write(
            "j, middle click: julia/mandel (mandel,%s)\n"
            % fractalmodes[currentfractalmode].__name__
        )
        sys.stdout.write("backspace: reset \n")
        sys.stdout.write("q: quit \n")
        sys.stdout.write("current x,y,h: %f, %f, %f \n" %
                         (xcenter, ycenter, yheight))

    # Initialize pygame
    pygame.init()
    screen_surface = pygame.display.set_mode(window_size, pygame.HWSURFACE)
    # Get the PixelArray object for the screen
    # screen_pixels = pygame.PixelArray(screen_surface)
    # Init the display
    reset()
    printhelp()
    redraw(screen_surface)
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
                redraw(screen_surface)
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
