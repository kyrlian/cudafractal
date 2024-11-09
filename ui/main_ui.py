#!python3
import cProfile
import sys
import pygame
import argparse

from ui.appState import AppState


def load_driver(cpu_only: bool):
    if cpu_only:
        from fractal_no_cuda.utils_no_cuda import create_image
        from fractal_no_cuda.fractal_no_cuda import FRACTAL_MODES

        return create_image, FRACTAL_MODES
    else:
        try:
            from numba import cuda
        except Exception:
            print(f"Error importing numba.cuda: {Exception}")
        # CUDA_AVAILABLE = cuda.detect()
        if cuda.is_available():
            print("cuda detected")
            from fractal_cuda.utils_cuda import create_image
            from fractal_cuda.fractal_cuda import FRACTAL_MODES

            return create_image, FRACTAL_MODES
        else:
            print("cuda NOT detected")
            from fractal_no_cuda.utils_no_cuda import create_image
            from fractal_no_cuda.fractal_no_cuda import FRACTAL_MODES

            return create_image, FRACTAL_MODES


def pygamemain(create_image, FRACTAL_MODES):
    def redraw(screen_surface, appstate):
        appstate.recalc_size()
        output_array = create_image(
            appstate.WINDOW_SIZE,
            appstate.xmax,
            appstate.xmin,
            appstate.ymin,
            appstate.ymax,
            appstate.fractalmode,
            appstate.maxiterations,
            appstate.power,
            appstate.escaper,
            appstate.epsilon,
            appstate.juliaxy,
            appstate.colormode,
            appstate.palette,
            appstate.colorwaves,
        )
        pygame.pixelcopy.array_to_surface(screen_surface, output_array)
        pygame.display.flip()

    def printhelp(appstate):
        print("Help: ")
        print("key(s): role, (default, current)")
        print("z, left click: zoom in")
        print("s, right click: zoom out")
        print("up, down, left, right: pan")
        print(f"k:  color mode (0, {appstate.colormode})")
        print(f"c:  color palette (0, {appstate.palette})")
        print(f"w:  color waves (1, {appstate.colorwaves})")
        print(f"i:  max_iterations (1000, {appstate.maxiterations})")
        print(f"p:  power(2, {appstate.power})")
        print(f"r:  escape radius(4, {appstate.escaper})")
        print(f"e:  epsilon (0.001, {appstate.epsilon})")
        print(f"a:  epsilon=0 (0.001, {appstate.epsilon})")
        print(f"j:  middle click: julia/mandelbrot (mandelbrot, {FRACTAL_MODES[appstate.fractalmode].__name__})")
        print("backspace: reset")
        print("q: quit")
        print(f"current x, y, h: {appstate.xcenter}, {appstate.ycenter}, {appstate.yheight}")

    # Initialize pygame
    pygame.init()
    appstate = AppState()
    screen_surface = pygame.display.set_mode(appstate.WINDOW_SIZE, pygame.HWSURFACE)
    # Get the PixelArray object for the screen
    # screen_pixels = pygame.PixelArray(screen_surface)
    # Init the display
    printhelp(appstate)
    redraw(screen_surface, appstate)
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
                        appstate.zoom_in()
                    case pygame.K_s:
                        appstate.zoom_out()
                    case pygame.K_UP:
                        appstate.pan(0, 1)
                    case pygame.K_DOWN:
                        appstate.pan(0, -1)
                    case pygame.K_LEFT:
                        appstate.pan(-1, 0)
                    case pygame.K_RIGHT:
                        appstate.pan(1, 0)
                    case pygame.K_i:
                        if shift:
                            appstate.changemaxiterations(0.9)
                        else:
                            appstate.changemaxiterations(1.1)
                    case pygame.K_r:
                        if shift:
                            appstate.changeescaper(0.5)
                        else:
                            appstate.changeescaper(2)
                    case pygame.K_a:
                        appstate.changeepsilon(0)
                    case pygame.K_e:
                        if shift:
                            appstate.changeepsilon(10)
                        else:
                            appstate.changeepsilon(0.1)
                    case pygame.K_p:
                        if shift:
                            appstate.changepower(-1)
                        else:
                            appstate.changepower(1)
                    case pygame.K_j:
                        appstate.changefractalmode(pygame.mouse.get_pos())
                    case pygame.K_k:
                        appstate.changecolormode()
                    case pygame.K_c:
                        appstate.changecolorpalette()
                    case pygame.K_w:
                        if shift:
                            appstate.changecolor_waves(-1)
                        else:
                            appstate.changecolor_waves(1)
                    case pygame.K_BACKSPACE:
                        appstate.reset()
                    case pygame.K_h:
                        printhelp(appstate)
                        doredraw = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                doredraw = True
                # 1 - left click, 2 - middle click, 3 - right click, 4 - scroll up, 5 - scroll down
                if event.button == 1 or event.button == 4:
                    appstate.zoom_in(pygame.mouse.get_pos())
                elif event.button == 3 or event.button == 5:
                    appstate.zoom_out(pygame.mouse.get_pos())
                elif event.button == 2:
                    appstate.changefractalmode(pygame.mouse.get_pos())
            if doredraw:
                redraw(screen_surface, appstate)
            # NOTE - get_pressed() gives current state, not state of event
            # pygame.key.get_pressed()[pygame.K_q]
            # pygame.mouse.get_pressed()[0]
    # Quit pygame
    pygame.quit()
    print("So Long, and Thanks for All the Fish!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--profile", help="activate profiling", action="store_true"
    )
    parser.add_argument("-c", "--cpu", help="compute on cpu only", action="store_true")
    args = parser.parse_args()
    create_image, FRACTAL_MODES = load_driver(args.cpu)
    if args.profile:
        # https://docs.python.org/3.8/library/profile.html#module-cProfile
        cProfile.run("pygamemain(create_image, FRACTAL_MODES)", sort="cumtime")
    else:
        pygamemain(create_image, FRACTAL_MODES)


if __name__ == "__main__":
    main()
