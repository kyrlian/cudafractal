#!python3
import cProfile
import pygame
import argparse
from ui.appState import AppState

def load_driver(cpu_only: bool):
    if not cpu_only:
        try:
            from numba import cuda
            if cuda.is_available():
                print("cuda detected")
                from fractal_cuda.fractal_cuda import compute_fractal, FRACTAL_MODES
                return compute_fractal, FRACTAL_MODES
        except Exception:
            print(f"Error importing numba.cuda: {Exception}")
    print("NOT using cuda")
    from fractal_no_cuda.fractal_no_cuda import compute_fractal, FRACTAL_MODES
    return compute_fractal, FRACTAL_MODES


def pygamemain(compute_fractal, FRACTAL_MODES):
    def redraw(screen_surface, appstate):
        appstate.recalc_size()
        # output_array_rgb = pygame.PixelArray(screen_surface)
        output_array_niter, output_array_z2, output_array_k, output_array_rgb = compute_fractal(
            appstate.WINDOW_SIZE,
            appstate.xmax,
            appstate.xmin,
            appstate.ymin,
            appstate.ymax,
            appstate.fractal_mode,
            appstate.max_iterations,
            appstate.power,
            appstate.escape_radius,
            appstate.epsilon,
            appstate.juliaxy,
            appstate.color_mode,
            appstate.palette,
            appstate.color_waves,
        )
        pygame.pixelcopy.array_to_surface(screen_surface, output_array_rgb)
        pygame.display.flip()

    def printhelp(appstate):
        print("Help:")
        print(f"    key(s): role, (default, current)")
        print(f"    z, left click: zoom in")
        print(f"    s, right click: zoom out")
        print(f"    up, down, left, right: pan")
        print(f"    k:  color mode (0, {appstate.color_mode})")
        print(f"    c:  color palette (0, {appstate.palette})")
        print(f"    w:  color waves (1, {appstate.color_waves})")
        print(f"    i:  max iterations (1000, {appstate.max_iterations})")
        print(f"    p:  power(2, {appstate.power})")
        print(f"    r:  escape radius(4, {appstate.escape_radius})")
        print(f"    e:  epsilon (0.001, {appstate.epsilon})")
        print(f"    a:  epsilon=0 (0.001, {appstate.epsilon})")
        print(f"    j:  middle click: julia/mandelbrot (mandelbrot, {FRACTAL_MODES[appstate.fractal_mode].__name__})")
        print(f"    backspace: reset")
        print(f"    q: quit")
        print(f"current x, y, h: {appstate.xcenter}, {appstate.ycenter}, {appstate.yheight}")

    # Initialize pygame
    pygame.init()
    appstate = AppState(FRACTAL_MODES)
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
                            appstate.change_max_iterations(0.9)
                        else:
                            appstate.change_max_iterations(1.1)
                    case pygame.K_r:
                        if shift:
                            appstate.change_escaper(0.5)
                        else:
                            appstate.change_escaper(2)
                    case pygame.K_a:
                        appstate.change_epsilon(0)
                    case pygame.K_e:
                        if shift:
                            appstate.change_epsilon(10)
                        else:
                            appstate.change_epsilon(0.1)
                    case pygame.K_p:
                        if shift:
                            appstate.change_power(-1)
                        else:
                            appstate.change_power(1)
                    case pygame.K_j:
                        appstate.change_fractal_mode(pygame.mouse.get_pos())
                    case pygame.K_k:
                        appstate.change_color_mode()
                    case pygame.K_c:
                        appstate.change_color_palette()
                    case pygame.K_w:
                        if shift:
                            appstate.change_color_waves(-1)
                        else:
                            appstate.change_color_waves(1)
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
                    appstate.change_fractal_mode(pygame.mouse.get_pos())
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
    compute_fractal, FRACTAL_MODES = load_driver(args.cpu)
    if args.profile:
        # https://docs.python.org/3.8/library/profile.html#module-cProfile
        cProfile.run("pygamemain(compute_fractal, FRACTAL_MODES)", sort="cumtime")
    else:
        pygamemain(compute_fractal, FRACTAL_MODES)


if __name__ == "__main__":
    main()
