#!python3
import cProfile
import sys
import pygame

from ui.appState import AppState
from fractal_cuda.utils_cuda import create_image


def pygamemain():
    def redraw(screen_surface, appstate):
        appstate.recalc_size()
        output_array = create_image(
            appstate.WINDOW_SIZE,
            appstate.xmax,
            appstate.xmin,
            appstate.ymin,
            appstate.ymax,
            appstate.fractalmode,
            appstate.maxiter,
            appstate.power,
            appstate.escaper,
            appstate.epsilon,
            appstate.juliaxy,
            appstate.currentcolormode,
            appstate.currentpalette,
            appstate.currentcolor_waves,
        )
        pygame.pixelcopy.array_to_surface(screen_surface, output_array)
        pygame.display.flip()

    def printhelp(appstate):
        sys.stdout.write("Help: \n")
        sys.stdout.write("key, role, (default, current)\n")
        sys.stdout.write("z, left click: zoom in\n")
        sys.stdout.write("s, right click: zoom out \n")
        sys.stdout.write("up, down, left, right: pan \n")
        sys.stdout.write(f"k: color mode (0,{appstate.currentcolormode})\n")
        sys.stdout.write(f"c: color palette (0,{appstate.currentpalette})\n")
        sys.stdout.write(f"w: color waves (1,{appstate.currentcolor_waves})\n")
        sys.stdout.write(f"i, max_iterations (1000,{appstate.maxiterations})\n")
        sys.stdout.write(f"p, power(2,{appstate.power})\n")
        sys.stdout.write(f"r, escape radius(4,{appstate.escaper})\n")
        sys.stdout.write(f"e, epsilon (0.001,{appstate.epsilon})\n")
        sys.stdout.write(f"a: epsilon=0 (0.001,{appstate.epsilon})\n")
        sys.stdout.write(
            f"j, middle click: julia/mandel (mandel,{appstate.FRACTAL_MODES[appstate.currentfractalmode].__name__}\n"
        )
        sys.stdout.write("backspace: reset \n")
        sys.stdout.write("q: quit \n")
        sys.stdout.write("current x,y,h: {xcenter}, {ycenter}, {yheight}\n")

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
    sys.stdout.write("So Long, and Thanks for All the Fish!\n")


def main():
    args = sys.argv[1:]
    if len(args) > 0 and args[0] == "--profile":
        # https://docs.python.org/3.8/library/profile.html#module-cProfile
        cProfile.run("create_image_profiling()", sort="cumtime")
    else:
        pygamemain()


if __name__ == "__main__":
    main()
