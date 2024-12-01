#!python3
import cProfile
import pygame
import pygame.freetype as ft
import argparse
from ui.appState import AppState
from fractal.fractal import compute_fractal, FRACTAL_NAMES
from fractal.colors import ColorMode, Palette


def pygamemain():
    def redraw(screen_surface, appstate, recalc=True):
        appstate.recalc_size()
        # output_array_rgb = pygame.PixelArray(screen_surface)
        if recalc:
            output_array_niter, output_array_z2, output_array_k, output_array_rgb = (
                compute_fractal(
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
            )
        pygame.pixelcopy.array_to_surface(screen_surface, output_array_rgb)
        if appstate.show_info:
            print_info(screen_surface)
        pygame.display.flip()
        return output_array_niter, output_array_z2, output_array_k, output_array_rgb

    def print_info(
        screen_surface,
        ni=None, z2=None, k=None, rgb=None,
        bgcolor=pygame.Color("white"),
        fgcolor=pygame.Color("black"),
    ):
        info_position = (10, 10)
        padding = 1
        line_spacing = 5
        lines = appstate.get_info()
        if ni is not None:
            lines.append(f"niter: {ni}")
        if z2 is not None:
            lines.append(f"z2: {z2}")
        if k is not None:
            lines.append(f"k: {k}")
        if rgb is not None:
            lines.append(f"rgb: {rgb}")
        font = ft.SysFont("Arial", 12)

        # draw a rectangle
        def blit_text(surface, lines, pos, font):
            x, y = pos
            maxx, maxy = x, y
            for line in lines:
                bounding_rect = font.render_to(surface, (x, y), line, fgcolor, bgcolor)
                maxx = max(bounding_rect.right, maxx)
                maxy = y
                y += bounding_rect.height + line_spacing  # Start on new row
            return (maxx, maxy)

        rect_bottom_left = blit_text(screen_surface, lines, info_position, font)
        pygame.draw.rect(
            screen_surface,
            bgcolor,
            (
                info_position[0] - padding,
                info_position[1] - padding,
                rect_bottom_left[0] + padding,
                rect_bottom_left[1] + padding,
            ),
        )
        blit_text(screen_surface, lines, info_position, font)
        pygame.display.flip()

    def printhelp(appstate):
        print("Help:")
        print(f"    key(s): role, (default, current)")
        print(f"    z, left click: zoom in")
        print(f"    s, right click: zoom out")
        print(f"    up, down, left, right: pan")
        print(
            f"    k:  color mode (0, {appstate.color_mode}:{ColorMode(appstate.color_mode).name})"
        )
        print(
            f"    c:  color palette (0, {appstate.palette}:{Palette(appstate.palette).name})"
        )
        print(f"    w:  color waves (1, {appstate.color_waves})")
        print(f"    i:  max iterations (1000, {appstate.max_iterations})")
        print(f"    p:  power(2, {appstate.power})")
        print(f"    r:  escape radius(4, {appstate.escape_radius})")
        print(f"    e:  epsilon (0.001, {appstate.epsilon})")
        print(f"    a:  epsilon=0 (0.001, {appstate.epsilon})")
        print(
            f"    j:  middle click: julia/mandelbrot (mandelbrot, {FRACTAL_NAMES[appstate.fractal_mode]})"
        )
        print(f"    d:  display info")
        print(f"    backspace: reset")
        print(f"    q: quit")
        print(
            f"current x, y, h: {appstate.xcenter}, {appstate.ycenter}, {appstate.yheight}"
        )

    # Initialize pygame
    pygame.init()
    ft.init()
    appstate = AppState(FRACTAL_NAMES)
    screen_surface = pygame.display.set_mode(appstate.WINDOW_SIZE, pygame.HWSURFACE)
    # Get the PixelArray object for the screen
    # screen_pixels = pygame.PixelArray(screen_surface)
    # Init the display
    printhelp(appstate)
    output_array_niter, output_array_z2, output_array_k, output_array_rgb = redraw(
        screen_surface, appstate
    )
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
                    case pygame.K_d:
                        appstate.toggle_info()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                doredraw = True
                # 1 - left click, 2 - middle click, 3 - right click, 4 - scroll up, 5 - scroll down
                if event.button == 1 or event.button == 4:
                    appstate.zoom_in(pygame.mouse.get_pos())
                elif event.button == 3 or event.button == 5:
                    appstate.zoom_out(pygame.mouse.get_pos())
                elif event.button == 2:
                    appstate.change_fractal_mode(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEMOTION:
                # show info at cursor (ni, k...)
                (mx, my) = pygame.mouse.get_pos()
                ni = output_array_niter[mx, my]
                z2 = output_array_z2[mx, my]
                k = output_array_k[mx, my]
                rgb = output_array_rgb[mx, my]
                print_info(screen_surface, ni, z2, k, rgb)
            if doredraw:
                (
                    output_array_niter,
                    output_array_z2,
                    output_array_k,
                    output_array_rgb,
                ) = redraw(screen_surface, appstate)
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
    if args.profile:
        # https://docs.python.org/3.8/library/profile.html#module-cProfile
        cProfile.run("pygamemain()", sort="cumtime")
    else:
        pygamemain()


if __name__ == "__main__":
    main()
