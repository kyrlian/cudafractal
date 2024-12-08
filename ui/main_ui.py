#!python3
import cProfile
import pygame
import pygame.freetype as ft
import argparse
from utils.appState import AppState
from fractal.fractal import init_arrays, compute_fractal
from ui.info import print_info, print_help
from ui.screenshot import screenshot, load_metada
from fractal.palette import prepare_palettes, palletes_definitions, get_computed_palette

def pygamemain(src_image=None):
    def redraw(
        screen_surface,
        appstate,
        output_array_niter,
        output_array_z2,
        output_array_k,
        output_array_rgb,
        recalc_fractal=True,
        recalc_color=False,
    ):
        appstate.recalc_size()
        # TODO get current custom pre-computed palette and pass it
        custom_palette = get_computed_palette(computed_palettes, appstate.custom_palette_name)
        output_array_niter, output_array_z2, output_array_k, output_array_rgb = (
            compute_fractal(
                output_array_niter,
                output_array_z2,
                output_array_k,
                output_array_rgb,
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
                appstate.k_mode,
                appstate.palette_mode,
                appstate.color_waves,
                recalc_fractal,
                recalc_color,
            )
        )
        pygame.pixelcopy.array_to_surface(screen_surface, output_array_rgb)
        if appstate.show_info:
            print_info(appstate, screen_surface)
        pygame.display.flip()
        return output_array_niter, output_array_z2, output_array_k, output_array_rgb

    # Initialize pygame
    pygame.init()
    ft.init()
    appstate = AppState()
    # Load metadata from image if present
    if src_image is not None:
        load_metada(src_image, appstate)
    # Init the display
    screen_surface = pygame.display.set_mode(appstate.WINDOW_SIZE, pygame.HWSURFACE)
    print_help(appstate)
    # Init palettes
    computed_palettes = prepare_palettes(palletes_definitions, appstate.max_iterations)
    # init matrices
    output_array_niter, output_array_z2, output_array_k, output_array_rgb = init_arrays(
        appstate.WINDOW_SIZE
    )
    # Initial draw
    output_array_niter, output_array_z2, output_array_k, output_array_rgb = redraw(
        screen_surface,
        appstate,
        output_array_niter,
        output_array_z2,
        output_array_k,
        output_array_rgb,
        True,
        False,
    )
    # Run the game loop
    running = True
    while running:
        for event in pygame.event.get():
            recalc_fractal = False
            recalc_color = False
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                shift = (
                    pygame.key.get_pressed()[pygame.K_LSHIFT]
                    or pygame.key.get_pressed()[pygame.K_RSHIFT]
                )
                match event.key:
                    case pygame.K_q:
                        running = False
                    case pygame.K_z:
                        if shift:
                            appstate.zoom_out()
                        else:
                            appstate.zoom_in()
                        recalc_fractal = True
                    case pygame.K_s:
                        screenshot(appstate)
                    case pygame.K_UP:
                        appstate.pan(0, 1)
                        recalc_fractal = True
                    case pygame.K_DOWN:
                        appstate.pan(0, -1)
                        recalc_fractal = True
                    case pygame.K_LEFT:
                        appstate.pan(-1, 0)
                        recalc_fractal = True
                    case pygame.K_RIGHT:
                        appstate.pan(1, 0)
                        recalc_fractal = True
                    case pygame.K_i:
                        if shift:
                            appstate.change_max_iterations(0.9)
                        else:
                            appstate.change_max_iterations(1.1)
                        # Recompute palettes
                        computed_palettes = prepare_palettes(palletes_definitions, appstate.max_iterations)
                        recalc_fractal = True
                    case pygame.K_r:
                        if shift:
                            appstate.change_escape_radius(0.5)
                        else:
                            appstate.change_escape_radius(2)
                        recalc_fractal = True
                    case pygame.K_a:
                        appstate.change_epsilon(0)
                        recalc_fractal = True
                    case pygame.K_e:
                        if shift:
                            appstate.change_epsilon(10)
                        else:
                            appstate.change_epsilon(0.1)
                        recalc_fractal = True
                    case pygame.K_p:
                        if shift:
                            appstate.change_power(-1)
                        else:
                            appstate.change_power(1)
                        recalc_fractal = True
                    case pygame.K_j:
                        appstate.change_fractal_mode(pygame.mouse.get_pos())
                        recalc_fractal = True
                    case pygame.K_k:
                        appstate.change_k_mode()
                        recalc_color = True
                    case pygame.K_c:
                        appstate.change_color_palette_mode()
                        recalc_color = True
                    case pygame.K_v:
                        appstate.change_color_palette_name()
                        recalc_color = True
                    case pygame.K_w:
                        if shift:
                            appstate.change_color_waves(-1)
                        else:
                            appstate.change_color_waves(1)
                        recalc_color = True
                    case pygame.K_BACKSPACE:
                        appstate.reset()
                        recalc_fractal = True
                    case pygame.K_h:
                        print_help(appstate)
                    case pygame.K_d:
                        appstate.toggle_info()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                recalc_fractal = True
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
                print_info(appstate, screen_surface, ni, z2, k, rgb)
            if recalc_fractal or recalc_color:
                (
                    output_array_niter,
                    output_array_z2,
                    output_array_k,
                    output_array_rgb,
                ) = redraw(
                    screen_surface,
                    appstate,
                    output_array_niter,
                    output_array_z2,
                    output_array_k,
                    output_array_rgb,
                    recalc_fractal,
                    recalc_color,
                )
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
    parser.add_argument("-s", "--source", help="source image")
    parser.add_argument("-c", "--cpu", help="compute on cpu only", action="store_true")
    args = parser.parse_args()
    if args.profile:
        # https://docs.python.org/3.8/library/profile.html#module-cProfile
        cProfile.run("pygamemain()", sort="cumtime")
    else:
        if args.source is not None:
            pygamemain(args.source)
        else:
            pygamemain()


if __name__ == "__main__":
    main()
