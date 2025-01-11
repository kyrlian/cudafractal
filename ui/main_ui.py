#!python3
import cProfile
import pygame
import pygame.freetype as ft
import argparse
from utils.appState import AppState
from fractal.fractal import init_arrays, compute_fractal
from ui.info import print_info, print_help
from ui.screenshot import screenshot, load_metada
from fractal.palette import (
    prepare_palettes,
    palettes_definitions,
    get_computed_palette,
)
from ui.keys_config import (
    key_shift,
    key_shift_r,
    key_quit,
    key_zoom,
    key_screenshot,
    key_pan_up,
    key_pan_down,
    key_pan_left,
    key_pan_right,
    key_iter,
    key_escape_radius,
    key_epsilon_reset,
    key_epsilon,
    key_power,
    key_julia,
    key_normalization_mode,
    key_palette_mode,
    key_color_palette,
    key_palette_shift,
    key_palette_width,
    key_reset,
    key_help,
    key_display_info,
    key_ctrl,
    key_ctrl_r,
)
from fractal.colors import Palette_Mode


def pygamemain(src_image=None):
    def redraw(
        screen_surface,
        appstate,
        host_array_niter,
        niter_min,
        niter_max,
        host_array_z2,
        z2_min,
        z2_max,
        host_array_der2,
        der2_min,
        der2_max,
        host_array_k,
        host_array_rgb,
        recalc_fractal=True,
        recalc_color=True,
    ):
        appstate.recalc_size()
        if appstate.palette_mode == Palette_Mode.CUSTOM:
            # Get custom palette
            custom_palette = get_computed_palette(
                computed_palettes, appstate.custom_palette_name
            )
        else:
            custom_palette = []
        # Compute fractal
        (
            host_array_niter,
            niter_min,
            niter_max,
            host_array_z2,
            z2_min,
            z2_max,
            host_array_der2,
            der2_min,
            der2_max,
            host_array_k,
            host_array_rgb,
        ) = compute_fractal(
            host_array_niter,
            niter_min,
            niter_max,
            host_array_z2,
            z2_min,
            z2_max,
            host_array_der2,
            der2_min,
            der2_max,
            host_array_k,
            host_array_rgb,
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
            appstate.normalization_mode,
            appstate.palette_mode,
            custom_palette,
            appstate.palette_width,
            appstate.palette_shift,
            recalc_fractal,
            recalc_color,
        )
        pygame.pixelcopy.array_to_surface(screen_surface, host_array_rgb)
        if appstate.show_info:
            print_info(appstate, screen_surface)
        pygame.display.flip()
        return (
            host_array_niter,
            niter_min,
            niter_max,
            host_array_z2,
            z2_min,
            z2_max,
            host_array_der2,
            der2_min,
            der2_max,
            host_array_k,
            host_array_rgb,
        )

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
    # TODO: add a specific "palette_steps" parameter instead of max iterations
    computed_palettes = prepare_palettes(palettes_definitions, appstate.max_iterations)
    # init matrices
    (
        host_array_niter,
        host_array_z2,
        host_array_der2,
        host_array_k,
        host_array_rgb,
    ) = init_arrays(appstate.WINDOW_SIZE)
    # Initial draw
    (
        host_array_niter,
        niter_min,
        niter_max,
        host_array_z2,
        z2_min,
        z2_max,
        host_array_der2,
        der2_min,
        der2_max,
        host_array_k,
        host_array_rgb,
    ) = redraw(
        screen_surface,
        appstate,
        host_array_niter,
        0,
        0,
        host_array_z2,
        0,
        0,
        host_array_der2,
        0,
        0,
        host_array_k,
        host_array_rgb,
        True,
        True,
    )


    def handle_event(event, appstate, screen_surface, host_array_niter, host_array_z2
    , host_array_der2
    , host_array_k
    , host_array_rgb
    ):
        recalc_fractal = False
        recalc_color = False
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            shift = (
                pygame.key.get_pressed()[key_shift]
                or pygame.key.get_pressed()[key_shift_r]
            )
            ctrl = (
                pygame.key.get_pressed()[key_ctrl]
                or pygame.key.get_pressed()[key_ctrl_r]
            )
            if event.key == key_quit:
                running = False
            elif event.key == key_zoom:
                if shift:
                    appstate.zoom_out()
                else:
                    appstate.zoom_in()
                recalc_fractal = True
            elif event.key == key_screenshot:
                screenshot(screen_surface,appstate)
            elif event.key == key_pan_up:
                appstate.pan(0, 1)
                recalc_fractal = True
            elif event.key == key_pan_down:
                appstate.pan(0, -1)
                recalc_fractal = True
            elif event.key == key_pan_left:
                appstate.pan(-1, 0)
                recalc_fractal = True
            elif event.key == key_pan_right:
                appstate.pan(1, 0)
                recalc_fractal = True
            elif event.key == key_iter:
                if shift:
                    appstate.change_max_iterations(1/1.1)
                else:
                    appstate.change_max_iterations(1.1)
                # Recompute palettes
                # TODO: add a specific "palette_steps" parameter instead of max iterations
                computed_palettes = prepare_palettes(
                    palettes_definitions, appstate.max_iterations
                )
                recalc_fractal = True
            elif event.key == key_escape_radius:
                if shift:
                    appstate.change_escape_radius(0.5)
                else:
                    appstate.change_escape_radius(2)
                recalc_fractal = True
            elif event.key == key_epsilon_reset:
                appstate.change_epsilon(0)
                recalc_fractal = True
            elif event.key == key_epsilon:
                if shift:
                    appstate.change_epsilon(10)
                else:
                    appstate.change_epsilon(0.1)
                recalc_fractal = True
            elif event.key == key_power:
                if shift:
                    appstate.change_power(-1)
                else:
                    appstate.change_power(1)
                recalc_fractal = True
            elif event.key == key_julia:
                appstate.change_fractal_mode(pygame.mouse.get_pos())
                recalc_fractal = True
            elif event.key == key_normalization_mode:
                appstate.change_normalization_mode()
                recalc_color = True
            elif event.key == key_palette_mode:
                appstate.change_color_palette_mode()
                recalc_color = True
            elif event.key == key_color_palette:
                appstate.change_color_palette_name()
                appstate.reset_palette_shift()
                recalc_color = True
            elif event.key == key_palette_shift:
                step = 0.01
                direction = 1
                if ctrl:
                    step = 0.1
                if shift:
                    direction = -1
                appstate.change_palette_shift(step * direction)
                recalc_color = True
            elif event.key == key_palette_width:
                if shift:
                    appstate.change_palette_width(1.1)
                else:
                    appstate.change_palette_width(0.9)
                recalc_color = True
            elif event.key == key_reset:
                appstate.reset()
                recalc_fractal = True
            elif event.key == key_help:
                print_help(appstate)
            elif event.key == key_display_info:
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
            ni = host_array_niter[mx, my]
            z2 = host_array_z2[mx, my]
            der2 = host_array_der2[mx, my]
            k = host_array_k[mx, my]
            rgb = host_array_rgb[mx, my]
            print_info(
                appstate,
                screen_surface,
                ni,
                niter_min,
                niter_max,
                z2,
                z2_min,
                z2_max,
                der2,
                der2_min,
                der2_max,
                k,
                rgb,
            )
        return recalc_fractal, recalc_color
    # Run the game loop
    running = True
    while running:
        for event in pygame.event.get():
            recalc_fractal, recalc_color = handle_event(event, appstate, screen_surface, host_array_niter, host_array_z2
            , host_array_der2
            , host_array_k
            , host_array_rgb)
            if recalc_fractal or recalc_color:
                (
                    host_array_niter,
                    niter_min,
                    niter_max,
                    host_array_z2,
                    z2_min,
                    z2_max,
                    host_array_der2,
                    der2_min,
                    der2_max,
                    host_array_k,
                    host_array_rgb,
                ) = redraw(
                    screen_surface,
                    appstate,
                    host_array_niter,
                    niter_min,
                    niter_max,
                    host_array_z2,
                    z2_min,
                    z2_max,
                    host_array_der2,
                    der2_min,
                    der2_max,
                    host_array_k,
                    host_array_rgb,
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
