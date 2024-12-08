#!python3
import cProfile
import pygame
import pygame.freetype as ft
import argparse
from utils.appState import AppState
from fractal.fractal import init_arrays, compute_fractal, Fractal_Mode
from fractal.colors import K_Mode, Palette_Mode
from PIL import Image
from PIL.PngImagePlugin import PngInfo


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
        # output_array_rgb = pygame.PixelArray(screen_surface)
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
            print_info(screen_surface)
        pygame.display.flip()
        return output_array_niter, output_array_z2, output_array_k, output_array_rgb

    def print_info(
        screen_surface,
        ni=None,
        z2=None,
        k=None,
        rgb=None,
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

    def print_help(appstate):
        print("Help:")
        print("    key(s): role, (default, current)")
        print("    z, left click: zoom in")
        print("    shift+z, right click: zoom out")
        print("    up, down, left, right: pan")
        k_mode_name = K_Mode(appstate.k_mode).name
        print(f"    k:  k mode (0, {appstate.k_mode}:{k_mode_name})")
        palette_mode_name = Palette_Mode(appstate.palette_mode).name
        print(f"    c:  palette mode (0, {appstate.palette_mode}:{palette_mode_name})")
        print(f"    w:  color waves (2, {appstate.color_waves})")
        print(f"    i:  max iterations (1000, {appstate.max_iterations})")
        print(f"    p:  power(2, {appstate.power})")
        print(f"    r:  escape radius(4, {appstate.escape_radius})")
        print(f"    e:  epsilon (0.001, {appstate.epsilon})")
        print(f"    a:  epsilon=0 (0.001, {appstate.epsilon})")
        fractal_mode_name = Fractal_Mode(appstate.fractal_mode).name
        print(
            f"    j:  middle click: julia/mandelbrot (0, {appstate.fractal_mode}:{fractal_mode_name})"
        )
        print("    d:  display info")
        print("    backspace: reset")
        print("    q: quit")
        print(
            f"current x, y, h: {appstate.xcenter}, {appstate.ycenter}, {appstate.yheight}"
        )

    def screenshot(appstate):
        filename = "screenshot.png"
        # TODO - add timestamp
        pygame.image.save(screen_surface, filename)
        # add metadata
        metadata_info = appstate.get_info_table()
        targetImage = Image.open(filename)
        metadata = PngInfo()
        for key, value in metadata_info.items():
            print(f"Adding metadata {key}:{value}")
            metadata.add_text(key, f"{value}")
        targetImage.save(filename, pnginfo=metadata)
        print(f"Saved screenshot to {filename}")

    def load_metada(filename, appstate):
        print(f"Loading metadata from {filename}")
        srcImage = Image.open(filename)
        srcImage.load()
        info_table = srcImage.info
        print(f"Metadata info_table: {info_table}")
        appstate.set_from_info_table(info_table)
        return info_table

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
                print_info(screen_surface, ni, z2, k, rgb)
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
