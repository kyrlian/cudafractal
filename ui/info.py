
import pygame
import pygame.freetype as ft
from fractal.colors import K_Mode, Palette_Mode
from fractal.fractal import Fractal_Mode

def print_info(
    appstate,
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
    print(f"    k: k mode (0, {appstate.k_mode}:{k_mode_name})")
    palette_mode_name = Palette_Mode(appstate.palette_mode).name
    print(f"    c: palette mode (0, {appstate.palette_mode}:{palette_mode_name})")
    print(f"    v: custom palette ({appstate.custom_palette_name})")
    print(f"    w: color waves (2, {appstate.color_waves})")
    print(f"    i: max iterations (1000, {appstate.max_iterations})")
    print(f"    p: power(2, {appstate.power})")
    print(f"    r: escape radius(4, {appstate.escape_radius})")
    print(f"    e: epsilon (0.001, {appstate.epsilon})")
    print(f"    a: epsilon=0 (0.001, {appstate.epsilon})")
    fractal_mode_name = Fractal_Mode(appstate.fractal_mode).name
    print(
        f"    j: middle click: julia/mandelbrot (0, {appstate.fractal_mode}:{fractal_mode_name})"
    )
    print("    d: display info")
    print("    backspace: reset")
    print("    q: quit")
    print(
        f"current x, y, h: {appstate.xcenter}, {appstate.ycenter}, {appstate.yheight}"
    )
