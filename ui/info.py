import pygame
import pygame.freetype as ft
from fractal.colors import Normalization_Mode, Palette_Mode
from fractal.fractal import Fractal_Mode
from ui.keys_config import (
    key_shift,
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
)
from pygame.key import name as key_name
from utils import defaults


def print_info(
    appstate,
    screen_surface,
    ni=None,niter_min=None, niter_max=None, 
    z2=None,z2_min=None, z2_max=None,
    der2=None, der2_min=None, der2_max=None,
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
        lines.append(f"niter: {ni} ({niter_min}-{niter_max})")
    if z2 is not None:
        lines.append(f"z2: {z2} ({z2_min}-{z2_max})")
    if der2 is not None:
        lines.append(f"der2: {der2} ({der2_min}-{der2_max})")
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
    # uses default values from defaults.py
    print("Help:")
    print("    key(s): role, (default, current)")
    print(f"    {key_name(key_zoom)}, left click: zoom in")
    print(f"    {key_name(key_shift)}+{key_name(key_zoom)}, right click: zoom out")
    print(
        f"    {key_name(key_pan_up)}, {key_name(key_pan_down)}, {key_name(key_pan_left)}, {key_name(key_pan_right)}: pan"
    )
    normalization_mode_name = Normalization_Mode(appstate.normalization_mode).name
    print(
        f"    {key_name(key_normalization_mode)}: normalization mode ({defaults.normalization_mode}, {appstate.normalization_mode}:{normalization_mode_name})"
    )
    palette_mode_name = Palette_Mode(appstate.palette_mode).name
    print(
        f"    {key_name(key_palette_mode)}: palette mode ({defaults.palette_mode}, {appstate.palette_mode}:{palette_mode_name})"
    )
    print(
        f"    {key_name(key_color_palette)}: custom palette ({appstate.custom_palette_name})"
    )
    print(
        f"    {key_name(key_palette_shift)}: palette shift ({defaults.palette_shift}, {appstate.palette_shift})"
    )
    print(
        f"    {key_name(key_palette_width)}: palette width ({defaults.palette_width}, {appstate.palette_width})"
    )
    print(
        f"    {key_name(key_iter)}: max iterations ({defaults.max_iterations}, {appstate.max_iterations})"
    )
    print(f"    {key_name(key_power)}: power({defaults.power}, {appstate.power})")
    print(
        f"    {key_name(key_escape_radius)}: escape radius({defaults.escape_radius}, {appstate.escape_radius})"
    )
    print(
        f"    {key_name(key_epsilon)}: epsilon ({defaults.epsilon}, {appstate.epsilon})"
    )
    print(
        f"    {key_name(key_epsilon_reset)}: epsilon=0 {defaults.epsilon}, {appstate.epsilon})"
    )
    fractal_mode_name = Fractal_Mode(appstate.fractal_mode).name
    print(
        f"    {key_name(key_julia)}: middle click: julia/mandelbrot ({defaults.fractal_mode}, {appstate.fractal_mode}:{fractal_mode_name})"
    )
    print(f"    {key_name(key_display_info)}: display info")
    print(f"    {key_name(key_screenshot)}: screenshot")
    print(f"    {key_name(key_help)}: help")
    print(f"    {key_name(key_reset)}: reset")
    print(f"    {key_name(key_quit)}: quit")
    print(
        f"current x, y, h: {appstate.xcenter}, {appstate.ycenter}, {appstate.yheight}"
    )
