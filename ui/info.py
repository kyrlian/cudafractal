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
        lines.append(f"z2: {z2:.4f} ({z2_min:.4f}-{z2_max:.4f})")
    if der2 is not None:
        lines.append(f"der2: {der2:.4f} ({der2_min}-{der2_max})")
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
    print("key(s): role, value")
    print(f"{key_name(key_zoom)}, left click: zoom in")
    print(f"{key_name(key_shift)}+{key_name(key_zoom)}, right click: zoom out")
    print(f"current center x, center y, height: {appstate.xcenter}, {appstate.ycenter}, {appstate.yheight}")
    print(f"{key_name(key_pan_up)}, {key_name(key_pan_down)}, {key_name(key_pan_left)}, {key_name(key_pan_right)}: pan")
    for i in appstate.get_info():
        print(i)
    print(f"{key_name(key_epsilon_reset)}: epsilon=0")
    print(f"{key_name(key_display_info)}: display info")
    print(f"{key_name(key_screenshot)}: screenshot")
    print(f"{key_name(key_help)}: help")
    print(f"{key_name(key_reset)}: reset")
    print(f"{key_name(key_quit)}: quit")
