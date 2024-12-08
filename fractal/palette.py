from typing import List, Tuple
from timeit import default_timer
from utils.types import (
    type_math_float,
    type_math_int,
    type_color_float,
    type_color_int,
    type_color_int_small,
)


palletes_definitions = {
    "black_red_white": ((0.0, 0, 0, 0), (0.5, 255, 0, 0), (1.0, 255, 255, 255)),
    "black_blue_white": ((0.0, 0, 0, 0), (0.5, 0, 0, 255), (1.0, 255, 255, 255)),
}


def rgb_to_packed(
    r: type_color_int_small, g: type_color_int_small, b: type_color_int_small
) -> type_color_int:
    # r, g, b should be [0:255]
    # Cant assert in device function
    # assert r >= 0 and r <= 255, f"r should be in [0:255], got {r}"
    # assert g >=  0 and g  <= 255, f"g should be in [0:255], got {g}"
    # assert b >=  0 and b  <= 255, f"b should be in [0:255], got {b}"
    packed = ((type_color_int(r) * 256) + g) * 256 + b
    return packed


def compute_color(
    palette_colors: List[
        Tuple[type_math_float, type_color_float, type_color_float, type_color_float]
    ],
    k: type_math_float,
) -> type_color_int:
    for i in range(len(palette_colors) - 1):
        color_a_k, color_a_red, color_a_green, color_a_blue = palette_colors[i]
        color_b_k, color_b_red, color_b_green, color_b_blue = palette_colors[i + 1]
        if k >= color_a_k and k < color_b_k:
            ratio_a = 1.0 - (k - color_a_k) / (color_b_k - color_a_k)
            ratio_b = 1.0 - ratio_a
            r = type_color_int_small(color_a_red * ratio_a + color_b_red * ratio_b)
            g = type_color_int_small(color_a_green * ratio_a + color_b_green * ratio_b)
            b = type_color_int_small(color_a_blue * ratio_a + color_b_blue * ratio_b)
            return rgb_to_packed(r, g, b)
    return type_color_int(0)


def prepare_palette(palette_colors, steps: type_math_int) -> List[type_color_int]:
    computed_palette = []
    for i in range(steps):
        k = i / steps
        computed_palette.append(compute_color(palette_colors, k))
    return computed_palette


def prepare_palettes(
    palletes_defs: dict, steps: type_math_int
) -> List[List[type_color_int]]:
    print("Precomputing palettes")
    timerstart = default_timer()
    computed_palettes = []
    for name, palette_def in palletes_defs.items():
        computed_palettes.append(prepare_palette(palette_def, steps))
    print(f"Palettes calculated in {(default_timer() - timerstart)}s")
    return computed_palettes
