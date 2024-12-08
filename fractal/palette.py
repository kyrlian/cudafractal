from typing import List
from fractal.colors import rgb_to_packed
from utils.cuda import cuda_jit
from utils.types import (
    type_color_int ,
)


palletes_definitions={
    "black_red_white":((0.0, 0, 0, 0), (0.5, 255, 0, 0), (1.0, 255, 255, 255)),
    "black_blue_white":((0.0, 0, 0, 0), (0.5, 0, 0, 255), (1.0, 255, 255, 255))

}

@cuda_jit(device=True)
def compute_color(colors, k) -> type_color_int:
    for i in range(len(colors) - 1):
        color_a_k, color_a_red, color_a_green, color_a_blue = colors[i]
        color_b_k, color_b_red, color_b_green, color_b_blue = colors[i + 1]
        if k >= color_a_k and k < color_b_k:
            ratio_a = 1.0 - (k - color_a_k) / (color_b_k - color_a_k)
            ratio_b = 1.0 - ratio_a
            r = int(color_a_red * ratio_a + color_b_red * ratio_b)
            g = int(color_a_green * ratio_a + color_b_green * ratio_b)
            b = int(color_a_blue * ratio_a + color_b_blue * ratio_b)
            return rgb_to_packed(r, g, b)
    return type_color_int(0)

def prepare_palette(colors, steps)->List[type_color_int]:
    palette=[]
    for i in range(steps):
        k = i / steps
        palette.append(compute_color(colors, k))
    return palette

def prepare_palettes(palletes_definition:dict, steps:int)->List[List[type_color_int]]:
    palettes=[]
    for name, palette_def in palletes_definitions.items():
        palettes.append(prepare_palette(palette_def,steps))
    return palettes
