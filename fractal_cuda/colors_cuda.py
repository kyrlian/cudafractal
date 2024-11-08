from math import log
from numba import float64
from numba import cuda

NB_COLOR_MODES = 6
NB_PALETTES = 2


@cuda.jit("void(uint32[:,:], int32, int32, int32, int32, int32)", device=True)
def set_color_rgb(device_array, x, y, r, g, b):
    # r, g, b should be [0:255]
    packed = (r * 256 + g) * 256 + b
    device_array[x, y] = packed


@cuda.jit("void(uint32[:,:], int32, int32, float64, float64, float64)", device=True)
def set_color_hsv(device_array, x, y, h, s, v):
    # h,s,v should be [0:1]
    r, g, b = 0, 0, 0
    if s > 0:
        if h == 1.0:
            h = 0.0
        i = int(h * 6.0)
        f = h * 6.0 - i

        w = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        if i == 0:
            r, g, b = v, t, w
        if i == 1:
            r, g, b = q, v, w
        if i == 2:
            r, g, b = w, v, t
        if i == 3:
            r, g, b = w, q, v
        if i == 4:
            r, g, b = t, w, v
        if i == 5:
            r, g, b = v, w, q
    else:
        r, g, b = v, v, v
    set_color_rgb(device_array, x, y, int(r * 255), int(g * 255), int(b * 255))


@cuda.jit(
    "void(uint32[:,:], int32, int32, int32, int32, float64, float64, float64, int32, int32, int32)",
    device=True,
)
def set_pixel_color(
    device_array, x, y, nbi, max_iter, z2, r, der2, cmode, palette, color_waves
):
    if z2 > r:
        # first calculate k based on color mode
        # k should be 0-1
        match cmode:
            case 0:
                mic = max_iter / color_waves
                k = float64(nbi % mic / mic)
            case 1:
                k = float64(nbi / max_iter)
            case 2:
                k = log(float64(nbi)) / log(float64(max_iter))
            case 3:
                # https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                # TODO : z2 is slightly bigger than r, so k doesnt cover 0-1
                k = float64(r) / float64(z2)
            case 4:
                k = log(float64(r)) / log(float64(z2))
            case 5:
                # k = math.sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                # k = sin(log(z2)) / 2 + 0.5 # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, sin table too big ?
                k = 1 / z2
        # then calculate color from k
        match palette:
            case 0:  # hue
                set_color_hsv(device_array, x, y, k, 1, 1)
            case 1:  # grayscale
                kk = int(k * 255)
                device_array[x, y] = (kk * 256 + kk) * 256 + kk
            case 3:  # custom palette k to rgb
                pass
    else:
        device_array[x, y] = 0


# TODO use list of color mode names
# TODO allow to change palette length
