from numba import complex128
from numba import cuda
from fractal_cuda.colors_cuda import set_pixel_color

@cuda.jit(
    "void(uint32[:,:], complex128, float64, float64, int32, int32, int32, float64, complex128, int32, int32, int32)"
)
def mandelbrot(
    device_array,
    topleft,
    xstep,
    ystep,
    maxiter,
    p,
    r,
    eps,
    juliaxy,
    cmode,
    palette,
    color_waves,
):
    x, y = cuda.grid(2)
    if x < device_array.shape[0] and y < device_array.shape[1]:
        c = complex128(topleft + x * xstep - 1j * y * ystep)
        z = c
        nbi = 0
        z2 = 0
        der = complex128(1 + 0j)
        der2 = 1
        while nbi < maxiter and z2 < r and der2 > eps:
            der = der * p * z
            z = z**p + c
            nbi += 1
            z2 = z.real**2 + z.imag**2
            der2 = der.real**2 + der.imag**2
        set_pixel_color(
            device_array, x, y, nbi, maxiter, z2, r, der2, cmode, palette, color_waves
        )


@cuda.jit(
    "void(uint32[:,:], complex128, float64, float64, int32, int32, int32, float64, complex128, int32, int32, int32)"
)
def julia(
    device_array,
    topleft,
    xstep,
    ystep,
    maxiter,
    p,
    r,
    eps,
    juliaxy,
    cmode,
    palette,
    color_waves,
):
    x, y = cuda.grid(2)
    if x < device_array.shape[0] and y < device_array.shape[1]:
        z = complex128(topleft + x * xstep - 1j * y * ystep)
        nbi = 0
        z2 = 0
        der = complex128(1 + 0j)
        der2 = 1
        while nbi < maxiter and z2 < r and der2 > eps:
            # TODO test julia with/without der
            der = der * p * z
            z = z**p + juliaxy
            nbi += 1
            z2 = z.real**2 + z.imag**2
            der2 = der.real**2 + der.imag**2
        set_pixel_color(
            device_array, x, y, nbi, maxiter, z2, r, der2, cmode, palette, color_waves
        )

FRACTAL_MODES = [mandelbrot, julia]
