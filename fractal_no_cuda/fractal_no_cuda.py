from fractal_no_cuda.colors_no_cuda import set_image_color
from numba import complex128


def mandelbrot(
    pixels, topleft, xstep, ystep, maxiter, p, r, eps, jx, jy, cmode, palette
):
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            c = complex128(topleft + x * xstep - 1j * y * ystep)
            z = c  # complex128(0 + 0j)
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
            set_image_color(pixels, x, y, nbi, maxiter, z2, r, der2, cmode, palette)


def julia(pixels, topleft, xstep, ystep, maxiter, p, r, eps, jx, jy, cmode, palette):
    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            c = complex128(jx + 1j * jy)
            z = complex128(topleft + x * xstep - 1j * y * ystep)
            nbi = 0
            z2 = 0
            der = complex128(1 + 0j)
            der2 = 1
            while nbi < maxiter and z2 < r and der2 > eps:
                # TODO test julia with/without der
                der = der * p * z
                z = z**p + c
                nbi += 1
                z2 = z.real**2 + z.imag**2
                der2 = der.real**2 + der.imag**2
            set_image_color(pixels, x, y, nbi, maxiter, z2, r, der2, cmode, palette)
