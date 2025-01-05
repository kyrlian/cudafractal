# from timeit import default_timer

from enum import IntEnum
from typing import Tuple
from utils.types import (
    type_math_int,
    type_math_float,
    type_math_complex,
    type_enum_int,
)
from utils.cuda import (
    cuda_jit,
)


class Fractal_Mode(IntEnum):
    MANDELBROT = 0
    JULIA = 1


@cuda_jit(
    "(int32, int32, complex128, float64, float64, uint8, int32, int32, int32, float64, complex128)",
    device=True,
)
def fractal_xy(
    x: type_math_int,
    y: type_math_int,
    topleft: type_math_complex,
    xstep: type_math_float,
    ystep: type_math_float,
    fractalmode: type_enum_int,
    max_iterations: type_math_int,
    power: type_math_int,
    escape_radius: type_math_int,
    epsilon: type_math_float,
    juliaxy: type_math_complex,
) -> Tuple[type_math_int, type_math_float, type_math_float]:
    z: type_math_complex = type_math_complex(
        topleft + type_math_float(x) * xstep - 1j * y * ystep
    )
    c: type_math_complex = z if fractalmode == Fractal_Mode.MANDELBROT else juliaxy
    nb_iter: type_math_int = type_math_int(0)
    z2: type_math_float = type_math_float(0)
    der: type_math_complex = type_math_complex(1 + 0j)
    der2: type_math_float = type_math_float(1)
    while nb_iter < max_iterations and z2 < escape_radius and der2 > epsilon:
        der = der * power * z
        z = z**power + c
        nb_iter += 1
        z2 = z.real**2 + z.imag**2
        der2 = der.real**2 + der.imag**2
    return nb_iter, z2, der2


