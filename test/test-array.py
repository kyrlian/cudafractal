import numpy
import sys
#pixel_array = numpy.zeros((256, 256, 3), dtype=float)
pixel_array = numpy.zeros((256, 256, 3))
x = y = 1
r = g = b = 255
pixel_array[x, y] = (r, g, b)
pixel_array[x, y, 1] = g
print("%f" % pixel_array[x, y, 1])