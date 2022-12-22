import pygame
import numpy
from numba import float64, int8, complex128

def randomarray(image_array):
    for y in range(image_array.shape[0] ):
        for x in range (image_array.shape[1]) :
            # image_array[y, x, 0] = int8(x % 255)
            # image_array[y, x, 1] = int8(y % 255)
            # image_array[y, x, 2] = int8((x+y) % 255)
            image_array[y, x] = ( x % 255, y % 255,(x+y) % 255)
    return image_array

if __name__ == "__main__":
    pygame.init()
    keep_running = True
    size = width, height = 320, 240
    black = 0, 0, 0
    screen = pygame.display.set_mode(size, pygame.HWSURFACE)
    screen_pixels = pygame.PixelArray(screen)
    while keep_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                pygame.display.set_caption("Shutting down...")
                keep_running = False
        randomarray(screen_pixels)
        pygame.display.flip()
        pygame.time.wait(10)
