import pygame
from pygame.locals import KEYDOWN, K_ESCAPE

# http://www.land-of-kain.de/docs/python_opengl_cuda_opencl/

if __name__ == "__main__":
    pygame.init()
    keep_running = True
    size = width, height = 320, 240
    speed = [2, 2]
    black = 0, 0, 0
    screen = pygame.display.set_mode(size, pygame.HWSURFACE)
    ball = pygame.image.load("ball.gif")
    pygame.image.frombuffer
    ballrect = ball.get_rect()
    while keep_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == KEYDOWN and event.key == K_ESCAPE
            ):
                pygame.display.set_caption("Shutting down...")
                keep_running = False
        # Move the ball, bounce back on borders.
        ballrect = ballrect.move(speed)
        speed[0] = (
            -speed[0] if ballrect.left < 0 or ballrect.right > width else speed[0]
        )
        speed[1] = (
            -speed[1] if ballrect.top < 0 or ballrect.bottom > height else speed[1]
        )
        # Fill the background screen buffer, then flip.
        screen.fill(black)
        screen.blit(ball, ballrect)
        pygame.display.flip()
        pygame.time.wait(10)
