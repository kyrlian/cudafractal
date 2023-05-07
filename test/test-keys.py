import pygame, sys
# Initialize pygame
pygame.init()
# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_a:
                sys.stdout.write("Key: a\n")
                sys.stdout.write("pygame.K_LSHIFT: %i\n" % pygame.key.get_pressed()[pygame.K_LSHIFT])
                sys.stdout.write("pygame.K_RSHIFT: %i\n" % pygame.key.get_pressed()[pygame.K_RSHIFT])
                sys.stdout.write("pygame.KMOD_SHIFT: %i\n" % pygame.key.get_pressed()[pygame.KMOD_SHIFT])
# Quit pygame
pygame.quit()