import pygame

# Initialize pygame
pygame.init()

# Set the window size
window_size = (400, 400)

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the background color to white
screen.fill((255, 255, 255))

# Get the PixelArray object for the screen
pixels = pygame.PixelArray(screen)

# Center of the circle
center_x, center_y = 200, 200

# Radius of the circle
radius = 150

# Iterate over the rows and columns of the PixelArray
for x in range(window_size[0]):
    for y in range(window_size[1]):
        # Calculate the distance between the current position and the center of the circle
        distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        # If the distance is less than the radius, set the pixel to black
        if distance < radius:
            pixels[x, y] = (0, 0, 0)

# Update the display
pygame.display.flip()

# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit pygame
pygame.quit()
