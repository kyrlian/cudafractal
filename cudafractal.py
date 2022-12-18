# Write python code using nvidia cuda to render a mandelbrot fractal. When the user left clicks, zoom in by a factor of 2 into the fractal at the cursor. when the user right clicks, zoom out by a factor of 2.

import numpy as np
from numba import cuda

@cuda.jit
def render_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iterations):
    # Calculate the pixel size of the image
    dx = (xmax - xmin)/width
    dy = (ymax - ymin)/height

    # Calculate the maximum number of iterations before a point is considered part of the set
    max_iterations = 255

    # Calculate the coordinates of each pixel
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    
    for x in range(startX, width, gridX):
        real = xmin + x * dx
        for y in range(startY, height, gridY):
            imag = ymax - y * dy
            c = complex(real, imag)
            z = 0.0j

            for i in range(max_iterations):
                z = z*z + c
                if (z.real*z.real + z.imag*z.imag) >= 4:
                    break

            # Set the colour of the pixel
            if i == max_iterations - 1:
                # Point is in the set
                color = 0
            else:
                # Point is not in the set
                color = i + 1
                
            # Set the colour of the pixel
            image[y, x] = color

# Initialize the variables for the fractal
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height = 1024, 1024
max_iterations = 255

# Allocate an empty image array
image = np.zeros((height, width), dtype = np.uint8)

# Set the grid size
threadsperblock = (8, 8)
blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Call the kernel to render the fractal
render_mandelbrot[blockspergrid, threadsperblock](xmin, xmax, ymin, ymax, width, height, max_iterations)

# Handle user click events
def click(event):
    x, y = int(event.xdata), int(event.ydata)
    if event.button == 'left':
        # Zoom in
        xmin = xmin + (xmax - xmin)*(x/width)
        xmax = xmin + (xmax - xmin)*((x+1)/width)
        ymin = ymin + (ymax - ymin)*(y/height)
        ymax = ymin + (ymax - ymin)*((y+1)/height)
    elif event.button == 'right':
        # Zoom out
        xmin = xmin - (xmax - xmin)*(x/width)
        xmax = xmin + (xmax - xmin)*((x+1)/width)
        ymin = ymin - (ymax - ymin)*(y/height)
        ymax = ymin + (ymax - ymin)*((y+1)/height)
    else:
        return
    
    # Call the kernel to render the fractal
    render_mandelbrot[blockspergrid, threadsperblock](xmin, xmax, ymin, ymax, width, height, max_iterations)
    
    # Show the image
    plt.imshow(image)
    plt.show()

# Connect the click event to the handler
plt.connect('button_press_event', click)

# Show the image
plt.imshow(image)
plt.show()