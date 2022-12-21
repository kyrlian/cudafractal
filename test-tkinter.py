#brew install python-tk
import tkinter as tk

# Create the canvas
canvas_width = 500
canvas_height = 500
canvas = tk.Canvas(width=canvas_width, height=canvas_height)
canvas.pack()

# Create a function to draw a white pixel at the cursor position when the user clicks
def draw_pixel(event):
    x, y = event.x, event.y
    canvas.create_rectangle(x, y, x+1, y+1, fill='white')

# Bind the click event to the draw_pixel function
canvas.bind('<Button-1>', draw_pixel)

# Run the Tkinter event loop
tk.mainloop()
