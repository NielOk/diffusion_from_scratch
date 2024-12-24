'''
This file contains a class that generates training data for the diffusion model. 
'''

from PIL import Image
import numpy as np

# Example matrix of RGB tuples
matrix = [
    [(255, 0, 0), (0, 255, 0), (0, 0, 255)],  # Red, Green, Blue
    [(255, 255, 0), (255, 255, 255), (0, 0, 0)]  # Yellow, White, Black
]

# Convert to a NumPy array
array = np.array(matrix, dtype=np.uint8)

# Create image from array
image = Image.fromarray(array)

# Save as PNG
image.save("output.png")