'''
This file contains a class that generates training data for the diffusion model. 
'''

from PIL import Image
import numpy as np
from typing import Tuple

class TrainingDataGenerator:

    def __init__(self):
        self.background_color = None
        self.square_color = None
        self.triangle_color = None
        self.image_size = None
        self.square_size = None
        self.triangle_size = None

    def draw_square(self, 
                    image_size: Tuple[int, int],
                    square_size: int, 
                    square_color: Tuple[int, int, int],
                    background_color: Tuple[int, int, int], # (255, 255, 255) is white, (0, 0, 0) is black
                    image_path: str
                    ) -> None:
        '''
        Draws a square on an image and saves it to a file.
        '''
        # Create a blank image
        canvas_matrix = np.full((image_size[0], image_size[1], 3), background_color, dtype=np.uint8)
        
        # Figure out coordinates and replace
        width, height = image_size
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
        canvas_matrix[top:bottom, left:right] = square_color

        image = Image.fromarray(canvas_matrix)
        image.save(image_path)

        # Save parameters to class
        self.background_color = background_color
        self.square_color = square_color
        self.image_size = image_size
        self.square_size = square_size

    def draw_triangle(self, 
                    image_size: Tuple[int, int],
                    triangle_size: Tuple[int, int],
                    triangle_color: Tuple[int, int, int],
                    background_color: Tuple[int, int, int], # (255, 255, 255) is white, (0, 0, 0) is black
                    image_path: str
                    ) -> None:
        '''
        Draws a triangle on an image and saves it to a file.
        '''
        # Create a blank image
        canvas_matrix = np.full((image_size[0], image_size[1], 3), background_color, dtype=np.uint8)
        
        # Figure out coordinates and draw triangle
        # Triangle dimensions and center
        width, height = image_size
        triangle_width, triangle_height = triangle_size

        center_x, center_y = width // 2, height // 2
        half_width = triangle_width // 2

        # Triangle vertices
        top_vertex = (center_x, center_y - triangle_height // 2)  # Top vertex
        left_vertex = (center_x - half_width, center_y + triangle_height // 2)  # Bottom-left vertex
        right_vertex = (center_x + half_width, center_y + triangle_height // 2)  # Bottom-right vertex

        # Fill the triangle region
        for y in range(height):
            for x in range(width):
                # Check if the point (x, y) is inside the triangle using the barycentric method
                b1 = (x - left_vertex[0]) * (top_vertex[1] - left_vertex[1]) - (y - left_vertex[1]) * (top_vertex[0] - left_vertex[0]) >= 0
                b2 = (x - right_vertex[0]) * (left_vertex[1] - right_vertex[1]) - (y - right_vertex[1]) * (left_vertex[0] - right_vertex[0]) >= 0
                b3 = (x - top_vertex[0]) * (right_vertex[1] - top_vertex[1]) - (y - top_vertex[1]) * (right_vertex[0] - top_vertex[0]) >= 0
                if b1 == b2 == b3:  # Point is inside the triangle
                    canvas_matrix[y, x] = triangle_color

        image = Image.fromarray(canvas_matrix)
        image.save(image_path)

        # Save parameters to class
        self.background_color = background_color
        self.triangle_color = triangle_color
        self.image_size = image_size
        self.triangle_size = triangle_size
        
def test():
    generator = TrainingDataGenerator()
    background_color = (255, 255, 255)

    # Draw square
    square_color = (0, 0, 0)
    image_size = (32, 32)
    square_size = 10
    square_path = "square.png"

    # Draw triangle
    triangle_color = (0, 0, 0)
    triangle_size = (20, 20)
    triangle_path = "triangle.png"

    generator.draw_square(image_size, square_size, square_color, background_color, square_path)
    generator.draw_triangle(image_size, triangle_size, triangle_color, background_color, triangle_path)

if __name__ == "__main__":
    test()