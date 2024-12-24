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
        
        # Still need to figure out square coordinates and replace background_matrix values with square
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
        
def test():
    generator = TrainingDataGenerator()
    background_color = (255, 255, 255)
    square_color = (0, 0, 0)
    image_size = (32, 32)
    square_size = 10
    image_path = "output.png"

    generator.draw_square(image_size, square_size, square_color, background_color, image_path)

if __name__ == "__main__":
    test()