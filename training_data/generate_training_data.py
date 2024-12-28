'''
Script that uses training_data_generator to generate training data for the model and check images. Must be edited manually
to change parameters for training data generation. 
'''

import json
import numpy as np
from PIL import Image

from training_data_generator import TrainingDataGenerator

def view_images(data):
    # Check square 0
    square_0_steps = data["squares"]["square_0_steps"]
    for step_id in square_0_steps:
        square_data = square_0_steps[step_id]
        square_matrix = np.array(square_data).astype(np.uint8)
        image = Image.fromarray(square_matrix)
        image.show()

    # Check triangle 1
    triangle_0_steps = data["triangles"]["triangle_0_steps"]
    for step_id in triangle_0_steps:
        triangle_data = triangle_0_steps[step_id]
        triangle_matrix = np.array(triangle_data).astype(np.uint8)
        image = Image.fromarray(triangle_matrix)
        image.show()

if __name__ == '__main__':
    generator = TrainingDataGenerator()
    num_images = 1000
    save_path = "training_data.json"

    generator.generate_training_data(num_images, save_path, forward_diffusion_method="discretized_time_continuous")

    with open(save_path, 'r') as f:
        data = json.load(f)

    view_images(data)