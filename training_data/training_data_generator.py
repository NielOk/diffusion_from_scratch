'''
This file contains a class that generates training data for the diffusion model. 
'''

from PIL import Image
import numpy as np
from typing import Tuple, Dict
import json

class TrainingDataGenerator:

    def __init__(self) -> None:
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
                    image_path: str = "",
                    inspect: bool = False
                    ) -> np.ndarray:
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

        if inspect and image_path != "":
            image = Image.fromarray(canvas_matrix)
            image.save(image_path)

        # Save parameters to class
        self.background_color = background_color
        self.square_color = square_color
        self.image_size = image_size
        self.square_size = square_size

        return canvas_matrix

    def draw_triangle(self, 
                    image_size: Tuple[int, int],
                    triangle_size: Tuple[int, int],
                    triangle_color: Tuple[int, int, int],
                    background_color: Tuple[int, int, int], # (255, 255, 255) is white, (0, 0, 0) is black
                    image_path: str = "",
                    inspect: bool = False
                    ) -> np.ndarray:
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

        if inspect and image_path != "":
            image = Image.fromarray(canvas_matrix)
            image.save(image_path)

        # Save parameters to class
        self.background_color = background_color
        self.triangle_color = triangle_color
        self.image_size = image_size
        self.triangle_size = triangle_size

        return canvas_matrix
    
    def beta_schedule_forward_diffusion(self,
                        x_0: np.ndarray,  # Initial state
                        T: int,  # Number of time steps
                        beta_schedule: np.ndarray,  # Schedule of beta values
                        ) -> Dict[int, list]: # Returns a dictionary of the noisy image at each time step, with the keys being the time steps from 0 (initial state) to T - 1
        '''
        Forward diffusion process for 8-bit RGB values
        according to beta-schedule.
        '''
        # Ensure x_0 is a float type for precision during calculations.
        x_t = (x_0.astype(np.float32) / 127.5) - 1  # Scale to [-1, 1]

        noisy_steps = {}
        
        for t in range(T):
            beta_t = beta_schedule[t]
            noise = np.random.normal(0, 1, size=x_t.shape)  # Generate noise with normal distribution
            
            # Add noise according to the diffusion process
            x_t = np.sqrt(1 - beta_t) * x_t + np.sqrt(beta_t) * noise
            
            # Clip the values to stay within [0, 255] (for 8-bit image values)
            x_t = np.clip(x_t, -1, 1)

            # Convert each step back to 8-bit RGB values
            noisy_step = np.clip((x_t + 1) * 127.5, 0, 255).astype(np.uint8)

            noisy_step_list_data = noisy_step.tolist()

            noisy_steps[t] = noisy_step_list_data # Convert array to list for json saving. Will be converted back to numpy array when doing computations. 

        return noisy_steps
    
    def discretized_time_continuous_forward_diffusion(self,
                                                      x_0: np.ndarray, # Initial state,
                                                      T: int, # Number of time steps
                                                      ) -> Dict[int, list]: # Returns a dictionary of the noisy image at each time step, with the keys being the time steps from 0 (initial state) to T - 1
        '''
        Forward diffusion process according to the 
        discretized-time continuous diffusion process.
        '''
        # Ensure x_0 is a float type for precision during calculations.
        x_t = (x_0.astype(np.float32) / 127.5) - 1  # Scale to [-1, 1]

        noisy_steps = {}

        for t in range(T):
            noise = np.random.normal(0, (t + 1) / T, size=x_t.shape) # Generate noise with normal distribution

            # Add noise according to the diffusion process
            x_t = x_t + noise

            # Clip the values to stay within [0, 255] (for 8-bit image values)
            x_t = np.clip(x_t, -1, 1)

            # Convert each step back to 8-bit RGB values
            noisy_step = np.clip((x_t + 1) * 127.5, 0, 255).astype(np.uint8)

            noisy_step_list_data = noisy_step.tolist()

            noisy_steps[t] = noisy_step_list_data # Convert array to list for json saving. Will be converted back to numpy array when doing computations. 

        return noisy_steps

    def generate_training_data(self, 
                            num_images: int,
                            save_path: str,
                            T: int = 25, # Number of time steps
                            squares: bool = True,
                            triangles: bool = True,
                            background_color: Tuple[int, int, int] = (255, 255, 255), # Default background color for all images is white
                            image_size: Tuple[int, int] = (32, 32), # Default image size is 32x32
                            forward_diffusion_method: str="beta_schedule", # other option is "discretized_time_continuous"
                            inspect: bool = False
                            ):
        '''
        Generates training data for the diffusion model. Image size is 
        32x32. Square size, color and triangle size, color is different 
        for every image. Matrix values are saved to a json file. 
        '''

        num_squares = 0
        num_triangles = 0

        if squares and triangles:
            num_squares = num_images // 2
            num_triangles = num_images - num_squares
        elif squares:
            num_squares = num_images
        elif triangles:
            num_triangles = num_images
        
        training_data_dict = { # Dictionary to store training data. Major keys are "squares" and "triangles". will try to use dictionaries because they are more efficient than lists.
            "squares": {},
            "triangles": {}
        }
        # Get square possible dimensions
        square_max_length = np.minimum(image_size[0], image_size[1])

        # Generate squares training data
        for i in range(num_squares):
            square_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            square_size = np.random.randint(1, square_max_length) # Give a buffer of 1 pixel for both miniumum and maximum size

            square_matrix = self.draw_square(image_size, square_size, square_color, background_color)

            if inspect and i % 100 == 0:
                image = Image.fromarray(square_matrix)
                image.save(f"square_{i}.png")

            if forward_diffusion_method == "beta_schedule":
                noisy_steps = self.beta_schedule_forward_diffusion(square_matrix, T=T, beta_schedule=np.linspace(0.0001, 0.02, T))
                training_data_dict["squares"][f"square_{i}_steps"] = noisy_steps
            elif forward_diffusion_method == "discretized_time_continuous":
                noisy_steps = self.discretized_time_continuous_forward_diffusion(square_matrix, T=T)
                training_data_dict["squares"][f"square_{i}_steps"] = noisy_steps
            else:
                raise ValueError("Invalid forward diffusion method")
            
        # Get triangle possible dimensions
        triangle_min_base_or_height = 6
        triangle_max_base = np.minimum(image_size[0], image_size[1])

        # Generate triangles training data
        for i in range(num_triangles):
            triangle_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            triangle_size = (np.random.randint(triangle_min_base_or_height, triangle_max_base), np.random.randint(triangle_min_base_or_height, triangle_max_base))

            triangle_matrix = self.draw_triangle(image_size, triangle_size, triangle_color, background_color)

            if inspect and i % 100 == 0:
                image = Image.fromarray(triangle_matrix)
                image.save(f"triangle_{i}.png")

            if forward_diffusion_method == "beta_schedule":
                noisy_steps = self.beta_schedule_forward_diffusion(triangle_matrix, T=T, beta_schedule=np.linspace(0.0001, 0.02, T))
                training_data_dict["triangles"][f"triangle_{i}_steps"] = noisy_steps
            elif forward_diffusion_method == "discretized_time_continuous":
                noisy_steps = self.discretized_time_continuous_forward_diffusion(triangle_matrix, T=T)
                training_data_dict["triangles"][f"triangle_{i}_steps"] = noisy_steps
            else:
                raise ValueError("Invalid forward diffusion method")
            
        with open(save_path, "w") as f:
            json.dump(training_data_dict, f, indent=4)

        print(f"Generated {num_squares} squares and {num_triangles} triangles.")