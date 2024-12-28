'''
Scripts that test training_data_generator.py
'''

from PIL import Image
import numpy as np

from training_data_generator import TrainingDataGenerator

def test1(): # Test draw_square and draw_triangle functions
    generator = TrainingDataGenerator()
    background_color = (255, 255, 255)

    # Draw square
    square_color = (0, 0, 0)
    image_size = (32, 32)
    square_size = 1
    square_path = "square.png"

    # Draw triangle
    triangle_color = (0, 0, 0)
    triangle_size = (6, 6)
    triangle_path = "triangle.png"

    square_matrix = generator.draw_square(image_size, square_size, square_color, background_color, square_path, inspect=False)
    triangle_matrix = generator.draw_triangle(image_size, triangle_size, triangle_color, background_color, triangle_path, inspect=True)

    print(f"Square matrix: {square_matrix}")
    print(f"Triangle matrix: {triangle_matrix}")

def test2(): # Test beta_schedule_forward_diffusion function
    generator = TrainingDataGenerator()
    background_color = (255, 255, 255)

    # Draw square
    square_color = (0, 0, 0)
    image_size = (32, 32)
    square_size = 10
    square_path = "square.png"

    square_matrix = generator.draw_square(image_size, square_size, square_color, background_color, square_path, inspect=False)

    # Add diffusion process
    T = 500
    beta_schedule = np.linspace(0.0001, 0.02, T) # Linear schedule from 1e-4 to 0.2
    x_t = generator.beta_schedule_forward_diffusion(square_matrix, T, beta_schedule)[T - 1]
    x_t_array = np.array(x_t, dtype=np.uint8)
    image = Image.fromarray(x_t_array)
    image.save("diffused_square.png")

    # Draw triangle
    triangle_color = (0, 0, 0)
    triangle_size = (25, 25)
    triangle_path = "triangle.png"

    triangle_matrix = generator.draw_triangle(image_size, triangle_size, triangle_color, background_color, triangle_path, inspect=False)

    # Add diffusion process
    T = 500
    beta_schedule = np.linspace(0.0001, 0.02, T) # Linear schedule from 1e-4 to 0.2
    x_t = generator.beta_schedule_forward_diffusion(triangle_matrix, T, beta_schedule)[T - 1]
    x_t_array = np.array(x_t, dtype=np.uint8)
    image = Image.fromarray(x_t_array)
    image.save("diffused_triangle.png")

def test3(): # Test generate_training_data function with beta schedule
    generator = TrainingDataGenerator()
    num_images = 1000
    save_path = "training_data.json"

    generator.generate_training_data(num_images, save_path, forward_diffusion_method="beta_schedule")

def test4(): # Test discretized_time_continuous_forward_diffusion function
    generator = TrainingDataGenerator()
    background_color = (255, 255, 255)

    # Draw square
    square_color = (0, 0, 0)
    image_size = (32, 32)
    square_size = 10
    square_path = "square.png"

    square_matrix = generator.draw_square(image_size, square_size, square_color, background_color, square_path, inspect=False)

    # Add diffusion process
    T = 30
    x_t = generator.discretized_time_continuous_forward_diffusion(square_matrix, T)[T - 1]
    x_t_array = np.array(x_t, dtype=np.uint8)
    image = Image.fromarray(x_t_array)
    image.save("diffused_square.png")

    # Draw triangle
    triangle_color = (0, 0, 0)
    triangle_size = (30, 30)
    triangle_path = "triangle.png"

    triangle_matrix = generator.draw_triangle(image_size, triangle_size, triangle_color, background_color, triangle_path, inspect=False)

    # Add diffusion process
    T = 30
    x_t = generator.discretized_time_continuous_forward_diffusion(triangle_matrix, T)[T - 1]
    x_t_array = np.array(x_t, dtype=np.uint8)
    image = Image.fromarray(x_t_array)
    image.save("diffused_triangle.png")

if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    test4()