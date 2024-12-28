'''
Test neural networks module with image classification task with non-noisy data.
'''

import numpy as np
import json
import os
from typing import Dict
from PIL import Image

from neural_network_module import NeuralNetwork

PROJECT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_BASE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "training_data")

def load_classification_data(
        filename: str,
        ) -> Dict[str, Dict[int, np.ndarray]]:
    
    '''
    Load image data from a diffusion training data json file.
    Returns a dictionary such that there is a key for each shape,
    and each shape key maps to a dictionary that contains a key
    for each image, and each image key maps to a numpy array
    '''
    
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, 'r') as f:
        data = json.load(f)

    non_noisy_data_dict = {'squares': {}, 'triangles': {}}
    for shape in data.keys():
        shape_data = data[shape]

        for i in range(len(shape_data.keys())):
            shape_data_key = list(shape_data.keys())[i]
            step_data = shape_data[shape_data_key]
            
            non_noisy_data = step_data["0"]
            non_noisy_matrix = np.array(non_noisy_data)
            image_array = non_noisy_matrix.astype(np.uint8)
            non_noisy_data_dict[shape][i] = image_array

    return non_noisy_data_dict

if __name__ == '__main__':
    data_filename = 'training_data.json'
    
    non_noisy_data_dict = load_classification_data(data_filename)

    print(non_noisy_data_dict['squares'][16].shape)
    image = Image.fromarray(non_noisy_data_dict['squares'][16])
    image.show()
