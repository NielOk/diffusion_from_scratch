'''
Test neural networks module with image classification task with non-noisy data.
'''

import numpy as np
import json
import os
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
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
    Returns a list of image arrays and a list of labels.
    '''
    
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, 'r') as f:
        data = json.load(f)

    array_list = []
    label_list = []

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

            array_list.append(image_array)
            label_list.append(shape)

    return array_list, label_list

def prepare_data(
        data_filename: str, 
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    '''
    Prepare image data for training.
    '''

    array_list, label_list = load_classification_data(data_filename)

    # Convert the labels to integers and create a label array
    label_list = [0.0 if label == 'squares' else 1.0 for label in label_list] # Basically, squares are 0 and triangles are 1
    label_array = np.array(label_list)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(array_list, label_array, test_size=0.2, random_state=42)

    # Flatten image data and combine into single numpy arrays
    x_train = np.stack(x_train).reshape(len(x_train), -1) / 255 # Normalize to [0, 1]
    x_test = np.stack(x_test).reshape(len(x_test), -1) / 255 # Normalize to [0, 1]

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    data_filename = 'training_data.json'
    
    x_train, x_test, y_train, y_test = prepare_data(data_filename)