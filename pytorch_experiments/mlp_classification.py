'''
Experiment simply using a multi-layer perceptron to classify images in the diffusion dataset.
'''

import numpy as np
import json
import os
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

PROJECT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_BASE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "training_data")

# Load the non-noisy image data from the diffusion dataset
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

# MLP model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, num_classes) # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image into a 1D vector
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == '__main__':
    
    data_filename = 'training_data.json'

    # Load the non-noisy image data from the diffusion dataset
    array_list, label_list = load_classification_data(data_filename)

    # Convert the labels to integers
    label_list = [0 if label == 'squares' else 1 for label in label_list] # Basically, squares are 0 and triangles are 1
    label_array = np.array(label_list)

    x_train, x_test, y_train, y_test = train_test_split(array_list, label_array, test_size=0.2, random_state=42)

    # Flatten image data and combine into single numpy arrays
    x_train = np.stack(x_train).reshape(len(x_train), -1)
    x_test = np.stack(x_test).reshape(len(x_test), -1)

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # Convert labels to PyTorch tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create custom dataset object