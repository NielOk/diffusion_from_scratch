'''
Experiment simply using a multi-layer perceptron to classify images in the diffusion dataset.
'''

import numpy as np
import json
import os
from typing import Dict, Tuple
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

# Custom dataset class for image data
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Define the MLP Model for Binary Classification
class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size):
        super(MLPBinaryClassifier, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_1_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_2_size, 1)  # Output layer with 1 unit

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Pass through first layer + ReLU
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (logits for BCEWithLogitsLoss)
        return x
    
def prepare_data(
        data_filename: str,
        ) -> Tuple[ImageDataset, ImageDataset]:
    
    array_list, label_list = load_classification_data(data_filename)

    # Convert the labels to integers and create a label array
    label_list = [0.0 if label == 'squares' else 1.0 for label in label_list] # Basically, squares are 0 and triangles are 1
    label_array = np.array(label_list)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(array_list, label_array, test_size=0.2, random_state=42)

    # Flatten image data and combine into single numpy arrays
    x_train = np.stack(x_train).reshape(len(x_train), -1) / 255 # Normalize to [0, 1]
    x_test = np.stack(x_test).reshape(len(x_test), -1) / 255 # Normalize to [0, 1]

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # Convert labels to PyTorch tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create data loader for batching and shuffling.
    train_dataset = ImageDataset(x_train_tensor, y_train_tensor)
    test_dataset = ImageDataset(x_test_tensor, y_test_tensor)

    return train_dataset, test_dataset

# Main function for loading data, training the model, and evaluating on test data
def main():
    data_filename = 'training_data.json'

    train_dataset, test_dataset = prepare_data(data_filename)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # Batch size of 16, shuffle the data for training
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) # No need to shuffle the test data

    # Model, Loss, and Optimizer
    input_size = 32 * 32 * 3  # Flattened size of 32x32x3 image
    hidden_1_size = 64  # Arbitrary hidden layer size
    hidden_2_size = 32  # Arbitrary hidden layer size
    model = MLPBinaryClassifier(input_size, hidden_1_size, hidden_2_size)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid activation and binary cross-entropy
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

    # Training Loop
    epochs = 10  # Number of epochs
    for epoch in range(epochs): # tqdm for progress bar
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images).squeeze(1)  # Remove the extra dimension
            loss = criterion(outputs, labels)  # Compute the loss
            
            # Backpropagation
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate the error
            optimizer.step()  # Update the model's weights
            
            running_loss += loss.item()
            
            # Accuracy calculation
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to binary predictions
            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Correct predictions

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Evaluation on Test Data
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, labels in test_loader:
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
if __name__ == '__main__':
    main()