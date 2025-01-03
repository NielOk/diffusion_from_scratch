'''
Module for flexibly defining and training neural networks.
'''

import numpy as np
from typing import Dict

class NeuralNetwork:

    def __init__(self) -> None: 
        self.layers = {} # dictionary to store information about the layers of the neural network, including activations, number of neurons, number of inputs, etc.
        self.layer_weights = {} # dictionary to store the tensor weights for each layer. weights are stacked as rows for each layer.
        self.layer_biases = {} # dictionary to store the vector biases for each layer


    ### DATA PREPARATION METHODS ###
    def create_batches(self, 
                       data: np.ndarray, # data to be batched, is a singular numpy array where each row is a data item
                       batch_size: int # size of each batch
                       ) -> np.ndarray:
        '''
        Create batches from a data array. This is to be used for both features and labels (the batch size should be the same for both for separate datasets).
        '''
        num_samples = data.shape[0]
        # Compute the number of batches
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Use slicing to create batches
        batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        return batches
    
    ### ACTIVATION METHODS ##
    def relu(self,
            array: np.ndarray
            ) -> np.ndarray:
        """
        Applies the ReLU activation function element-wise to the input array.

        Parameters:
            array (numpy.ndarray): Input NumPy array.

        Returns:
            numpy.ndarray: The result of applying ReLU activation.
        """
        return np.maximum(0, array)
    
    def sigmoid(self,
                array: np.ndarray
                ) -> np.ndarray:
        """
        Applies the sigmoid activation function element-wise to the input array.

        Parameters:
            array (numpy.ndarray): Input NumPy array.

        Returns:
            numpy.ndarray: The result of applying sigmoid activation.
        """
        return 1 / (1 + np.exp(-array))
            
    ### NEURAL NETWORK METHODS ###

    def forward_pass(self, 
                     data: np.ndarray # data to be passed through the network
                     ) -> np.ndarray:
        '''
        Forward pass of the neural network.
        '''

        # loop through the layers of the network
        for i in range(len(self.layers)):
            layer_dict = self.layers[i]
            weights = self.layer_weights[i]

            # Check the type of layer
            if layer_dict["type"] == "dense": # dense layer case
                biases = self.layer_biases[i]
                data = np.dot(data, weights.T) + biases
                activation = layer_dict["activation"]
                if activation == "relu":
                    data = self.relu(data)
                elif activation == "sigmoid":
                    data = self.sigmoid(data)
                else:
                    raise ValueError("Activation function not supported.")
            else:
                raise ValueError("Layer type not supported.")
            
        return data
        
    def add_dense_layer(self,
                        num_inputs: int, # length of input vector to the layer. inputs must be flattened before being passed to the layer in forward propagation.
                        num_neurons: int, # number of neurons in the layer, also the number of outputs
                        activation: str = "relu", # activation function for the layer
                        ) -> None:
        '''
        Add a dense layer to the neural network.
        '''

        self.layers[len(self.layers)] = {"type": "dense","num_inputs": num_inputs, "num_outputs": num_neurons, "activation": activation}
        self.layer_weights[len(self.layer_weights)] = np.random.randn(num_neurons, num_inputs) # initialize the weights for the layer
        self.layer_biases[len(self.layer_biases)] = np.zeros(num_neurons)