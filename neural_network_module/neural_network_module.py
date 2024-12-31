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
        
    def add_dense_layer(self,
                        num_inputs: int, # length of input vector to the layer. inputs must be flattened before being passed to the layer in forward propagation.
                        values_per_input: int, # number of values per item in the input vector
                        num_neurons: int, # number of neurons in the layer, also the number of outputs
                        activation: str = "relu", # activation function for the layer
                        ) -> None:
        '''
        Add a dense layer to the neural network.
        '''

        self.layers[len(self.layers)] = {"type": "dense","num_inputs": num_inputs * values_per_input, "num_outputs": num_neurons, "activation": activation}
        self.layer_weights[len(self.layer_weights)] = np.random.randn(num_neurons, num_inputs * values_per_input) # initialize the weights for the layer
        self.layer_biases[len(self.layer_biases)] = np.zeros(num_neurons)