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
        self.pre_activations = {} # dictionary to store the pre-activation values for each layer. 
        self.activations = {} # dictionary to store the activation values for each layer.


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
    

    ### LOSS METHODS ###
    def bce_with_logits_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Computes Binary Cross-Entropy Loss with Logits.

        Parameters:
            logits (numpy.ndarray): Output predictions (logits) from the neural network before applying sigmoid.
            targets (numpy.ndarray): Ground truth binary labels (0 or 1).

        Returns:
            float: The binary cross-entropy loss.
        """
        loss = np.mean(
            np.maximum(0, logits) - logits * targets + np.log(1 + np.exp(-np.abs(logits))) # numerical stability trick
        )
        return loss
    

    ### DERIVATIVE METHODS ###
    def relu_derivative(self, 
                        x: np.ndarray
                        ) -> np.ndarray:
        """Derivative of ReLU."""
        return np.where(x > 0, 1, 0)

    def sigmoid_derivative(self, 
                           x: np.ndarray
                           ) -> np.ndarray:
        """Derivative of Sigmoid."""
        return x * (1 - x)

    def bce_with_logits_loss_derivative(self, 
                                        logits: np.ndarray, 
                                        targets: np.ndarray
                                        ) -> np.ndarray:
        """Derivative of Binary Cross-Entropy loss with logits."""
        return self.sigmoid(logits) - targets
            

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
                self.pre_activations[i] = data # store the pre-activations for the layer
                if activation == "relu":
                    data = self.relu(data)
                elif activation == "sigmoid":
                    data = self.sigmoid(data)
                elif activation == "None":
                    data = data
                else:
                    raise ValueError("Activation function not supported.")
                self.activations[i] = data # store the activations for the layer
            else:
                raise ValueError("Layer type not supported.")
            
        return data
    
    def backward_pass(self, 
                      input_data: np.ndarray, # input data to the network
                      target: np.ndarray, # target values for the output layer
                      learning_rate: float = 0.01, # learning rate for gradient descent,
                      loss_function: str = "bce_with_logits_loss" # loss function to use for training
                      ) -> None:
        
        '''
        Perform the backward pass and update weights and biases for each layer.
        This function uses the stored activations and pre-activations from the forward pass.
        '''

        batch_size = target.shape[0] # get the batch size

        # start with output layer
        if loss_function == "bce_with_logits_loss":
            dz = self.bce_with_logits_loss_derivative(self.pre_activations[len(self.pre_activations)-1], target.reshape(batch_size, 1)) # derivative of loss with respect to pre-activation of output layer

        # loop through the layers in reverse order
        for i in reversed(range(len(self.layers))):
            layers_dict = self.layers[i]
            activation = layers_dict["activation"]
        
            # Derivative of the activation function with respect to the pre-activation
            if activation == "relu":
                dz = dz * self.relu_derivative(self.pre_activations[i])
            elif activation == "sigmoid":
                dz = dz * self.sigmoid_derivative(self.pre_activations[i])
            
            # Compute the gradients for weights and biases
            dW = np.dot(dz.T, self.activations[i-1] if i > 0 else input_data) / batch_size  # Gradient for weights
            db = np.sum(dz, axis=0, keepdims=True) / batch_size  # Gradient for biases
            
            # Update the weights and biases using gradient descent
            self.layer_weights[i] -= learning_rate * dW
            self.layer_biases[i] -= learning_rate * db
            
            # Calculate the error (dz) for the previous layer
            dz = np.dot(dz, self.layer_weights[i])  # Backpropagate the error
            
    def add_dense_layer(self,
                        num_inputs: int, # length of input vector to the layer. inputs must be flattened before being passed to the layer in forward propagation.
                        num_neurons: int, # number of neurons in the layer, also the number of outputs
                        activation: str = "None", # activation function for the layer
                        ) -> None:
        '''
        Add a dense layer to the neural network.
        '''

        self.layers[len(self.layers)] = {"type": "dense","num_inputs": num_inputs, "num_outputs": num_neurons, "activation": activation}
        self.layer_weights[len(self.layer_weights)] = np.random.randn(num_neurons, num_inputs) # initialize the weights for the layer to shape (num_neurons, num_inputs)
        self.layer_biases[len(self.layer_biases)] = np.zeros((1, num_neurons)) # initialize the biases for the layer to shape (1, num_neurons)