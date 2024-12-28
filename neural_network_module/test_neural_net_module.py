'''
Test neural networks module.
'''

import numpy as np
from neural_network_module import NeuralNetwork

def test1(): # Check dense layer definition and initialization
    net = NeuralNetwork()
    net.add_dense_layer(32*32, 3, 32, "relu")
    net.add_dense_layer(32, 1, 32, "relu")

    for i in range(len(net.layers)):
        print(net.layers[i])
        print(net.layer_weights[i].shape)
        print(net.layer_weights[i])
        print(net.layer_biases[i].shape)
        print(net.layer_biases[i])

if __name__ == '__main__':
    test1()