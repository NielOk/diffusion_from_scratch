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

def test2(): # Check batching
    net = NeuralNetwork()
    data = np.random.randn(100, 32*32*3)
    labels = np.random.randn(100, 1)
    batch_size = 10
    data_batches = net.create_batches(data, 10)
    label_batches = net.create_batches(labels, 10)

    print(len(data_batches))
    print(len(label_batches))

    for i in range(len(data_batches)):
        print(data_batches[i].shape)
        print(label_batches[i].shape)
    
if __name__ == '__main__':
    #test1()
    test2()