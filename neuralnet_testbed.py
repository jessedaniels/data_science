
import numpy as np
import math


data_dim = 7
hidden_nodes = 5
output_nodes = 1

W1 = np.random.rand(hidden_nodes, data_dim + 1)/50.0
W2 = np.random.rand(output_nodes, hidden_nodes + 1)/50.0

# Activations in first and output layers
a1 = None
a2 = None

# Error values for each layer
delta_1 = None
delta_2 = None

def sigmoid(x):
    return 1/(1+np.exp(-x))

def do_pass(x, y):
    # Add bias and make examples columns; ith training example is x[:,i]
    x = np.hstack([np.ones((x.shape[0], 1)), x]).T

    # Activations in hidden and output layers
    a1 = sigmoid(W1.dot(x))
    a2 = sigmoid(W2.dot(a1))

    # Backpropagate errors
    delta_2 = -(y - a2) * a2*(1 - a2)
    delta_1 = w21.T.dot(d2) * a1*(1-a1)
    
    

