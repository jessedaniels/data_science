
import numpy as np
import math

hidden_nodes = 50
output_nodes = 10
lam = 0.001 # regulation parameter
alpha = 0.01 # learning rate
W1 = None
W2 = None

def initialize(x):
    global hidden_nodes, output_nodes, W1, W2
    data_dim = x.shape[1]

    W1 = np.random.rand(hidden_nodes, data_dim + 1)/20.0
    W2 = np.random.rand(output_nodes, hidden_nodes + 1)/20.0

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Accepts a minibatch of (x,y) vectors to compute gradients and update weights.
def do_pass(x, y):
    global W1, W2 
    
    # Add ones column to x for bias
    x_w_b = np.hstack([np.ones((x.shape[0], 1)), x])
    # Forward pass
    z2 = W1.dot(x_w_b.T)
    a2 = sigmoid(z2)
    
    # Add ones column to a2 for bias
    a2_w_b = np.vstack([np.ones((1, a2.shape[1])), a2])
    z3 = W2.dot(a2_w_b)
    a3 = sigmoid(z3)

    # Backpropagate errors
    delta3 = -(y - a3) * a3*(1 - a3)
    # Derivative wrt activations removes bias from W2 at this point, so have to slice it out
    delta2 = W2[:,1:].T.dot(delta3) * a2*(1 - a2)

    # Derivatives wrt weights adds a row of 1s for the biases since it is a constant weight not multiplied by anything
    # Dimensions of grad3 and grad2 should equal W2 and W1, respectively
    grad2 = 1.0 / x.shape[0] * delta3.dot(a2_w_b.T)
    grad1 = 1.0 / x.shape[0] * delta2.dot(x_w_b) # x is already the right dimension - no need to transpose

    # Add in regularization derivatives; no regularization on biases
    grad2 += np.hstack([np.zeros([W2.shape[0], 1]), W2[:,1:] * lam])
    grad1 += np.hstack([np.zeros([W1.shape[0], 1]), W1[:,1:] * lam])
    
    # Update the weight arrays
    W2 -= alpha * grad2
    W1 -= alpha * grad1
    
    print np.mean(0.5 * (y - a3) ** 2)

def main(x, y):

    # Break up x into minibatches


# TODO adaptive learning rate?

def gradient_checking(x, y, W1, W2):
    # Note for gradient checking - make sure the regularization derivatives are incorporated into the gradient computed above before comparing.
    epsilon = 0.0001
    t = np.zeros_like(W1)
    t[4,10] = epsilon
    grad_estimate = (cost_function(x, y, W1+t, W2) - cost_function(x, y, W1-t, W2)) / (2 * epsilon)     


def cost_function(x, y, W1, W2):
    z2 = W1.dot(np.hstack([np.ones((x.shape[0], 1)), x]).T)  
    a2 = sigmoid(z2)
         
    z3 = W2.dot(np.vstack([np.ones((1, a2.shape[1])), a2]))
    a3 = sigmoid(z3)

    # Combine weight vectors (without biases) into one long vector for easy sum of squares 
    rav = np.hstack((W1[:,1:].ravel(), W2[:,1:].ravel()))
    
    return np.mean(0.5 * (y - a3) ** 2) + 0.5 * lam * rav.dot(rav.T)
    
