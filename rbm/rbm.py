

import numpy as np
import numexpr as ne
import scipy.io

num_hid = 100
num_vis = 256
mini_batch_size = 10


# Initialize weight matrix, num_hid by num_vis
# TODO: biases
rbm_w = np.random.uniform(low=-1, high=1, size=(num_hid, num_vis))

# Returns a nice object to access the different parts of the data set.
data_sets = scipy.io.loadmat('/home/jdaniels/pydev/src/data_science/rbm/a4/data_set.mat', struct_as_record=False, squeeze_me=True)['data']
randomness_source = scipy.io.loadmat('/home/jdaniels/pydev/src/data_science/rbm/a4/a4_randomness_source.mat', struct_as_record=False, squeeze_me=True)['randomness_source']
report_calls_to_sample_bernoulli = True

# Note: Matlab's reshape uses Column-Major form (Fortran). Default for Numpy is C ordering (row)
def a4_rand(requested_size, seed):
    global randomness_source
    start_i = np.mod(round(seed), round(randomness_source.shape[0]) / 10.0)
    if start_i + np.prod(requested_size) >= randomness_source.shape[0]: 
        print('a4_rand failed to generate an array of that size (too big)')
    return np.reshape(randomness_source[start_i : start_i + np.prod(requested_size)], requested_size, order='F')

def sample_bernoulli(probabilities):
    global report_calls_to_sample_bernoulli
    if report_calls_to_sample_bernoulli: 
        print 'sample_bernoulli() was called with a matrix of shape', probabilities.shape
    seed = np.sum(probabilities)
    return (probabilities > a4_rand(probabilities.shape, seed)).astype(int)

def initialize_data():
    test_rbm_w = a4_rand((100, 256), 0) * 2 - 1;
    small_test_rbm_w = a4_rand((10, 256), 0) * 2 - 1;

    temp = extract_mini_batch(data_sets.training, 0, 1)
    data_1_case = sample_bernoulli(temp.inputs)
    temp = extract_mini_batch(data_sets.training, 99, 10)
    data_10_cases = sample_bernoulli(temp.inputs)
    temp = extract_mini_batch(data_sets.training, 199, 37)
    data_37_cases = sample_bernoulli(temp.inputs)

    test_hidden_state_1_case = sample_bernoulli(a4_rand((100, 1), 0))
    test_hidden_state_10_cases = sample_bernoulli(a4_rand((100, 10), 1))
    test_hidden_state_37_cases = sample_bernoulli(a4_rand((100, 37), 2))

    report_calls_to_sample_bernoulli = True

    del temp

class mini_batch:
    def __init__(self):
        self.inputs=None
        self.targets=None

def extract_mini_batch(data_set, start_i, n_cases):
    batch = mini_batch()
    batch.inputs = data_set.inputs[:, start_i : start_i + n_cases]
    batch.targets = data_set.targets[:, start_i : start_i + n_cases]
    return batch

# Computes vectorized sigmoid
def sigmoid(matr):
    return ne.evaluate("1 / (1 + exp(-matr))")

def visible_to_hidden(rbm_w, visible_state, binary=True):
    # Needs to return a num_hid by num_examples matrix of hidden activation probabilities
    # num_hid x num_vis * num_vis x num_examples = num_hid x num_examples
    probs = sigmoid(np.dot(rbm_w, visible_state))
    return probs > np.random.random(probs.shape) if binary else probs

def hidden_to_visible(rbm_w, hidden_state, binary=False):
    # Need to return a num_vis x num_examples matrix
    # n_vis x n_hid * n_hid x examples
    probs = sigmoid(np.dot(rbm_w.transpose(), hidden_state))
    return probs > np.random.random(probs.shape) if binary else probs

def configuration_goodness(rbm_w, visible_state, hidden_state):
    # return np.mean(np.sum((np.dot(hid.transpose(), rbm_w) * vis.transpose()), axis=1))
    # This method is slightly faster than the one above, even though it has to do more unnecessary computations.
    return np.mean(np.diag(np.dot(np.dot(hidden_state.transpose(), rbm_w), visible_state)))

def configuration_goodness_gradient(visible_state, hidden_state):
    # This is the average of the product of visible_state and hidden_state across all
    # cases in the minibatch. The dot product gets the sum of each pair and then we
    # divide by the number of cases in the minibatch (second component of the state shape)
    return np.dot(hidden_state, visible_state.transpose()) / float(visible_state.shape[1])

def k_gibbs_sample(rbm_w, vis, k=1):
    for i in range(1,k):
        hid = sample_bernoulli(visible_to_hidden(rbm_w, vis, binary=False))
        vis = sample_bernoulli(hidden_to_visible(rbm_w, hid, binary=False))
    return vis, hid

def cd_k(rbm_w, vis, k=1):

    hid_0 = sample_bernoulli(visible_to_hidden(rbm_w, vis, binary=False))
    vis_1 = sample_bernoulli(hidden_to_visible(rbm_w, hid_0, binary=False))
    hid_1 = sample_bernoulli(visible_to_hidden(rbm_w, vis_1, binary=False))

    pos_stats = configuration_goodness_gradient(vis, hid_0)
    neg_stats = configuration_goodness_gradient(vis_1, hid_1)

    return pos_stats - neg_stats

    
def describe_matrix(mat):
    print '%s, %s' %(np.mean(mat), np.sum(mat))

