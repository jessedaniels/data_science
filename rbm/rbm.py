

import numpy as np
import numexpr as ne
import scipy.io

class mini_batch:
    def __init__(self):
        self.inputs=None
        self.targets=None

    def extract_mini_batch(data_set, start_i, n_cases):
        batch = mini_batch()
        batch.inputs = data_set.inputs[:, start_i : start_i + n_cases]
        # Add a row of 1s to account for bias unit
        batch.inputs = np.insert(batch.inputs, 0, 1, axis=0)
        batch.targets = data_set.targets[:, start_i : start_i + n_cases]
        return batch

class RBM:
    def __init__(self, num_visible, num_hidden, batch_size=6, learning_rate=0.1, l2=0.006, momentum=0.5):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2 = l2
        self.momentum = momentum

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.1 * np.random.randn(self.num_hidden+1, self.num_visible+1)    

    def class_init(self):       
        # Returns a nice object to access the different parts of the data set.
        data_sets = scipy.io.loadmat('/home/jdaniels/pydev/src/data_science/rbm/a4/data_set.mat', struct_as_record=False, squeeze_me=True)['data']
        randomness_source = scipy.io.loadmat('/home/jdaniels/pydev/src/data_science/rbm/a4/a4_randomness_source.mat', struct_as_record=False, squeeze_me=True)['randomness_source']
        report_calls_to_sample_bernoulli = True

    # Note: Matlab's reshape uses Column-Major form (Fortran). Default for Numpy is C ordering (row)
    def a4_rand(self, requested_size, seed):
        global randomness_source
        start_i = np.mod(round(seed), round(randomness_source.shape[0]) / 10.0)
        if start_i + np.prod(requested_size) >= randomness_source.shape[0]: 
            print('a4_rand failed to generate an array of that size (too big)')
        return np.reshape(randomness_source[start_i : start_i + np.prod(requested_size)], requested_size, order='F')

    def sample_bernoulli(self, probabilities):
        global report_calls_to_sample_bernoulli
        if report_calls_to_sample_bernoulli: 
            print 'sample_bernoulli() was called with a matrix of shape', probabilities.shape
        seed = np.sum(probabilities)
        return (probabilities > a4_rand(probabilities.shape, seed)).astype(int)

    def initialize_data(self):
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


    # Computes vectorized sigmoid
    def sigmoid(self, x):
        return ne.evaluate("1 / (1 + exp(-x))")

    def visible_to_hidden(self, visible_state, binary=True):
        #assert visible_state.shape == (self.num_visible + 1, self.batch_size) 
        probs = self.sigmoid(np.dot(self.weights, visible_state))
        probs = probs > np.random.random(probs.shape) if binary else probs
        probs[0,:] = 1 # fix bias row. Assume it already has the right shape 
        return probs

    def hidden_to_visible(self, hidden_state, binary=False):
        #assert hidden_state.shape == (self.num_hidden + 1, self.batch_size)
        probs = self.sigmoid(np.dot(self.weights.T, hidden_state))
        probs = probs > np.random.random(probs.shape) if binary else probs
        probs[0,:] = 1 # fix bias row
        return probs

    def configuration_goodness(self, visible_state, hidden_state):
        # return np.mean(np.sum((np.dot(hid.transpose(), rbm_w) * vis.transpose()), axis=1))
        # This method is slightly faster than the one above, even though it has to do more unnecessary computations.
        return np.mean(np.diag(np.dot(np.dot(hidden_state.T, self.weights), visible_state)))

    def configuration_goodness_gradient(self, visible_state, hidden_state):
        # This is the average of the product of visible_state and hidden_state across all
        # cases in the minibatch. The dot product gets the sum of each pair and then we
        # divide by the number of cases in the minibatch (second component of the state shape)
        return np.dot(hidden_state, visible_state.T) / float(visible_state.shape[1])

    def k_gibbs_sample(self, vis, k=1):
        hid_0 = None
        for i in range(k):
            #hid = sample_bernoulli(visible_to_hidden(rbm_w, vis, binary=False))
            #vis = sample_bernoulli(hidden_to_visible(rbm_w, hid, binary=False))
            hid = self.visible_to_hidden(vis, binary=True)
            vis = self.hidden_to_visible(hid, binary=False)
            if i == 0:
                hid_0 = hid
        return hid_0, vis

    def cd_k(self, vis_0, k=1):
        hid_0, vis_k = self.k_gibbs_sample(vis_0, k=k)
        hid_k = self.visible_to_hidden(vis_k, binary=False)

        pos_stats = self.configuration_goodness_gradient(vis_0, hid_0)
        neg_stats = self.configuration_goodness_gradient(vis_k, hid_k)
        grad = pos_stats - neg_stats

        return grad, vis_k

        
    def describe_matrix(mat):
        print 'Mean: %s, Sum: %s' %(np.mean(mat), np.sum(mat))

    def train(self, data, max_epochs=1000):
        # reg_w stores 
        reg_w = np.zeros(self.weights.shape)
        prev_grad = np.zeros(self.weights.shape)
        for epoch in range(max_epochs):
            grad, vis_k = self.cd_k(data, k=3)
            reg_w[1:,1:] = self.weights[1:,1:]
            self.weights += self.momentum * prev_grad + self.learning_rate * (grad - self.l2 * reg_w)
            prev_grad = grad
            error = np.sum((data - vis_k) ** 2)
            print("Epoch %s: error is %s" % (epoch, error))    

if __name__ == '__main__':
    movies = ['bias', 'harry potter', 'avatar', 'lotr', 'gladiator', 'titanic', 'glitter']
    r = RBM(num_visible = 6, num_hidden = 2)
    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,1],[0,0,1,1,1,0],[1,0,0,0,0,1]]).T
    training_data = np.insert(training_data, 0, 1, axis=0)
    r.train(training_data, max_epochs = 5000)
    print r.weights
    user = np.array([[1,0,0,0,1,1,0]]).T
    print zip(r.hidden_to_visible(r.visible_to_hidden(user), binary=False).flatten(), movies)



#TODO:
# Finish train_rbm - look at a4_main. ...have it read in mini batches and actually do SGD - learning rates, etc. At the end we should have a trained RBM with weights.
# Do assignment part 3 - use weights from above to seed a NN.
# Add in momentum


