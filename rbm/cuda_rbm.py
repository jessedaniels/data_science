import numpy as np
from datetime import datetime

class RBM:
    def __init__(self, dat, epsilon=0.1, lam=0.006, momentum=0.9, batch_size=7, num_hid=2, epochs=2000):
        self.dat = dat
        # training parameters
        self.epochs = epochs
        self.epsilon = epsilon # learning rate
        self.lam = lam # l2 regularization
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_batches = dat.shape[0]/batch_size
        # model parameters
        self.num_vis = dat.shape[1]
        self.num_hid = num_hid
        # initialize weights
        self.w_vh = 0.1 * np.random.randn(self.num_vis, self.num_hid)
        self.w_v = np.zeros(self.num_vis)
        self.w_h = -4. * np.ones(self.num_hid)
        # initialize weight updates
        self.wu_vh = np.zeros((self.num_vis, self.num_hid))
        self.wu_v = np.zeros(self.num_vis)
        self.wu_h = np.zeros(self.num_hid)

    def logistic(self, x):
        return 1/(1+np.exp(-x))

    def visible_to_hidden(self, visible_state, binary=True):
        #assert visible_state.shape == (self.num_visible + 1, self.batch_size) 
        probs = self.logistic(np.dot(visible_state, self.w_vh) + self.w_h)
        probs = probs > np.random.random(probs.shape) if binary else probs
        return probs

    def hidden_to_visible(self, hidden_state, binary=False):
        #assert hidden_state.shape == (self.num_hidden + 1, self.batch_size)
        probs = self.logistic(np.dot(hidden_state, self.w_vh.T) + self.w_v)
        probs = probs > np.random.random(probs.shape) if binary else probs
        return probs

    def k_gibbs_sample(self, vis, k=1):
        for i in range(k):
            # For hiddens, sample the binary state
            hid = self.visible_to_hidden(vis, binary=True)
            vis = self.hidden_to_visible(hid, binary=False)
        return vis

    def train(self):
        #import gnumpy as np
        # load data. <dat> is 2 dimensional: 60000 X 784
        #dat = np.garray(load('mnist_cudaTest').T/255.) 
        for epoch in range(self.epochs):
            err = []
            epoch_start = datetime.now()
            for batch in range(self.num_batches):
                #batch_start = datetime.now()
                # positive phase
                v1 = self.dat[batch*self.batch_size : (batch + 1)*self.batch_size]
                h1 = self.visible_to_hidden(v1, binary=False) # Get probs

                # negative phase
                vk = self.k_gibbs_sample(v1, k=1)
                hk = self.visible_to_hidden(vk, binary=False) # Get probs

                # Update weights
                self.wu_vh = self.wu_vh * self.momentum + np.dot(v1.T, h1) - np.dot(vk.T, hk)
                self.wu_v = self.wu_v * self.momentum + v1.sum(0) - vk.sum(0)
                self.wu_h = self.wu_h * self.momentum + h1.sum(0) - hk.sum(0)
                # The weight updates are averaged across the data in the batch
                self.w_vh += self.wu_vh * (self.epsilon/self.batch_size) - self.lam * self.w_vh
                self.w_v += self.wu_v * (self.epsilon/self.batch_size)
                self.w_h += self.wu_h * (self.epsilon/self.batch_size)

                # calculate reconstruction error
                err.append((vk-v1)**2/(self.num_vis*self.batch_size))
                #print 'Epoch %s, batch %s duration: %s' % (epoch, batch, datetime.now() - batch_start)
            print 'Epoch %s duration %s' % (epoch, datetime.now() - epoch_start)
        print "Mean squared error: " + str(np.mean(err))

def pl(index):
    # can import training label data and use np.where(target == 1) to train on only 1s
    plt.subplot(211)
    plt.imshow((dat[index] > 0.5).astype(int).reshape(28,28), interpolation='nearest')
    plt.subplot(212)
    plt.imshow(r.hidden_to_visible(r.visible_to_hidden(dat[index]), binary=True).astype(int).reshape(28,28), interpolation='nearest')



if __name__ == '__main__':
    movies = ['harry potter', 'avatar', 'lotr', 'gladiator', 'titanic', 'glitter']
    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,1],[0,0,1,1,1,0],[1,0,0,0,0,1]])
    
    r = RBM(training_data, epsilon=0.1, lam=0.006, momentum=0.9, batch_size=7, num_hid=2, epochs=2000)
    r.train()
    print r.w_vh, r.w_v, r.w_h
    user = np.array([[0,0,0,1,1,0]])
    print zip(r.hidden_to_visible(r.visible_to_hidden(user), binary=False).flatten(), movies)
    
    ## MNIST TESTING ##
    # data is already the right shape and between 0 and 1
    dat = np.load('mnist/train.npy')


