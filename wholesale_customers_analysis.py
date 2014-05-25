
import pylab as pl
import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.cross_validation import ShuffleSplit
import sklearn.metrics as metrics

import neuralnet_testbed as nn

x = None
y = None
ss = None # Test/train shuffle split

def preprocess_data():
    global x, y, ss

    dat = pd.read_csv('wholesale_customers_data.csv')
    # Used to scale the integer features
    scalerx = preprocessing.StandardScaler()
    scalery = preprocessing.StandardScaler()
    # First two columns are categorical and need to be one-hot encoded
    enc = preprocessing.OneHotEncoder(categorical_features=[0])

    # want to predict Delicassen, real-valued quantity (spelled wrong in data)
    x = dat[dat.columns.drop(['Channel'])].values.astype('float')
    y = dat.Channel.values.astype('float')

    # Normalize only the integer features, hstack back on the categorical features (columns 0 and 1)
    x = np.hstack((x[:,0:1], scalerx.fit_transform(x[:,1:])))
    # Now encode the categorical columns using the OneHotEncoder, configured to only encode categoricals
    x = enc.fit_transform(x).toarray()

    y = preprocessing.Binarizer(threshold=1.5).transform(y)

    # Set up train, test folds
    ss = ShuffleSplit(len(y), n_iter=1)

def train_and_evaluate():
    nn_training_error = 0
    nn_test_error = 0
    training_error = 0
    test_error = 0

    for train, test in ss:
        # Train NN
        nn.initialize(x[train])
        #print 'NN pre-training train error: %f' % metrics.mean_absolute_error(y[train], nn.predict(x[train]).reshape(x[train].shape[0],))
        #print 'NN pre-training f1 score: %f' %metrics.f1_score(y[train], preprocessing.Binarizer(threshold=0.5).transform(nn.predict(x[train])).T)
        print 'NN pre-training auc score: %f' %metrics.roc_auc_score(y[train], nn.predict(x[train]).T)
        
        nn.train(x[train], y[train], passes=1000, alpha=2.0, lam=0.001)

        nn_training_auc = metrics.roc_auc_score(y[train], nn.predict(x[train]).T)
        nn_test_auc = metrics.roc_auc_score(y[test], nn.predict(x[test]).T)
        #nn_training_error = metrics.f1_score(y[train], preprocessing.Binarizer(threshold=0.5).transform(nn.predict(x[train])).T)
        #nn_test_error = metrics.f1_score(y[test], preprocessing.Binarizer(threshold=0.5).transform(nn.predict(x[test])).T)

        #nn_training_error += metrics.mean_absolute_error(y[train], nn.predict(x[train]).reshape(x[train].shape[0],))
        #nn_test_error += metrics.mean_absolute_error(y[test], nn.predict(x[test]).reshape(x[test].shape[0],))



        #print 'NN: %f, %f' %(nn_training_error, nn_test_error)
        print 'NN: %f, %f' %(nn_training_auc, nn_test_auc)
        plot_roc(y[test], nn.predict(x[test]).T)

        # Test accuracy against a sklearn sgd algorithm.
    #    clf = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.001, l1_ratio=0.15, fit_intercept=True, n_iter=10, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, rho=None)
    #    clf.fit(x[train], y[train])

    #    training_error += metrics.mean_absolute_error(y[train], clf.predict(x[train]))
    #    test_error += metrics.mean_absolute_error(y[test], clf.predict(x[test]))

        #print 'SGD: %f, %f\n' %(training_error, test_error)


    #print 'NN: %f, %f' %(nn_training_auc, nn_test_auc)
    #print 'SGD: %f, %f\n' %(training_error, test_error)

def plot_roc(y_true, y_predictions):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predictions)
    roc_auc = metrics.auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='Test ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.show()


def main():
    preprocess_data()
    train_and_evaluate()

if __name__ == '__main__':
    print 'main is called'
    main()


# Next things to try:
# Classification - pick one if the categorical variables. 0 is a binary, 1 is a multi - softmax??
# Breakout regression and classification examples - avoid all the commenting.
# One against all
# Abstract the kind of output unit - make it configurable. Linear, Binary Sigmoid, Softmax

