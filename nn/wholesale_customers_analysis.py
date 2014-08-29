
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
    # Data columns: Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen
    # Channel is categorical, 2 levels. Region is categorical

    # Used to scale the integer features
    scalerx = preprocessing.StandardScaler()
    scalery = preprocessing.StandardScaler()
    # First two columns are categorical and need to be one-hot encoded
    # Can do this directly in pandas with get_dummies
    #enc = preprocessing.OneHotEncoder(categorical_features=[0])

    #x = dat[dat.columns.drop(['Region'])].values.astype('float')
    x = pd.concat([pd.get_dummies(dat.Channel), dat.drop(['Channel', 'Region'], axis=1)], axis=1).values
    y = pd.get_dummies(dat.Region).values

    # Normalize only the integer features, hstack back on the categorical features (column 0)
    x = np.hstack((x[:,:2], scalerx.fit_transform(x[:,2:])))
    # Now encode the categorical columns using the OneHotEncoder, configured to only encode categoricals
    #x = enc.fit_transform(x).toarray()

    #y = preprocessing.Binarizer(threshold=1.5).transform(y)

    # Set up train, test folds
    ss = ShuffleSplit(len(y), n_iter=1, test_size=0.15)

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
        #print 'NN pre-training auc score: %f' %metrics.roc_auc_score(y[train], nn.predict(x[train]).T)
        
        nn.train(x[train], y[train], passes=500, alpha=0.7, lam=0.0)

        cat=1
        nn_training_auc = metrics.roc_auc_score(y[train][:,cat], nn.predict(x[train]).T[:,cat])
        nn_test_auc = metrics.roc_auc_score(y[test][:,cat], nn.predict(x[test]).T[:,cat])
        nn_training_error = metrics.f1_score(y[train][:,cat], preprocessing.Binarizer(threshold=0.5).transform(nn.predict(x[train])).T[:,cat])
        nn_test_error = metrics.f1_score(y[test][:,cat], preprocessing.Binarizer(threshold=0.5).transform(nn.predict(x[test])).T[:,cat])

        #nn_training_error += metrics.mean_absolute_error(y[train], nn.predict(x[train]).reshape(x[train].shape[0],))
        #nn_test_error += metrics.mean_absolute_error(y[test], nn.predict(x[test]).reshape(x[test].shape[0],))



        print 'NN F1: (Training) %f, (Test) %f' %(nn_training_error, nn_test_error)
        print 'NN AUC: (Training) %f, (Test) %f' %(nn_training_auc, nn_test_auc)
        #plot_roc(y[test][:,cat], nn.predict(x[test]).T[:,cat])

# The nn AUC pretty much sucks on this data set as is. I think due to squared loss. F1 is ok.
# Good time to try AUC loss as the data set is quite unbalanced. Can't really do this with multiple outputs?
# Very good example of how AUC can suck even though F1 is ok.



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
    main()


# Next things to try:
# Classification - pick one if the categorical variables. 0 is a binary, 1 is a multi - softmax??
# Breakout regression and classification examples - avoid all the commenting.
# One against all
# Abstract the kind of output unit - make it configurable. Linear, Binary Sigmoid, Softmax

