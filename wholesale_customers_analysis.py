
import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn import cross_validation
import sklearn.metrics as metrics

import neuralnet_testbed as nn

dat = pd.read_csv('/home/jdaniels/pydev/src/data_science/wholesale_customers_data.csv')
# Used to scale the integer features
scalerx = preprocessing.StandardScaler()
scalery = preprocessing.StandardScaler()
# First two columns are categorical and need to be one-hot encoded
enc = preprocessing.OneHotEncoder(categorical_features=[0,1])

# want to predict Delicassen, real-valued quantity (spelled wrong in data)
x = dat[dat.columns.drop(['Delicassen'])].values.astype('float')
y = dat.Delicassen.values.astype('float')

# Normalize only the integer features, hstack back on the categorical features (columns 0 and 1)
x = np.hstack((x[:,0:2], scalerx.fit_transform(x[:,2:])))
# Now encode the categorical columns using the OneHotEncoder, configured to only encode categoricals
x = enc.fit_transform(x).toarray()
# Normalize the targets since they're numbers - regression
y = scalery.fit_transform(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=0)

# Train NN
nn.initialize(X_train)
nn.train(X_train, y_train, alpha=0.5, lam=0.0001)

nn_training_error = metrics.mean_squared_error(y_train, nn.predict(X_train).reshape(X_train.shape[0],))
nn_test_error = metrics.mean_squared_error(y_test, nn.predict(X_test).reshape(X_test.shape[0],))

print 'NN'
print nn_training_error, nn_test_error

# Test accuracy against a sklearn sgd algorithm.
clf = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=10, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, rho=None)
clf.fit(X_train, y_train)

training_error = metrics.mean_squared_error(y_train, clf.predict(X_train))
test_error = metrics.mean_squared_error(y_test, clf.predict(X_test))

print 'SGD'
print training_error, test_error


# Next things to try:
# Classification - pick one if the categorical variables. 0 is a binary, 1 is a multi - softmax??
# One against all
# Abstract the kind of output unit - make it configurable. Linear, Binary Sigmoid, Softmax

