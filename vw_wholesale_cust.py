
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
    enc = preprocessing.OneHotEncoder(categorical_features=[1])

    # want to predict Delicassen, real-valued quantity (spelled wrong in data)
    data = dat.values.astype('float')
    #y = dat.Channel.values.astype('float')

    # Normalize only the integer features, hstack back on the categorical features (columns 0 and 1)
    data = np.hstack((data[:,0:2], scalerx.fit_transform(data[:,2:])))
    # Now encode the categorical columns using the OneHotEncoder, configured to only encode categoricals
    data = enc.fit_transform(data).toarray()
    
    for row in data:
        print '%s | %s' %(str(1 if row[3] == 1.0 else -1), ' '.join([str(val) for val in np.concatenate((row[0:3], row[4:]))]))
        # Below - comment out enc.fit_tranform line above - tests not one-hot encoding the categorical variable
        #print '%s | %s' %(str(1 if row[0] == 1.0 else -1), ' '.join([str(val) for val in row[1:]]))
    
def main():
    preprocess_data()

if __name__ == '__main__':
    main()

