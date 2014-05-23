

import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from sklearn import preprocessing

dat = pd.read_csv('/home/jdaniels/pydev/src/data_science/wholesale_customers_data.csv')
scalerx = preprocessing.StandardScaler()
scalery = preprocessing.StandardScaler()

#x = dat[dat.columns.drop(['Grocery', 'Channel', 'Region'])].values.astype('float')
x = dat.Region.values.astype('float')
y = dat.Grocery.values.astype('float')

x = scalerx.fit_transform(x)
y = scalery.fit_transform(y)


