

import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from sklearn import preprocessing

dat = pd.read_csv('/home/jdaniels/pydev/src/data_science/wholesale_customers_data.csv')
scaler = preprocessing.MinMaxScaler()

x = dat[dat.columns[dat.columns != 'Grocery']].values.astype('float')
y = dat.Grocery.values.astype('float')

scalerX.fit_transform(x)
scalerY.fit_transform(y)




