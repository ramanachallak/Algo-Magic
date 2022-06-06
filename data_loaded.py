import numpy as np
import pandas as pd

import hvplot.pandas
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#from pandas.plotting import lag_plot
# from datetime import datetime

from sklearn.metrics import mean_squared_error
from pandas_datareader import data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

window_size = 5

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)


# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window ):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)